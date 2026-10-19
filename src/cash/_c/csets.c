/*
 * cash._cwatch, part 6: a stored set iterates in the order the original did.
 *
 * A set iterates in the order of its hash table's slots, and where an element
 * sits depends on what was in its way when it went in: the elements before it,
 * the deleted ones whose markers ("dummies") are still there, the table's size
 * then. A set rebuilt from its elements -- by pickle, by a copy -- puts them in
 * again one after the other and lands many in other slots, so a restored set
 * of ints iterated in another order than the run that stored it, and
 * `{x: i for i, x in enumerate(s)}` numbered its elements differently.
 *
 * `set_layouts(value)` records, for each set and frozenset in *value*, where
 * its elements and dummies sit; `apply_set_layouts(value, layouts)` puts the
 * restored sets' elements back in those slots. The pickle is not touched.
 *
 * Elements are matched by hash: the layout lists the slots in the order of
 * the elements' hashes, sorted, and a restored set's elements, sorted by their
 * hashes in THIS process, take them in that order. The layout also holds a
 * check over the sorted hashes; a set whose hashes are not those (strings
 * under another PYTHONHASHSEED) keeps the order its rebuild gave it. Every
 * table written this way is one CPython built: the same hashes, in the same
 * slots, with the same dummies -- so every lookup probes as it did there.
 * Two elements with one hash share a probe path, and may trade places.
 *
 * Only exact lists, tuples, dicts, sets and frozensets are walked, depth
 * first, each object once; sets are counted in that order, and a restored set
 * is laid out before its own elements are walked, so a frozenset inside a set
 * is met in the same place in both walks. Nothing in Python runs while a walk
 * reads or writes: the garbage collector is paused, and only C memory is
 * allocated until the walk is done.
 *
 * Layout record, little-endian: ordinal u32, used u32, mask u64, dummies u32,
 * check u64, then a u32 slot per element and a u32 slot per dummy.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>

#include "cwalk.h"

#ifndef Py_GIL_DISABLED

#define MIN_USED 2
#define HEAD_BYTES (4 + 4 + 8 + 4 + 8)

typedef struct {
    PyObject **items;
    Py_ssize_t n, cap;
} Stack;

typedef struct {
    uintptr_t *slots;
    size_t mask, n;
} Seen;

typedef struct {
    char *data;
    size_t n, cap;
} Buf;

typedef struct {
    Py_hash_t hash;
    PyObject *key;
    uint32_t slot;
} Item;

static int stack_push(Stack *s, PyObject *o) {
    if (s->n == s->cap) {
        Py_ssize_t cap = s->cap ? s->cap * 2 : 256;
        PyObject **items = PyMem_Realloc(s->items, (size_t)cap * sizeof *items);
        if (!items) return -1;
        s->items = items;
        s->cap = cap;
    }
    s->items[s->n++] = o;
    return 0;
}

static uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

/* 1 when *o* was not seen before (and is now), 0 when it was, -1 on no memory. */
static int seen_add(Seen *s, PyObject *o) {
    if ((s->n + 1) * 2 > s->mask + 1) {
        size_t cap = s->mask ? (s->mask + 1) * 2 : 1024;
        uintptr_t *slots = PyMem_Calloc(cap, sizeof *slots);
        if (!slots) return -1;
        for (size_t i = 0; s->slots && i <= s->mask; i++) {
            uintptr_t p = s->slots[i];
            if (!p) continue;
            size_t j = (size_t)mix64(p) & (cap - 1);
            while (slots[j]) j = (j + 1) & (cap - 1);
            slots[j] = p;
        }
        PyMem_Free(s->slots);
        s->slots = slots;
        s->mask = cap - 1;
    }
    uintptr_t p = (uintptr_t)o;
    size_t j = (size_t)mix64(p) & s->mask;
    while (s->slots[j]) {
        if (s->slots[j] == p) return 0;
        j = (j + 1) & s->mask;
    }
    s->slots[j] = p;
    s->n++;
    return 1;
}

static int buf_put(Buf *b, const void *data, size_t n) {
    if (b->n + n > b->cap) {
        size_t cap = b->cap ? b->cap : 4096;
        while (cap < b->n + n) cap *= 2;
        char *d = PyMem_Realloc(b->data, cap);
        if (!d) return -1;
        b->data = d;
        b->cap = cap;
    }
    memcpy(b->data + b->n, data, n);
    b->n += n;
    return 0;
}

static int put_u32(Buf *b, uint32_t v) {
    unsigned char c[4] = {(unsigned char)v, (unsigned char)(v >> 8), (unsigned char)(v >> 16), (unsigned char)(v >> 24)};
    return buf_put(b, c, 4);
}

static int put_u64(Buf *b, uint64_t v) {
    if (put_u32(b, (uint32_t)v) < 0) return -1;
    return put_u32(b, (uint32_t)(v >> 32));
}

static uint32_t get_u32(const unsigned char *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

static uint64_t get_u64(const unsigned char *p) { return (uint64_t)get_u32(p) | (uint64_t)get_u32(p + 4) << 32; }

/* Sort by hash, then by slot (so equal hashes keep the table's order): LSD
 * radix on the hash, as unsigned, 16 bits a pass; the slot order is the
 * input order, which radix passes keep. */
static int sort_items(Item *items, size_t n) {
    if (n < 2) return 0;
    Item *tmp = PyMem_Malloc(n * sizeof *items);
    size_t *count = PyMem_Calloc(4 * 65536, sizeof *count);
    if (!tmp || !count) {
        PyMem_Free(tmp);
        PyMem_Free(count);
        return -1;
    }
    /* The four passes' digit counts, from one read of the hashes. */
    for (size_t i = 0; i < n; i++) {
        uint64_t h = (uint64_t)items[i].hash;
        count[h & 0xffff]++;
        count[65536 + ((h >> 16) & 0xffff)]++;
        count[2 * 65536 + ((h >> 32) & 0xffff)]++;
        count[3 * 65536 + (h >> 48)]++;
    }
    Item *src = items, *dst = tmp;
    for (int pass = 0; pass < 4; pass++) {
        int shift = pass * 16;
        size_t *c = count + (size_t)pass * 65536;
        if (c[((uint64_t)src[0].hash >> shift) & 0xffff] == n) continue; /* one digit for all: in order already */
        size_t sum = 0;
        for (size_t d = 0; d < 65536; d++) {
            size_t k = c[d];
            c[d] = sum;
            sum += k;
        }
        for (size_t i = 0; i < n; i++) dst[c[((uint64_t)src[i].hash >> shift) & 0xffff]++] = src[i];
        Item *t = src;
        src = dst;
        dst = t;
    }
    if (src != items) memcpy(items, src, n * sizeof *items);
    PyMem_Free(tmp);
    PyMem_Free(count);
    return 0;
}

static uint64_t hashes_check(const Item *items, size_t n) {
    uint64_t h = 0x243f6a8885a308d3ULL ^ (uint64_t)n;
    for (size_t i = 0; i < n; i++) h = mix64(h ^ (uint64_t)items[i].hash) + 0x9e3779b97f4a7c15ULL;
    return h;
}

/* The set's elements with their hashes and slots, sorted by hash; NULL with
 * *n* 0 for none, NULL with an exception on no memory. *dummies* (when given)
 * gets the dummy slots. */
static Item *read_table(PySetObject *so, PyObject *dummy, size_t *n, uint32_t **dummies, size_t *ndummies) {
    size_t used = (size_t)so->used, k = 0, d = 0;
    Item *items = PyMem_Malloc((used ? used : 1) * sizeof *items);
    uint32_t *ds = dummies ? PyMem_Malloc(((size_t)(so->fill - so->used) + 1) * sizeof *ds) : NULL;
    if (!items || (dummies && !ds)) {
        PyMem_Free(items);
        PyMem_Free(ds);
        PyErr_NoMemory();
        return NULL;
    }
    for (size_t i = 0; i <= (size_t)so->mask; i++) {
        PyObject *key = so->table[i].key;
        if (!key) continue;
        if (key == dummy) {
            if (ds && d < (size_t)(so->fill - so->used)) ds[d++] = (uint32_t)i;
            continue;
        }
        if (k == used) {
            k++; /* more elements than it counts: refused below */
            break;
        }
        items[k].hash = so->table[i].hash;
        items[k].key = key;
        items[k].slot = (uint32_t)i;
        k++;
    }
    if (k != used || sort_items(items, k) < 0) {
        PyMem_Free(items);
        PyMem_Free(ds);
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, "a set's table does not hold what it says");
        return NULL;
    }
    *n = k;
    if (dummies) {
        *dummies = ds;
        *ndummies = d;
    }
    return items;
}

static int is_set(PyObject *o) { return PySet_CheckExact(o) || PyFrozenSet_CheckExact(o); }

static int walked(PyObject *o) {
    return PyList_CheckExact(o) || PyTuple_CheckExact(o) || PyDict_CheckExact(o) || is_set(o);
}

/* Whether walking *o* can meet a set: a tuple of numbers and strings (a
 * million of them in a dict's values, say) is passed over without a look in
 * the table of objects seen. */
static int worth(PyObject *o) {
    if (!PyTuple_CheckExact(o)) return walked(o);
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(o); i++)
        if (walked(PyTuple_GET_ITEM(o, i))) return 1;
    return 0;
}

/* Push *o*'s children that can hold a set, last first, so they are walked
 * in order. */
static int push_children(Stack *st, PyObject *o) {
    if (PyList_CheckExact(o) || PyTuple_CheckExact(o)) {
        Py_ssize_t n = PySequence_Fast_GET_SIZE(o);
        PyObject **items = PySequence_Fast_ITEMS(o);
        for (Py_ssize_t i = n - 1; i >= 0; i--)
            if (worth(items[i]) && stack_push(st, items[i]) < 0) return -1;
    } else if (PyDict_CheckExact(o)) {
        Py_ssize_t n = PyDict_GET_SIZE(o), pos = 0, i = 0;
        PyObject **pairs = PyMem_Malloc(((size_t)n * 2 + 1) * sizeof *pairs);
        /* At most 2n go in: a dict that grows while read cannot overrun it. */
        if (!pairs) return -1;
        PyObject *k, *v;
        while (PyDict_Next(o, &pos, &k, &v)) {
            if (i < 2 * n && worth(k)) pairs[i++] = k;
            if (i < 2 * n && worth(v)) pairs[i++] = v;
        }
        for (Py_ssize_t j = i - 1; j >= 0; j--)
            if (stack_push(st, pairs[j]) < 0) {
                PyMem_Free(pairs);
                return -1;
            }
        PyMem_Free(pairs);
    } else if (is_set(o)) {
        PySetObject *so = (PySetObject *)o;
        PyObject *dummy = cwatch_set_dummy();
        Py_ssize_t m = 0;
        PyObject **keys = PyMem_Malloc(((size_t)so->used + 1) * sizeof *keys);
        if (!keys) return -1;
        for (Py_ssize_t i = 0; i <= so->mask && m < so->used; i++) {
            PyObject *key = so->table[i].key;
            if (key && key != dummy && worth(key)) keys[m++] = key;
        }
        for (Py_ssize_t j = m - 1; j >= 0; j--)
            if (stack_push(st, keys[j]) < 0) {
                PyMem_Free(keys);
                return -1;
            }
        PyMem_Free(keys);
    }
    return 0;
}


/* set_layouts(value, max_nodes) -> bytes | None */
static PyObject *py_set_layouts(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "set_layouts(value, max_nodes)");
        return NULL;
    }
    Py_ssize_t budget = PyLong_AsSsize_t(args[1]);
    if (budget == -1 && PyErr_Occurred()) return NULL;
    PyObject *dummy = cwatch_set_dummy();
    Stack st = {0};
    Seen seen = {0};
    Buf out = {0};
    PyObject *result = NULL;
    uint32_t ordinal = 0;
    int was = PyGC_Disable(), failed = 0, over = 0;
    if (stack_push(&st, args[0]) < 0) failed = 1;
    while (!failed && st.n) {
        PyObject *o = st.items[--st.n];
        if (!walked(o)) continue;
        int fresh = seen_add(&seen, o);
        if (fresh < 0) { failed = 1; break; }
        if (!fresh) continue;
        if (--budget < 0) { over = 1; break; }
        if (is_set(o)) {
            PySetObject *so = (PySetObject *)o;
            uint32_t here = ordinal++;
            if (so->used >= MIN_USED && (uint64_t)so->mask < UINT32_MAX) {
                size_t n = 0, nd = 0;
                uint32_t *ds = NULL;
                Item *items = read_table(so, dummy, &n, &ds, &nd);
                if (!items) { failed = 1; break; }
                int bad = put_u32(&out, here) < 0 || put_u32(&out, (uint32_t)n) < 0 ||
                          put_u64(&out, (uint64_t)so->mask) < 0 || put_u32(&out, (uint32_t)nd) < 0 ||
                          put_u64(&out, hashes_check(items, n)) < 0;
                for (size_t i = 0; !bad && i < n; i++) bad = put_u32(&out, items[i].slot) < 0;
                for (size_t i = 0; !bad && i < nd; i++) bad = put_u32(&out, ds[i]) < 0;
                PyMem_Free(items);
                PyMem_Free(ds);
                if (bad) { failed = 1; break; }
            }
        }
        if (push_children(&st, o) < 0) { failed = 1; break; }
    }
    if (was) PyGC_Enable();
    if (failed) {
        if (!PyErr_Occurred()) PyErr_NoMemory();
    } else if (over || out.n == 0) {
        result = Py_NewRef(Py_None);
    } else {
        result = PyBytes_FromStringAndSize(out.data, (Py_ssize_t)out.n);
    }
    PyMem_Free(st.items);
    PyMem_Free(seen.slots);
    PyMem_Free(out.data);
    return result;
}

/* Lay *so* out as the record at *p* says, when its hashes are the recorded
 * ones: 1 laid out, 0 left as it is, -1 on no memory. */
static int lay_out(PySetObject *so, const unsigned char *p, uint32_t used, uint64_t mask, uint32_t nd, uint64_t check) {
    PyObject *dummy = cwatch_set_dummy();
    if ((uint64_t)so->used != used) return 0;
    size_t n = 0;
    Item *items = read_table(so, dummy, &n, NULL, NULL);
    if (!items) return -1;
    const unsigned char *slots = p + HEAD_BYTES, *dslots = slots + 4 * (size_t)used;
    int result = 0;
    if (hashes_check(items, n) != check) goto done; /* other hashes here: the rebuild's order stands */
    size_t size = (size_t)mask + 1;
    int same = (uint64_t)so->mask == mask && (size_t)(so->fill - so->used) == nd;
    for (size_t i = 0; same && i < n; i++) same = items[i].slot == get_u32(slots + 4 * i);
    if (same) goto done; /* already laid out so: the original itself, or a copy that kept it */
    for (size_t i = 0; i < n; i++)
        if (get_u32(slots + 4 * i) >= size) goto done;
    for (size_t i = 0; i < nd; i++)
        if (get_u32(dslots + 4 * i) >= size) goto done;
    setentry *table = size <= PySet_MINSIZE ? NULL : PyMem_Calloc(size, sizeof(setentry));
    if (size > PySet_MINSIZE && !table) {
        result = -1;
        goto done;
    }
    if (!table) {
        table = so->smalltable;
        if (so->table != so->smalltable) PyMem_Free(so->table);
        memset(table, 0, sizeof(so->smalltable));
    } else if (so->table != so->smalltable) {
        PyMem_Free(so->table);
    } else {
        memset(so->smalltable, 0, sizeof(so->smalltable));
    }
    /* The keys' references move from the old table to the new one. */
    for (size_t i = 0; i < n; i++) {
        setentry *e = &table[get_u32(slots + 4 * i)];
        e->key = items[i].key;
        e->hash = items[i].hash;
    }
    for (size_t i = 0; i < nd; i++) {
        setentry *e = &table[get_u32(dslots + 4 * i)];
        e->key = dummy;
        e->hash = -1;
    }
    so->table = table;
    so->mask = (Py_ssize_t)mask;
    so->fill = (Py_ssize_t)(n + nd);
    so->used = (Py_ssize_t)n;
    so->finger = 0;
    result = 1;
done:
    PyMem_Free(items);
    return result;
}

/* apply_set_layouts(value, layouts, max_nodes) -> (laid out, left as they were) */
static PyObject *py_apply_set_layouts(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 3 || !PyBytes_Check(args[1])) {
        PyErr_SetString(PyExc_TypeError, "apply_set_layouts(value, layouts: bytes, max_nodes)");
        return NULL;
    }
    Py_ssize_t budget = PyLong_AsSsize_t(args[2]);
    if (budget == -1 && PyErr_Occurred()) return NULL;
    const unsigned char *p = (const unsigned char *)PyBytes_AS_STRING(args[1]);
    const unsigned char *end = p + PyBytes_GET_SIZE(args[1]);
    Stack st = {0};
    Seen seen = {0};
    uint32_t ordinal = 0;
    Py_ssize_t laid = 0, left = 0;
    int was = PyGC_Disable(), failed = 0;
    if (stack_push(&st, args[0]) < 0) failed = 1;
    while (!failed && st.n && p < end) {
        PyObject *o = st.items[--st.n];
        if (!walked(o)) continue;
        int fresh = seen_add(&seen, o);
        if (fresh < 0) { failed = 1; break; }
        if (!fresh) continue;
        if (--budget < 0) break;
        if (is_set(o)) {
            uint32_t here = ordinal++;
            if (end - p < HEAD_BYTES) break;
            if (get_u32(p) == here) {
                uint32_t used = get_u32(p + 4), nd = get_u32(p + 16);
                uint64_t mask = get_u64(p + 8), check = get_u64(p + 20);
                size_t len = HEAD_BYTES + 4 * ((size_t)used + nd);
                if ((size_t)(end - p) < len) break;
                int r = lay_out((PySetObject *)o, p, used, mask, nd, check);
                if (r < 0) { failed = 1; break; }
                if (r) laid++; else left++;
                p += len;
            }
        }
        if (push_children(&st, o) < 0) { failed = 1; break; }
    }
    if (was) PyGC_Enable();
    PyMem_Free(st.items);
    PyMem_Free(seen.slots);
    if (failed) {
        if (!PyErr_Occurred()) PyErr_NoMemory();
        return NULL;
    }
    return Py_BuildValue("(nn)", laid, left);
}

static PyMethodDef csets_methods[] = {
    {"set_layouts", (PyCFunction)(void (*)(void))py_set_layouts, METH_FASTCALL,
     "set_layouts(value, max_nodes) -> bytes | None"},
    {"apply_set_layouts", (PyCFunction)(void (*)(void))py_apply_set_layouts, METH_FASTCALL,
     "apply_set_layouts(value, layouts, max_nodes) -> (laid out, left)"},
    {NULL, NULL, 0, NULL}};

int csets_add_functions(PyObject *m) { return PyModule_AddFunctions(m, csets_methods); }

#else

int csets_add_functions(PyObject *m) { return 0; }

#endif
