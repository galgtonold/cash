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
 * first, each object once (an object only one reference holds is met once
 * anyway, and is not looked up). Each set is recorded with its path from the
 * value: the index of each step in a list or tuple, 2k or 2k+1 for the k-th
 * key or value of a dict, k for a set's k-th element in slot order. A hit
 * follows the paths, not the whole value, and a set is laid out before the
 * paths through it are followed. Nothing in Python runs while a walk reads or
 * writes: the garbage collector is paused, and only C memory is allocated
 * until the walk is done.
 *
 * A record is checked before a table is touched: a checksum over all of it,
 * then for each set a table size that is a power of two, room for an empty
 * slot, distinct slots, and every element on its hash's probe path with each
 * slot before it on the path taken -- a table a lookup can use.
 *
 * Layouts, little-endian: magic u32, then per set: depth u32, depth u32 path
 * steps, used u32, mask u64, dummies u32, check u64, a u32 slot per element
 * and a u32 slot per dummy; then a u64 checksum of everything before it.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>

#include "cwalk.h"

#ifndef Py_GIL_DISABLED

#define MIN_USED 2
#define MAGIC 0x324c5343u /* "CSL2" */
#define SET_BYTES (4 + 8 + 4 + 8)
/* CPython's set probing (Objects/setobject.c, 3.12 to 3.14). */
#define LINEAR_PROBES 9
#define PERTURB_SHIFT 5

typedef struct {
    PyObject *o;
    uint32_t depth, idx;
} Frame;

typedef struct {
    Frame *items;
    size_t n, cap;
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

static int stack_push(Stack *s, PyObject *o, uint32_t depth, uint32_t idx) {
    if (s->n == s->cap) {
        size_t cap = s->cap ? s->cap * 2 : 256;
        Frame *items = PyMem_Realloc(s->items, cap * sizeof *items);
        if (!items) return -1;
        s->items = items;
        s->cap = cap;
    }
    s->items[s->n].o = o;
    s->items[s->n].depth = depth;
    s->items[s->n].idx = idx;
    s->n++;
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

static int seen_has(const Seen *s, PyObject *o) {
    if (!s->n) return 0;
    uintptr_t p = (uintptr_t)o;
    size_t j = (size_t)mix64(p) & s->mask;
    while (s->slots[j]) {
        if (s->slots[j] == p) return 1;
        j = (j + 1) & s->mask;
    }
    return 0;
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

/* The checksum the layouts end with, over their first *n* bytes. */
static uint64_t blob_check(const unsigned char *p, size_t n) {
    uint64_t h = 0x6a09e667f3bcc908ULL ^ (uint64_t)n;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) h = mix64(h ^ get_u64(p + i)) + 0x9e3779b97f4a7c15ULL;
    uint64_t tail = 0;
    for (size_t k = 0; i < n; i++, k++) tail |= (uint64_t)p[i] << (8 * k);
    return mix64(h ^ tail);
}

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

/* A path step is a u32: containers too big for one are not walked into. */
#define MAX_STEPS ((Py_ssize_t)(UINT32_MAX / 2))

/* Push *o*'s children that can hold a set, last first, so they are walked
 * in order; each with its step from *o* (see the top of this file). */
static int push_children(Stack *st, PyObject *o, uint32_t depth) {
    uint32_t d = depth + 1;
    if (d == UINT32_MAX) return 0;
    if (PyList_CheckExact(o) || PyTuple_CheckExact(o)) {
        Py_ssize_t n = PySequence_Fast_GET_SIZE(o);
        if (n > MAX_STEPS) return 0;
        PyObject **items = PySequence_Fast_ITEMS(o);
        for (Py_ssize_t i = n - 1; i >= 0; i--)
            if (worth(items[i]) && stack_push(st, items[i], d, (uint32_t)i) < 0) return -1;
    } else if (PyDict_CheckExact(o)) {
        Py_ssize_t n = PyDict_GET_SIZE(o), pos = 0, i = 0;
        if (n > MAX_STEPS / 2) return 0;
        /* A row of a few fields needs no allocation. */
        Frame local[64], *pairs = (size_t)n * 2 <= 64 ? local : PyMem_Malloc((size_t)n * 2 * sizeof *pairs);
        if (!pairs) return -1;
        PyObject *k, *v;
        uint32_t entry = 0;
        /* At most 2n go in: a dict that grows while read cannot overrun it. */
        while (PyDict_Next(o, &pos, &k, &v) && entry < (uint32_t)n) {
            if (worth(k)) {
                pairs[i].o = k;
                pairs[i++].idx = 2 * entry;
            }
            if (worth(v)) {
                pairs[i].o = v;
                pairs[i++].idx = 2 * entry + 1;
            }
            entry++;
        }
        int bad = 0;
        for (Py_ssize_t j = i - 1; j >= 0 && !bad; j--) bad = stack_push(st, pairs[j].o, d, pairs[j].idx) < 0;
        if (pairs != local) PyMem_Free(pairs);
        if (bad) return -1;
    } else if (is_set(o)) {
        PySetObject *so = (PySetObject *)o;
        PyObject *dummy = cwatch_set_dummy();
        if (so->used > MAX_STEPS) return 0;
        Py_ssize_t m = 0;
        uint32_t k = 0;
        Frame *keys = PyMem_Malloc(((size_t)so->used + 1) * sizeof *keys);
        if (!keys) return -1;
        for (Py_ssize_t i = 0; i <= so->mask && (Py_ssize_t)k < so->used; i++) {
            PyObject *key = so->table[i].key;
            if (!key || key == dummy) continue;
            if (worth(key)) {
                keys[m].o = key;
                keys[m++].idx = k;
            }
            k++;
        }
        int bad = 0;
        for (Py_ssize_t j = m - 1; j >= 0 && !bad; j--) bad = stack_push(st, keys[j].o, d, keys[j].idx) < 0;
        PyMem_Free(keys);
        if (bad) return -1;
    }
    return 0;
}

/* set_layouts(value, max_nodes) -> bytes, b"" when it holds no set to lay
 * out, None when it is bigger than *max_nodes* */
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
    uint32_t *path = NULL;
    size_t path_cap = 0, sets = 0;
    PyObject *result = NULL;
    int was = PyGC_Disable(), failed = 0, over = 0;
    if (put_u32(&out, MAGIC) < 0 || stack_push(&st, args[0], 0, 0) < 0) failed = 1;
    while (!failed && st.n) {
        Frame f = st.items[--st.n];
        PyObject *o = f.o;
        if (!walked(o)) continue;
        /* One reference: only the one container holding it reaches it. */
        int fresh = Py_REFCNT(o) == 1 ? 1 : seen_add(&seen, o);
        if (fresh < 0) { failed = 1; break; }
        if (!fresh) continue;
        if (--budget < 0) { over = 1; break; }
        if (f.depth) {
            if (f.depth > path_cap) {
                size_t cap = path_cap ? path_cap * 2 : 64;
                while (cap < f.depth) cap *= 2;
                uint32_t *p = PyMem_Realloc(path, cap * sizeof *p);
                if (!p) { failed = 1; break; }
                path = p;
                path_cap = cap;
            }
            path[f.depth - 1] = f.idx;
        }
        if (is_set(o)) {
            PySetObject *so = (PySetObject *)o;
            if (so->used >= MIN_USED && (uint64_t)so->mask < UINT32_MAX) {
                size_t n = 0, nd = 0;
                uint32_t *ds = NULL;
                Item *items = read_table(so, dummy, &n, &ds, &nd);
                if (!items) { failed = 1; break; }
                int bad = put_u32(&out, f.depth) < 0;
                for (uint32_t i = 0; !bad && i < f.depth; i++) bad = put_u32(&out, path[i]) < 0;
                bad = bad || put_u32(&out, (uint32_t)n) < 0 || put_u64(&out, (uint64_t)so->mask) < 0 ||
                      put_u32(&out, (uint32_t)nd) < 0 || put_u64(&out, hashes_check(items, n)) < 0;
                for (size_t i = 0; !bad && i < n; i++) bad = put_u32(&out, items[i].slot) < 0;
                for (size_t i = 0; !bad && i < nd; i++) bad = put_u32(&out, ds[i]) < 0;
                PyMem_Free(items);
                PyMem_Free(ds);
                if (bad) { failed = 1; break; }
                sets++;
            }
        }
        if (push_children(&st, o, f.depth) < 0) { failed = 1; break; }
    }
    if (was) PyGC_Enable();
    if (!failed && !over && sets && put_u64(&out, blob_check((const unsigned char *)out.data, out.n)) < 0) failed = 1;
    if (failed) {
        if (!PyErr_Occurred()) PyErr_NoMemory();
    } else if (over) {
        result = Py_NewRef(Py_None);
    } else if (!sets) {
        result = PyBytes_FromStringAndSize(NULL, 0);
    } else {
        result = PyBytes_FromStringAndSize(out.data, (Py_ssize_t)out.n);
    }
    PyMem_Free(st.items);
    PyMem_Free(seen.slots);
    PyMem_Free(out.data);
    PyMem_Free(path);
    return result;
}

/* Whether the slots a record gives are a table CPython could have built for
 * these elements: a power-of-two size with room for an empty slot, no slot
 * twice, and each element where a lookup of its hash finds it -- every slot
 * the lookup passes before it taken, so it does not stop short. */
static int table_is_sound(const Item *items, size_t n, uint64_t mask, size_t nd, const unsigned char *slots,
                          const unsigned char *dslots) {
    uint64_t size = mask + 1;
    if (size < PySet_MINSIZE || (size & mask) || mask >= UINT32_MAX) return 0;
    if ((uint64_t)n + nd >= size || size > 16 * ((uint64_t)n + nd) + 64) return 0;
    unsigned char *taken = PyMem_Calloc((size_t)size, 1);
    if (!taken) return -1;
    int ok = 1;
    for (size_t i = 0; ok && i < n + nd; i++) {
        uint32_t s = i < n ? get_u32(slots + 4 * i) : get_u32(dslots + 4 * (i - n));
        if (s > mask || taken[s]) ok = 0;
        else taken[s] = 1;
    }
    for (size_t i = 0; ok && i < n; i++) {
        size_t want = get_u32(slots + 4 * i), m = (size_t)mask;
        size_t perturb = (size_t)items[i].hash, at = (size_t)items[i].hash & m;
        /* Each turn looks at most LINEAR_PROBES + 1 slots; once perturb is
         * spent, at = 5 * at + 1 visits every slot within size turns. */
        uint64_t turns = 0, limit = 64 + size;
        for (;;) {
            size_t probes = at + LINEAR_PROBES <= m ? LINEAR_PROBES : 0, j = at;
            int found = 0, stop = 0;
            for (size_t k = 0; k <= probes; k++, j++) {
                if (j == want) { found = 1; break; }
                if (!taken[j]) { stop = 1; break; }
            }
            if (found) break;
            if (stop || ++turns > limit) { ok = 0; break; }
            perturb >>= PERTURB_SHIFT;
            at = (at * 5 + 1 + perturb) & m;
        }
    }
    PyMem_Free(taken);
    return ok;
}

enum { LEFT = 0, LAID = 1, IN_ORDER = 2 };

/* Lay *so* out as the record says, when its hashes are the recorded ones and
 * the record makes a sound table: LAID, IN_ORDER (laid out so already), LEFT
 * (the rebuild's order stands), or -1 on no memory. */
static int lay_out(PySetObject *so, uint32_t used, uint64_t mask, uint32_t nd, uint64_t check,
                   const unsigned char *slots) {
    PyObject *dummy = cwatch_set_dummy();
    if ((uint64_t)so->used != used) return LEFT;
    const unsigned char *dslots = slots + 4 * (size_t)used;
    size_t n = 0;
    Item *items = read_table(so, dummy, &n, NULL, NULL);
    if (!items) return -1;
    int result = LEFT;
    if (hashes_check(items, n) != check) goto done; /* other hashes here: the rebuild's order stands */
    int same = (uint64_t)so->mask == mask && (size_t)(so->fill - so->used) == nd;
    for (size_t i = 0; same && i < n; i++) same = items[i].slot == get_u32(slots + 4 * i);
    if (same) { /* the original itself, or a copy that kept it */
        result = IN_ORDER;
        goto done;
    }
    int sound = table_is_sound(items, n, mask, nd, slots, dslots);
    if (sound <= 0) {
        result = sound;
        goto done;
    }
    size_t size = (size_t)mask + 1;
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
    result = LAID;
done:
    PyMem_Free(items);
    return result;
}

/* Where a path last went in a dict or a set at one depth: the next step into
 * the same one, further on, carries on from there. */
typedef struct {
    PyObject *o, *key, *value;
    Py_ssize_t pos;
    int64_t k;
} Cursor;

static void cursor_start(Cursor *c, PyObject *o) {
    c->o = o;
    c->key = c->value = NULL;
    c->pos = 0;
    c->k = -1;
}

/* The object *path* leads to from *root*, or NULL: a step that is not there,
 * or one through a set left in the rebuild's order (*left*). */
static PyObject *follow(PyObject *root, const unsigned char *path, uint32_t depth, Cursor *cur, const Seen *left) {
    PyObject *o = root, *dummy = cwatch_set_dummy();
    for (uint32_t j = 0; j < depth; j++) {
        uint32_t t = get_u32(path + 4 * (size_t)j);
        Cursor *c = &cur[j];
        if (PyList_CheckExact(o) || PyTuple_CheckExact(o)) {
            if ((Py_ssize_t)t >= PySequence_Fast_GET_SIZE(o)) return NULL;
            o = PySequence_Fast_ITEMS(o)[t];
        } else if (PyDict_CheckExact(o)) {
            int64_t k = t >> 1;
            if (c->o != o || c->k > k) cursor_start(c, o);
            while (c->k < k) {
                if (!PyDict_Next(o, &c->pos, &c->key, &c->value)) return NULL;
                c->k++;
            }
            o = t & 1 ? c->value : c->key;
        } else if (is_set(o)) {
            if (seen_has(left, o)) return NULL;
            PySetObject *so = (PySetObject *)o;
            int64_t k = t;
            if (c->o != o || c->k > k) cursor_start(c, o);
            while (c->k < k) {
                while (c->pos <= so->mask && (!so->table[c->pos].key || so->table[c->pos].key == dummy)) c->pos++;
                if (c->pos > so->mask) return NULL;
                c->key = so->table[c->pos++].key;
                c->k++;
            }
            o = c->key;
        } else {
            return NULL;
        }
    }
    return o;
}

/* apply_set_layouts(value, layouts, max_depth) -> (laid out, in order already,
 * left as they were) */
static PyObject *py_apply_set_layouts(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 3 || !PyBytes_Check(args[1])) {
        PyErr_SetString(PyExc_TypeError, "apply_set_layouts(value, layouts: bytes, max_depth)");
        return NULL;
    }
    Py_ssize_t max_depth = PyLong_AsSsize_t(args[2]);
    if (max_depth == -1 && PyErr_Occurred()) return NULL;
    const unsigned char *p = (const unsigned char *)PyBytes_AS_STRING(args[1]);
    size_t total = (size_t)PyBytes_GET_SIZE(args[1]);
    if (total < 12 || get_u32(p) != MAGIC || blob_check(p, total - 8) != get_u64(p + total - 8)) {
        PyErr_SetString(PyExc_ValueError, "set layouts are damaged");
        return NULL;
    }
    const unsigned char *end = p + total - 8;
    p += 4;
    Cursor *cur = NULL;
    size_t cur_cap = 0;
    Seen left = {0};
    Py_ssize_t laid = 0, in_order = 0, kept = 0;
    int was = PyGC_Disable(), failed = 0, damaged = 0;
    while (p < end) {
        if (end - p < 4) { damaged = 1; break; }
        uint32_t depth = get_u32(p);
        if ((uint64_t)depth > (uint64_t)max_depth || (uint64_t)(end - p - 4) / 4 < depth) { damaged = 1; break; }
        const unsigned char *path = p + 4, *head = path + 4 * (size_t)depth;
        if ((size_t)(end - head) < SET_BYTES) { damaged = 1; break; }
        uint32_t used = get_u32(head), nd = get_u32(head + 12);
        uint64_t mask = get_u64(head + 4), check = get_u64(head + 16);
        if ((uint64_t)(end - head - SET_BYTES) / 4 < (uint64_t)used + nd) { damaged = 1; break; }
        const unsigned char *slots = head + SET_BYTES;
        p = slots + 4 * ((size_t)used + nd);
        if (depth > cur_cap) {
            Cursor *c = PyMem_Realloc(cur, depth * sizeof *c);
            if (!c) { failed = 1; break; }
            memset(c + cur_cap, 0, (depth - cur_cap) * sizeof *c);
            cur = c;
            cur_cap = depth;
        }
        PyObject *o = follow(args[0], path, depth, cur, &left);
        int r = o && is_set(o) ? lay_out((PySetObject *)o, used, mask, nd, check, slots) : LEFT;
        if (r < 0) { failed = 1; break; }
        if (r == LAID) {
            laid++;
            /* Its table is new: a cursor into it starts again. */
            for (size_t j = 0; j < cur_cap; j++)
                if (cur[j].o == o) cur[j].o = NULL;
        } else if (r == IN_ORDER) {
            in_order++;
        } else {
            kept++;
            /* What lies inside it was found by slot order there: not here. */
            if (o && is_set(o) && seen_add(&left, o) < 0) { failed = 1; break; }
        }
    }
    if (was) PyGC_Enable();
    PyMem_Free(cur);
    PyMem_Free(left.slots);
    if (failed) {
        if (!PyErr_Occurred()) PyErr_NoMemory();
        return NULL;
    }
    if (damaged) {
        PyErr_SetString(PyExc_ValueError, "set layouts are damaged");
        return NULL;
    }
    return Py_BuildValue("(nnn)", laid, in_order, kept);
}

static PyMethodDef csets_methods[] = {
    {"set_layouts", (PyCFunction)(void (*)(void))py_set_layouts, METH_FASTCALL,
     "set_layouts(value, max_nodes) -> bytes (empty: no set) | None (too big)"},
    {"apply_set_layouts", (PyCFunction)(void (*)(void))py_apply_set_layouts, METH_FASTCALL,
     "apply_set_layouts(value, layouts, max_depth) -> (laid out, in order already, left)"},
    {NULL, NULL, 0, NULL}};

int csets_add_functions(PyObject *m) { return PyModule_AddFunctions(m, csets_methods); }

#else

int csets_add_functions(PyObject *m) { return 0; }

#endif
