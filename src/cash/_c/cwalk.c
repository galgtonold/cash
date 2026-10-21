/*
 * cash._cwatch, part 2: C kernels for cash's hottest Python walks.
 *
 * Each function here computes EXACTLY what one Python expression or helper
 * of cash computes (named in its comment), at C speed. The Python code stays
 * the reference and the fallback: a kernel that meets anything it cannot
 * answer the same way -- an object of a subclass, a class whose metaclass
 * hashes or compares in its own way -- returns NotImplemented, and the
 * caller runs its Python version. tests/test_core/test_cwalk.py runs both on
 * random structures and real objects and requires identical results.
 *
 * No Python code runs inside a kernel: the garbage collector is paused for
 * its duration (a finalizer could otherwise change a container while a
 * kernel reads it), and only exact builtin containers are read.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <limits.h>
#include <stdint.h>
#include <string.h>

#include "cwalk.h"

/* Overflow-checked arithmetic without compiler builtins (MSVC has none):
 * 1 on overflow, else 0 and *out set. */
static int mul_overflows(long long a, long long b, long long *out) {
    /* a, b >= 0 here: sizes and counts */
    if (a < 0 || b < 0 || (a && b > LLONG_MAX / a)) return 1;
    *out = a * b;
    return 0;
}
static int add_overflows(long long a, long long b, long long *out) {
    if ((b > 0 && a > LLONG_MAX - b) || (b < 0 && a < LLONG_MIN - b)) return 1;
    *out = a + b;
    return 0;
}

#ifndef Py_GIL_DISABLED

/* ------------------------------------------------------------ leaf test */

/* A class whose type hashes and compares as `type` does: by identity. Its
 * membership in a set of types is then pointer membership, and asking runs
 * no Python code. */
static int plain_meta(PyTypeObject *t) {
    PyTypeObject *m = Py_TYPE(t);
    return m == &PyType_Type ||
           (m->tp_hash == PyType_Type.tp_hash && m->tp_richcompare == PyType_Type.tp_richcompare);
}

#define LT_CACHE 8
typedef struct {
    PyObject *set; /* a set or frozenset of types */
    PyTypeObject *yes[LT_CACHE], *no[LT_CACHE];
    int ny, nn, ry, rn;
} TypeSet;

static void ts_init(TypeSet *ts, PyObject *set) {
    memset(ts, 0, sizeof *ts);
    ts->set = set;
}

/* 1: t in the set, 0: not, -1: error, -2: a metaclass of its own (unknown) */
static int ts_has(TypeSet *ts, PyTypeObject *t) {
    for (int i = 0; i < ts->ny; i++)
        if (ts->yes[i] == t) return 1;
    for (int i = 0; i < ts->nn; i++)
        if (ts->no[i] == t) return 0;
    if (!plain_meta(t)) return -2;
    int r = PySet_Contains(ts->set, (PyObject *)t);
    if (r < 0) return -1;
    if (r) {
        if (ts->ny < LT_CACHE) ts->yes[ts->ny++] = t;
        else ts->yes[ts->ry++ % LT_CACHE] = t;
    } else {
        if (ts->nn < LT_CACHE) ts->no[ts->nn++] = t;
        else ts->no[ts->rn++ % LT_CACHE] = t;
    }
    return r;
}

static int is_typeset(PyObject *o) { return PyFrozenSet_CheckExact(o) || PyAnySet_CheckExact(o); }

/* ------------------------------------------------------------ gc pause */
typedef struct { int was; } GCPause;
static void gc_pause(GCPause *p) { p->was = PyGC_Disable(); }
static void gc_resume(GCPause *p) {
    if (p->was) PyGC_Enable();
}

#define NOT_IMPL() Py_NewRef(Py_NotImplemented)

/* Is every item of the exact list/tuple/set/frozenset *c* of a type in *ts*?
 * 1/0, -1 error, -2 unknown */
static int seq_all_in(TypeSet *ts, PyObject *c) {
    if (PyList_CheckExact(c) || PyTuple_CheckExact(c)) {
        Py_ssize_t n = PySequence_Fast_GET_SIZE(c);
        PyObject **items = PySequence_Fast_ITEMS(c);
        for (Py_ssize_t i = 0; i < n; i++) {
            int r = ts_has(ts, Py_TYPE(items[i]));
            if (r != 1) return r;
        }
        return 1;
    }
    /* exact set / frozenset: their own iterator, in C */
    PyObject *it = PyObject_GetIter(c);
    if (!it) return -1;
    PyObject *x;
    int out = 1;
    while ((x = PyIter_Next(it))) {
        int r = ts_has(ts, Py_TYPE(x));
        Py_DECREF(x);
        if (r != 1) { out = r; break; }
    }
    Py_DECREF(it);
    if (out == 1 && PyErr_Occurred()) return -1;
    return out;
}

static int is_exact_seq(PyObject *o) {
    return PyList_CheckExact(o) || PyTuple_CheckExact(o) || PyFrozenSet_CheckExact(o) ||
           (PyAnySet_CheckExact(o) && !PyFrozenSet_CheckExact(o));
}

/* 1/0/-1/-2: keys (want_keys) and/or values (want_values) of exact dict d all in ts */
static int dict_all_in(TypeSet *ts, PyObject *d, int want_keys, int want_values) {
    Py_ssize_t pos = 0;
    PyObject *k, *v;
    while (PyDict_Next(d, &pos, &k, &v)) {
        if (want_keys) {
            int r = ts_has(ts, Py_TYPE(k));
            if (r != 1) return r;
        }
        if (want_values) {
            int r = ts_has(ts, Py_TYPE(v));
            if (r != 1) return r;
        }
    }
    return 1;
}

/*
 * leaf_items(group, mode, leaves) -> bool | NotImplemented
 *
 *   mode 0: set(map(type, chain.from_iterable(group))) <= leaves, every one
 *           of group an exact list, tuple, set or frozenset;
 *   mode 1: ... over chain.from_iterable(map(dict.values, group)), exact dicts;
 *   mode 2: keys and values of exact dicts (held_functions._plain_records);
 *   mode 3: keys of exact dicts.
 */
static PyObject *py_leaf_items(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 3 || !PyList_CheckExact(args[0]) || !PyLong_CheckExact(args[1]) || !is_typeset(args[2])) {
        PyErr_SetString(PyExc_TypeError, "leaf_items(list, mode, set of types)");
        return NULL;
    }
    PyObject *group = args[0];
    long mode = PyLong_AsLong(args[1]);
    if (mode < 0 || mode > 3) {
        PyErr_SetString(PyExc_ValueError, "mode 0..3");
        return NULL;
    }
    TypeSet ts;
    ts_init(&ts, args[2]);
    Py_ssize_t n = PyList_GET_SIZE(group);
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *c = PyList_GET_ITEM(group, i);
        if (mode == 0 ? !is_exact_seq(c) : !PyDict_CheckExact(c)) return NOT_IMPL();
    }
    GCPause gp;
    gc_pause(&gp);
    int out = 1;
    Py_INCREF(group);
    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(group) && out == 1; i++) {
        PyObject *c = PyList_GET_ITEM(group, i);
        out = mode == 0 ? seq_all_in(&ts, c) : dict_all_in(&ts, c, mode != 1, mode != 3);
    }
    Py_DECREF(group);
    gc_resume(&gp);
    if (out == -1) return NULL;
    if (out == -2) return NOT_IMPL();
    return PyBool_FromLong(out);
}

/* append every item of exact seq c to out */
static int extend_seq(PyObject *out, PyObject *c) {
    if (PyList_CheckExact(c) || PyTuple_CheckExact(c)) {
        Py_ssize_t n = PySequence_Fast_GET_SIZE(c);
        for (Py_ssize_t i = 0; i < n; i++)
            if (PyList_Append(out, PySequence_Fast_ITEMS(c)[i]) < 0) return -1;
        return 0;
    }
    PyObject *it = PyObject_GetIter(c);
    if (!it) return -1;
    PyObject *x;
    while ((x = PyIter_Next(it))) {
        int r = PyList_Append(out, x);
        Py_DECREF(x);
        if (r < 0) { Py_DECREF(it); return -1; }
    }
    Py_DECREF(it);
    return PyErr_Occurred() ? -1 : 0;
}

/*
 * seq_children(group, leaves, children) -> None | NotImplemented
 *
 * held_functions._visit_group for a group of exact lists, tuples, sets or
 * frozensets:
 *     if not set(map(type, chain.from_iterable(group))) <= leaves:
 *         children.extend(chain.from_iterable(group))
 */
static PyObject *py_seq_children(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 3 || !PyList_CheckExact(args[0]) || !is_typeset(args[1]) || !PyList_CheckExact(args[2])) {
        PyErr_SetString(PyExc_TypeError, "seq_children(list, set of types, list)");
        return NULL;
    }
    PyObject *group = args[0], *children = args[2];
    if (group == children) {
        PyErr_SetString(PyExc_ValueError, "group is children");
        return NULL;
    }
    Py_ssize_t n = PyList_GET_SIZE(group);
    for (Py_ssize_t i = 0; i < n; i++)
        if (!is_exact_seq(PyList_GET_ITEM(group, i))) return NOT_IMPL();
    TypeSet ts;
    ts_init(&ts, args[1]);
    GCPause gp;
    gc_pause(&gp);
    Py_INCREF(group);
    int all = 1;
    for (Py_ssize_t i = 0; i < n && all == 1; i++) all = seq_all_in(&ts, PyList_GET_ITEM(group, i));
    int err = all == -1;
    if (all == 0)
        for (Py_ssize_t i = 0; i < n && !err; i++) err = extend_seq(children, PyList_GET_ITEM(group, i)) < 0;
    Py_DECREF(group);
    gc_resume(&gp);
    if (err) return NULL;
    if (all == -2) return NOT_IMPL();
    Py_RETURN_NONE;
}

/*
 * dict_children(group, leaves, children) -> None | NotImplemented
 *
 * held_functions._visit_group for a group of exact dicts:
 *     keys = list(chain.from_iterable(group))
 *     key_kinds = set(map(type, keys))
 *     if not key_kinds <= leaves and not (
 *         key_kinds <= leaves | {tuple}
 *         and set(map(type, chain.from_iterable(_of_type(keys, tuple)))) <= leaves):
 *         children.extend(keys)
 *     if not set(map(type, chain.from_iterable(map(dict.values, group)))) <= leaves:
 *         children.extend(chain.from_iterable(map(dict.values, group)))
 */
static PyObject *py_dict_children(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 3 || !PyList_CheckExact(args[0]) || !is_typeset(args[1]) || !PyList_CheckExact(args[2])) {
        PyErr_SetString(PyExc_TypeError, "dict_children(list, set of types, list)");
        return NULL;
    }
    PyObject *group = args[0], *children = args[2];
    if (group == children) {
        PyErr_SetString(PyExc_ValueError, "group is children");
        return NULL;
    }
    Py_ssize_t n = PyList_GET_SIZE(group);
    for (Py_ssize_t i = 0; i < n; i++)
        if (!PyDict_CheckExact(PyList_GET_ITEM(group, i))) return NOT_IMPL();
    TypeSet ts;
    ts_init(&ts, args[1]);
    GCPause gp;
    gc_pause(&gp);
    Py_INCREF(group);
    int keys_leaf = 1, keys_alt = 1, values_leaf = 1, err = 0, unknown = 0;
    for (Py_ssize_t i = 0; i < n && !err && !unknown; i++) {
        PyObject *d = PyList_GET_ITEM(group, i), *k, *v;
        Py_ssize_t pos = 0;
        while (PyDict_Next(d, &pos, &k, &v)) {
            if (keys_leaf || keys_alt) {
                int r = ts_has(&ts, Py_TYPE(k));
                if (r < 0) { err = r == -1; unknown = r == -2; break; }
                if (!r) {
                    keys_leaf = 0;
                    if (keys_alt && !PyTuple_CheckExact(k)) keys_alt = 0;
                }
                if (keys_alt && PyTuple_CheckExact(k)) {
                    /* every exact tuple key's items, leaf tuple or not */
                    int a = seq_all_in(&ts, k);
                    if (a < 0) { err = a == -1; unknown = a == -2; break; }
                    if (!a) keys_alt = 0;
                }
            }
            if (values_leaf) {
                int r = ts_has(&ts, Py_TYPE(v));
                if (r < 0) { err = r == -1; unknown = r == -2; break; }
                if (!r) values_leaf = 0;
            }
            if (!keys_leaf && !keys_alt && !values_leaf) break;
        }
        if (!keys_leaf && !keys_alt && !values_leaf) break;
    }
    if (!err && !unknown && !keys_leaf && !keys_alt) {
        for (Py_ssize_t i = 0; i < n && !err; i++) {
            PyObject *d = PyList_GET_ITEM(group, i), *k, *v;
            Py_ssize_t pos = 0;
            while (PyDict_Next(d, &pos, &k, &v))
                if (PyList_Append(children, k) < 0) { err = 1; break; }
        }
    }
    if (!err && !unknown && !values_leaf) {
        for (Py_ssize_t i = 0; i < n && !err; i++) {
            PyObject *d = PyList_GET_ITEM(group, i), *k, *v;
            Py_ssize_t pos = 0;
            while (PyDict_Next(d, &pos, &k, &v))
                if (PyList_Append(children, v) < 0) { err = 1; break; }
        }
    }
    Py_DECREF(group);
    gc_resume(&gp);
    if (err) return NULL;
    if (unknown) return NOT_IMPL();
    Py_RETURN_NONE;
}

/* ------------------------------------------------------------ pointer map */
typedef struct { uintptr_t key; Py_ssize_t val; } PMEntry;
typedef struct { PMEntry *t; size_t cap, used; } PMap;

static size_t pm_hash(uintptr_t p) { return (size_t)((p >> 4) * 0x9E3779B97F4A7C15ull); }

static int pm_init(PMap *m, size_t want) {
    size_t cap = 16;
    while (cap < want * 2) cap <<= 1;
    m->t = PyMem_Calloc(cap, sizeof(PMEntry));
    if (!m->t) { PyErr_NoMemory(); return -1; }
    m->cap = cap;
    m->used = 0;
    return 0;
}
static void pm_free(PMap *m) { PyMem_Free(m->t); m->t = NULL; }

/* index of key's slot (empty: key 0) */
static PMEntry *pm_slot(PMap *m, uintptr_t key) {
    size_t mask = m->cap - 1, i = pm_hash(key) & mask;
    while (m->t[i].key && m->t[i].key != key) i = (i + 1) & mask;
    return &m->t[i];
}
static int pm_grow(PMap *m) {
    PMap n;
    if (pm_init(&n, m->cap) < 0) return -1;
    for (size_t i = 0; i < m->cap; i++)
        if (m->t[i].key) { *pm_slot(&n, m->t[i].key) = m->t[i]; n.used++; }
    pm_free(m);
    *m = n;
    return 0;
}

/*
 * split_level(children, skipped, extra) -> [(type, [objects])] | NotImplemented
 *
 * held_functions._next_level up to the dedup:
 *     others = [c for c in children if type(c) not in skipped]; others.extend(extra)
 *     _by_type(others)
 * Groups in _by_type's order: the types holding more than 1/16 of the
 * objects in first-met order, then the others in first-met order; each
 * group's objects in their order. One type: one group.
 */
static PyObject *py_split_level(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 3 || !PyList_CheckExact(args[0]) || !is_typeset(args[1]) || !is_typeset(args[2])) {
        PyErr_SetString(PyExc_TypeError, "split_level(list, set of types, set)");
        return NULL;
    }
    PyObject *children = args[0];
    TypeSet ts;
    ts_init(&ts, args[1]);
    PyObject *others = PyList_New(0), *out = NULL, *extra = NULL;
    if (!others) return NULL;
    GCPause gp;
    gc_pause(&gp);
    PMap map = {0};
    PyTypeObject **kinds = NULL;
    Py_ssize_t *counts = NULL, *fill = NULL;
    PyObject **lists = NULL;
    Py_ssize_t nkinds = 0;
    int unknown = 0;
    Py_INCREF(children);
    Py_ssize_t n = PyList_GET_SIZE(children);
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *c = PyList_GET_ITEM(children, i);
        int r = ts_has(&ts, Py_TYPE(c));
        if (r == -1) goto fail;
        if (r == -2) { unknown = 1; goto done; }
        if (!r && PyList_Append(others, c) < 0) goto fail;
    }
    extra = PySequence_List(args[2]); /* the set's own order */
    if (!extra) goto fail;
    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(extra); i++)
        if (PyList_Append(others, PyList_GET_ITEM(extra, i)) < 0) goto fail;
    Py_ssize_t m = PyList_GET_SIZE(others);
    if (pm_init(&map, 16) < 0) goto fail;
    Py_ssize_t capk = 16;
    kinds = PyMem_Malloc(capk * sizeof *kinds);
    counts = PyMem_Malloc(capk * sizeof *counts);
    if (!kinds || !counts) { PyErr_NoMemory(); goto fail; }
    PyTypeObject *last = NULL;
    Py_ssize_t last_ix = -1;
    for (Py_ssize_t i = 0; i < m; i++) {
        PyTypeObject *t = Py_TYPE(PyList_GET_ITEM(others, i));
        if (t == last) { counts[last_ix]++; continue; }
        PMEntry *e = pm_slot(&map, (uintptr_t)t);
        if (!e->key) {
            if ((map.used + 1) * 2 > map.cap) {
                if (pm_grow(&map) < 0) goto fail;
                e = pm_slot(&map, (uintptr_t)t);
            }
            if (nkinds == capk) {
                capk *= 2;
                PyTypeObject **k2 = PyMem_Realloc(kinds, capk * sizeof *kinds);
                if (!k2) { PyErr_NoMemory(); goto fail; }
                kinds = k2;
                Py_ssize_t *c2 = PyMem_Realloc(counts, capk * sizeof *counts);
                if (!c2) { PyErr_NoMemory(); goto fail; }
                counts = c2;
            }
            e->key = (uintptr_t)t;
            e->val = nkinds;
            map.used++;
            kinds[nkinds] = t;
            counts[nkinds++] = 0;
        }
        last = t;
        last_ix = e->val;
        counts[last_ix]++;
    }
    out = PyList_New(0);
    if (!out) goto fail;
    if (nkinds == 1) {
        PyObject *pair = PyTuple_Pack(2, (PyObject *)kinds[0], others);
        if (!pair || PyList_Append(out, pair) < 0) { Py_XDECREF(pair); goto fail; }
        Py_DECREF(pair);
        goto done;
    }
    /* order: majors first-met, then minors first-met */
    Py_ssize_t *order = PyMem_Malloc((nkinds + 1) * sizeof *order);
    fill = PyMem_Calloc(nkinds + 1, sizeof *fill);
    lists = PyMem_Calloc(nkinds + 1, sizeof *lists);
    if (!order || !fill || !lists) { PyMem_Free(order); PyErr_NoMemory(); goto fail; }
    Py_ssize_t no = 0;
    for (Py_ssize_t k = 0; k < nkinds; k++)
        if (counts[k] * 16 > m) order[no++] = k;
    for (Py_ssize_t k = 0; k < nkinds; k++)
        if (!(counts[k] * 16 > m)) order[no++] = k;
    for (Py_ssize_t k = 0; k < nkinds; k++) {
        lists[k] = PyList_New(counts[k]);
        if (!lists[k]) { PyMem_Free(order); goto fail; }
    }
    last = NULL;
    for (Py_ssize_t i = 0; i < m; i++) {
        PyObject *o = PyList_GET_ITEM(others, i);
        PyTypeObject *t = Py_TYPE(o);
        if (t != last) { last = t; last_ix = pm_slot(&map, (uintptr_t)t)->val; }
        PyList_SET_ITEM(lists[last_ix], fill[last_ix]++, Py_NewRef(o));
    }
    for (Py_ssize_t j = 0; j < no; j++) {
        Py_ssize_t k = order[j];
        PyObject *pair = PyTuple_Pack(2, (PyObject *)kinds[k], lists[k]);
        if (!pair || PyList_Append(out, pair) < 0) { Py_XDECREF(pair); PyMem_Free(order); goto fail; }
        Py_DECREF(pair);
    }
    PyMem_Free(order);
done:
    if (lists)
        for (Py_ssize_t k = 0; k < nkinds; k++) Py_XDECREF(lists[k]);
    PyMem_Free(lists);
    PyMem_Free(fill);
    PyMem_Free(kinds);
    PyMem_Free(counts);
    pm_free(&map);
    Py_XDECREF(extra);
    Py_DECREF(others);
    Py_DECREF(children);
    gc_resume(&gp);
    if (unknown) { Py_XDECREF(out); return NOT_IMPL(); }
    return out;
fail:
    Py_CLEAR(out);
    if (lists)
        for (Py_ssize_t k = 0; k < nkinds; k++) Py_XDECREF(lists[k]);
    PyMem_Free(lists);
    PyMem_Free(fill);
    PyMem_Free(kinds);
    PyMem_Free(counts);
    pm_free(&map);
    Py_XDECREF(extra);
    Py_DECREF(others);
    Py_DECREF(children);
    gc_resume(&gp);
    return NULL;
}

/*
 * new_in(group, by_identity, seen) -> list
 *
 * held_functions._next_level's dedup of one group:
 *     found = dict(zip(group if by_identity else map(id, group), group))
 *     for key in seen & found.keys(): del found[key]
 *     seen.update(found)
 *     list(found.values())
 * each object once, in first-met order, none already in *seen* (by the
 * object itself, or by its id), and *seen* updated. The ids are made without
 * id(): no audit event per object.
 */
static PyObject *py_new_in(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 3 || !PyList_CheckExact(args[0]) || !PyAnySet_CheckExact(args[2]) ||
        PyFrozenSet_CheckExact(args[2])) {
        PyErr_SetString(PyExc_TypeError, "new_in(list, bool, set)");
        return NULL;
    }
    PyObject *group = args[0], *seen = args[2];
    int by_identity = PyObject_IsTrue(args[1]);
    if (by_identity < 0) return NULL;
    PyObject *out = PyList_New(0);
    if (!out) return NULL;
    GCPause gp;
    gc_pause(&gp);
    Py_INCREF(group);
    int err = 0;
    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(group); i++) {
        PyObject *o = PyList_GET_ITEM(group, i);
        PyObject *key = by_identity ? Py_NewRef(o) : PyLong_FromVoidPtr(o);
        if (!key) { err = 1; break; }
        Py_ssize_t before = PySet_GET_SIZE(seen);
        if (PySet_Add(seen, key) < 0) { Py_DECREF(key); err = 1; break; }
        Py_DECREF(key);
        if (PySet_GET_SIZE(seen) != before && PyList_Append(out, o) < 0) { err = 1; break; }
    }
    Py_DECREF(group);
    gc_resume(&gp);
    if (err) { Py_DECREF(out); return NULL; }
    return out;
}

/*
 * type_set(objects) -> set: set(map(type, objects)) for a list.
 */
static PyObject *py_type_set(PyObject *self, PyObject *lst) {
    if (!PyList_CheckExact(lst)) {
        PyErr_SetString(PyExc_TypeError, "type_set(list)");
        return NULL;
    }
    PyObject *out = PySet_New(NULL);
    if (!out) return NULL;
    PyTypeObject *last = NULL;
    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(lst); i++) {
        PyTypeObject *t = Py_TYPE(PyList_GET_ITEM(lst, i));
        if (t == last) continue;
        last = t;
        if (PySet_Add(out, (PyObject *)t) < 0) { Py_DECREF(out); return NULL; }
    }
    return out;
}


/* ============================================================ store walks */

static PyObject *getsizeof_fn = NULL; /* sys.getsizeof */

/* sys.getsizeof(o) as a size, or -1 with an exception */
static Py_ssize_t sizeof_of(PyObject *o) {
    if (!getsizeof_fn) {
        getsizeof_fn = PySys_GetObject("getsizeof"); /* borrowed */
        if (!getsizeof_fn) {
            PyErr_SetString(PyExc_RuntimeError, "no sys.getsizeof");
            return -1;
        }
        Py_INCREF(getsizeof_fn);
    }
    PyObject *r = PyObject_CallOneArg(getsizeof_fn, o);
    if (!r) return -1;
    Py_ssize_t n = PyLong_AsSsize_t(r);
    Py_DECREF(r);
    return n;
}

/* a growable array of borrowed object pointers */
typedef struct { PyObject **v; Py_ssize_t n, cap; } PtrVec;

static int pv_push(PtrVec *a, PyObject *o) {
    if (a->n == a->cap) {
        Py_ssize_t cap = a->cap ? a->cap * 2 : 64;
        PyObject **v = PyMem_Realloc(a->v, cap * sizeof *v);
        if (!v) { PyErr_NoMemory(); return -1; }
        a->v = v;
        a->cap = cap;
    }
    a->v[a->n++] = o;
    return 0;
}
static void pv_free(PtrVec *a) { PyMem_Free(a->v); a->v = NULL; a->n = a->cap = 0; }

static PyObject *list_of(PtrVec *a) {
    PyObject *l = PyList_New(a->n);
    if (!l) return NULL;
    for (Py_ssize_t i = 0; i < a->n; i++) PyList_SET_ITEM(l, i, Py_NewRef(a->v[i]));
    return l;
}

#define SIZE_EXACT_UP_TO ((Py_ssize_t)1 << 16)
#define SIZE_SAMPLE ((Py_ssize_t)1 << 14)

static PyObject *picks_fn = NULL; /* _plain_data._picks, handed in */

/* _plain_data._level_size: getsizeof summed over v (n items), from
 * `_picks(n)` of them when n > SIZE_EXACT_UP_TO. -1 on error, -2 overflow */
static long long level_size(PyObject **v, Py_ssize_t n) {
    long long total = 0;
    if (n <= SIZE_EXACT_UP_TO) {
        for (Py_ssize_t i = 0; i < n; i++) {
            Py_ssize_t s = sizeof_of(v[i]);
            if (s < 0) return -1;
            total += s;
        }
        return total;
    }
    PyObject *count = PyLong_FromSsize_t(n);
    if (!count) return -1;
    PyObject *picks = PyObject_CallOneArg(picks_fn, count);
    Py_DECREF(count);
    if (!picks) return -1;
    if (!PyTuple_CheckExact(picks)) {
        Py_DECREF(picks);
        PyErr_SetString(PyExc_TypeError, "_picks gave no tuple");
        return -1;
    }
    for (Py_ssize_t j = 0; j < PyTuple_GET_SIZE(picks); j++) {
        Py_ssize_t at = PyLong_AsSsize_t(PyTuple_GET_ITEM(picks, j));
        if (at < 0 || at >= n) {
            Py_DECREF(picks);
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_IndexError, "pick out of range");
            return -1;
        }
        Py_ssize_t s = sizeof_of(v[at]);
        if (s < 0) { Py_DECREF(picks); return -1; }
        total += s;
    }
    Py_DECREF(picks);
    long long out;
    if (mul_overflows(total, (long long)n, &out)) return -2;
    return out / SIZE_SAMPLE;
}

/* a set of types met: a small array, then a frozenset/set at the end */
typedef struct { PyTypeObject **t; Py_ssize_t n, cap; } TypeList;

static int tl_add(TypeList *l, PyTypeObject *t) {
    for (Py_ssize_t i = 0; i < l->n; i++)
        if (l->t[i] == t) return 0;
    if (l->n == l->cap) {
        Py_ssize_t cap = l->cap ? l->cap * 2 : 16;
        PyTypeObject **v = PyMem_Realloc(l->t, cap * sizeof *v);
        if (!v) { PyErr_NoMemory(); return -1; }
        l->t = v;
        l->cap = cap;
    }
    l->t[l->n++] = t;
    return 0;
}
static PyObject *tl_set(TypeList *l, int frozen) {
    PyObject *s = frozen ? PyFrozenSet_New(NULL) : PySet_New(NULL);
    if (!s) return NULL;
    for (Py_ssize_t i = 0; i < l->n; i++)
        if (PySet_Add(s, (PyObject *)l->t[i]) < 0) { Py_DECREF(s); return NULL; }
    return s;
}

static int is_node_type(PyTypeObject *t) { return t == &PyDict_Type || t == &PyList_Type || t == &PyTuple_Type; }

#define MAX_LEVELS 16

/*
 * tree_facts(value, leaves, want_walk, top_extra, picks, mode=0)
 *   -> (types, held_once, size, nodes, walk | None) | None | NotImplemented
 *
 * _plain_data._tree_facts(value, look=None, walk, top_extra) for JSON-like
 * *value* (exact dicts, lists and tuples over *leaves*, dict keys leaves):
 * None where it raises _NotPlain (anything else, deeper than MAX_LEVELS, a
 * cycle). Walked one level at a time in the same order, so each level's
 * items, their sampled sizes and the walk's lists are the same. held_once
 * reads the reference counts the Python walk reads, less the references its
 * own lists hold: a container below the top held by its parent alone has
 * one (plus *top_extra* at the first level).
 *
 * *mode* 1: the size is `tree_size`'s (`_tree_walk` over the same leaves),
 * which sizes a level's keys from a sample of the keys, never of the dicts.
 * *mode* 2: plain data (`_levels`, `size_of`): exact lists and tuples over
 * the leaves, no dict anywhere, the top included.
 * *mode* 3: the walk alone (`tree_walk`), no size (0).
 */
static int is_seq_type(PyTypeObject *t) { return t == &PyList_Type || t == &PyTuple_Type; }

static PyObject *py_tree_facts(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs < 5 || nargs > 6 || !is_typeset(args[1]) || !PyLong_CheckExact(args[3]) || !PyCallable_Check(args[4])) {
        PyErr_SetString(PyExc_TypeError, "tree_facts(value, leaves, want_walk, top_extra, picks, mode=0)");
        return NULL;
    }
    long mode = nargs > 5 ? PyLong_AsLong(args[5]) : 0;
    if (mode == -1 && PyErr_Occurred()) return NULL;
    int (*node_type)(PyTypeObject *) = mode == 2 ? is_seq_type : is_node_type;
    PyObject *value = args[0];
    if (!node_type(Py_TYPE(value))) Py_RETURN_NONE;
    int want_walk = PyObject_IsTrue(args[2]);
    if (want_walk < 0) return NULL;
    Py_ssize_t top_extra = PyLong_AsSsize_t(args[3]);
    if (top_extra == -1 && PyErr_Occurred()) return NULL;
    Py_XSETREF(picks_fn, Py_NewRef(args[4]));
    TypeSet leaves;
    ts_init(&leaves, args[1]);

    PtrVec level = {0}, flat = {0}, keys = {0}, dicts = {0};
    TypeList found = {0}, types = {0}, ktypes = {0};
    PyObject *walk = NULL, *out = NULL;
    /* the walk's levels, kept as pointer arrays until held_once is read */
    PtrVec *wl = NULL, *wf = NULL;
    PyObject **wtypes = NULL, **wktypes = NULL;
    int wall_nodes[MAX_LEVELS + 1];
    Py_ssize_t nw = 0;
    int notplain = 0, unknown = 0, held_once = 1;
    long long size = 0, nodes = 0;
    GCPause gp;
    gc_pause(&gp);
    Py_INCREF(value);

    if (want_walk) {
        wl = PyMem_Calloc(MAX_LEVELS + 1, sizeof *wl);
        wf = PyMem_Calloc(MAX_LEVELS + 1, sizeof *wf);
        wtypes = PyMem_Calloc(MAX_LEVELS + 1, sizeof *wtypes);
        wktypes = PyMem_Calloc(MAX_LEVELS + 1, sizeof *wktypes);
        if (!wl || !wf || !wtypes || !wktypes) { PyErr_NoMemory(); goto fail; }
    }
    if (tl_add(&found, Py_TYPE(value)) < 0) goto fail;
    int sizes = mode != 3;
    if (sizes) {
        Py_ssize_t s = sizeof_of(value);
        if (s < 0) goto fail;
        size = s;
    }
    if (pv_push(&level, value) < 0) goto fail;
    for (int depth = 0; depth <= MAX_LEVELS; depth++) {
        if (depth == MAX_LEVELS) { notplain = 1; goto done; }
        nodes += level.n;
        flat.n = keys.n = dicts.n = 0;
        types.n = ktypes.n = 0;
        int any_dict = 0;
        for (Py_ssize_t i = 0; i < level.n; i++) {
            PyObject *c = level.v[i];
            if (PyDict_CheckExact(c)) {
                any_dict = 1;
                if (pv_push(&dicts, c) < 0) goto fail;
                PyObject *k, *v;
                Py_ssize_t pos = 0;
                while (PyDict_Next(c, &pos, &k, &v)) {
                    int r = ts_has(&leaves, Py_TYPE(k));
                    if (r == -1) goto fail;
                    if (r == -2) { unknown = 1; goto done; }
                    if (!r) { notplain = 1; goto done; }
                    if (tl_add(&ktypes, Py_TYPE(k)) < 0 || (sizes && pv_push(&keys, k) < 0) || pv_push(&flat, v) < 0)
                        goto fail;
                }
            } else {
                Py_ssize_t n = PySequence_Fast_GET_SIZE(c);
                PyObject **items = PySequence_Fast_ITEMS(c);
                for (Py_ssize_t j = 0; j < n; j++)
                    if (pv_push(&flat, items[j]) < 0) goto fail;
            }
        }
        if (any_dict) {
            for (Py_ssize_t i = 0; i < ktypes.n; i++)
                if (tl_add(&found, ktypes.t[i]) < 0) goto fail;
        }
        if (any_dict && sizes) {
            /* _keys_size(dicts) */
            long long ks;
            if (mode == 1 || dicts.n <= SIZE_EXACT_UP_TO) {
                ks = level_size(keys.v, keys.n);
            } else {
                PyObject *count = PyLong_FromSsize_t(dicts.n);
                PyObject *picks = count ? PyObject_CallOneArg(picks_fn, count) : NULL;
                Py_XDECREF(count);
                if (!picks) goto fail;
                ks = 0;
                for (Py_ssize_t j = 0; j < PyTuple_GET_SIZE(picks) && ks >= 0; j++) {
                    Py_ssize_t at = PyLong_AsSsize_t(PyTuple_GET_ITEM(picks, j));
                    if (at < 0 || at >= dicts.n) { ks = -1; if (!PyErr_Occurred()) PyErr_SetString(PyExc_IndexError, "pick"); break; }
                    PyObject *k, *v;
                    Py_ssize_t pos = 0;
                    while (PyDict_Next(dicts.v[at], &pos, &k, &v)) {
                        Py_ssize_t s = sizeof_of(k);
                        if (s < 0) { ks = -1; break; }
                        ks += s;
                    }
                }
                Py_DECREF(picks);
                if (ks >= 0 && mul_overflows(ks, (long long)dicts.n, &ks)) ks = -2;
                if (ks >= 0) ks /= SIZE_SAMPLE;
            }
            if (ks == -1) goto fail;
            if (ks == -2) { unknown = 1; goto done; }
            size += ks;
        }
        int all_nodes = 1, no_nodes = 1;
        PyTypeObject *last = NULL;
        for (Py_ssize_t i = 0; i < flat.n; i++) {
            PyTypeObject *t = Py_TYPE(flat.v[i]);
            if (t == last) continue;
            last = t;
            int node = node_type(t);
            if (!node) {
                int r = ts_has(&leaves, t);
                if (r == -1) goto fail;
                if (r == -2) { unknown = 1; goto done; }
                if (!r) { notplain = 1; goto done; }
            }
            if (tl_add(&types, t) < 0) goto fail;
            if (node) no_nodes = 0; else all_nodes = 0;
        }
        for (Py_ssize_t i = 0; i < types.n; i++)
            if (tl_add(&found, types.t[i]) < 0) goto fail;
        if (sizes) {
            long long ls = level_size(flat.v, flat.n);
            if (ls == -1) goto fail;
            if (ls == -2) { unknown = 1; goto done; }
            size += ls;
        }
        if (want_walk) {
            /* the level's containers and items, by pointer: made lists below */
            PtrVec *a = &wl[nw], *b = &wf[nw];
            for (Py_ssize_t i = 0; i < level.n; i++) if (pv_push(a, level.v[i]) < 0) goto fail;
            for (Py_ssize_t i = 0; i < flat.n; i++) if (pv_push(b, flat.v[i]) < 0) goto fail;
            wtypes[nw] = tl_set(&types, 0);
            wktypes[nw] = tl_set(&ktypes, 0);
            wall_nodes[nw] = all_nodes;
            nw++;
            if (!wtypes[nw - 1] || !wktypes[nw - 1]) goto fail;
        }
        if (no_nodes) break;
        /* the next level: the containers among the items */
        level.n = 0;
        if (all_nodes) {
            for (Py_ssize_t i = 0; i < flat.n; i++) if (pv_push(&level, flat.v[i]) < 0) goto fail;
        } else {
            for (Py_ssize_t i = 0; i < flat.n; i++)
                if (node_type(Py_TYPE(flat.v[i])) && pv_push(&level, flat.v[i]) < 0) goto fail;
        }
        Py_ssize_t limit = 1 + (depth == 0 ? top_extra : 0);
        if (held_once)
            for (Py_ssize_t i = 0; i < level.n; i++)
                if (Py_REFCNT(level.v[i]) > limit) { held_once = 0; break; }
    }
done:
    if (!notplain && !unknown) {
        PyObject *ft = tl_set(&found, 1);
        if (!ft) goto fail;
        if (want_walk) {
            /* As the Python walk makes them: a level of containers only is
             * the list of the items above it, the same list object (what
             * `_spine_copy` reads reference counts against). */
            walk = PyList_New(nw);
            if (!walk) { Py_DECREF(ft); goto fail; }
            PyObject *prev_flat = NULL; /* borrowed: walk holds it */
            for (Py_ssize_t i = 0; i < nw; i++) {
                PyObject *a = (i && wall_nodes[i - 1]) ? Py_NewRef(prev_flat) : list_of(&wl[i]);
                PyObject *b = a ? list_of(&wf[i]) : NULL;
                PyObject *t = b ? PyTuple_Pack(4, a, b, wtypes[i], wktypes[i]) : NULL;
                Py_XDECREF(a);
                Py_XDECREF(b);
                if (!t) { Py_DECREF(ft); goto fail; }
                PyList_SET_ITEM(walk, i, t);
                prev_flat = b;
            }
        }
        out = Py_BuildValue("(NOLLO)", ft, held_once ? Py_True : Py_False, size, nodes, walk ? walk : Py_None);
        if (!out) goto fail;
    }
    goto cleanup;
fail:
    Py_CLEAR(out);
    notplain = unknown = 0;
cleanup:
    Py_XDECREF(walk);
    if (want_walk && wl) {
        for (Py_ssize_t i = 0; i <= MAX_LEVELS; i++) {
            pv_free(&wl[i]);
            pv_free(&wf[i]);
            Py_XDECREF(wtypes[i]);
            Py_XDECREF(wktypes[i]);
        }
    }
    PyMem_Free(wl);
    PyMem_Free(wf);
    PyMem_Free(wtypes);
    PyMem_Free(wktypes);
    pv_free(&level);
    pv_free(&flat);
    pv_free(&keys);
    pv_free(&dicts);
    PyMem_Free(found.t);
    PyMem_Free(types.t);
    PyMem_Free(ktypes.t);
    Py_DECREF(value);
    gc_resume(&gp);
    if (out) return out;
    if (PyErr_Occurred()) return NULL;
    if (unknown) return NOT_IMPL();
    Py_RETURN_NONE; /* not plain */
}

/*
 * shared_parts(value, min_rows, max_depth, max_keys, immutable_below) -> dict
 *
 * memory_backend._shared_parts(value): {id(part): part} for the lists and
 * tuples of at least *min_rows* items whose first item is a tuple and that
 * `immutable_below` accepts, among the values of exact dict *value* and of
 * the exact dicts of at most *max_keys* keys in it, *max_depth* down.
 */
static int shared_parts_in(PyObject *value, int depth, Py_ssize_t min_rows, int max_depth, Py_ssize_t max_keys,
                           PyObject *check, PyObject *found) {
    PyObject *k, *item;
    Py_ssize_t pos = 0;
    while (PyDict_Next(value, &pos, &k, &item)) {
        if (PyList_CheckExact(item) || PyTuple_CheckExact(item)) {
            Py_ssize_t n = PySequence_Fast_GET_SIZE(item);
            if (n < min_rows || !PyTuple_CheckExact(PySequence_Fast_ITEMS(item)[0])) continue;
            PyObject *key = PyLong_FromVoidPtr(item);
            if (!key) return -1;
            int there = PyDict_Contains(found, key);
            if (there) { Py_DECREF(key); if (there < 0) return -1; continue; }
            Py_INCREF(item);
            PyObject *ok = PyObject_CallOneArg(check, item);
            int yes = ok ? PyObject_IsTrue(ok) : -1;
            Py_XDECREF(ok);
            int r = yes > 0 ? PyDict_SetItem(found, key, item) : yes;
            Py_DECREF(item);
            Py_DECREF(key);
            if (r < 0) return -1;
        } else if (PyDict_CheckExact(item) && depth < max_depth && PyDict_GET_SIZE(item) <= max_keys) {
            Py_INCREF(item);
            int r = shared_parts_in(item, depth + 1, min_rows, max_depth, max_keys, check, found);
            Py_DECREF(item);
            if (r < 0) return -1;
        }
    }
    return 0;
}

static PyObject *py_shared_parts(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 5 || !PyDict_CheckExact(args[0]) || !PyCallable_Check(args[4])) {
        PyErr_SetString(PyExc_TypeError, "shared_parts(dict, min_rows, max_depth, max_keys, immutable_below)");
        return NULL;
    }
    Py_ssize_t min_rows = PyLong_AsSsize_t(args[1]);
    long max_depth = PyLong_AsLong(args[2]);
    Py_ssize_t max_keys = PyLong_AsSsize_t(args[3]);
    if (PyErr_Occurred()) return NULL;
    if (min_rows < 1) {
        PyErr_SetString(PyExc_ValueError, "min_rows >= 1");
        return NULL;
    }
    PyObject *found = PyDict_New();
    if (!found) return NULL;
    GCPause gp;
    gc_pause(&gp);
    Py_INCREF(args[0]);
    int r = shared_parts_in(args[0], 0, min_rows, (int)max_depth, max_keys, args[4], found);
    Py_DECREF(args[0]);
    gc_resume(&gp);
    if (r < 0) { Py_DECREF(found); return NULL; }
    return found;
}

/*
 * items_estimate(container, depth, max_depth, no_arrays, no_arrays_at_limit, estimate, seen) -> int
 *
 * sizing._items_estimate(items of *container*, depth, seen) for an exact
 * tuple, list or dict (its values): the exact tuples, lists and dicts among
 * the items are summed here, as `pickled_size_estimate` recurses into them;
 * any other item not of a skipped type is handed to *estimate*
 * (`pickled_size_estimate`) with its depth and *seen*, in the same order as
 * the Python loop, and its answer added as Python adds it. The estimate runs
 * Python code: a list is read live by index as a list iterator reads it,
 * and a dict changed meanwhile raises as its values iterator would.
 */
typedef struct {
    int max_depth;
    TypeSet skip, skip_at_limit;
    PyObject *estimate, *seen;
} EstCtx;

/* `type(item) in skip`, asking the set itself for a metaclass of its own */
static int est_skips(TypeSet *ts, PyTypeObject *t) {
    int r = ts_has(ts, t);
    if (r == -2) r = PySet_Contains(ts->set, (PyObject *)t);
    return r;
}

/* total += x, in C while it fits, else as Python ints (*big* holds it) */
static int est_add(long long *total, PyObject **big, PyObject *x) {
    if (!*big && PyLong_CheckExact(x)) {
        int overflow = 0;
        long long v = PyLong_AsLongLongAndOverflow(x, &overflow);
        if (v == -1 && PyErr_Occurred()) return -1;
        if (!overflow && !add_overflows(*total, v, total)) return 0;
    }
    if (!*big) {
        *big = PyLong_FromLongLong(*total);
        if (!*big) return -1;
    }
    PyObject *sum = PyNumber_Add(*big, x);
    if (!sum) return -1;
    Py_SETREF(*big, sum);
    return 0;
}

static PyObject *est_items(EstCtx *cx, PyObject *container, int depth) {
    TypeSet *skip = depth < cx->max_depth ? &cx->skip : &cx->skip_at_limit;
    int is_dict = PyDict_CheckExact(container);
    Py_ssize_t pos = 0, size0 = is_dict ? PyDict_GET_SIZE(container) : 0;
    long long total = 0;
    PyObject *big = NULL, *k, *item;
    for (Py_ssize_t i = 0;; i++) {
        if (is_dict) {
            if (PyDict_GET_SIZE(container) != size0) {
                PyErr_SetString(PyExc_RuntimeError, "dictionary changed size during iteration");
                goto fail;
            }
            if (!PyDict_Next(container, &pos, &k, &item)) break;
        } else {
            if (i >= PySequence_Fast_GET_SIZE(container)) break;
            item = PySequence_Fast_ITEMS(container)[i];
        }
        Py_INCREF(item);
        int r = est_skips(skip, Py_TYPE(item));
        if (r) {
            Py_DECREF(item);
            if (r < 0) goto fail;
            continue;
        }
        PyObject *got;
        if (is_node_type(Py_TYPE(item))) {
            /* pickled_size_estimate(item, depth): 0 at the limit, else its items one down */
            got = depth >= cx->max_depth ? PyLong_FromLong(0) : est_items(cx, item, depth + 1);
        } else {
            PyObject *d = PyLong_FromLong(depth);
            got = d ? PyObject_CallFunctionObjArgs(cx->estimate, item, d, cx->seen, NULL) : NULL;
            Py_XDECREF(d);
        }
        Py_DECREF(item);
        if (!got) goto fail;
        int bad = est_add(&total, &big, got);
        Py_DECREF(got);
        if (bad) goto fail;
    }
    return big ? big : PyLong_FromLongLong(total);
fail:
    Py_XDECREF(big);
    return NULL;
}

static PyObject *py_items_estimate(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 7 || !is_typeset(args[3]) || !is_typeset(args[4]) || !PyCallable_Check(args[5])) {
        PyErr_SetString(PyExc_TypeError,
                        "items_estimate(container, depth, max_depth, no_arrays, at_limit, estimate, seen)");
        return NULL;
    }
    PyObject *container = args[0];
    if (!is_node_type(Py_TYPE(container))) return NOT_IMPL();
    long depth = PyLong_AsLong(args[1]), max_depth = PyLong_AsLong(args[2]);
    if (PyErr_Occurred()) return NULL;
    EstCtx cx;
    cx.max_depth = (int)max_depth;
    ts_init(&cx.skip, args[3]);
    ts_init(&cx.skip_at_limit, args[4]);
    cx.estimate = args[5];
    cx.seen = args[6];
    Py_INCREF(container);
    PyObject *total = est_items(&cx, container, (int)depth);
    Py_DECREF(container);
    return total;
}

/*
 * referrers_in(heap, ids) -> list | NotImplemented
 *
 * shared_objects._referrers_of's read of the heap:
 *     [obj for obj in heap if not ids.isdisjoint(map(id, gc.get_referents(obj)))]
 * in heap order. gc.get_referents reads an object the collector tracks
 * through its type's tp_traverse, and so does this. The ids are sorted once
 * and each referent looked up by its address: no id() per referent (an
 * audit event each) and no int made for it. NotImplemented when an id is not
 * an exact int.
 */
typedef struct {
    const uintptr_t *ids;
    size_t n;
    int found;
} RefCtx;

static int cmp_address(const void *a, const void *b) {
    uintptr_t x = *(const uintptr_t *)a, y = *(const uintptr_t *)b;
    return x < y ? -1 : x > y;
}

static int visit_referent(PyObject *o, void *arg) {
    RefCtx *cx = (RefCtx *)arg;
    uintptr_t key = (uintptr_t)o;
    if (cx->n && bsearch(&key, cx->ids, cx->n, sizeof(uintptr_t), cmp_address)) {
        cx->found = 1;
        return 1; /* stops the traverse: one is enough */
    }
    return 0;
}

static PyObject *py_referrers_in(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 2 || !PyList_CheckExact(args[0]) || !PyAnySet_Check(args[1])) {
        PyErr_SetString(PyExc_TypeError, "referrers_in(list, set)");
        return NULL;
    }
    PyObject *heap = args[0], *ids = args[1];
    Py_ssize_t count = PySet_Size(ids);
    if (count < 0) return NULL;
    uintptr_t *sorted = PyMem_Malloc((size_t)(count ? count : 1) * sizeof(uintptr_t));
    if (!sorted) return PyErr_NoMemory();
    size_t n = 0;
    PyObject *it = PyObject_GetIter(ids), *item;
    if (!it) {
        PyMem_Free(sorted);
        return NULL;
    }
    while ((item = PyIter_Next(it))) {
        void *address = PyLong_CheckExact(item) ? PyLong_AsVoidPtr(item) : NULL;
        Py_DECREF(item);
        if (!address || (Py_ssize_t)n >= count) {
            /* not an exact int, or not an address an id() gives */
            PyErr_Clear();
            Py_DECREF(it);
            PyMem_Free(sorted);
            return NOT_IMPL();
        }
        sorted[n++] = (uintptr_t)address;
    }
    Py_DECREF(it);
    if (PyErr_Occurred()) {
        PyMem_Free(sorted);
        return NULL;
    }
    qsort(sorted, n, sizeof(uintptr_t), cmp_address);
    PyObject *out = PyList_New(0);
    if (!out) {
        PyMem_Free(sorted);
        return NULL;
    }
    RefCtx cx = {sorted, n, 0};
    GCPause gp;
    gc_pause(&gp);
    Py_INCREF(heap);
    int err = 0;
    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(heap); i++) {
        PyObject *o = PyList_GET_ITEM(heap, i);
        if (!PyObject_IS_GC(o)) continue;
        traverseproc traverse = Py_TYPE(o)->tp_traverse;
        if (!traverse) continue;
        cx.found = 0;
        Py_INCREF(o);
        traverse(o, visit_referent, &cx);
        if (cx.found && PyList_Append(out, o) < 0) err = 1;
        Py_DECREF(o);
        if (err) break;
    }
    Py_DECREF(heap);
    gc_resume(&gp);
    PyMem_Free(sorted);
    if (err) {
        Py_DECREF(out);
        return NULL;
    }
    return out;
}

static PyMethodDef cwalk_methods[] = {
    {"leaf_items", (PyCFunction)(void (*)(void))py_leaf_items, METH_FASTCALL,
     "leaf_items(group, mode, leaves) -> bool | NotImplemented"},
    {"seq_children", (PyCFunction)(void (*)(void))py_seq_children, METH_FASTCALL,
     "seq_children(group, leaves, children) -> None | NotImplemented"},
    {"dict_children", (PyCFunction)(void (*)(void))py_dict_children, METH_FASTCALL,
     "dict_children(group, leaves, children) -> None | NotImplemented"},
    {"split_level", (PyCFunction)(void (*)(void))py_split_level, METH_FASTCALL,
     "split_level(children, skipped, extra) -> [(type, list)] | NotImplemented"},
    {"new_in", (PyCFunction)(void (*)(void))py_new_in, METH_FASTCALL, "new_in(group, by_identity, seen) -> list"},
    {"type_set", py_type_set, METH_O, "type_set(list) -> set(map(type, list))"},
    {"tree_facts", (PyCFunction)(void (*)(void))py_tree_facts, METH_FASTCALL,
     "tree_facts(value, leaves, want_walk, top_extra, picks, mode=0) -> (types, held_once, size, nodes, walk) | None | "
     "NotImplemented"},
    {"shared_parts", (PyCFunction)(void (*)(void))py_shared_parts, METH_FASTCALL,
     "shared_parts(dict, min_rows, max_depth, max_keys, immutable_below) -> dict"},
    {"items_estimate", (PyCFunction)(void (*)(void))py_items_estimate, METH_FASTCALL,
     "items_estimate(container, depth, max_depth, no_arrays, at_limit, estimate, seen) -> int | NotImplemented"},
    {"referrers_in", (PyCFunction)(void (*)(void))py_referrers_in, METH_FASTCALL,
     "referrers_in(heap, ids) -> list | NotImplemented"},
    {NULL, NULL, 0, NULL}};

int cwalk_add_functions(PyObject *m) { return PyModule_AddFunctions(m, cwalk_methods); }

#else /* Py_GIL_DISABLED: no kernels; cash runs the Python walks */

int cwalk_add_functions(PyObject *m) {
    (void)m;
    return 0;
}

#endif
