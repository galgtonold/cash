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
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "cwalk.h"
#include "datetime.h"

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
 * held_beyond(value, leaves, limit) -> list | False | None | NotImplemented
 *
 * The walk of `_plain_data.held_beyond_parents` in one pass: the exact
 * dicts, lists and tuples below *value* whose reference count says that
 * something besides their parent holds them, or their parent more than
 * once, level by level (a level's containers in order). The walk holds no
 * references, so a container held by its parent alone has a count of one.
 * Tuples are listed whatever they hold; the caller drops those holding
 * leaves alone. False when more than *limit* containers other than tuples
 * are found, None when *value* is not JSON-like data over *leaves* (as
 * `tree_levels` raises `_NotPlain`), NotImplemented for a type with a
 * metaclass of its own.
 */
static PyObject *py_held_beyond(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 3 || !is_typeset(args[1]) || !PyLong_CheckExact(args[2])) {
        PyErr_SetString(PyExc_TypeError, "held_beyond(value, leaves, limit)");
        return NULL;
    }
    Py_ssize_t limit = PyLong_AsSsize_t(args[2]);
    if (limit == -1 && PyErr_Occurred()) return NULL;
    PyObject *value = args[0];
    if (!is_node_type(Py_TYPE(value))) Py_RETURN_NONE;
    TypeSet leaves;
    ts_init(&leaves, args[1]);
    PtrVec level = {0}, flat = {0}, found = {0};
    int notplain = 0, unknown = 0, over = 0;
    Py_ssize_t others = 0;
    PyObject *out = NULL;
    GCPause gp;
    gc_pause(&gp);
    Py_INCREF(value);
    if (pv_push(&level, value) < 0) goto fail;
    for (int depth = 0;; depth++) {
        if (depth == MAX_LEVELS) { notplain = 1; goto done; }
        flat.n = 0;
        for (Py_ssize_t i = 0; i < level.n; i++) {
            PyObject *c = level.v[i];
            if (PyDict_CheckExact(c)) {
                PyObject *k, *v;
                Py_ssize_t pos = 0;
                while (PyDict_Next(c, &pos, &k, &v)) {
                    int r = ts_has(&leaves, Py_TYPE(k));
                    if (r == -1) goto fail;
                    if (r == -2) { unknown = 1; goto done; }
                    if (!r) { notplain = 1; goto done; }
                    if (pv_push(&flat, v) < 0) goto fail;
                }
            } else {
                Py_ssize_t n = PySequence_Fast_GET_SIZE(c);
                PyObject **items = PySequence_Fast_ITEMS(c);
                for (Py_ssize_t j = 0; j < n; j++)
                    if (pv_push(&flat, items[j]) < 0) goto fail;
            }
        }
        /* As `tree_levels`: a type among the leaves is a leaf; the level is
         * the last when all its types are; any other must be a node. */
        int all_leaves = 1;
        PyTypeObject *last = NULL;
        for (Py_ssize_t i = 0; i < flat.n; i++) {
            PyTypeObject *t = Py_TYPE(flat.v[i]);
            if (t == last) continue;
            last = t;
            int r = ts_has(&leaves, t);
            if (r == -1) goto fail;
            if (r == -2) { unknown = 1; goto done; }
            if (!r) {
                all_leaves = 0;
                if (!is_node_type(t)) { notplain = 1; goto done; }
            }
        }
        if (all_leaves) break;
        level.n = 0;
        for (Py_ssize_t i = 0; i < flat.n; i++) {
            PyObject *o = flat.v[i];
            if (!is_node_type(Py_TYPE(o))) continue;
            if (pv_push(&level, o) < 0) goto fail;
            if (Py_REFCNT(o) > 1 && !over) {
                if (!PyTuple_CheckExact(o) && ++others > limit) over = 1;
                else if (pv_push(&found, o) < 0) goto fail;
            }
        }
    }
done:
    if (!notplain && !unknown) out = over ? Py_NewRef(Py_False) : list_of(&found);
    goto cleanup;
fail:
    notplain = unknown = 0;
cleanup:
    pv_free(&level);
    pv_free(&flat);
    pv_free(&found);
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

/*
 * The _asyncio module's own traverse on 3.12.0-3.12.5 can loop for ever:
 * after the module's state it visits each entry of its free list of
 * FutureIter objects, and that list can come back on itself. An entry is a
 * freed iterator, cleared, whose one field links the next entry: never an
 * object of the program's, and 3.12.6 no longer visits the list at all.
 * Only the state comes before the list, and the module's dict comes after
 * the traverse. So the walk stops after FREE_LIST_BOUND visits -- the state
 * is ~20 and the list holds 255 at most -- and the dict is looked at here:
 * the references 3.12.6 gives, none lost.
 */
#define FREE_LIST_BOUND 4096
#define STOPPED_SHORT 2

typedef struct {
    RefCtx *cx;
    long left;
} BoundCtx;

static int visit_bounded(PyObject *o, void *arg) {
    BoundCtx *b = (BoundCtx *)arg;
    if (b->left-- <= 0) return STOPPED_SHORT;
    return visit_referent(o, b->cx);
}

static int loops_on_traverse(PyObject *o) {
#if PY_VERSION_HEX >= 0x030C0000 && PY_VERSION_HEX < 0x030D0000
    /* 3.12.6 on too: the same answer there, so the stop is tested on every 3.12 */
    if (!PyModule_Check(o)) return 0;
    PyModuleDef *def = PyModule_GetDef(o);
    return def && def->m_name && strcmp(def->m_name, "_asyncio") == 0;
#else
    (void)o;
    return 0;
#endif
}

static void traverse_bounded(PyObject *o, traverseproc traverse, RefCtx *cx) {
    BoundCtx b = {cx, FREE_LIST_BOUND};
    if (traverse(o, visit_bounded, &b) == STOPPED_SHORT && !cx->found) {
        PyObject *dict = PyModule_GetDict(o);
        if (dict) visit_referent(dict, cx);
    }
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
        if (loops_on_traverse(o))
            traverse_bounded(o, traverse, &cx);
        else
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

/* key_alias_id(seen, atoms, by_name, kinds, identityless, outside, sets, obj):
 * kept_state.key_aliases's persistent_id, answer for answer. A key pickle
 * asks it of every object it writes, and the Python one took most of the
 * time of keying a column of a million timestamps. *kinds* is
 * {type: bool} of the answers of *identityless* (a numpy dtype or number
 * type?) asked so far; *outside* is a dict or None. *sets* is None or the
 * callable asked of an exact frozenset, and of an exact set met for the
 * first time (kept_state.sorted_set_ids): its answer is this one. */
static PyObject *py_key_alias_id(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 8 || !PyDict_CheckExact(args[0]) || !PyAnySet_Check(args[1]) || !PyAnySet_Check(args[2]) ||
        !PyDict_CheckExact(args[3]) || (args[5] != Py_None && !PyDict_CheckExact(args[5]))) {
        PyErr_SetString(PyExc_TypeError, "key_alias_id(dict, set, set, dict, callable, dict | None, sets, obj)");
        return NULL;
    }
    PyObject *seen = args[0], *kinds = args[3], *outside = args[5], *sets = args[6], *obj = args[7];
    if (sets != Py_None && Py_IS_TYPE(obj, &PyFrozenSet_Type)) return PyObject_CallOneArg(sets, obj);
    PyObject *t = (PyObject *)Py_TYPE(obj);
    int in = PySet_Contains(args[1], t);
    if (in == 0) in = PySet_Contains(args[2], t);
    if (in < 0) return NULL;
    if (in || PyType_IsSubtype(Py_TYPE(obj), &PyType_Type)) Py_RETURN_NONE;
    PyObject *known = PyDict_GetItemWithError(kinds, t); /* borrowed */
    if (known == NULL) {
        if (PyErr_Occurred()) {
            if (!PyErr_ExceptionMatches(PyExc_TypeError)) return NULL;
            PyErr_Clear(); /* an unhashable type: not a numpy one */
        } else {
            PyObject *answer = PyObject_CallOneArg(args[4], t);
            if (!answer) return NULL;
            int truth = PyObject_IsTrue(answer);
            Py_DECREF(answer);
            if (truth < 0) return NULL;
            if (PyDict_SetItem(kinds, t, truth ? Py_True : Py_False) < 0) return NULL;
            if (truth) Py_RETURN_NONE;
        }
    } else if (known == Py_True) {
        Py_RETURN_NONE;
    }
    PyObject *key = PyLong_FromVoidPtr(obj);
    if (!key) return NULL;
    PyObject *found = PyDict_GetItemWithError(seen, key); /* borrowed */
    if (found) {
        Py_DECREF(key);
        return Py_BuildValue("(sO)", "__cash_alias__", PyTuple_GET_ITEM(found, 0));
    }
    if (PyErr_Occurred()) {
        Py_DECREF(key);
        return NULL;
    }
    if (outside != Py_None) {
        PyObject *where = PyDict_GetItemWithError(outside, key); /* borrowed */
        if (where && where != Py_None) {
            Py_DECREF(key);
            return Py_BuildValue("(sO)", "__cash_held__", where);
        }
        if (PyErr_Occurred()) {
            Py_DECREF(key);
            return NULL;
        }
    }
    PyObject *entry = Py_BuildValue("(nO)", PyDict_GET_SIZE(seen), obj);
    if (!entry) {
        Py_DECREF(key);
        return NULL;
    }
    int err = PyDict_SetItem(seen, key, entry);
    Py_DECREF(entry);
    Py_DECREF(key);
    if (err < 0) return NULL;
    if (sets != Py_None && Py_IS_TYPE(obj, &PySet_Type)) return PyObject_CallOneArg(sets, obj);
    Py_RETURN_NONE;
}

/* set_ops_scan(state, data) -> bool: does the pickle written so far hold a
 * set with an item in it, or a frozenset with one? *data* is the next piece
 * of a protocol 2-5 pickle as the pickler hands it to its file; *state* a
 * bytearray from set_scan_state(), carried from one piece to the next. The
 * opcodes are read one by one, each argument and each item's raw bytes
 * skipped by their length, so the bytes of a number, a string or an array
 * never read as a set (stored_pickle._Sink). A set's items are added by
 * ADDITEMS, which nothing else writes; a frozenset with items is FROZENSET
 * not straight after its MARK.
 * An opcode not known here answers True: the caller then sorts. Once True,
 * it stays True. */
typedef struct {
    uint64_t skip;
    uint8_t hdr[9];
    uint8_t have, need, varlen, last_mark, found;
} SetScan;

static int set_scan_shape(uint8_t op, uint8_t *argb, uint8_t *varlen) {
    *varlen = 0;
    switch (op) {
    case '(': case ')': case '.': case '0': case '1': case '2': case 'N': case '}': case 'a': case 'b':
    case 'd': case ']': case 'e': case 'l': case 's': case 'u': case 't': case 'R': case 'o': case 'Q':
    case 0x81: case 0x85: case 0x86: case 0x87: case 0x88: case 0x89: case 0x8f: case 0x90: case 0x91:
    case 0x92: case 0x93: case 0x94: case 0x97: case 0x98:
        *argb = 0;
        return 1;
    case 'K': case 'h': case 'q': case 0x80: case 0x82:
        *argb = 1;
        return 1;
    case 'U': case 'C': case 0x8a: case 0x8c:
        *argb = 1;
        *varlen = 1;
        return 1;
    case 'M': case 0x83:
        *argb = 2;
        return 1;
    case 'J': case 'j': case 'r': case 0x84:
        *argb = 4;
        return 1;
    case 'X': case 'T': case 'B': case 0x8b:
        *argb = 4;
        *varlen = 1;
        return 1;
    case 'G': case 0x95:
        *argb = 8;
        return 1;
    case 0x8d: case 0x8e: case 0x96:
        *argb = 8;
        *varlen = 1;
        return 1;
    default:
        return 0;
    }
}

/* One whole opcode: *op*, its argument *arg* of *argb* bytes. */
static void set_scan_op(SetScan *s, uint8_t op, const uint8_t *arg, uint8_t argb, uint8_t varlen) {
    if (varlen) {
        uint64_t n = 0;
        for (int k = argb - 1; k >= 0; k--) n = (n << 8) | arg[k];
        s->skip = n;
        if (argb == 4 && (n & 0x80000000u) && (op == 'T' || op == 0x8b)) s->found = 1; /* a negative length */
    }
    /* What the memo or a frame puts between an opcode and the next says nothing. */
    if (op == 0x94 || op == 'q' || op == 'r' || op == 0x95) return;
    if (op == 0x90 || (op == 0x91 && !s->last_mark)) s->found = 1;
    s->last_mark = op == '(';
}

static PyObject *py_set_scan_state(PyObject *self, PyObject *unused) {
    PyObject *state = PyByteArray_FromStringAndSize(NULL, (Py_ssize_t)sizeof(SetScan));
    if (state) memset(PyByteArray_AS_STRING(state), 0, sizeof(SetScan));
    return state;
}

static PyObject *py_set_ops_scan(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 2 || !PyByteArray_CheckExact(args[0]) || PyByteArray_GET_SIZE(args[0]) != (Py_ssize_t)sizeof(SetScan)) {
        PyErr_SetString(PyExc_TypeError, "set_ops_scan(state from set_scan_state(), data)");
        return NULL;
    }
    SetScan s;
    memcpy(&s, PyByteArray_AS_STRING(args[0]), sizeof s);
    if (s.found) Py_RETURN_TRUE;
    Py_buffer view;
    if (PyObject_GetBuffer(args[1], &view, PyBUF_SIMPLE) < 0) return NULL;
    const uint8_t *p = (const uint8_t *)view.buf;
    uint64_t n = (uint64_t)view.len, i = 0;
    while (i < n && !s.found) {
        if (s.skip) {
            uint64_t k = n - i < s.skip ? n - i : s.skip;
            i += k;
            s.skip -= k;
            continue;
        }
        if (s.have == 0) {
            uint8_t op = p[i], argb, varlen;
            if (!set_scan_shape(op, &argb, &varlen)) {
                s.found = 1;
                break;
            }
            if (i + 1 + argb <= n) { /* the whole opcode in this piece */
                set_scan_op(&s, op, p + i + 1, argb, varlen);
                i += 1 + (uint64_t)argb;
                continue;
            }
            s.hdr[0] = op;
            s.have = 1;
            s.need = (uint8_t)(1 + argb);
            s.varlen = varlen;
            i++;
            continue;
        }
        s.hdr[s.have++] = p[i++];
        if (s.have == s.need) {
            set_scan_op(&s, s.hdr[0], s.hdr + 1, (uint8_t)(s.need - 1), s.varlen);
            s.have = 0;
        }
    }
    PyBuffer_Release(&view);
    memcpy(PyByteArray_AS_STRING(args[0]), &s, sizeof s);
    return PyBool_FromLong(s.found);
}

/* Does *o* hash the same in every process (kept_state._hashed_alike)? An
 * int or bool, a float or complex number other than NaN, None when
 * *none_alike* (Python 3.12+), or a tuple or frozenset of those. -1 on error. */
/* None hashes alike in every process from Python 3.12 on (before, its address). */
#define NONE_HASHED_ALIKE (PY_VERSION_HEX >= 0x030C0000)

static int hashed_alike(PyObject *o, int none_alike) {
    PyTypeObject *t = Py_TYPE(o);
    if (t == &PyLong_Type || t == &PyBool_Type) return 1;
    if (o == Py_None) return none_alike;
    if (t == &PyFloat_Type) return !isnan(PyFloat_AS_DOUBLE(o));
    if (t == &PyComplex_Type) {
        Py_complex c = PyComplex_AsCComplex(o);
        return !isnan(c.real) && !isnan(c.imag);
    }
    if (t != &PyTuple_Type && t != &PyFrozenSet_Type) return 0;
    if (Py_EnterRecursiveCall(" while keying a set's order")) return -1;
    int alike = 1;
    if (t == &PyTuple_Type) {
        for (Py_ssize_t i = 0; alike == 1 && i < PyTuple_GET_SIZE(o); i++) alike = hashed_alike(PyTuple_GET_ITEM(o, i), none_alike);
    } else {
        PyObject *it = PyObject_GetIter(o), *item;
        if (!it) alike = -1;
        while (alike == 1 && it && (item = PyIter_Next(it))) {
            alike = hashed_alike(item, none_alike);
            Py_DECREF(item);
        }
        Py_XDECREF(it);
        if (alike == 1 && PyErr_Occurred()) alike = -1;
    }
    Py_LeaveRecursiveCall();
    return alike;
}

typedef struct {
    PyObject *item;
    uint32_t at;
} placed_t;

static int by_address(const void *a, const void *b) {
    uintptr_t x = (uintptr_t)((const placed_t *)a)->item, y = (uintptr_t)((const placed_t *)b)->item;
    return x < y ? -1 : x > y;
}

static void put_place(unsigned char *bytes, Py_ssize_t i, size_t at) {
    bytes[4 * i] = (unsigned char)(at & 0xff);
    bytes[4 * i + 1] = (unsigned char)((at >> 8) & 0xff);
    bytes[4 * i + 2] = (unsigned char)((at >> 16) & 0xff);
    bytes[4 * i + 3] = (unsigned char)((at >> 24) & 0xff);
}

/* Each of *walk*'s *n* items' place in *order* (sorted, the same objects),
 * found by its value in an array of the sorted values: for a set of floats
 * alone, or of ints alone that fit 64 bits, *order* ascending by value. 1
 * when done, 0 when not (the caller places them by address), -1 on error. */
static int place_by_value(PyObject **walk, PyObject **order, Py_ssize_t n, unsigned char *bytes) {
    PyTypeObject *kind = Py_TYPE(order[0]);
    if (kind != &PyFloat_Type && kind != &PyLong_Type) return 0;
    for (Py_ssize_t i = 0; i < n; i++)
        if (!Py_IS_TYPE(order[i], kind)) return 0;
    if (kind == &PyFloat_Type) {
        double *keys = PyMem_Malloc((size_t)n * sizeof(double));
        if (!keys) return (PyErr_NoMemory(), -1);
        for (Py_ssize_t i = 0; i < n; i++) {
            keys[i] = PyFloat_AS_DOUBLE(order[i]);
            if (i && !(keys[i - 1] < keys[i])) { /* not in value order (canonical_form's) */
                PyMem_Free(keys);
                return 0;
            }
        }
        for (Py_ssize_t i = 0; i < n; i++) {
            double v = PyFloat_AS_DOUBLE(walk[i]);
            size_t lo = 0, hi = (size_t)n;
            while (lo < hi) {
                size_t mid = lo + (hi - lo) / 2;
                if (keys[mid] < v) lo = mid + 1;
                else hi = mid;
            }
            /* -0.0 and 0.0 are one item: whichever is held is at lo */
            if (lo >= (size_t)n || keys[lo] != v) {
                PyMem_Free(keys);
                PyErr_SetString(PyExc_ValueError, "set_walk_order: an item walked is not among the sorted items");
                return -1;
            }
            put_place(bytes, i, lo);
        }
        PyMem_Free(keys);
        return 1;
    }
    long long *keys = PyMem_Malloc((size_t)n * sizeof(long long));
    if (!keys) return (PyErr_NoMemory(), -1);
    for (Py_ssize_t i = 0; i < n; i++) {
        int overflow = 0;
        keys[i] = PyLong_AsLongLongAndOverflow(order[i], &overflow);
        if (overflow || (keys[i] == -1 && PyErr_Occurred()) || (i && !(keys[i - 1] < keys[i]))) {
            PyErr_Clear();
            PyMem_Free(keys);
            return 0;
        }
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        long long v = PyLong_AsLongLong(walk[i]);
        size_t lo = 0, hi = (size_t)n;
        while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            if (keys[mid] < v) lo = mid + 1;
            else hi = mid;
        }
        if (lo >= (size_t)n || keys[lo] != v) {
            PyMem_Free(keys);
            PyErr_SetString(PyExc_ValueError, "set_walk_order: an item walked is not among the sorted items");
            return -1;
        }
        put_place(bytes, i, lo);
    }
    PyMem_Free(keys);
    return 1;
}

/* Each of the *k* items *alike*'s place in *order* (*n* objects), found by
 * address in one sorted table. 1 when done, -1 on error. */
static int place_by_address(PyObject **alike, Py_ssize_t k, PyObject **order, Py_ssize_t n, unsigned char *bytes) {
    placed_t *table = PyMem_Malloc((size_t)n * sizeof(placed_t));
    if (!table) return (PyErr_NoMemory(), -1);
    for (Py_ssize_t i = 0; i < n; i++) {
        table[i].item = order[i];
        table[i].at = (uint32_t)i;
    }
    qsort(table, (size_t)n, sizeof(placed_t), by_address);
    int done = 1;
    for (Py_ssize_t i = 0; i < k; i++) {
        placed_t key = {alike[i], 0};
        placed_t *found = bsearch(&key, table, (size_t)n, sizeof(placed_t), by_address);
        if (!found) {
            PyErr_SetString(PyExc_ValueError, "set_walk_order: an item not in sorted");
            done = -1;
            break;
        }
        put_place(bytes, i, found->at);
    }
    PyMem_Free(table);
    return done;
}

/* set_walk_order(walked, sorted, none_alike): kept_state.set_walk_order,
 * answer for answer. *walked* is a set's items in the order it iterates,
 * *sorted* the same objects sorted. None when fewer than two of them hash
 * alike in every process (hashed_alike), or when those iterate in sorted
 * order; else bytes: each such item's place in *sorted*, in the order the set
 * iterates them, as little-endian uint32. Found by value in the sorted
 * values for floats or 64-bit ints alone, else by address in one sorted
 * table, never a dict: a 100,000-item set took 0.1 s in Python. */
static PyObject *set_walk_order(PyObject *walked, PyObject *sorted, int none_alike) {
    Py_ssize_t n = PySequence_Fast_GET_SIZE(walked);
    if (n != PySequence_Fast_GET_SIZE(sorted) || n > (Py_ssize_t)UINT32_MAX) {
        PyErr_SetString(PyExc_ValueError, "set_walk_order: walked and sorted differ in length");
        return NULL;
    }
    PyObject **walk = PySequence_Fast_ITEMS(walked), **order = PySequence_Fast_ITEMS(sorted);
    PyObject **alike = PyMem_Malloc((size_t)(n ? n : 1) * sizeof(PyObject *));
    if (!alike) return PyErr_NoMemory();
    Py_ssize_t k = 0;
    for (Py_ssize_t i = 0; i < n; i++) {
        int is = hashed_alike(walk[i], none_alike);
        if (is < 0) {
            PyMem_Free(alike);
            return NULL;
        }
        if (is) alike[k++] = walk[i];
    }
    if (k < 2) {
        PyMem_Free(alike);
        Py_RETURN_NONE;
    }
    PyObject *out = PyBytes_FromStringAndSize(NULL, k * 4);
    if (!out) {
        PyMem_Free(alike);
        return NULL;
    }
    unsigned char *bytes = (unsigned char *)PyBytes_AS_STRING(out);
    int found = k == n ? place_by_value(alike, order, n, bytes) : 0;
    if (found == 0) found = place_by_address(alike, k, order, n, bytes);
    PyMem_Free(alike);
    if (found < 0) {
        Py_DECREF(out);
        return NULL;
    }
    /* in sorted order: the plain one, keyed as before */
    uint32_t last = 0;
    for (Py_ssize_t i = 0; i < k; i++) {
        uint32_t at = (uint32_t)bytes[4 * i] | (uint32_t)bytes[4 * i + 1] << 8 | (uint32_t)bytes[4 * i + 2] << 16 |
                      (uint32_t)bytes[4 * i + 3] << 24;
        if (i && at <= last) return out;
        last = at;
    }
    Py_DECREF(out);
    Py_RETURN_NONE;
}

static PyObject *py_set_walk_order(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 3 || !PyList_CheckExact(args[0]) || !(PyList_CheckExact(args[1]) || PyTuple_CheckExact(args[1]))) {
        PyErr_SetString(PyExc_TypeError, "set_walk_order(list, list | tuple, none_alike)");
        return NULL;
    }
    int none_alike = PyObject_IsTrue(args[2]);
    if (none_alike < 0) return NULL;
    return set_walk_order(args[0], args[1], none_alike);
}

/* sorted_set_id(forms, made, spec, obj): kept_state.sorted_set_ids's
 * persistent_id, answer for answer: None for anything but an exact set or
 * frozenset of two or more items; else the form a key writes in its place,
 * (spec[0], spec[1] for a set or spec[2] for a frozenset, its items sorted,
 * and, for numbers, the order it iterates in where that is not the plain
 * one: set_walk_order),
 * kept in *made* ({id: (set, form)}) and given again for the same set. A set
 * of strings alone, ints alone, bytes alone or floats alone (no NaN) is sorted here; any other is
 * left to *forms*, which keeps its own in *made*. A key pickle asks it of
 * every object it writes, and a memo of 245,760 small frozensets took a
 * Python call apiece. */
static PyObject *py_sorted_set_id(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 4 || !PyDict_CheckExact(args[1]) || !PyTuple_CheckExact(args[2]) || PyTuple_GET_SIZE(args[2]) != 3) {
        PyErr_SetString(PyExc_TypeError, "sorted_set_id(callable, dict, (mark, set name, frozenset name), obj)");
        return NULL;
    }
    PyObject *made = args[1], *spec = args[2], *obj = args[3];
    int frozen = Py_IS_TYPE(obj, &PyFrozenSet_Type);
    if ((!frozen && !Py_IS_TYPE(obj, &PySet_Type)) || PySet_GET_SIZE(obj) < 2) Py_RETURN_NONE;
    PyObject *key = PyLong_FromVoidPtr(obj);
    if (!key) return NULL;
    PyObject *found = PyDict_GetItemWithError(made, key); /* borrowed */
    if (found) {
        Py_DECREF(key);
        if (!PyTuple_CheckExact(found) || PyTuple_GET_SIZE(found) != 2) {
            PyErr_SetString(PyExc_TypeError, "sorted_set_id: made holds (set, form) pairs");
            return NULL;
        }
        return Py_NewRef(PyTuple_GET_ITEM(found, 1));
    }
    if (PyErr_Occurred()) {
        Py_DECREF(key);
        return NULL;
    }
    PyObject *items = PySequence_List(obj);
    if (!items) {
        Py_DECREF(key);
        return NULL;
    }
    PyTypeObject *kind = Py_TYPE(PyList_GET_ITEM(items, 0));
    int plain = kind == &PyUnicode_Type || kind == &PyLong_Type || kind == &PyBytes_Type || kind == &PyFloat_Type;
    for (Py_ssize_t i = 1; plain && i < PyList_GET_SIZE(items); i++) plain = Py_IS_TYPE(PyList_GET_ITEM(items, i), kind);
    /* floats sort as themselves only with no NaN among them (kept_state._sorted_items) */
    for (Py_ssize_t i = 0; plain && kind == &PyFloat_Type && i < PyList_GET_SIZE(items); i++)
        plain = !isnan(PyFloat_AS_DOUBLE(PyList_GET_ITEM(items, i)));
    if (!plain) {
        Py_DECREF(items);
        Py_DECREF(key);
        return PyObject_CallOneArg(args[0], obj);
    }
    /* Only numbers hash alike in every process; a set of strings or bytes
     * keys order-free (kept_state.set_walk_order), and its walk is not kept. */
    int numbers = kind == &PyLong_Type || kind == &PyFloat_Type;
    PyObject *walked = numbers ? PyList_GetSlice(items, 0, PyList_GET_SIZE(items)) : NULL;
    PyObject *sorted = (numbers && walked == NULL) || PyList_Sort(items) < 0 ? NULL : PyList_AsTuple(items);
    Py_DECREF(items);
    PyObject *order = !sorted ? NULL : numbers ? set_walk_order(walked, sorted, NONE_HASHED_ALIKE) : Py_NewRef(Py_None);
    Py_XDECREF(walked);
    PyObject *form = NULL;
    if (order == Py_None)
        form = PyTuple_Pack(3, PyTuple_GET_ITEM(spec, 0), PyTuple_GET_ITEM(spec, frozen ? 2 : 1), sorted);
    else if (order)
        form = PyTuple_Pack(4, PyTuple_GET_ITEM(spec, 0), PyTuple_GET_ITEM(spec, frozen ? 2 : 1), sorted, order);
    Py_XDECREF(order);
    Py_XDECREF(sorted);
    PyObject *entry = form ? PyTuple_Pack(2, obj, form) : NULL;
    int err = entry ? PyDict_SetItem(made, key, entry) : -1;
    Py_XDECREF(entry);
    Py_DECREF(key);
    if (err < 0) {
        Py_XDECREF(form);
        return NULL;
    }
    return form;
}

/* reduce_dict_apart(full, apart, protocol, obj): kept_state's reducer of a
 * type whose own reduce drops its instances' __dict__ (`_reducer_with_dict`
 * under `_names_apart`), answer for answer. An instance with something in
 * its __dict__ goes to full(obj); an empty one is obj.__reduce_ex__(protocol),
 * a reduce with a state handed to apart(reduced). A key pickle asks it of
 * every pandas Timestamp it writes: two Python calls each, most of keying a
 * column of a million of them. */
static PyObject *str_dunder_dict, *str_reduce_ex;

static PyObject *py_reduce_dict_apart(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 4) {
        PyErr_SetString(PyExc_TypeError, "reduce_dict_apart(full, apart, protocol, obj)");
        return NULL;
    }
    if (!str_dunder_dict && !(str_dunder_dict = PyUnicode_InternFromString("__dict__"))) return NULL;
    if (!str_reduce_ex && !(str_reduce_ex = PyUnicode_InternFromString("__reduce_ex__"))) return NULL;
    PyObject *obj = args[3];
    PyObject *attrs = PyObject_GetAttr(obj, str_dunder_dict);
    int empty;
    if (!attrs) {
        if (!PyErr_ExceptionMatches(PyExc_AttributeError)) return NULL;
        PyErr_Clear();
        empty = 1;
    } else {
        int truth = PyDict_CheckExact(attrs) ? PyDict_GET_SIZE(attrs) > 0 : PyObject_IsTrue(attrs);
        Py_DECREF(attrs);
        if (truth < 0) return NULL;
        empty = !truth;
    }
    if (!empty) return PyObject_CallOneArg(args[0], obj);
    PyObject *reduced = PyObject_CallMethodOneArg(obj, str_reduce_ex, args[2]);
    if (!reduced || !PyTuple_CheckExact(reduced) || PyTuple_GET_SIZE(reduced) < 3) return reduced;
    PyObject *state = PyTuple_GET_ITEM(reduced, 2);
    if (!PyDict_CheckExact(state) && !PyTuple_CheckExact(state)) return reduced;
    PyObject *apart = PyObject_CallOneArg(args[1], reduced);
    Py_DECREF(reduced);
    return apart;
}

/*
 * values_in_tuples(items, exact, levels) -> bool | NotImplemented
 *
 * shared_objects._values_in_tuples(items, exact) with _TUPLE_LEVELS =
 * *levels*: every item of the exact list *items* is of a type in *exact*,
 * or an exact tuple whose items are so in turn, a tuple at most *levels* - 2
 * tuples deep (the Python version reads *levels* levels, each a level of
 * the tuples of the one above). Read depth first: the answer is whether
 * every item passes, whichever is met first.
 */
static int tuples_hold_values(TypeSet *ts, PyObject *obj, int level, int last) {
    int r = ts_has(ts, Py_TYPE(obj));
    if (r != 0) return r;
    if (!PyTuple_CheckExact(obj) || level > last) return 0;
    Py_ssize_t n = PyTuple_GET_SIZE(obj);
    for (Py_ssize_t i = 0; i < n; i++) {
        r = tuples_hold_values(ts, PyTuple_GET_ITEM(obj, i), level + 1, last);
        if (r != 1) return r;
    }
    return 1;
}

static PyObject *py_values_in_tuples(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 3 || !PyList_CheckExact(args[0]) || !is_typeset(args[1]) || !PyLong_CheckExact(args[2])) {
        PyErr_SetString(PyExc_TypeError, "values_in_tuples(list, set of types, int)");
        return NULL;
    }
    long levels = PyLong_AsLong(args[2]);
    if (levels < 1 || levels > 64) {
        PyErr_SetString(PyExc_ValueError, "levels 1..64");
        return NULL;
    }
    PyObject *items = args[0];
    TypeSet ts;
    ts_init(&ts, args[1]);
    GCPause gp;
    gc_pause(&gp);
    Py_INCREF(items);
    int out = 1;
    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(items) && out == 1; i++)
        out = tuples_hold_values(&ts, PyList_GET_ITEM(items, i), 0, (int)levels - 2);
    Py_DECREF(items);
    gc_resume(&gp);
    if (out == -1) return NULL;
    if (out == -2) return NOT_IMPL();
    return PyBool_FromLong(out);
}

/*
 * nested_leaves(items, leaves, levels, longest)
 *   -> (nodes, texts, ints, kinds, lists) | False | NotImplemented
 *
 * identity_holders._Walk._read_nested's read of the exact list *items*, a
 * level at a time, the containers being exact lists and tuples: False
 * where a level holds anything but those and types of *leaves*, or where
 * containers are still met at level *levels* - 1. Else *nodes*: the
 * containers, level by level, each level's in order; *kinds*: the types
 * of the leaves; *lists*: how many of the nodes are lists. Of the leaves,
 * what _read_leaves reads: *texts*, the exact strs of at most *longest*
 * characters or holding a line break, each once in the order first met
 * (dict.fromkeys), and *ints*, the exact ints in order.
 */
static PyObject *py_nested_leaves(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 4 || !PyList_CheckExact(args[0]) || !is_typeset(args[1]) || !PyLong_CheckExact(args[2]) ||
        !PyLong_CheckExact(args[3])) {
        PyErr_SetString(PyExc_TypeError, "nested_leaves(list, set of types, int, int)");
        return NULL;
    }
    long levels = PyLong_AsLong(args[2]);
    Py_ssize_t longest = PyLong_AsSsize_t(args[3]);
    if (levels < 1 || levels > 64 || longest < 0) {
        PyErr_SetString(PyExc_ValueError, "levels 1..64, longest >= 0");
        return NULL;
    }
    TypeSet ts;
    ts_init(&ts, args[1]);
    PyObject *nodes = PyList_New(0), *texts = PyDict_New(), *ints = PyList_New(0), *kinds = PySet_New(NULL);
    PyObject *level = Py_NewRef(args[0]), *next = NULL, *out = NULL;
    Py_ssize_t lists = 0;
    int verdict = 0; /* 1: read, 0: False, -1: error, -2: NotImplemented */
    GCPause gp;
    gc_pause(&gp);
    if (!nodes || !texts || !ints || !kinds) {
        verdict = -1;
        goto done;
    }
    for (long at = 0; at < levels; at++) {
        next = PyList_New(0);
        if (!next) {
            verdict = -1;
            goto done;
        }
        Py_ssize_t n = PyList_GET_SIZE(level);
        int met = 0;
        for (Py_ssize_t i = 0; i < n; i++) {
            PyObject *obj = PyList_GET_ITEM(level, i);
            if (PyList_CheckExact(obj) || PyTuple_CheckExact(obj)) {
                met = 1;
                if (PyList_Append(nodes, obj) < 0) {
                    verdict = -1;
                    goto done;
                }
                if (PyList_CheckExact(obj)) lists++;
                Py_ssize_t m = PySequence_Fast_GET_SIZE(obj);
                PyObject **inner = PySequence_Fast_ITEMS(obj);
                for (Py_ssize_t j = 0; j < m; j++)
                    if (PyList_Append(next, inner[j]) < 0) {
                        verdict = -1;
                        goto done;
                    }
                continue;
            }
            int r = ts_has(&ts, Py_TYPE(obj));
            if (r != 1) {
                verdict = r;
                goto done;
            }
            if (PySet_Add(kinds, (PyObject *)Py_TYPE(obj)) < 0) {
                verdict = -1;
                goto done;
            }
            if (PyUnicode_CheckExact(obj)) {
                Py_ssize_t len = PyUnicode_GET_LENGTH(obj);
                if (len <= longest || PyUnicode_FindChar(obj, '\n', 0, len, 1) >= 0) {
                    PyObject *had = PyDict_SetDefault(texts, obj, Py_None);
                    if (!had) {
                        verdict = -1;
                        goto done;
                    }
                }
            } else if (PyLong_CheckExact(obj)) {
                if (PyList_Append(ints, obj) < 0) {
                    verdict = -1;
                    goto done;
                }
            }
        }
        if (!met) {
            /* this level held leaves alone: the read ends here */
            verdict = 1;
            goto done;
        }
        Py_SETREF(level, next);
        next = NULL;
    }
    verdict = 0; /* containers still met past the last level */
done:
    gc_resume(&gp);
    Py_XDECREF(next);
    Py_XDECREF(level);
    if (verdict == 1) {
        PyObject *found = PyDict_Keys(texts);
        if (found) out = Py_BuildValue("(NNOOn)", (PyObject *)nodes, found, ints, kinds, lists), nodes = NULL;
    } else if (verdict == 0) {
        out = Py_NewRef(Py_False);
    } else if (verdict == -2) {
        out = NOT_IMPL();
    }
    Py_XDECREF(nodes);
    Py_XDECREF(texts);
    Py_XDECREF(ints);
    Py_XDECREF(kinds);
    return out;
}

/*
 * shape_witness(items, levels) -> (int, int) | None
 *
 * A fingerprint of what the exact list *items* holds, for a memo whose
 * answer depends on that content alone (identity_holders._read_flat's
 * read, kept across the statements of a cell). It holds no reference to
 * anything it read: two equal witnesses mean the same objects at the same
 * addresses, of the same types, nested the same way, and every str and int
 * among them of the same value.
 *
 * Every item, at every depth, adds its address and its type's; an exact
 * list or tuple its length, then its items (so a list changed in place, or
 * one freed and another built at its address, differs); an exact str its
 * length and hash, an exact int its value: an object of another type at
 * the same address (one freed, another made there) is told apart by type,
 * and a str or int by value. Anything else is taken as it is: the memo's
 * answer is kept only for immutable leaves, whose type alone is what the
 * answer reads of them.
 *
 * None when a list or tuple is met at depth *levels* - 1 or below (the
 * read gives up there too), or when an exact list or tuple is not what an
 * item claims to be. Two 64-bit lanes, mixed apart: a change that left the
 * witness equal would take a 128-bit collision.
 */
#define SW_P1 0x9E3779B97F4A7C15ULL
#define SW_P2 0xC2B2AE3D27D4EB4FULL
typedef struct {
    uint64_t a, b;
} Witness;
static inline void sw_mix(Witness *w, uint64_t v) {
    w->a = (w->a ^ v) * SW_P1;
    w->a ^= w->a >> 29;
    w->b = (w->b + v + 0x165667B19E3779F9ULL) * SW_P2;
    w->b = (w->b << 31) | (w->b >> 33);
}

static PyObject *py_shape_witness(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 2 || !PyList_CheckExact(args[0]) || !PyLong_CheckExact(args[1])) {
        PyErr_SetString(PyExc_TypeError, "shape_witness(list, int)");
        return NULL;
    }
    long levels = PyLong_AsLong(args[1]);
    if (levels < 1 || levels > 64) {
        PyErr_SetString(PyExc_ValueError, "levels 1..64");
        return NULL;
    }
    struct {
        PyObject *seq;
        Py_ssize_t at;
    } stack[64];
    Witness w = {0x243F6A8885A308D3ULL, 0x13198A2E03707344ULL};
    PyObject *items = args[0];
    Py_INCREF(items);
    int depth = 0;
    stack[0].seq = items;
    stack[0].at = 0;
    sw_mix(&w, (uint64_t)PyList_GET_SIZE(items));
    int ok = 1;
    while (depth >= 0) {
        PyObject *seq = stack[depth].seq;
        Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
        if (stack[depth].at >= n) {
            depth--;
            if (depth >= 0) sw_mix(&w, 0xFFFFFFFFFFFFFFFFULL); /* the end of a container */
            continue;
        }
        PyObject *obj = PySequence_Fast_ITEMS(seq)[stack[depth].at++];
        PyTypeObject *t = Py_TYPE(obj);
        sw_mix(&w, (uint64_t)(uintptr_t)obj);
        sw_mix(&w, (uint64_t)(uintptr_t)t);
        if (t == &PyList_Type || t == &PyTuple_Type) {
            /* items of *items* are at level 0: a container on level levels - 1 is past the read */
            if (depth + 1 >= levels) {
                ok = 0;
                break;
            }
            sw_mix(&w, (uint64_t)PySequence_Fast_GET_SIZE(obj));
            depth++;
            stack[depth].seq = obj;
            stack[depth].at = 0;
        } else if (t == &PyUnicode_Type) {
            Py_hash_t h = PyObject_Hash(obj); /* an exact str: cached, never Python code */
            if (h == -1) {
                ok = -1;
                break;
            }
            sw_mix(&w, (uint64_t)PyUnicode_GET_LENGTH(obj));
            sw_mix(&w, (uint64_t)h);
        } else if (t == &PyLong_Type) {
            int overflow = 0;
            long long v = PyLong_AsLongLongAndOverflow(obj, &overflow);
            if (v == -1 && PyErr_Occurred()) {
                ok = -1;
                break;
            }
            if (overflow) {
                Py_hash_t h = PyObject_Hash(obj);
                if (h == -1) {
                    ok = -1;
                    break;
                }
                sw_mix(&w, (uint64_t)overflow);
                sw_mix(&w, (uint64_t)h);
            } else {
                sw_mix(&w, (uint64_t)v);
            }
        }
    }
    Py_DECREF(items);
    if (ok < 0) return NULL;
    if (!ok) Py_RETURN_NONE;
    return Py_BuildValue("(KK)", (unsigned long long)w.a, (unsigned long long)w.b);
}


/*
 * dated_records(items, probe, levels) -> bytes | None
 *
 * dated_records.dated_records_bytes(items, probe, levels): the content of
 * the exact list *items* as one byte stream, for a column of records that
 * holds dates (student_8's sessions of ``(action, datetime)``), whose
 * pickle cost a microsecond and more a datetime. Every value is a tag and
 * its bytes, little-endian:
 *
 *   list    '[' u64 length, then the items    tuple   '(' u64 length, ...
 *   str     's' u64 n, n bytes of UTF-8 (surrogates passed through)
 *   int     'i' i64 (an exact int that fits; any other: None)
 *   float   'f' the 8 bytes of the double     bool    'T' / 'F'
 *   None    'N'                               bytes   'b' u64 n, the n bytes
 *   datetime 'D' the 10 bytes pickle keeps of a naive exact datetime: year
 *           (2, big-endian), month (fold in its top bit), day, hour,
 *           minute, second, microsecond (3, big-endian)
 *   date    'd' year (2, big-endian), month, day
 *
 * None when anything else is met (a subclass, an aware datetime, a big
 * int, another type), when a list or tuple is met *levels* deep or more
 * (items of *items* are at depth 0), and when no datetime or date is among
 * what the first *probe* items hold: the caller pickles those as before.
 */
typedef struct {
    char *buf;
    Py_ssize_t len, cap;
} DRBuf;

static int dr_grow(DRBuf *b, Py_ssize_t more) {
    if (b->len + more <= b->cap) return 0;
    Py_ssize_t cap = b->cap ? b->cap : 4096;
    while (cap < b->len + more) {
        if (cap > PY_SSIZE_T_MAX / 2) {
            PyErr_NoMemory();
            return -1;
        }
        cap *= 2;
    }
    char *nb = PyMem_Realloc(b->buf, (size_t)cap);
    if (!nb) {
        PyErr_NoMemory();
        return -1;
    }
    b->buf = nb;
    b->cap = cap;
    return 0;
}

static inline void dr_u64(char *at, uint64_t v) {
    for (int k = 0; k < 8; k++) at[k] = (char)((v >> (8 * k)) & 0xFF);
}

static int dr_tagged(DRBuf *b, char tag, const char *data, Py_ssize_t n, int with_length) {
    if (dr_grow(b, 1 + (with_length ? 8 : 0) + n) < 0) return -1;
    b->buf[b->len++] = tag;
    if (with_length) {
        dr_u64(b->buf + b->len, (uint64_t)n);
        b->len += 8;
    }
    if (n) memcpy(b->buf + b->len, data, (size_t)n);
    b->len += n;
    return 0;
}

/* 1 written, 0 not this form, -1 error; *dated set when a date was written */
static int dr_write(DRBuf *b, PyObject *obj, int depth, int levels, int *dated) {
    PyTypeObject *t = Py_TYPE(obj);
    if (t == &PyList_Type || t == &PyTuple_Type) {
        if (depth >= levels) return 0;
        Py_ssize_t n = PySequence_Fast_GET_SIZE(obj);
        char head[8];
        dr_u64(head, (uint64_t)n);
        if (dr_tagged(b, t == &PyList_Type ? '[' : '(', head, 8, 0) < 0) return -1;
        PyObject **items = PySequence_Fast_ITEMS(obj);
        for (Py_ssize_t i = 0; i < n; i++) {
            int r = dr_write(b, items[i], depth + 1, levels, dated);
            if (r != 1) return r;
        }
        return 1;
    }
    if (t == &PyUnicode_Type) {
        if (PyUnicode_IS_ASCII(obj))
            return dr_tagged(b, 's', (const char *)PyUnicode_DATA(obj), PyUnicode_GET_LENGTH(obj), 1) < 0 ? -1 : 1;
        PyObject *utf8 = PyUnicode_AsEncodedString(obj, "utf-8", "surrogatepass");
        if (!utf8) return -1;
        int r = dr_tagged(b, 's', PyBytes_AS_STRING(utf8), PyBytes_GET_SIZE(utf8), 1);
        Py_DECREF(utf8);
        return r < 0 ? -1 : 1;
    }
    if (t == &PyLong_Type) {
        int overflow = 0;
        long long v = PyLong_AsLongLongAndOverflow(obj, &overflow);
        if (v == -1 && PyErr_Occurred()) return -1;
        if (overflow) return 0;
        char data[8];
        dr_u64(data, (uint64_t)v);
        return dr_tagged(b, 'i', data, 8, 0) < 0 ? -1 : 1;
    }
    if (t == &PyFloat_Type) {
        double d = PyFloat_AS_DOUBLE(obj);
        uint64_t bits;
        memcpy(&bits, &d, 8);
        char data[8];
        dr_u64(data, bits);
        return dr_tagged(b, 'f', data, 8, 0) < 0 ? -1 : 1;
    }
    if (t == &PyBool_Type) return dr_tagged(b, obj == Py_True ? 'T' : 'F', NULL, 0, 0) < 0 ? -1 : 1;
    if (obj == Py_None) return dr_tagged(b, 'N', NULL, 0, 0) < 0 ? -1 : 1;
    if (t == &PyBytes_Type) return dr_tagged(b, 'b', PyBytes_AS_STRING(obj), PyBytes_GET_SIZE(obj), 1) < 0 ? -1 : 1;
    if (t == PyDateTimeAPI->DateTimeType) {
        if (PyDateTime_DATE_GET_TZINFO(obj) != Py_None) return 0;
        int year = PyDateTime_GET_YEAR(obj), us = PyDateTime_DATE_GET_MICROSECOND(obj);
        char data[10] = {(char)(year >> 8),
                         (char)(year & 0xFF),
                         (char)(PyDateTime_GET_MONTH(obj) | (PyDateTime_DATE_GET_FOLD(obj) << 7)),
                         (char)PyDateTime_GET_DAY(obj),
                         (char)PyDateTime_DATE_GET_HOUR(obj),
                         (char)PyDateTime_DATE_GET_MINUTE(obj),
                         (char)PyDateTime_DATE_GET_SECOND(obj),
                         (char)(us >> 16),
                         (char)((us >> 8) & 0xFF),
                         (char)(us & 0xFF)};
        *dated = 1;
        return dr_tagged(b, 'D', data, 10, 0) < 0 ? -1 : 1;
    }
    if (t == PyDateTimeAPI->DateType) {
        int year = PyDateTime_GET_YEAR(obj);
        char data[4] = {(char)(year >> 8), (char)(year & 0xFF), (char)PyDateTime_GET_MONTH(obj),
                        (char)PyDateTime_GET_DAY(obj)};
        *dated = 1;
        return dr_tagged(b, 'd', data, 4, 0) < 0 ? -1 : 1;
    }
    return 0;
}

static PyObject *py_dated_records(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 3 || !PyList_CheckExact(args[0]) || !PyLong_CheckExact(args[1]) || !PyLong_CheckExact(args[2])) {
        PyErr_SetString(PyExc_TypeError, "dated_records(list, int, int)");
        return NULL;
    }
    Py_ssize_t probe = PyLong_AsSsize_t(args[1]);
    long levels = PyLong_AsLong(args[2]);
    if (probe < 0 || levels < 1 || levels > 64) {
        PyErr_SetString(PyExc_ValueError, "probe >= 0, levels 1..64");
        return NULL;
    }
#if PY_BIG_ENDIAN
    return NOT_IMPL();
#endif
    if (!PyDateTimeAPI) {
        PyDateTime_IMPORT;
        if (!PyDateTimeAPI) return NULL;
    }
    PyObject *items = args[0];
    Py_INCREF(items);
    DRBuf b = {NULL, 0, 0};
    GCPause gp;
    gc_pause(&gp);
    int r = 1, dated = 0;
    /* The list's own head, then the probe: no date among what its first
     * items hold, and nothing more is read. */
    Py_ssize_t n = PyList_GET_SIZE(items);
    char head[8];
    dr_u64(head, (uint64_t)n);
    if (dr_tagged(&b, '[', head, 8, 0) < 0) r = -1;
    for (Py_ssize_t i = 0; r == 1 && i < n; i++) {
        if (i == probe && !dated) {
            r = 0;
            break;
        }
        if (i >= PyList_GET_SIZE(items)) { /* cannot shrink with the GC paused; checked anyway */
            r = 0;
            break;
        }
        r = dr_write(&b, PyList_GET_ITEM(items, i), 1, (int)levels, &dated);
    }
    if (r == 1 && !dated) r = 0;
    gc_resume(&gp);
    Py_DECREF(items);
    PyObject *out = NULL;
    if (r == 1)
        out = PyBytes_FromStringAndSize(b.buf, b.len);
    else if (r == 0)
        out = Py_NewRef(Py_None);
    PyMem_Free(b.buf);
    return out;
}

static PyMethodDef cwalk_methods[] = {
    {"shape_witness", (PyCFunction)(void (*)(void))py_shape_witness, METH_FASTCALL,
     "shape_witness(items, levels) -> (int, int) | None"},
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
    {"values_in_tuples", (PyCFunction)(void (*)(void))py_values_in_tuples, METH_FASTCALL,
     "values_in_tuples(items, exact, levels) -> bool | NotImplemented"},
    {"nested_leaves", (PyCFunction)(void (*)(void))py_nested_leaves, METH_FASTCALL,
     "nested_leaves(items, leaves, levels, longest) -> (nodes, texts, ints, kinds, lists) | False | NotImplemented"},
    {"tree_facts", (PyCFunction)(void (*)(void))py_tree_facts, METH_FASTCALL,
     "tree_facts(value, leaves, want_walk, top_extra, picks, mode=0) -> (types, held_once, size, nodes, walk) | None | "
     "NotImplemented"},
    {"held_beyond", (PyCFunction)(void (*)(void))py_held_beyond, METH_FASTCALL,
     "held_beyond(value, leaves, limit) -> list | False | None | NotImplemented"},
    {"shared_parts", (PyCFunction)(void (*)(void))py_shared_parts, METH_FASTCALL,
     "shared_parts(dict, min_rows, max_depth, max_keys, immutable_below) -> dict"},
    {"items_estimate", (PyCFunction)(void (*)(void))py_items_estimate, METH_FASTCALL,
     "items_estimate(container, depth, max_depth, no_arrays, at_limit, estimate, seen) -> int | NotImplemented"},
    {"referrers_in", (PyCFunction)(void (*)(void))py_referrers_in, METH_FASTCALL,
     "referrers_in(heap, ids) -> list | NotImplemented"},
    {"key_alias_id", (PyCFunction)(void (*)(void))py_key_alias_id, METH_FASTCALL,
     "key_alias_id(seen, atoms, by_name, kinds, identityless, outside, sets, obj) -> None | mark"},
    {"set_scan_state", (PyCFunction)py_set_scan_state, METH_NOARGS, "set_scan_state() -> bytearray"},
    {"set_ops_scan", (PyCFunction)(void (*)(void))py_set_ops_scan, METH_FASTCALL,
     "set_ops_scan(state, data) -> bool"},
    {"set_walk_order", (PyCFunction)(void (*)(void))py_set_walk_order, METH_FASTCALL,
     "set_walk_order(walked, sorted, none_alike) -> None | bytes"},
    {"sorted_set_id", (PyCFunction)(void (*)(void))py_sorted_set_id, METH_FASTCALL,
     "sorted_set_id(forms, made, spec, obj) -> None | form"},
    {"dated_records", (PyCFunction)(void (*)(void))py_dated_records, METH_FASTCALL,
     "dated_records(items, probe, levels) -> bytes | None | NotImplemented"},
    {"reduce_dict_apart", (PyCFunction)(void (*)(void))py_reduce_dict_apart, METH_FASTCALL,
     "reduce_dict_apart(full, apart, protocol, obj) -> reduce value"},
    {NULL, NULL, 0, NULL}};

int cwalk_add_functions(PyObject *m) { return PyModule_AddFunctions(m, cwalk_methods); }

#else /* Py_GIL_DISABLED: no kernels; cash runs the Python walks */

int cwalk_add_functions(PyObject *m) {
    (void)m;
    return 0;
}

#endif
