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
#include <stdint.h>
#include <string.h>

#include "cwalk.h"

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
    {NULL, NULL, 0, NULL}};

int cwalk_add_functions(PyObject *m) { return PyModule_AddFunctions(m, cwalk_methods); }

#else /* Py_GIL_DISABLED: no kernels; cash runs the Python walks */

int cwalk_add_functions(PyObject *m) {
    (void)m;
    return 0;
}

#endif
