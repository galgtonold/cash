/*
 * cash._cwatch, part 8: is a value held by nothing but its holder?
 *
 * `held_alone(holder, index)` walks the value `holder[index]` (holder a
 * list) through each object's own `tp_traverse`, the way the garbage
 * collector finds what an object references, and counts, for every object
 * the walk reaches, the references to it from inside the value. When each
 * object's reference count is exactly that count -- plus the holder's one
 * reference for the value itself -- nothing outside holds any part of it:
 * no variable, no frame, no other container. The memory tier hands such a
 * value out again (`handed_copies`) where it would unpickle a new copy.
 *
 * Objects every copy shares anyway are not counted: atoms (None, bool,
 * int, float, complex, str, bytes), classes, functions and modules (pickle
 * names them, so a new copy holds the same ones), builtin functions bound
 * to nothing, a module or such a shared object (`len`, `math.sqrt`; not
 * `items.append`, which holds `items`), and tuples and
 * frozensets holding only those. A class with `__del__` (a copy kept
 * alive would delay it), or an object of any type other than an exact
 * dict, list, set, frozenset or tuple or an instance of a class defined in
 * Python over `object`, answers None: not provable.
 *
 * Returns None, or the list of the weak references to the value's objects
 * (a weak reference adds no count, and one can find the object later):
 * the caller tells cash's own from the user's (`_weak.is_cash_ref`).
 *
 * `held_still` asks again after the watchers proved the value unchanged
 * (`cwatch.peek`): the same objects with the same references between them,
 * so the sum of their counts alone says whether anything outside took one
 * -- a pass over the recorded addresses, not a walk.
 *
 * Runs no Python code: the holder keeps every object it reads alive.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>

#include "cwalk.h"

#ifndef Py_GIL_DISABLED

typedef struct {
    uintptr_t *addr;
    size_t cap, n;
} AddrCount;

typedef struct {
    PyObject **v;
    size_t n, cap;
} ObjVec;

static inline size_t ac_hash(uintptr_t a) {
    uint64_t x = (uint64_t)a >> 4;
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33;
    return (size_t)x;
}

static int ac_grow(AddrCount *m) {
    size_t nc = m->cap ? m->cap * 2 : 1024;
    uintptr_t *na = calloc(nc, sizeof *na);
    if (!na) { PyErr_NoMemory(); return -1; }
    for (size_t i = 0; i < m->cap; i++) {
        if (!m->addr[i]) continue;
        size_t j = ac_hash(m->addr[i]) & (nc - 1);
        while (na[j]) j = (j + 1) & (nc - 1);
        na[j] = m->addr[i];
    }
    free(m->addr);
    m->addr = na; m->cap = nc;
    return 0;
}

/* note `a`: 1 when first seen, 0 when seen before, -1 error */
static int ac_add(AddrCount *m, uintptr_t a) {
    if ((m->n + 1) * 2 > m->cap && ac_grow(m) < 0) return -1;
    size_t j = ac_hash(a) & (m->cap - 1);
    while (m->addr[j]) {
        if (m->addr[j] == a) return 0;
        j = (j + 1) & (m->cap - 1);
    }
    m->addr[j] = a; m->n++;
    return 1;
}

static int ov_push(ObjVec *v, PyObject *o) {
    if (v->n == v->cap) {
        size_t nc = v->cap ? v->cap * 2 : 1024;
        PyObject **p = realloc(v->v, nc * sizeof *p);
        if (!p) { PyErr_NoMemory(); return -1; }
        v->v = p; v->cap = nc;
    }
    v->v[v->n++] = o;
    return 0;
}

/* shared by every copy, or immutable all the way down: not counted */
static int is_shared(PyObject *o, int depth) {
    PyTypeObject *t = Py_TYPE(o);
    if (o == Py_None || o == Py_Ellipsis || o == Py_NotImplemented) return 1;
    if (t == &PyLong_Type || t == &PyBool_Type || t == &PyFloat_Type || t == &PyComplex_Type ||
        t == &PyUnicode_Type || t == &PyBytes_Type)
        return 1;
    if (PyType_Check(o) || PyModule_CheckExact(o) || PyFunction_Check(o)) return 1;
    if (PyCFunction_Check(o)) {
        /* a builtin function, or one bound to a module or class; a bound
           method such as `items.append` holds its `__self__`, which is part
           of the value unless shared itself */
        PyObject *self = PyCFunction_GET_SELF(o);
        return !self || PyModule_Check(self) || (depth < 8 && is_shared(self, depth + 1));
    }
    if (depth < 8 && t == &PyTuple_Type) {
        for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(o); i++)
            if (!is_shared(PyTuple_GET_ITEM(o, i), depth + 1)) return 0;
        return 1;
    }
    if (depth < 8 && t == &PyFrozenSet_Type) {
        PySetObject *so = (PySetObject *)o;
        for (Py_ssize_t i = 0; i <= so->mask; i++) {
            PyObject *k = so->table[i].key;
            if (!k || k == cwatch_set_dummy()) continue;
            if (!is_shared(k, depth + 1)) return 0;
        }
        return 1;
    }
    return 0;
}

/* a type whose objects the walk reads: 1 yes, 0 not provable */
static int walkable_type(PyTypeObject *t) {
    if (t == &PyDict_Type || t == &PyList_Type || t == &PySet_Type || t == &PyFrozenSet_Type ||
        t == &PyTuple_Type)
        return 1;
    if (!(t->tp_flags & Py_TPFLAGS_HEAPTYPE) || PyType_IsSubtype(t, &PyType_Type)) return 0;
    if (t->tp_finalize || t->tp_del || !t->tp_traverse) return 0;
    PyObject *mro = t->tp_mro;
    if (!mro || !PyTuple_Check(mro) || PyTuple_GET_SIZE(mro) < 2) return 0;
    Py_ssize_t n = PyTuple_GET_SIZE(mro);
    if (PyTuple_GET_ITEM(mro, n - 1) != (PyObject *)&PyBaseObject_Type) return 0;
    for (Py_ssize_t i = 0; i < n - 1; i++) {
        PyTypeObject *k = (PyTypeObject *)PyTuple_GET_ITEM(mro, i);
        if (!(k->tp_flags & Py_TPFLAGS_HEAPTYPE)) return 0;
    }
    return 1;
}

/* the weak references to `o` added to `out`: 0 ok, -1 error */
static int weakrefs_of(PyObject *o, PyObject *out) {
    PyTypeObject *t = Py_TYPE(o);
    if (t->tp_weaklistoffset == 0) return 0;
    PyObject **head = (PyObject **)((char *)o + t->tp_weaklistoffset);
    for (PyWeakReference *wr = (PyWeakReference *)*head; wr; wr = wr->wr_next)
        if (PyList_Append(out, (PyObject *)wr) < 0) return -1;
    return 0;
}

typedef struct {
    AddrCount *seen;
    PyObject *weak; /* the weak references to the objects met */
    ObjVec *stack;  /* those still to walk */
    ObjVec *all;    /* with `record`: every object met */
    ObjVec *weakable; /* with `record`: those that can be weakly referenced */
    size_t edges;   /* references met from inside the value (and the holder's) */
    size_t counts;  /* the reference counts of the objects met, each once */
    int failed;     /* 1 not provable, -1 error */
} Visit;

/* a reference to `o` from inside the value, `o` walked in turn when `walk`.
   An object with one reference is met by that one alone: only one with
   more is remembered, to be met once (a tree's objects all have one). */
static int count_ref_as(PyObject *o, Visit *V, int walk) {
    if (V->failed) return 0;
    if (!o || is_shared(o, 0)) return 0;
    if (!walkable_type(Py_TYPE(o))) { V->failed = 1; return 0; }
    V->edges++;
    Py_ssize_t rc = Py_REFCNT(o);
    if (rc > 1) {
        int a = ac_add(V->seen, (uintptr_t)o);
        if (a < 0) { V->failed = -1; return 0; }
        if (a == 0) return 0;
    }
    V->counts += (size_t)rc;
    if (Py_TYPE(o)->tp_weaklistoffset) {
        if (weakrefs_of(o, V->weak) < 0 || (V->weakable && ov_push(V->weakable, o) < 0)) { V->failed = -1; return 0; }
    }
    if ((walk && ov_push(V->stack, o) < 0) || (V->all && ov_push(V->all, o) < 0)) { V->failed = -1; return 0; }
    return 0;
}

static int count_ref(PyObject *o, void *arg) { return count_ref_as(o, (Visit *)arg, 1); }

/* An instance's managed dict, once made (`vars(x)`, `x.__dict__`, pickle
   setting its state), or NULL: read in place, never made here. */
static PyObject *managed_dict(PyObject *o) {
    char *at = (char *)o - 3 * (Py_ssize_t)sizeof(PyObject *);
#if PY_VERSION_HEX >= 0x030D0000
    return *(PyObject **)at;
#else
    uintptr_t p = *(uintptr_t *)at;   /* 3.12: the dict, or its values tagged 1 */
    return (p & 1) ? NULL : (PyObject *)p;
#endif
}

typedef struct {
    Visit *V;
    PyObject *dict;
    int met;
} InstVisit;

static int count_inst_ref(PyObject *o, void *arg) {
    InstVisit *I = (InstVisit *)arg;
    if (o && o == I->dict) I->met = 1;
    return count_ref(o, I->V);
}

static int collect_ref(PyObject *o, void *arg) {
    return o ? ov_push((ObjVec *)arg, o) : 0;
}

static int cmp_ptr(const void *a, const void *b) {
    uintptr_t x = (uintptr_t)*(PyObject *const *)a, y = (uintptr_t)*(PyObject *const *)b;
    return x < y ? -1 : x > y;
}

/* Walk an instance with a managed dict. Its `tp_traverse` reaches its
   attributes through the dict, or -- where the instance still keeps them
   inline (3.13 and later) -- through those inline values alone, leaving the
   dict unseen: a reference to the dict held outside (`d = vars(x)`) would
   go uncounted, and a reused copy share `d` with the caller. So a dict the
   traverse did not reach is counted here as the instance's reference to
   it, once its every value is shown met already (the dict and the inline
   values are the same attributes), and not walked again. */
static void walk_instance(PyObject *o, traverseproc tr, Visit *V, ObjVec *scratch) {
    InstVisit I = {V, managed_dict(o), 0};
    if (tr(o, count_inst_ref, &I) != 0) { V->failed = 1; return; }
    if (V->failed || !I.dict || I.met) return;
    if (!PyDict_CheckExact(I.dict)) { V->failed = 1; return; }
    scratch->n = 0;
    if (tr(o, collect_ref, scratch) != 0) { V->failed = PyErr_Occurred() ? -1 : 1; return; }
    if (scratch->n > 1) qsort(scratch->v, scratch->n, sizeof *scratch->v, cmp_ptr);
    Py_ssize_t pos = 0;
    PyObject *k, *v;
    while (PyDict_Next(I.dict, &pos, &k, &v)) {
        /* inline values have names only; anything else, or a value the
           traverse did not reach, is not provable */
        if (!PyUnicode_CheckExact(k)) { V->failed = 1; return; }
        if (is_shared(v, 0)) continue;
        if (!scratch->n || !bsearch(&v, scratch->v, scratch->n, sizeof *scratch->v, cmp_ptr)) {
            V->failed = 1;
            return;
        }
    }
    count_ref_as(I.dict, V, 0);
}

static PyObject *as_bytes(ObjVec *v) {
    return PyBytes_FromStringAndSize((const char *)v->v, (Py_ssize_t)(v->n * sizeof(PyObject *)));
}

/* held_alone(holder, index, record=False): see the top of this file. With
   `record`, (weak references, objects, weakable, count): the addresses of
   the value's objects and of those of them that take weak references, as
   bytes, and the sum of their reference counts, for `held_still`. */
static PyObject *py_held_alone(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if ((nargs != 2 && nargs != 3) || !PyList_CheckExact(args[0])) {
        PyErr_SetString(PyExc_TypeError, "held_alone(holder: list, index, record=False)");
        return NULL;
    }
    Py_ssize_t index = PyLong_AsSsize_t(args[1]);
    if (index == -1 && PyErr_Occurred()) return NULL;
    if (index < 0 || index >= PyList_GET_SIZE(args[0])) {
        PyErr_SetString(PyExc_IndexError, "held_alone: index out of range");
        return NULL;
    }
    int record = nargs == 3 ? PyObject_IsTrue(args[2]) : 0;
    if (record < 0) return NULL;
    PyObject *root = PyList_GET_ITEM(args[0], index);   /* borrowed: the holder keeps it */
    if (is_shared(root, 0) || !walkable_type(Py_TYPE(root))) Py_RETURN_NONE;
    PyObject *weak = PyList_New(0);
    if (!weak) return NULL;
    AddrCount seen = {0};
    ObjVec stack = {0}, all = {0}, weakable = {0}, scratch = {0};
    PyObject *result = NULL;
    Visit V = {&seen, weak, &stack, record ? &all : NULL, record ? &weakable : NULL, 0, 0, 0};
    count_ref(root, &V);   /* the holder's reference */
    /* depth first: near the order pickle wrote the objects in and so, in a
       copy just read, the order they sit in memory */
    while (stack.n && !V.failed) {
        PyObject *o = stack.v[--stack.n];
        traverseproc tr = Py_TYPE(o)->tp_traverse;
        if (!tr) continue;
        if (Py_TYPE(o)->tp_flags & Py_TPFLAGS_MANAGED_DICT) walk_instance(o, tr, &V, &scratch);
        else if (tr(o, count_ref, &V) != 0) { V.failed = 1; break; }
    }
    if (V.failed < 0) goto done;
    /* each object's count is at least the references met to it: the sums
       are equal only when every one is exactly that */
    if (V.failed || V.counts != V.edges) { result = Py_NewRef(Py_None); goto done; }
    if (!record) { result = Py_NewRef(weak); goto done; }
    PyObject *objs = as_bytes(&all), *wk = objs ? as_bytes(&weakable) : NULL;
    if (objs && wk) result = Py_BuildValue("(OOOn)", weak, objs, wk, (Py_ssize_t)V.counts);
    Py_XDECREF(objs); Py_XDECREF(wk);
done:
    free(seen.addr); free(stack.v); free(all.v); free(weakable.v); free(scratch.v);
    Py_DECREF(weak);
    return result;
}

/* held_still(holder, index, objects, weakable, count): the value
   `held_alone(..., record=True)` recorded, ONLY while the watchers prove
   it unchanged since (so `objects` names the very objects it holds, all
   alive): None when the sum of their reference counts is not `count` any
   more (something outside took a reference), else the weak references to
   them. */
static PyObject *py_held_still(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 5 || !PyList_CheckExact(args[0]) || !PyBytes_CheckExact(args[2]) || !PyBytes_CheckExact(args[3])) {
        PyErr_SetString(PyExc_TypeError, "held_still(holder: list, index, objects: bytes, weakable: bytes, count)");
        return NULL;
    }
    Py_ssize_t index = PyLong_AsSsize_t(args[1]);
    if (index == -1 && PyErr_Occurred()) return NULL;
    Py_ssize_t count = PyLong_AsSsize_t(args[4]);
    if (count == -1 && PyErr_Occurred()) return NULL;
    if (index < 0 || index >= PyList_GET_SIZE(args[0])) {
        PyErr_SetString(PyExc_IndexError, "held_still: index out of range");
        return NULL;
    }
    PyObject *const *objs = (PyObject *const *)PyBytes_AS_STRING(args[2]);
    size_t n = (size_t)PyBytes_GET_SIZE(args[2]) / sizeof(PyObject *);
    if (n == 0 || objs[0] != PyList_GET_ITEM(args[0], index)) Py_RETURN_NONE;
    size_t sum = 0;
    for (size_t i = 0; i < n; i++) sum += (size_t)Py_REFCNT(objs[i]);
    if (sum != (size_t)count) Py_RETURN_NONE;
    PyObject *weak = PyList_New(0);
    if (!weak) return NULL;
    PyObject *const *wk = (PyObject *const *)PyBytes_AS_STRING(args[3]);
    size_t nw = (size_t)PyBytes_GET_SIZE(args[3]) / sizeof(PyObject *);
    for (size_t i = 0; i < nw; i++)
        if (weakrefs_of(wk[i], weak) < 0) { Py_DECREF(weak); return NULL; }
    return weak;
}

static PyMethodDef cprivate_methods[] = {
    {"held_alone", (PyCFunction)(void (*)(void))py_held_alone, METH_FASTCALL,
     "held_alone(holder, index) -> the weak references to holder[index]'s objects, or None when anything "
     "outside it holds part of it (or that cannot be told)"},
    {"held_still", (PyCFunction)(void (*)(void))py_held_still, METH_FASTCALL,
     "held_still(holder, index, objects, weakable, count) -> held_alone's answer again, for a value the "
     "watchers prove unchanged since held_alone(record=True) recorded it"},
    {NULL, NULL, 0, NULL}};

int cprivate_add_functions(PyObject *m) { return PyModule_AddFunctions(m, cprivate_methods); }

#else

int cprivate_add_functions(PyObject *m) {
    (void)m;
    return 0;
}

#endif
