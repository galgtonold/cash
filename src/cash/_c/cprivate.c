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
 * `held_still` asks again while the watchers prove the value unchanged:
 * the same objects with the same references between them, so the sum of
 * their counts alone says whether anything outside took one -- a pass over
 * the recorded addresses, not a walk. It asks the watchers itself
 * (`cwatch_holds_token`), in the same call: a thread that still holds the
 * value may change it, or free parts of it, between any two bytecodes of
 * the asking thread, so a proof read by an earlier call says nothing about
 * the addresses read here. `clone_held` copies under the same rule.
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

/* held_still(holder, index, objects, weakable, count, kind, token): the
   value `held_alone(..., record=True)` recorded, while the watchers keep
   `token` for it under `kind` and prove it unchanged since (so `objects`
   names the very objects it holds, all alive): None when they do not, or
   when the sum of their reference counts is not `count` any more
   (something outside took a reference), else the weak references to
   them. */
static PyObject *py_held_still(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 7 || !PyList_CheckExact(args[0]) || !PyBytes_CheckExact(args[2]) || !PyBytes_CheckExact(args[3])) {
        PyErr_SetString(PyExc_TypeError,
                        "held_still(holder: list, index, objects: bytes, weakable: bytes, count, kind, token)");
        return NULL;
    }
    Py_ssize_t index = PyLong_AsSsize_t(args[1]);
    if (index == -1 && PyErr_Occurred()) return NULL;
    Py_ssize_t count = PyLong_AsSsize_t(args[4]);
    if (count == -1 && PyErr_Occurred()) return NULL;
    long kind = PyLong_AsLong(args[5]);
    if (kind == -1 && PyErr_Occurred()) return NULL;
    if (index < 0 || index >= PyList_GET_SIZE(args[0])) {
        PyErr_SetString(PyExc_IndexError, "held_still: index out of range");
        return NULL;
    }
    PyObject *const *objs = (PyObject *const *)PyBytes_AS_STRING(args[2]);
    size_t n = (size_t)PyBytes_GET_SIZE(args[2]) / sizeof(PyObject *);
    if (n == 0 || objs[0] != PyList_GET_ITEM(args[0], index)) Py_RETURN_NONE;
    /* the proof and the read of the recorded addresses: nothing between */
    if (!cwatch_holds_token(objs[0], kind, args[6])) Py_RETURN_NONE;
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

/* ------------------------------------------------------------ clone_held
 *
 * `clone_held(holder, index)` builds a new copy of `holder[index]`, a copy
 * the memory tier handed out before and the watchers prove unchanged since
 * it was read out of its pickle bytes (`handed_copies.reuse`), for a hit
 * while someone still holds that copy: the same objects pickle would make
 * again, without reading the bytes. Objects every copy shares (`is_shared`)
 * are the same objects; every other one is new: exact dicts (their keys
 * atoms, or tuples of them), lists and tuples, and instances of a class
 * defined in Python whose copy pickle makes by calling `object.__new__` and
 * setting its slots and its __dict__ (none of the hooks that let a class
 * make its copy otherwise, nor a `__new__` or `__setattr__` of its own).
 * Anything else -- a set, a subclass of a builtin, an instance keeping its
 * attributes inline with no __dict__ made -- answers None, and the hit
 * reads the bytes as before. Runs no Python code: keys are atoms, and the
 * cyclic collector is held off while the copy is half made. */

typedef struct {
    uintptr_t *key;
    PyObject **val;
    size_t cap, n;
} CloneMemo;

static PyObject *cm_get(CloneMemo *m, PyObject *o) {
    if (!m->cap) return NULL;
    size_t j = ac_hash((uintptr_t)o) & (m->cap - 1);
    while (m->key[j]) {
        if (m->key[j] == (uintptr_t)o) return m->val[j];
        j = (j + 1) & (m->cap - 1);
    }
    return NULL;
}

static int cm_put(CloneMemo *m, PyObject *o, PyObject *copy) {
    if ((m->n + 1) * 2 > m->cap) {
        size_t nc = m->cap ? m->cap * 2 : 1024;
        uintptr_t *nk = calloc(nc, sizeof *nk);
        PyObject **nv = calloc(nc, sizeof *nv);
        if (!nk || !nv) { free(nk); free(nv); PyErr_NoMemory(); return -1; }
        for (size_t i = 0; i < m->cap; i++) {
            if (!m->key[i]) continue;
            size_t j = ac_hash(m->key[i]) & (nc - 1);
            while (nk[j]) j = (j + 1) & (nc - 1);
            nk[j] = m->key[i]; nv[j] = m->val[i];
        }
        free(m->key); free(m->val);
        m->key = nk; m->val = nv; m->cap = nc;
    }
    size_t j = ac_hash((uintptr_t)o) & (m->cap - 1);
    while (m->key[j]) j = (j + 1) & (m->cap - 1);
    m->key[j] = (uintptr_t)o; m->val[j] = copy; m->n++;
    return 0;
}

/* A dict key a new dict takes without running Python code: an atom whose
   hash is C's own, or a tuple of them. */
static int atom_key(PyObject *k, int depth) {
    PyTypeObject *t = Py_TYPE(k);
    if (k == Py_None || t == &PyUnicode_Type || t == &PyLong_Type || t == &PyBool_Type || t == &PyFloat_Type ||
        t == &PyBytes_Type || t == &PyComplex_Type)
        return 1;
    if (t == &PyTuple_Type && depth < 8) {
        for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(k); i++)
            if (!atom_key(PyTuple_GET_ITEM(k, i), depth + 1)) return 0;
        return 1;
    }
    return 0;
}

/* The hooks that let a class make its pickled copy otherwise than from its
   slots and __dict__, or run code of its own while pickle sets them. */
static const char *const CLONE_HOOKS[] = {"__reduce__", "__reduce_ex__", "__getstate__", "__setstate__",
                                          "__getnewargs__", "__getnewargs_ex__"};
#define NCLONE_HOOKS (sizeof CLONE_HOOKS / sizeof *CLONE_HOOKS)
static PyObject *clone_hook_names[NCLONE_HOOKS];

/* 1 when pickle's copy of an instance of `t` is `object.__new__(t)` with
   the original's slots and __dict__ set the default way, 0 otherwise. */
static int clonable_type(PyTypeObject *t) {
    if (!walkable_type(t) || t->tp_itemsize != 0) return 0;
    if (t->tp_new != PyBaseObject_Type.tp_new || t->tp_alloc != PyType_GenericAlloc) return 0;
    if (t->tp_setattro != PyObject_GenericSetAttr || t->tp_getattro != PyObject_GenericGetAttr) return 0;
    if (t->tp_dictoffset != 0 && !(t->tp_flags & Py_TPFLAGS_MANAGED_DICT)) return 0;
    if (!clone_hook_names[0]) {
        for (size_t i = 0; i < NCLONE_HOOKS; i++) {
            PyObject *s = PyUnicode_InternFromString(CLONE_HOOKS[i]);
            if (!s) { PyErr_Clear(); return 0; }
            clone_hook_names[i] = s; /* kept for the life of the process */
        }
    }
    PyObject *mro = t->tp_mro;
    Py_ssize_t n = PyTuple_GET_SIZE(mro);
    for (Py_ssize_t i = 0; i < n - 1; i++) {   /* the last is object */
        PyObject *d = ((PyTypeObject *)PyTuple_GET_ITEM(mro, i))->tp_dict;
        if (!d || !PyDict_CheckExact(d)) return 0;
        for (size_t k = 0; k < NCLONE_HOOKS; k++) {
            /* a hook written in C (a method descriptor) is object's own */
            PyObject *v = PyDict_GetItemWithError(d, clone_hook_names[k]);
            if (PyErr_Occurred()) { PyErr_Clear(); return 0; }
            if (v) return 0;
        }
    }
    return 1;
}

static int is_slot(const PyMemberDef *m) {
    return m->type == Py_T_OBJECT_EX && !(m->flags & Py_READONLY) && m->offset > 0;
}

typedef struct {
    CloneMemo memo;
    ObjVec todo;   /* pairs: an original, then its new copy still to fill */
    int failed;    /* 1 not clonable, -1 error */
} Clone;

/* The copy of `o`, a new reference: `o` itself when shared, the copy made
   already when `o` was met before, else a new empty copy, filled later
   (`clone_fill`). NULL with C->failed set. */
static PyObject *clone_shell(PyObject *o, Clone *C) {
    if (is_shared(o, 0)) return Py_NewRef(o);
    int many = Py_REFCNT(o) > 1;
    if (many) {
        PyObject *made = cm_get(&C->memo, o);
        if (made) return Py_NewRef(made);
    }
    PyTypeObject *t = Py_TYPE(o);
    PyObject *copy;
    if (t == &PyDict_Type) copy = PyDict_New();
    else if (t == &PyList_Type) copy = PyList_New(PyList_GET_SIZE(o));
    else if (t == &PyTuple_Type) copy = PyTuple_New(PyTuple_GET_SIZE(o));
    else if (clonable_type(t)) copy = t->tp_alloc(t, 0);
    else { C->failed = 1; return NULL; }
    if (!copy) { C->failed = -1; return NULL; }
    if ((many && cm_put(&C->memo, o, copy) < 0) || ov_push(&C->todo, o) < 0 || ov_push(&C->todo, copy) < 0) {
        Py_DECREF(copy);
        C->failed = -1;
        return NULL;
    }
    return copy;
}

/* Fill `copy`, made empty for `o` by `clone_shell`: 0, or -1 with C->failed set. */
static int clone_fill(PyObject *o, PyObject *copy, Clone *C) {
    PyTypeObject *t = Py_TYPE(o);
    if (t == &PyDict_Type) {
        Py_ssize_t pos = 0;
        PyObject *k, *v;
        while (PyDict_Next(o, &pos, &k, &v)) {
            if (!atom_key(k, 0)) { C->failed = 1; return -1; }
            PyObject *nv = clone_shell(v, C);
            if (!nv) return -1;
            int r = PyDict_SetItem(copy, k, nv);
            Py_DECREF(nv);
            if (r < 0) { C->failed = -1; return -1; }
        }
        return 0;
    }
    if (t == &PyList_Type || t == &PyTuple_Type) {
        int list = t == &PyList_Type;
        Py_ssize_t n = list ? PyList_GET_SIZE(o) : PyTuple_GET_SIZE(o);
        for (Py_ssize_t i = 0; i < n; i++) {
            PyObject *nv = clone_shell(list ? PyList_GET_ITEM(o, i) : PyTuple_GET_ITEM(o, i), C);
            if (!nv) return -1;
            if (list) PyList_SET_ITEM(copy, i, nv);
            else PyTuple_SET_ITEM(copy, i, nv);
        }
        return 0;
    }
    PyObject *mro = t->tp_mro;
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(mro); i++) {
        PyTypeObject *k = (PyTypeObject *)PyTuple_GET_ITEM(mro, i);
        if (!(k->tp_flags & Py_TPFLAGS_HEAPTYPE)) continue;
        for (PyMemberDef *m = k->tp_members; m && m->name; m++) {
            if (!is_slot(m)) continue;
            PyObject *v = *(PyObject **)((char *)o + m->offset);
            if (!v) continue;
            PyObject *nv = clone_shell(v, C);
            if (!nv) return -1;
            Py_XSETREF(*(PyObject **)((char *)copy + m->offset), nv);
        }
    }
    if (t->tp_flags & Py_TPFLAGS_MANAGED_DICT) {
        PyObject *d = managed_dict(o);
        /* attributes kept inline with no dict made: not read here */
        if (!d) { C->failed = 1; return -1; }
        /* a dict held elsewhere too (`d = vars(x)`) is not read here */
        if (!PyDict_CheckExact(d) || Py_REFCNT(d) != 1) { C->failed = 1; return -1; }
        /* filled the way pickle fills it: the copy's own __dict__, made
           by Python, its items set one by one */
        PyObject *nd = PyObject_GenericGetDict(copy, NULL);
        if (!nd) { C->failed = -1; return -1; }
        int r = clone_fill(d, nd, C);
        Py_DECREF(nd);
        if (r < 0) return -1;
    }
    return 0;
}

static PyObject *py_clone_held(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if ((nargs != 2 && nargs != 4) || !PyList_CheckExact(args[0])) {
        PyErr_SetString(PyExc_TypeError, "clone_held(holder: list, index[, kind, token])");
        return NULL;
    }
    Py_ssize_t index = PyLong_AsSsize_t(args[1]);
    if (index == -1 && PyErr_Occurred()) return NULL;
    long kind = nargs == 4 ? PyLong_AsLong(args[2]) : 0;
    if (kind == -1 && PyErr_Occurred()) return NULL;
    if (index < 0 || index >= PyList_GET_SIZE(args[0])) {
        PyErr_SetString(PyExc_IndexError, "clone_held: index out of range");
        return NULL;
    }
    PyObject *root = PyList_GET_ITEM(args[0], index);   /* borrowed: the holder keeps it */
    /* With a token: only a value the watchers prove unchanged now. Whoever
       holds it may write it between any two bytecodes of this thread, and
       the copy of a value changed since an earlier proof would hand its
       holder's edit to the next caller. */
    if (nargs == 4 && !cwatch_holds_token(root, kind, args[3])) Py_RETURN_NONE;
    if (is_shared(root, 0)) Py_RETURN_NONE;
    int gc_was_on = PyGC_Disable();
    Clone C = {{0}, {0}, 0};
    PyObject *copy = clone_shell(root, &C);
    /* depth first, as `held_alone` walks */
    while (copy && !C.failed && C.todo.n) {
        PyObject *made = C.todo.v[--C.todo.n];
        PyObject *o = C.todo.v[--C.todo.n];
        if (clone_fill(o, made, &C) < 0) break;
    }
    free(C.memo.key); free(C.memo.val); free(C.todo.v);
    if (C.failed) {
        Py_XDECREF(copy);
        copy = NULL;
    }
    if (gc_was_on) PyGC_Enable();
    if (C.failed < 0) return NULL;
    if (!copy) Py_RETURN_NONE;
    return copy;
}

static PyMethodDef cprivate_methods[] = {
    {"held_alone", (PyCFunction)(void (*)(void))py_held_alone, METH_FASTCALL,
     "held_alone(holder, index) -> the weak references to holder[index]'s objects, or None when anything "
     "outside it holds part of it (or that cannot be told)"},
    {"held_still", (PyCFunction)(void (*)(void))py_held_still, METH_FASTCALL,
     "held_still(holder, index, objects, weakable, count, kind, token) -> held_alone's answer again, "
     "while the watchers keep token for the value: unchanged since held_alone(record=True) recorded it"},
    {"clone_held", (PyCFunction)(void (*)(void))py_clone_held, METH_FASTCALL,
     "clone_held(holder, index[, kind, token]) -> a new copy of holder[index] as pickle would make it, or "
     "None when it holds an object that cannot be copied so, or the watchers do not keep token for it"},
    {NULL, NULL, 0, NULL}};

int cprivate_add_functions(PyObject *m) { return PyModule_AddFunctions(m, cprivate_methods); }

#else

int cprivate_add_functions(PyObject *m) {
    (void)m;
    return 0;
}

#endif
