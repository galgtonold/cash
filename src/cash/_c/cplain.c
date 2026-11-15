/*
 * cash._cwatch, part 9: the key walk of plain data (cplain.c).
 *
 * `PlainWalk` gives the form `canonical_form.stable_key_repr` gives an exact
 * dict, list or tuple and everything below it, and an exact numpy array of
 * numbers, with the same side effects on the walk: the same entries in its
 * `seen` (numbered in the same order), the same ids on its `stack` while a
 * container's parts are walked, the same digests in the decorator's quiet
 * window. A part that is anything else -- a subclass, an object, a set, a
 * bytearray, an array of objects or records, a big array, an array that
 * is not C-contiguous, a list whose
 * items could be records -- is handed to the Python walk (`_walk_below`),
 * which may come back here for the plain parts under it. The Python walk
 * stays the reference: tests/test_core/arguments/
 * test_the_c_key_walk_forms_what_the_python_walk_forms.py runs both on
 * random structures and requires identical bytes.
 *
 * Unlike the kernels in cwalk.c, Python code runs inside this one (the
 * Python walk of a part it hands over, numpy's buffer export, hashlib), so
 * every item is held while it is walked and a dict that changes size under
 * the walk raises as iterating it in Python does.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "cwalk.h"

#ifndef Py_GIL_DISABLED

/* Set once by plain_walk_setup. */
static PyObject *pw_tag_type;     /* "__cash_type__" */
static PyObject *pw_tag_alias;    /* "__cash_alias__" */
static PyObject *pw_tag_content;  /* "__cash_content__" */
static PyObject *pw_family;       /* "numpy" */
static PyObject *pw_tag_dict, *pw_tag_list, *pw_tag_tuple;
static PyObject *pw_cyclic;       /* CyclicValueError */
static PyObject *pw_walk_below;   /* _walk_below(value, walk) */
static PyObject *pw_dtype_text;   /* dtype -> its text, or None when not read here */
static PyObject *pw_layout;       /* array_layout(array) */
static PyObject *pw_sha256;       /* hashlib.sha256 */
static PyObject *pw_not_fast;     /* the answer for a value left to Python */
static PyObject *pw_dtype_cache;  /* {id(dtype): (dtype, text or None)} */
static PyObject *s_dtype, *s_hexdigest, *pw_layout_c;
static Py_ssize_t pw_direct_max;  /* bulk_digest.DIRECT_MAX_BYTES */
static Py_ssize_t pw_records_from; /* canonical_form.RECORDS_FROM */

/* How deep this C walk recurses, across all its open walks, before handing
 * a part to the Python walk, and how many walks may be open at once (each
 * hands parts to Python, whose walk may open another): the C stack stays
 * bounded -- a few tens of KB -- however deep a value, so a thread with a
 * small stack walks it as the Python walk alone does. */
#define PW_MAX_DEPTH 48
#define PW_MAX_OPEN 8
#define PW_DTYPE_CACHE_MAX 256

typedef struct {
    PyObject_HEAD
    vectorcallfunc vectorcall;
    PyObject *seen;    /* dict: id -> (number, value) */
    PyObject *stack;   /* set of ids */
    PyObject *quiet;   /* dict id -> (value, digest), or NULL */
    PyObject *walk;    /* the Python _Walk */
    PyObject *ndarray; /* numpy.ndarray, or NULL */
    int open;  /* walks open: each hands parts to Python, which may open another */
    int depth; /* containers being walked, across all open walks */
} PlainWalk;

static int is_prim(PyTypeObject *t) {
    return t == &PyUnicode_Type || t == &PyLong_Type || t == &PyFloat_Type || t == &PyBool_Type ||
           t == Py_TYPE(Py_None) || t == &PyBytes_Type || t == &PyComplex_Type;
}

static PyObject *below(PlainWalk *w, PyObject *value) {
    PyObject *args[2] = {value, w->walk};
    return PyObject_Vectorcall(pw_walk_below, args, 2, NULL);
}

/* ("__cash_alias__", number) when *value* was met before, NULL with no
 * error when not, NULL with an error on failure. */
static PyObject *alias_of(PlainWalk *w, PyObject *key) {
    PyObject *first = PyDict_GetItemWithError(w->seen, key);
    if (!first) return NULL;
    if (!PyTuple_Check(first) || PyTuple_GET_SIZE(first) < 1) {
        PyErr_SetString(PyExc_TypeError, "a walk's seen entry is not a tuple");
        return NULL;
    }
    return PyTuple_Pack(2, pw_tag_alias, PyTuple_GET_ITEM(first, 0));
}

static int note_seen(PlainWalk *w, PyObject *key, PyObject *value) {
    PyObject *n = PyLong_FromSsize_t(PyDict_GET_SIZE(w->seen));
    if (!n) return -1;
    PyObject *entry = PyTuple_Pack(2, n, value);
    Py_DECREF(n);
    if (!entry) return -1;
    int r = PyDict_SetItem(w->seen, key, entry);
    Py_DECREF(entry);
    return r;
}

/* The text of an array's dtype when this walk reads such an array (numbers,
 * strings: no fields, no objects), else None. Borrowed from the cache, or
 * a new reference with *owned* set. NULL on error. */
static PyObject *dtype_text(PyObject *dtype, int *owned) {
    *owned = 0;
    PyObject *key = PyLong_FromVoidPtr(dtype);
    if (!key) return NULL;
    PyObject *hit = PyDict_GetItemWithError(pw_dtype_cache, key);
    if (hit) {
        Py_DECREF(key);
        if (PyTuple_GET_ITEM(hit, 0) == dtype) return PyTuple_GET_ITEM(hit, 1);
        PyObject *text = PyObject_CallOneArg(pw_dtype_text, dtype);
        if (text) *owned = 1;
        return text;
    }
    if (PyErr_Occurred()) {
        Py_DECREF(key);
        return NULL;
    }
    PyObject *text = PyObject_CallOneArg(pw_dtype_text, dtype);
    if (!text || (text != Py_None && !PyUnicode_Check(text))) {
        Py_DECREF(key);
        if (text) {
            Py_DECREF(text);
            Py_RETURN_NONE;
        }
        return NULL;
    }
    if (PyDict_GET_SIZE(pw_dtype_cache) < PW_DTYPE_CACHE_MAX) {
        PyObject *entry = PyTuple_Pack(2, dtype, text);
        if (!entry || PyDict_SetItem(pw_dtype_cache, key, entry) < 0) {
            Py_XDECREF(entry);
            Py_DECREF(key);
            Py_DECREF(text);
            return NULL;
        }
        Py_DECREF(entry);
    }
    Py_DECREF(key);
    *owned = 1;
    return text;
}

/* `content_hashers.hash_numpy` of an array of a plain dtype, at most
 * DIRECT_MAX_BYTES big: SHA-256 of "{shape}:{dtype}:{layout}:", then
 * `bulk_digest.fold_buffer`'s b"r", the length and the bytes in C order. */
static PyObject *array_digest(PyObject *value, Py_buffer *view, PyObject *text) {
    Py_ssize_t size = 1;
    for (int i = 0; i < view->ndim; i++) size *= view->shape[i];
    PyObject *layout;
    if (view->ndim < 2 || (size && PyBuffer_IsContiguous(view, 'C')))
        layout = Py_NewRef(pw_layout_c);
    else
        layout = PyObject_CallOneArg(pw_layout, value);
    if (!layout) return NULL;
    PyObject *payload = NULL, *h = NULL, *out = NULL;
    char *shape = NULL;
    Py_ssize_t text_len, layout_len;
    const char *text_utf8 = PyUnicode_AsUTF8AndSize(text, &text_len);
    const char *layout_utf8 = text_utf8 && PyUnicode_Check(layout) ? PyUnicode_AsUTF8AndSize(layout, &layout_len) : NULL;
    if (!layout_utf8) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_TypeError, "array_layout gave no text");
        goto done;
    }
    /* repr(shape): "()", "(9,)", "(15, 9)". */
    shape = PyMem_Malloc((size_t)view->ndim * 24 + 4);
    if (!shape) {
        PyErr_NoMemory();
        goto done;
    }
    Py_ssize_t shape_len = 0;
    shape[shape_len++] = '(';
    for (int i = 0; i < view->ndim; i++) {
        if (i) {
            shape[shape_len++] = ',';
            shape[shape_len++] = ' ';
        }
        shape_len += snprintf(shape + shape_len, 24, "%lld", (long long)view->shape[i]);
    }
    if (view->ndim == 1) shape[shape_len++] = ',';
    shape[shape_len++] = ')';
    Py_ssize_t n = view->len;
    Py_ssize_t head_len = shape_len + 1 + text_len + 1 + layout_len + 1;
    payload = PyBytes_FromStringAndSize(NULL, head_len + 9 + n);
    if (!payload) goto done;
    char *p = PyBytes_AS_STRING(payload);
    memcpy(p, shape, (size_t)shape_len);
    p += shape_len;
    *p++ = ':';
    memcpy(p, text_utf8, (size_t)text_len);
    p += text_len;
    *p++ = ':';
    memcpy(p, layout_utf8, (size_t)layout_len);
    p += layout_len;
    *p++ = ':';
    *p++ = 'r';
    unsigned long long len = (unsigned long long)n;
    for (int i = 0; i < 8; i++) *p++ = (char)((len >> (8 * i)) & 0xff);
    if (n) {
        if (PyBuffer_IsContiguous(view, 'C')) {
            memcpy(p, view->buf, (size_t)n);
        } else if (PyBuffer_ToContiguous(p, view, n, 'C') < 0) {
            goto done;
        }
    }
    h = PyObject_CallOneArg(pw_sha256, payload);
    if (!h) goto done;
    out = PyObject_CallMethodNoArgs(h, s_hexdigest);
done:
    PyMem_Free(shape);
    Py_XDECREF(payload);
    Py_XDECREF(h);
    Py_DECREF(layout);
    return out;
}

/* The form of an exact ndarray, or pw_not_fast (a new reference) when the
 * Python walk is to read it. */
static PyObject *array_form(PlainWalk *w, PyObject *value) {
    PyObject *dtype = PyObject_GetAttr(value, s_dtype);
    if (!dtype) return NULL;
    int owned;
    PyObject *text = dtype_text(dtype, &owned);
    Py_DECREF(dtype);
    if (!text) return NULL;
    if (text == Py_None) {
        if (owned) Py_DECREF(text);
        return Py_NewRef(pw_not_fast);
    }
    if (!owned) Py_INCREF(text);
    Py_buffer view;
    if (PyObject_GetBuffer(value, &view, PyBUF_RECORDS_RO) < 0) {
        Py_DECREF(text);
        if (!PyErr_ExceptionMatches(PyExc_BufferError) && !PyErr_ExceptionMatches(PyExc_ValueError) &&
            !PyErr_ExceptionMatches(PyExc_TypeError))
            return NULL;
        PyErr_Clear();
        return Py_NewRef(pw_not_fast);
    }
    PyObject *out = NULL, *digest = NULL;
    Py_ssize_t size = 1;
    for (int i = 0; i < view.ndim; i++) size *= view.shape[i];
    if (view.len > pw_direct_max || (size == 0 && view.ndim != 1) || !PyBuffer_IsContiguous(&view, 'C')) {
        /* Big: read on threads. Empty of two axes or more: hash_numpy has
         * no digest for it (memoryview refuses the cast). Not C-contiguous
         * (Fortran order, a strided view): its layout and the order its
         * bytes are read in are hash_numpy's alone to say. */
        out = Py_NewRef(pw_not_fast);
        goto done;
    }
    PyObject *key = PyLong_FromVoidPtr(value);
    if (!key) goto done;
    out = alias_of(w, key);
    if (out || PyErr_Occurred()) {
        Py_DECREF(key);
        goto done;
    }
    if (w->quiet) {
        PyObject *held = PyDict_GetItemWithError(w->quiet, key);
        if (held && PyTuple_Check(held) && PyTuple_GET_SIZE(held) == 2 && PyTuple_GET_ITEM(held, 0) == value)
            digest = Py_NewRef(PyTuple_GET_ITEM(held, 1));
        else if (PyErr_Occurred()) {
            Py_DECREF(key);
            goto done;
        }
    }
    if (!digest) {
        digest = array_digest(value, &view, text);
        if (!digest) {
            Py_DECREF(key);
            goto done;
        }
        if (w->quiet) {
            PyObject *entry = PyTuple_Pack(2, value, digest);
            if (!entry || PyDict_SetItem(w->quiet, key, entry) < 0) {
                Py_XDECREF(entry);
                Py_DECREF(key);
                goto done;
            }
            Py_DECREF(entry);
        }
    }
    if (digest == Py_None) {
        /* A quiet window holds no None; read in Python all the same. */
        Py_DECREF(key);
        out = Py_NewRef(pw_not_fast);
        goto done;
    }
    if (note_seen(w, key, value) < 0) {
        Py_DECREF(key);
        goto done;
    }
    Py_DECREF(key);
    out = PyTuple_Pack(3, pw_tag_content, pw_family, digest);
done:
    Py_XDECREF(digest);
    PyBuffer_Release(&view);
    Py_DECREF(text);
    return out;
}

/* Could a list or tuple of these items be read as records, numbers or
 * dates by the Python walk (`_object_steps`)? Not when every item is a
 * primitive, a plain container or an array. */
static int items_plain(PlainWalk *w, PyObject *seq) {
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    PyObject **items = PySequence_Fast_ITEMS(seq);
    for (Py_ssize_t i = 0; i < n; i++) {
        PyTypeObject *t = Py_TYPE(items[i]);
        if (is_prim(t) || t == &PyDict_Type || t == &PyList_Type || t == &PyTuple_Type ||
            (w->ndarray && (PyObject *)t == w->ndarray))
            continue;
        return 0;
    }
    return 1;
}

static PyObject *form_of(PlainWalk *w, PyObject *value);

static PyObject *part_form(PlainWalk *w, PyObject *value) {
    if (is_prim(Py_TYPE(value))) return Py_NewRef(value);
    return form_of(w, value);
}

static PyObject *cyclic(PyObject *value) {
    PyObject *name = PyType_GetQualName(Py_TYPE(value));
    if (!name) return NULL;
    PyErr_Format(pw_cyclic, "a %U that contains itself has no stable form to key on", name);
    Py_DECREF(name);
    return NULL;
}

static PyObject *container_form(PlainWalk *w, PyObject *value) {
    PyTypeObject *t = Py_TYPE(value);
    PyObject *key = PyLong_FromVoidPtr(value);
    if (!key) return NULL;
    int on = PySet_Contains(w->stack, key);
    if (on != 0) {
        Py_DECREF(key);
        return on < 0 ? NULL : cyclic(value);
    }
    if (t != &PyTuple_Type) {
        PyObject *alias = alias_of(w, key);
        if (alias || PyErr_Occurred() || note_seen(w, key, value) < 0) {
            Py_DECREF(key);
            return alias;
        }
    }
    if (PySet_Add(w->stack, key) < 0) {
        Py_DECREF(key);
        return NULL;
    }
    PyObject *items = PyList_New(0), *out = NULL;
    w->depth++;
    if (!items) goto done;
    if (t == &PyDict_Type) {
        Py_ssize_t pos = 0, size = PyDict_GET_SIZE(value);
        PyObject *k, *v;
        while (PyDict_Next(value, &pos, &k, &v)) {
            Py_INCREF(k);
            Py_INCREF(v);
            PyObject *kf = part_form(w, k);
            PyObject *vf = kf ? part_form(w, v) : NULL;
            Py_DECREF(k);
            Py_DECREF(v);
            PyObject *pair = vf ? PyTuple_Pack(2, kf, vf) : NULL;
            Py_XDECREF(kf);
            Py_XDECREF(vf);
            if (!pair) goto done;
            int r = PyList_Append(items, pair);
            Py_DECREF(pair);
            if (r < 0) goto done;
            if (PyDict_GET_SIZE(value) != size) {
                PyErr_SetString(PyExc_RuntimeError, "dictionary changed size during iteration");
                goto done;
            }
        }
    } else {
        /* As `for v in value` reads a list: by index, its length asked anew. */
        for (Py_ssize_t i = 0; i < (t == &PyList_Type ? PyList_GET_SIZE(value) : PyTuple_GET_SIZE(value)); i++) {
            PyObject *v = t == &PyList_Type ? PyList_GET_ITEM(value, i) : PyTuple_GET_ITEM(value, i);
            Py_INCREF(v);
            PyObject *vf = part_form(w, v);
            Py_DECREF(v);
            if (!vf) goto done;
            int r = PyList_Append(items, vf);
            Py_DECREF(vf);
            if (r < 0) goto done;
        }
    }
    {
        PyObject *canon = PyList_AsTuple(items);
        if (!canon) goto done;
        PyObject *tag = t == &PyDict_Type ? pw_tag_dict : t == &PyList_Type ? pw_tag_list : pw_tag_tuple;
        out = PyTuple_Pack(3, pw_tag_type, tag, canon);
        Py_DECREF(canon);
    }
done:
    w->depth--;
    Py_XDECREF(items);
    {
        PyObject *exc = PyErr_GetRaisedException();
        if (PySet_Discard(w->stack, key) < 0) {
            Py_CLEAR(out);
            Py_XDECREF(exc);
        } else if (exc) {
            PyErr_SetRaisedException(exc);
        }
    }
    Py_DECREF(key);
    return out;
}

/* The form of a value met as a part (not a primitive). */
static PyObject *form_of(PlainWalk *w, PyObject *value) {
    PyTypeObject *t = Py_TYPE(value);
    if (t == &PyDict_Type || t == &PyList_Type || t == &PyTuple_Type) {
        if (w->depth >= PW_MAX_DEPTH) return below(w, value);
        if (t != &PyDict_Type && Py_SIZE(value) >= pw_records_from && !items_plain(w, value)) return below(w, value);
        return container_form(w, value);
    }
    if (w->ndarray && (PyObject *)t == w->ndarray) {
        PyObject *form = array_form(w, value);
        if (form == pw_not_fast) {
            Py_DECREF(form);
            return below(w, value);
        }
        return form;
    }
    return below(w, value);
}

static PyObject *pw_call(PlainWalk *w, PyObject *const *args, size_t nargsf, PyObject *kwnames) {
    if (PyVectorcall_NARGS(nargsf) != 1 || kwnames) {
        PyErr_SetString(PyExc_TypeError, "PlainWalk takes one value");
        return NULL;
    }
    PyObject *value = args[0];
    PyTypeObject *t = Py_TYPE(value);
    if (w->open >= PW_MAX_OPEN || w->depth >= PW_MAX_DEPTH || !pw_walk_below)
        return Py_NewRef(pw_not_fast ? pw_not_fast : Py_None);
    PyObject *out;
    w->open++;
    if (t == &PyDict_Type || t == &PyList_Type || t == &PyTuple_Type) {
        if (t != &PyDict_Type && Py_SIZE(value) >= pw_records_from && !items_plain(w, value))
            out = Py_NewRef(pw_not_fast);
        else
            out = container_form(w, value);
    } else if (w->ndarray && (PyObject *)t == w->ndarray) {
        out = array_form(w, value);
    } else {
        out = Py_NewRef(pw_not_fast);
    }
    w->open--;
    return out;
}

static PyObject *pw_vcall(PyObject *self, PyObject *const *args, size_t nargsf, PyObject *kwnames) {
    return pw_call((PlainWalk *)self, args, nargsf, kwnames);
}

static PyObject *pw_tp_call(PyObject *self, PyObject *args, PyObject *kwargs) {
    return PyVectorcall_Call(self, args, kwargs);
}

static PyObject *pw_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    PyObject *seen, *stack, *quiet, *walk, *ndarray;
    if (kwargs && PyDict_GET_SIZE(kwargs)) {
        PyErr_SetString(PyExc_TypeError, "PlainWalk takes no keywords");
        return NULL;
    }
    if (!PyArg_UnpackTuple(args, "PlainWalk", 5, 5, &seen, &stack, &quiet, &walk, &ndarray)) return NULL;
    if (!PyDict_CheckExact(seen) || !PySet_CheckExact(stack) || (quiet != Py_None && !PyDict_CheckExact(quiet)) ||
        (ndarray != Py_None && !PyType_Check(ndarray))) {
        PyErr_SetString(PyExc_TypeError, "PlainWalk(seen: dict, stack: set, quiet: dict | None, walk, ndarray | None)");
        return NULL;
    }
    PlainWalk *w = (PlainWalk *)type->tp_alloc(type, 0);
    if (!w) return NULL;
    w->seen = Py_NewRef(seen);
    w->stack = Py_NewRef(stack);
    w->quiet = quiet == Py_None ? NULL : Py_NewRef(quiet);
    w->walk = Py_NewRef(walk);
    w->ndarray = ndarray == Py_None ? NULL : Py_NewRef(ndarray);
    w->open = 0;
    w->depth = 0;
    w->vectorcall = pw_vcall;
    return (PyObject *)w;
}

static int pw_traverse(PlainWalk *w, visitproc visit, void *arg) {
    Py_VISIT(w->seen);
    Py_VISIT(w->stack);
    Py_VISIT(w->quiet);
    Py_VISIT(w->walk);
    Py_VISIT(w->ndarray);
    return 0;
}

static int pw_clear(PlainWalk *w) {
    Py_CLEAR(w->seen);
    Py_CLEAR(w->stack);
    Py_CLEAR(w->quiet);
    Py_CLEAR(w->walk);
    Py_CLEAR(w->ndarray);
    return 0;
}

static void pw_dealloc(PlainWalk *w) {
    PyObject_GC_UnTrack(w);
    pw_clear(w);
    Py_TYPE(w)->tp_free((PyObject *)w);
}

static PyTypeObject PlainWalkType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "cash._cwatch.PlainWalk",
    .tp_basicsize = sizeof(PlainWalk),
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_HAVE_VECTORCALL,
    .tp_doc = "PlainWalk(seen, stack, quiet, walk, ndarray): the key form of plain data (cplain.c)",
    .tp_new = pw_new,
    .tp_dealloc = (destructor)pw_dealloc,
    .tp_traverse = (traverseproc)pw_traverse,
    .tp_clear = (inquiry)pw_clear,
    .tp_call = pw_tp_call,
    .tp_vectorcall_offset = 0,
};

/* plain_walk_setup(tag_type, tag_alias, tag_content, family, tag_dict,
 * tag_list, tag_tuple, cyclic_error, walk_below, dtype_text, layout,
 * sha256, not_fast, direct_max, records_from) */
static PyObject *py_plain_walk_setup(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 15) {
        PyErr_SetString(PyExc_TypeError, "plain_walk_setup takes 15 arguments");
        return NULL;
    }
    Py_ssize_t direct_max = PyLong_AsSsize_t(args[13]);
    Py_ssize_t records_from = PyLong_AsSsize_t(args[14]);
    if ((direct_max == -1 || records_from == -1) && PyErr_Occurred()) return NULL;
    PyObject **slots[] = {&pw_tag_type, &pw_tag_alias, &pw_tag_content, &pw_family, &pw_tag_dict,
                          &pw_tag_list, &pw_tag_tuple, &pw_cyclic,    &pw_walk_below, &pw_dtype_text,
                          &pw_layout,   &pw_sha256,    &pw_not_fast};
    for (int i = 0; i < 13; i++) Py_XSETREF(*slots[i], Py_NewRef(args[i]));
    if (!pw_dtype_cache && !(pw_dtype_cache = PyDict_New())) return NULL;
    PyDict_Clear(pw_dtype_cache);
    pw_direct_max = direct_max;
    pw_records_from = records_from;
    Py_RETURN_NONE;
}

static PyMethodDef cplain_methods[] = {
    {"plain_walk_setup", (PyCFunction)(void (*)(void))py_plain_walk_setup, METH_FASTCALL,
     "plain_walk_setup(tag_type, tag_alias, tag_content, family, tag_dict, tag_list, tag_tuple, cyclic_error, "
     "walk_below, dtype_text, layout, sha256, not_fast, direct_max, records_from) -> None"},
    {NULL, NULL, 0, NULL}};

int cplain_add_functions(PyObject *m) {
    if (!s_dtype && !(s_dtype = PyUnicode_InternFromString("dtype"))) return -1;
    if (!s_hexdigest && !(s_hexdigest = PyUnicode_InternFromString("hexdigest"))) return -1;
    if (!pw_layout_c && !(pw_layout_c = PyUnicode_InternFromString("C"))) return -1;
    PlainWalkType.tp_vectorcall_offset = offsetof(PlainWalk, vectorcall);
    if (PyType_Ready(&PlainWalkType) < 0) return -1;
    if (PyModule_AddObjectRef(m, "PlainWalk", (PyObject *)&PlainWalkType) < 0) return -1;
    return PyModule_AddFunctions(m, cplain_methods);
}

#else /* Py_GIL_DISABLED */

int cplain_add_functions(PyObject *m) {
    (void)m;
    return 0;
}

#endif
