/*
 * cash._cwatch, part 7: each garbage collection runs in a context of its own.
 *
 * `cash._gc_context` puts a callback on gc.callbacks that enters a fresh,
 * empty context when a collection starts and leaves it when it stops. Written
 * in Python, the callback is a Python frame, which a collection that starts a
 * frame short of the recursion limit cannot call: on Python 3.14 the
 * collection then reported "Exception ignored while calling GC callback" and
 * ran in the interrupted code's context. Called from C, this one takes no
 * Python frame and reads no recursion limit, whatever the depth.
 *
 * `gc_own_context(phase, info)` does what the Python callback does: on
 * "start" it enters a new context, sets `gc_collecting` (a context variable)
 * to True in it and keeps it, on any other phase it leaves the context it
 * entered last, if any. Code a collection runs reads `gc_collecting` True,
 * and the trace of a statement marks it (`trace_recorder.Unit.collected`).
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "cwalk.h"

#ifndef Py_GIL_DISABLED

/* The contexts entered for collections not yet stopped, newest last. */
static PyObject *entered = NULL;
static PyObject *s_start = NULL;
/* True in the context of a collection, unset anywhere else. */
static PyObject *collecting = NULL;

static PyObject *py_gc_own_context(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "gc_own_context(phase, info)");
        return NULL;
    }
    int start = PyUnicode_Check(args[0]) ? PyUnicode_Compare(args[0], s_start) == 0 : 0;
    if (start) {
        PyObject *context = PyContext_New();
        if (!context) return NULL;
        if (PyContext_Enter(context) < 0) {
            Py_DECREF(context);
            return NULL;
        }
        PyObject *token = PyContextVar_Set(collecting, Py_True);
        if (!token) {
            PyContext_Exit(context);
            Py_DECREF(context);
            return NULL;
        }
        Py_DECREF(token);
        int failed = PyList_Append(entered, context);
        Py_DECREF(context);
        if (failed < 0) return NULL;
        Py_RETURN_NONE;
    }
    Py_ssize_t n = PyList_GET_SIZE(entered);
    if (n == 0) Py_RETURN_NONE;
    PyObject *context = Py_NewRef(PyList_GET_ITEM(entered, n - 1));
    if (PyList_SetSlice(entered, n - 1, n, NULL) < 0) {
        Py_DECREF(context);
        return NULL;
    }
    int failed = PyContext_Exit(context);
    Py_DECREF(context);
    if (failed < 0) return NULL;
    Py_RETURN_NONE;
}

static PyMethodDef cgc_methods[] = {
    {"gc_own_context", (PyCFunction)(void (*)(void))py_gc_own_context, METH_FASTCALL,
     "gc_own_context(phase, info): a gc callback; each collection runs in a fresh context"},
    {NULL, NULL, 0, NULL}};

int cgc_add_functions(PyObject *m) {
    if (!entered) {
        entered = PyList_New(0);
        if (!entered) return -1;
    }
    if (!s_start) {
        s_start = PyUnicode_InternFromString("start");
        if (!s_start) return -1;
    }
    if (!collecting) {
        collecting = PyContextVar_New("cash_gc_collecting", NULL);
        if (!collecting) return -1;
    }
    if (PyModule_AddObjectRef(m, "gc_collecting", collecting) < 0) return -1;
    return PyModule_AddFunctions(m, cgc_methods);
}

#else

int cgc_add_functions(PyObject *m) {
    (void)m;
    return 0;
}

#endif
