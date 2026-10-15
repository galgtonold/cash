/*
 * cash._cwatch, part 3: cash's audit hook, filtered in C.
 *
 * Python calls every audit hook for every audited event in the process: each
 * id(), each import, each compile, each unpickled class. cash needs a handful
 * of them (open, the directory listings, a connect, a process start, and the
 * "exec" / "function.__new__" count of `held_functions`), and its two Python
 * hooks cost ~0.25 us per event they ignore: id() over a million objects took
 * 0.35 s instead of 0.10 s, most of it cash's own walks asking for ids.
 *
 * This hook is installed once, in C, in place of the Python hooks:
 *   - it counts "exec" and "function.__new__" itself (`audit_new_code`),
 *     where held_functions._note_new_code counted them;
 *   - it calls the Python dispatcher (io_watch._hook) only for an event named
 *     in the set io_watch hands it (`audit_dispatch`), a superset of what
 *     that dispatcher acts on: the dispatcher sees exactly the events it acted
 *     on before, and decides on them as before.
 * Like a hook added by sys.addaudithook, the dispatcher runs with tracing off
 * (a profiler or debugger does not step into it) and an exception it raises
 * fails the audited operation. Events of other interpreters and events after
 * the interpreter started finalizing are not dispatched.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>

#include "cwalk.h"

#ifndef Py_GIL_DISABLED

#if PY_VERSION_HEX >= 0x030D0000
#define CA_TSTATE() PyThreadState_GetUnchecked()
#define CA_FINALIZING() Py_IsFinalizing()
#else
#define CA_TSTATE() _PyThreadState_UncheckedGet()
#define CA_FINALIZING() _Py_IsFinalizing()
#endif

static int installed = 0;
static PyInterpreterState *home = NULL;
static unsigned long long new_code = 0;
static PyObject *dispatcher = NULL; /* strong; never freed once set (the hook outlives the module) */

/* the names dispatched: copies, and the first bytes they start with */
static char **names = NULL;
static Py_ssize_t nnames = 0;
static unsigned char first_byte[256];

static int wanted(const char *event) {
    if (!first_byte[(unsigned char)event[0]]) return 0;
    for (Py_ssize_t i = 0; i < nnames; i++)
        if (strcmp(names[i], event) == 0) return 1;
    return 0;
}

static int hook(const char *event, PyObject *args, void *data) {
    (void)data;
    PyThreadState *ts = CA_TSTATE();
    if (!ts || PyThreadState_GetInterpreter(ts) != home) return 0;
    if ((event[0] == 'e' && strcmp(event, "exec") == 0) ||
        (event[0] == 'f' && strcmp(event, "function.__new__") == 0))
        new_code++;
    if (!dispatcher || !wanted(event) || CA_FINALIZING()) return 0;
    PyObject *fn = Py_NewRef(dispatcher);
    PyObject *name = PyUnicode_FromString(event);
    if (!name) { Py_DECREF(fn); return -1; }
    PyObject *argv[2] = {name, args};
    PyThreadState_EnterTracing(ts);
    PyObject *r = PyObject_Vectorcall(fn, argv, 2, NULL);
    PyThreadState_LeaveTracing(ts);
    Py_DECREF(name);
    Py_DECREF(fn);
    if (!r) return -1;
    Py_DECREF(r);
    return 0;
}

/* audit_hook_install() -> bool: the hook is in (installed now or before) */
static PyObject *py_install(PyObject *self, PyObject *noargs) {
    if (!installed) {
        PyThreadState *ts = CA_TSTATE();
        if (!ts) Py_RETURN_FALSE;
        if (PySys_AddAuditHook(hook, NULL) < 0) {
            /* refused by an existing hook: cash adds its Python hook instead */
            PyErr_Clear();
            Py_RETURN_FALSE;
        }
        home = PyThreadState_GetInterpreter(ts);
        installed = 1;
    }
    Py_RETURN_TRUE;
}

/* audit_dispatch(dispatcher, names): call *dispatcher*(event, args) for the
 * events named in the iterable *names* (str), and only those */
static PyObject *py_dispatch(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 2 || (args[0] != Py_None && !PyCallable_Check(args[0]))) {
        PyErr_SetString(PyExc_TypeError, "audit_dispatch(callable or None, names)");
        return NULL;
    }
    PyObject *seq = PySequence_List(args[1]);
    if (!seq) return NULL;
    Py_ssize_t n = PyList_GET_SIZE(seq);
    char **copy = PyMem_Calloc(n ? n : 1, sizeof *copy);
    if (!copy) { Py_DECREF(seq); return PyErr_NoMemory(); }
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *s = PyList_GET_ITEM(seq, i);
        Py_ssize_t len;
        const char *utf8 = PyUnicode_Check(s) ? PyUnicode_AsUTF8AndSize(s, &len) : NULL;
        if (!utf8 || len == 0 || (Py_ssize_t)strlen(utf8) != len) {
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "audit_dispatch: event names are non-empty str");
            for (Py_ssize_t j = 0; j < i; j++) PyMem_Free(copy[j]);
            PyMem_Free(copy);
            Py_DECREF(seq);
            return NULL;
        }
        copy[i] = PyMem_Malloc(len + 1);
        if (!copy[i]) {
            for (Py_ssize_t j = 0; j < i; j++) PyMem_Free(copy[j]);
            PyMem_Free(copy);
            Py_DECREF(seq);
            return PyErr_NoMemory();
        }
        memcpy(copy[i], utf8, len + 1);
    }
    Py_DECREF(seq);
    /* no Python code runs from here: the hook never sees a half-made table */
    char **old = names;
    Py_ssize_t nold = nnames;
    names = copy;
    nnames = n;
    memset(first_byte, 0, sizeof first_byte);
    for (Py_ssize_t i = 0; i < n; i++) first_byte[(unsigned char)copy[i][0]] = 1;
    for (Py_ssize_t j = 0; j < nold; j++) PyMem_Free(old[j]);
    PyMem_Free(old);
    if (args[0] != dispatcher) {
        /* a hook running further up the stack holds its own reference */
        Py_XSETREF(dispatcher, args[0] == Py_None ? NULL : Py_NewRef(args[0]));
    }
    Py_RETURN_NONE;
}

/* audit_new_code() -> int: "exec" and "function.__new__" events seen in this
 * interpreter since the hook went in */
static PyObject *py_new_code(PyObject *self, PyObject *noargs) { return PyLong_FromUnsignedLongLong(new_code); }

/* audit_state() -> (installed, dispatched names): for tests */
static PyObject *py_state(PyObject *self, PyObject *noargs) {
    PyObject *out = PyList_New(nnames);
    if (!out) return NULL;
    for (Py_ssize_t i = 0; i < nnames; i++) {
        PyObject *s = PyUnicode_FromString(names[i]);
        if (!s) { Py_DECREF(out); return NULL; }
        PyList_SET_ITEM(out, i, s);
    }
    return Py_BuildValue("(ON)", installed ? Py_True : Py_False, out);
}

static PyMethodDef caudit_methods[] = {
    {"audit_hook_install", py_install, METH_NOARGS, "audit_hook_install() -> bool"},
    {"audit_dispatch", (PyCFunction)(void (*)(void))py_dispatch, METH_FASTCALL,
     "audit_dispatch(dispatcher, names): dispatch only these events"},
    {"audit_new_code", py_new_code, METH_NOARGS, "audit_new_code() -> int"},
    {"audit_state", py_state, METH_NOARGS, "audit_state() -> (installed, names)"},
    {NULL, NULL, 0, NULL}};

int caudit_add_functions(PyObject *m) { return PyModule_AddFunctions(m, caudit_methods); }

#else

int caudit_add_functions(PyObject *m) {
    (void)m;
    return 0;
}

#endif
