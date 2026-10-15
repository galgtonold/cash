/* Placeholder for cash._cwatch, so the build and the import fallback can be
 * tested before the watcher module lands. It exports only its API version.
 * The real module replaces this file. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

static int
cwatch_exec(PyObject *module)
{
    if (PyModule_AddIntConstant(module, "API_VERSION", 1) < 0) {
        return -1;
    }
    return 0;
}

static PyModuleDef_Slot cwatch_slots[] = {
    {Py_mod_exec, cwatch_exec},
#ifdef Py_mod_multiple_interpreters
    {Py_mod_multiple_interpreters, Py_MOD_PER_INTERPRETER_GIL_SUPPORTED},
#endif
#ifdef Py_GIL_DISABLED
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
    {0, NULL},
};

static struct PyModuleDef cwatch_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "cash._cwatch",
    .m_doc = "cash's watcher extension (placeholder).",
    .m_size = 0,
    .m_slots = cwatch_slots,
};

PyMODINIT_FUNC
PyInit__cwatch(void)
{
    return PyModuleDef_Init(&cwatch_module);
}
