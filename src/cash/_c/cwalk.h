/* cash._cwatch, part 2: C kernels for cash's hottest Python walks (cwalk.c). */
#ifndef CASH_CWALK_H
#define CASH_CWALK_H
#include <Python.h>

/* Add the walk kernels to the module; 0 or -1 with an exception set. */
int cwalk_add_functions(PyObject *m);

#endif
