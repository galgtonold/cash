/* cash._cwatch, part 2: C kernels for cash's hottest Python walks (cwalk.c). */
#ifndef CASH_CWALK_H
#define CASH_CWALK_H
#include <Python.h>

/* Add the walk kernels to the module; 0 or -1 with an exception set. */
int cwalk_add_functions(PyObject *m);
/* Add the audit hook's functions (caudit.c). */
int caudit_add_functions(PyObject *m);
/* Add the native file hook's functions (cfilehook.c). */
int cfilehook_add_functions(PyObject *m);
/* Add the Arrow view column readers (cviews.c). */
int cviews_add_functions(PyObject *m);
/* Add the set layout recorder (csets.c). */
int csets_add_functions(PyObject *m);
/* The marker a set leaves where it deleted a key (cwatch.c). */
PyObject *cwatch_set_dummy(void);

#endif
