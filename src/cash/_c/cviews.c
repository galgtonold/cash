/*
 * cash._cwatch, part 5: the parts of an Arrow string or binary VIEW column
 * as `content_hashers.fold_arrow_column` reads the large_string or
 * large_binary column it converts to (polars hands out views since 1.0).
 *
 * That fold hashes each value's length as int64 and the values' bytes one
 * after the other. Converting the column to get them copied all its bytes
 * into new memory, and the lengths were an array of their own; these two
 * functions write both streams into a block the caller hashes and reuses.
 *
 * A view is 16 bytes: the length (int32), then either the value itself (12
 * bytes at most) or its first 4 bytes, the index of the data buffer holding
 * it (int32) and its offset there (int32). A missing value has length 0 in
 * the converted column, whatever its view holds. Arrow data is
 * little-endian, as every platform cash builds for is.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>

#include "cwalk.h"

#define VIEW_BYTES 16
#define INLINE_BYTES 12

static int32_t read_i32(const char *at) {
    int32_t v;
    memcpy(&v, at, 4);
    return v;
}

static int is_valid(const unsigned char *validity, Py_ssize_t row) {
    return !validity || (validity[row >> 3] >> (row & 7)) & 1;
}

/* Rows' views and validity as buffers, checked to cover rows [0, stop). */
static int get_views(PyObject *views_obj, PyObject *validity_obj, Py_ssize_t start, Py_ssize_t stop, Py_buffer *views,
                     Py_buffer *validity, int *has_validity) {
    if (start < 0 || stop < start) {
        PyErr_SetString(PyExc_ValueError, "bad row range");
        return -1;
    }
    if (PyObject_GetBuffer(views_obj, views, PyBUF_SIMPLE) < 0) return -1;
    *has_validity = 0;
    if (views->len / VIEW_BYTES < stop) {
        PyBuffer_Release(views);
        PyErr_SetString(PyExc_ValueError, "views buffer too short");
        return -1;
    }
    if (validity_obj != Py_None) {
        if (PyObject_GetBuffer(validity_obj, validity, PyBUF_SIMPLE) < 0) {
            PyBuffer_Release(views);
            return -1;
        }
        *has_validity = 1;
        if (validity->len < (stop + 7) / 8) {
            PyBuffer_Release(validity);
            PyBuffer_Release(views);
            PyErr_SetString(PyExc_ValueError, "validity buffer too short");
            return -1;
        }
    }
    return 0;
}

/* view_lengths(views, validity, start, stop, out) -> rows written
 *
 * The int64 length of each of rows start.. into the writable buffer out, as
 * many as fit and at most up to stop; 0 for a missing one. */
static PyObject *py_view_lengths(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 5) {
        PyErr_SetString(PyExc_TypeError, "view_lengths takes 5 arguments");
        return NULL;
    }
    Py_ssize_t start = PyLong_AsSsize_t(args[2]), stop = PyLong_AsSsize_t(args[3]);
    if ((start == -1 || stop == -1) && PyErr_Occurred()) return NULL;
    Py_buffer views, validity, out;
    int has_validity;
    if (get_views(args[0], args[1], start, stop, &views, &validity, &has_validity) < 0) return NULL;
    if (PyObject_GetBuffer(args[4], &out, PyBUF_WRITABLE) < 0) {
        if (has_validity) PyBuffer_Release(&validity);
        PyBuffer_Release(&views);
        return NULL;
    }
    Py_ssize_t rows = stop - start;
    if (rows > out.len / 8) rows = out.len / 8;
    const char *v = (const char *)views.buf;
    const unsigned char *bits = has_validity ? (const unsigned char *)validity.buf : NULL;
    char *o = (char *)out.buf;
    int bad = 0;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < rows; i++) {
        Py_ssize_t row = start + i;
        int64_t length = 0;
        if (is_valid(bits, row)) {
            int32_t n = read_i32(v + row * VIEW_BYTES);
            if (n < 0) { bad = 1; break; }
            length = n;
        }
        memcpy(o + i * 8, &length, 8);
    }
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&out);
    if (has_validity) PyBuffer_Release(&validity);
    PyBuffer_Release(&views);
    if (bad) {
        PyErr_SetString(PyExc_ValueError, "negative view length");
        return NULL;
    }
    return PyLong_FromSsize_t(rows);
}

/* view_values(views, datas, validity, start, skip, stop, out) -> (row, skip, written)
 *
 * The bytes of rows start.. one after the other, the first `skip` bytes of
 * row start left out (written by the call before), into the writable
 * buffer out until it is full or row stop is reached. A value that does not
 * fit is split: the call returns the row it stopped in and how much of it
 * is written. datas is the tuple of the column's data buffers. */
static PyObject *py_view_values(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 7) {
        PyErr_SetString(PyExc_TypeError, "view_values takes 7 arguments");
        return NULL;
    }
    if (!PyTuple_Check(args[1])) {
        PyErr_SetString(PyExc_TypeError, "datas must be a tuple");
        return NULL;
    }
    Py_ssize_t start = PyLong_AsSsize_t(args[3]), skip = PyLong_AsSsize_t(args[4]), stop = PyLong_AsSsize_t(args[5]);
    if ((start == -1 || skip == -1 || stop == -1) && PyErr_Occurred()) return NULL;
    if (skip < 0) {
        PyErr_SetString(PyExc_ValueError, "negative skip");
        return NULL;
    }
    Py_buffer views, validity, out;
    int has_validity;
    if (get_views(args[0], args[2], start, stop, &views, &validity, &has_validity) < 0) return NULL;
    Py_ssize_t ndatas = PyTuple_GET_SIZE(args[1]);
    /* A data buffer is asked for the first time a row points into it, not
     * all of them up front: every chunk of a polars text column lists all
     * of the column's data buffers (thousands), and this is called once per
     * block, so taking each one each call cost more than the copy. */
    Py_buffer *datas = PyMem_Calloc(ndatas ? (size_t)ndatas : 1, sizeof(Py_buffer));
    char *have = PyMem_Calloc(ndatas ? (size_t)ndatas : 1, 1);
    PyObject *result = NULL;
    int has_out = 0;
    if (!datas || !have) {
        PyErr_NoMemory();
        goto done;
    }
    if (PyObject_GetBuffer(args[6], &out, PyBUF_WRITABLE) < 0) goto done;
    has_out = 1;
    const char *v = (const char *)views.buf;
    const unsigned char *bits = has_validity ? (const unsigned char *)validity.buf : NULL;
    char *o = (char *)out.buf;
    Py_ssize_t room = out.len, written = 0, row = start;
    int bad = 0;
    for (;;) {
        Py_ssize_t need = -1;
        Py_BEGIN_ALLOW_THREADS
        for (; row < stop && written < room; row++, skip = 0) {
            if (!is_valid(bits, row)) continue;
            const char *view = v + row * VIEW_BYTES;
            int32_t n = read_i32(view);
            if (n < 0 || skip > n) { bad = 1; break; }
            const char *src;
            if (n <= INLINE_BYTES) {
                src = view + 4;
            } else {
                int32_t index = read_i32(view + 8), at = read_i32(view + 12);
                if (index < 0 || index >= ndatas || at < 0) { bad = 1; break; }
                if (!have[index]) { need = index; break; } /* taken with the GIL, then this row again */
                if ((Py_ssize_t)at + n > datas[index].len) { bad = 1; break; }
                src = (const char *)datas[index].buf + at;
            }
            Py_ssize_t take = n - skip;
            if (take > room - written) {
                take = room - written;
                memcpy(o + written, src + skip, (size_t)take);
                written += take;
                skip += take;
                break; /* row stays: the rest of it goes to the next call */
            }
            memcpy(o + written, src + skip, (size_t)take);
            written += take;
        }
        Py_END_ALLOW_THREADS
        if (bad || need < 0) break;
        if (PyObject_GetBuffer(PyTuple_GET_ITEM(args[1], need), &datas[need], PyBUF_SIMPLE) < 0) goto done;
        have[need] = 1;
    }
    if (bad) {
        PyErr_SetString(PyExc_ValueError, "a view points outside its data buffers");
        goto done;
    }
    result = Py_BuildValue("(nnn)", row, row < stop ? skip : (Py_ssize_t)0, written);
done:
    if (has_out) PyBuffer_Release(&out);
    if (have)
        for (Py_ssize_t i = 0; i < ndatas; i++)
            if (have[i]) PyBuffer_Release(&datas[i]);
    PyMem_Free(have);
    PyMem_Free(datas);
    if (has_validity) PyBuffer_Release(&validity);
    PyBuffer_Release(&views);
    return result;
}

static PyMethodDef cviews_methods[] = {
    {"view_lengths", (PyCFunction)(void (*)(void))py_view_lengths, METH_FASTCALL,
     "view_lengths(views, validity, start, stop, out) -> rows written"},
    {"view_values", (PyCFunction)(void (*)(void))py_view_values, METH_FASTCALL,
     "view_values(views, datas, validity, start, skip, stop, out) -> (row, skip, written)"},
    {NULL, NULL, 0, NULL}};

int cviews_add_functions(PyObject *m) { return PyModule_AddFunctions(m, cviews_methods); }
