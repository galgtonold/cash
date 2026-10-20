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

/* A piece of a column: one chunk's views, data buffers and validity, and
 * the rows [start, stop) of it to read, as a tuple
 * (views, datas, validity, start, stop). */
typedef struct {
    Py_buffer views, validity;
    int has_validity;
    PyObject *datas; /* borrowed tuple */
    Py_ssize_t start, stop;
} piece_t;

static int get_piece(PyObject *pieces, Py_ssize_t index, piece_t *p) {
    PyObject *item = PyList_GET_ITEM(pieces, index);
    if (!PyTuple_Check(item) || PyTuple_GET_SIZE(item) != 5 || !PyTuple_Check(PyTuple_GET_ITEM(item, 1))) {
        PyErr_SetString(PyExc_TypeError, "a piece is (views, datas, validity, start, stop)");
        return -1;
    }
    p->datas = PyTuple_GET_ITEM(item, 1);
    p->start = PyLong_AsSsize_t(PyTuple_GET_ITEM(item, 3));
    p->stop = PyLong_AsSsize_t(PyTuple_GET_ITEM(item, 4));
    if ((p->start == -1 || p->stop == -1) && PyErr_Occurred()) return -1;
    return get_views(PyTuple_GET_ITEM(item, 0), PyTuple_GET_ITEM(item, 2), p->start, p->stop, &p->views, &p->validity,
                     &p->has_validity);
}

static void release_piece(piece_t *p) {
    if (p->has_validity) PyBuffer_Release(&p->validity);
    PyBuffer_Release(&p->views);
}

static int parse_many(PyObject *const *args, Py_ssize_t nargs, Py_ssize_t want, const char *name, Py_ssize_t *index,
                      Py_ssize_t *row) {
    if (nargs != want) {
        PyErr_Format(PyExc_TypeError, "%s takes %zd arguments", name, want);
        return -1;
    }
    if (!PyList_Check(args[0])) {
        PyErr_SetString(PyExc_TypeError, "pieces must be a list");
        return -1;
    }
    *index = PyLong_AsSsize_t(args[1]);
    *row = PyLong_AsSsize_t(args[2]);
    if ((*index == -1 || *row == -1) && PyErr_Occurred()) return -1;
    if (*index < 0 || *index > PyList_GET_SIZE(args[0])) {
        PyErr_SetString(PyExc_ValueError, "bad piece index");
        return -1;
    }
    return 0;
}

/* view_lengths_many(pieces, index, row, out) -> (index, row, written)
 *
 * view_lengths over a column's pieces, one after the other: from row `row`
 * of pieces[index] on, into out until it has no room for another length or
 * the last piece is read. Returns where the next call starts and the bytes
 * written; index is len(pieces) once all are read. A row before a piece's
 * start is read as its start. One call fills a whole block where a call
 * per chunk handed back after a few thousand rows: p08's log frame comes in
 * 1,280 chunks. */
static PyObject *py_view_lengths_many(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    Py_ssize_t index, row;
    if (parse_many(args, nargs, 4, "view_lengths_many", &index, &row) < 0) return NULL;
    PyObject *pieces = args[0];
    Py_buffer out;
    if (PyObject_GetBuffer(args[3], &out, PyBUF_WRITABLE) < 0) return NULL;
    char *o = (char *)out.buf;
    Py_ssize_t room = out.len / 8, written = 0;
    int bad = 0;
    while (index < PyList_GET_SIZE(pieces) && written < room) {
        piece_t p;
        if (get_piece(pieces, index, &p) < 0) {
            PyBuffer_Release(&out);
            return NULL;
        }
        if (row < p.start) row = p.start;
        const char *v = (const char *)p.views.buf;
        const unsigned char *bits = p.has_validity ? (const unsigned char *)p.validity.buf : NULL;
        Py_BEGIN_ALLOW_THREADS
        for (; row < p.stop && written < room; row++, written++) {
            int64_t length = 0;
            if (is_valid(bits, row)) {
                int32_t n = read_i32(v + row * VIEW_BYTES);
                if (n < 0) { bad = 1; break; }
                length = n;
            }
            memcpy(o + written * 8, &length, 8);
        }
        Py_END_ALLOW_THREADS
        release_piece(&p);
        if (bad) break;
        if (row >= p.stop) {
            index++;
            row = 0;
        }
    }
    PyBuffer_Release(&out);
    if (bad) {
        PyErr_SetString(PyExc_ValueError, "negative view length");
        return NULL;
    }
    return Py_BuildValue("(nnn)", index, row, written * 8);
}

/* view_values_many(pieces, index, row, skip, out) -> (index, row, skip, written)
 *
 * view_values over a column's pieces, one after the other, as
 * view_lengths_many reads their lengths: from row `row` of pieces[index],
 * its first `skip` bytes left out, into out until it is full or the last
 * piece is read. */
static PyObject *py_view_values_many(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    Py_ssize_t index, row;
    if (parse_many(args, nargs, 5, "view_values_many", &index, &row) < 0) return NULL;
    Py_ssize_t skip = PyLong_AsSsize_t(args[3]);
    if (skip == -1 && PyErr_Occurred()) return NULL;
    if (skip < 0) {
        PyErr_SetString(PyExc_ValueError, "negative skip");
        return NULL;
    }
    PyObject *pieces = args[0];
    Py_buffer out;
    if (PyObject_GetBuffer(args[4], &out, PyBUF_WRITABLE) < 0) return NULL;
    char *o = (char *)out.buf;
    Py_ssize_t room = out.len, written = 0;
    int bad = 0, failed = 0;
    while (index < PyList_GET_SIZE(pieces) && written < room) {
        piece_t p;
        if (get_piece(pieces, index, &p) < 0) {
            failed = 1;
            break;
        }
        if (row < p.start) {
            row = p.start;
            skip = 0;
        }
        Py_ssize_t ndatas = PyTuple_GET_SIZE(p.datas);
        /* Taken the first time a row points into it, as view_values does. */
        Py_buffer *datas = PyMem_Calloc(ndatas ? (size_t)ndatas : 1, sizeof(Py_buffer));
        char *have = PyMem_Calloc(ndatas ? (size_t)ndatas : 1, 1);
        if (!datas || !have) {
            PyErr_NoMemory();
            failed = 1;
        }
        const char *v = (const char *)p.views.buf;
        const unsigned char *bits = p.has_validity ? (const unsigned char *)p.validity.buf : NULL;
        while (!failed) {
            Py_ssize_t need = -1;
            Py_BEGIN_ALLOW_THREADS
            for (; row < p.stop && written < room; row++, skip = 0) {
                if (!is_valid(bits, row)) continue;
                const char *view = v + row * VIEW_BYTES;
                int32_t n = read_i32(view);
                if (n < 0 || skip > n) { bad = 1; break; }
                const char *src;
                if (n <= INLINE_BYTES) {
                    src = view + 4;
                } else {
                    int32_t at_index = read_i32(view + 8), at = read_i32(view + 12);
                    if (at_index < 0 || at_index >= ndatas || at < 0) { bad = 1; break; }
                    if (!have[at_index]) { need = at_index; break; }
                    if ((Py_ssize_t)at + n > datas[at_index].len) { bad = 1; break; }
                    src = (const char *)datas[at_index].buf + at;
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
            if (PyObject_GetBuffer(PyTuple_GET_ITEM(p.datas, need), &datas[need], PyBUF_SIMPLE) < 0) {
                failed = 1;
                break;
            }
            have[need] = 1;
        }
        if (have)
            for (Py_ssize_t i = 0; i < ndatas; i++)
                if (have[i]) PyBuffer_Release(&datas[i]);
        PyMem_Free(have);
        PyMem_Free(datas);
        Py_ssize_t stop = p.stop;
        release_piece(&p);
        if (bad || failed) break;
        if (row >= stop) {
            index++;
            row = 0;
            skip = 0;
        }
    }
    PyBuffer_Release(&out);
    if (failed) return NULL;
    if (bad) {
        PyErr_SetString(PyExc_ValueError, "a view points outside its data buffers");
        return NULL;
    }
    return Py_BuildValue("(nnnn)", index, row, skip, written);
}

/* copy_pieces(pieces, index, skip, out) -> (index, skip, written)
 *
 * The bytes of the buffers in the list pieces one after the other, from
 * byte `skip` of pieces[index] on, into out until it is full or the last
 * piece is copied: a fixed-width column's chunks' values gathered into the
 * block `fold_arrow_column` hashes, in one call where hashing each chunk's
 * slice was a call per chunk. index is len(pieces) once all are copied. */
static PyObject *py_copy_pieces(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    Py_ssize_t index, skip;
    if (parse_many(args, nargs, 4, "copy_pieces", &index, &skip) < 0) return NULL;
    if (skip < 0) {
        PyErr_SetString(PyExc_ValueError, "negative skip");
        return NULL;
    }
    PyObject *pieces = args[0];
    Py_buffer out;
    if (PyObject_GetBuffer(args[3], &out, PyBUF_WRITABLE) < 0) return NULL;
    Py_ssize_t room = out.len, written = 0;
    while (index < PyList_GET_SIZE(pieces) && written < room) {
        Py_buffer piece;
        if (PyObject_GetBuffer(PyList_GET_ITEM(pieces, index), &piece, PyBUF_SIMPLE) < 0) {
            PyBuffer_Release(&out);
            return NULL;
        }
        if (skip > piece.len) {
            PyBuffer_Release(&piece);
            PyBuffer_Release(&out);
            PyErr_SetString(PyExc_ValueError, "skip past the piece");
            return NULL;
        }
        Py_ssize_t take = piece.len - skip;
        if (take > room - written) take = room - written;
        if (take >= (1 << 16)) { /* a small copy is quicker than handing the lock over */
            Py_BEGIN_ALLOW_THREADS
            memcpy((char *)out.buf + written, (const char *)piece.buf + skip, (size_t)take);
            Py_END_ALLOW_THREADS
        } else {
            memcpy((char *)out.buf + written, (const char *)piece.buf + skip, (size_t)take);
        }
        written += take;
        skip += take;
        if (skip >= piece.len) {
            index++;
            skip = 0;
        }
        PyBuffer_Release(&piece);
    }
    PyBuffer_Release(&out);
    return Py_BuildValue("(nnn)", index, skip, written);
}

/* The `rows` bits of `src` from bit `start` on, to `dst` from bit 0, with
 * the bits past them in the last byte cleared. */
static void copy_bits(unsigned char *dst, const unsigned char *src, Py_ssize_t start, Py_ssize_t rows) {
    Py_ssize_t bytes = (rows + 7) / 8, first = start >> 3;
    int shift = (int)(start & 7);
    if (!bytes) return;
    if (!shift) {
        memcpy(dst, src + first, (size_t)bytes);
    } else {
        Py_ssize_t last = (start + rows - 1) >> 3; /* the last source byte holding a bit */
        for (Py_ssize_t i = 0; i < bytes; i++) {
            Py_ssize_t at = first + i;
            unsigned v = src[at] >> shift;
            if (at + 1 <= last) v |= (unsigned)src[at + 1] << (8 - shift);
            dst[i] = (unsigned char)v;
        }
    }
    if (rows & 7) dst[bytes - 1] &= (unsigned char)((1u << (rows & 7)) - 1);
}

/* view_data_sizes(pieces) -> [bytes]
 *
 * For each piece (views, datas, validity, start, stop), the bytes of its
 * present values longer than a view holds inline: what copy_view_chunks
 * writes of it into the data buffer. */
static PyObject *py_view_data_sizes(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 1 || !PyList_Check(args[0])) {
        PyErr_SetString(PyExc_TypeError, "view_data_sizes takes a list of pieces");
        return NULL;
    }
    PyObject *pieces = args[0];
    Py_ssize_t count = PyList_GET_SIZE(pieces);
    PyObject *out = PyList_New(count);
    if (!out) return NULL;
    for (Py_ssize_t i = 0; i < count; i++) {
        piece_t p;
        if (get_piece(pieces, i, &p) < 0) {
            Py_DECREF(out);
            return NULL;
        }
        const char *v = (const char *)p.views.buf;
        const unsigned char *bits = p.has_validity ? (const unsigned char *)p.validity.buf : NULL;
        long long total = 0;
        int bad = 0;
        Py_BEGIN_ALLOW_THREADS
        for (Py_ssize_t row = p.start; row < p.stop; row++) {
            if (!is_valid(bits, row)) continue;
            int32_t n = read_i32(v + row * VIEW_BYTES);
            if (n < 0) { bad = 1; break; }
            if (n > INLINE_BYTES) total += n;
        }
        Py_END_ALLOW_THREADS
        release_piece(&p);
        if (bad) {
            Py_DECREF(out);
            PyErr_SetString(PyExc_ValueError, "negative view length");
            return NULL;
        }
        PyObject *n = PyLong_FromLongLong(total);
        if (!n) {
            Py_DECREF(out);
            return NULL;
        }
        PyList_SET_ITEM(out, i, n);
    }
    return out;
}

/* copy_view_chunks(pieces, views_out, data_out, validity_out) -> None
 *
 * Each piece (views, datas, validity, start, stop, views_at, data_at,
 * validity_at) copied into the three buffers: its rows' views from byte
 * views_at, every value longer than a view holds inline from byte data_at
 * of data_out, one after the other, its view pointing there as buffer 0 of
 * the piece, and when validity_at is not None its validity bits from byte
 * validity_at, from bit 0. A missing row's view is written as an empty
 * value. The copy of a polars text chunk, which lists every data buffer of
 * its column, holds only the bytes its own rows use: what casting it to
 * large_string and back wrote, in one pass instead of three, for all of a
 * column's chunks in one call. */
static PyObject *py_copy_view_chunks(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 4 || !PyList_Check(args[0])) {
        PyErr_SetString(PyExc_TypeError, "copy_view_chunks takes pieces and three buffers");
        return NULL;
    }
    PyObject *pieces = args[0];
    Py_buffer vout, dout, bout;
    if (PyObject_GetBuffer(args[1], &vout, PyBUF_WRITABLE) < 0) return NULL;
    if (PyObject_GetBuffer(args[2], &dout, PyBUF_WRITABLE) < 0) {
        PyBuffer_Release(&vout);
        return NULL;
    }
    int has_bout = 0;
    if (args[3] != Py_None) {
        if (PyObject_GetBuffer(args[3], &bout, PyBUF_WRITABLE) < 0) {
            PyBuffer_Release(&dout);
            PyBuffer_Release(&vout);
            return NULL;
        }
        has_bout = 1;
    }
    int failed = 0;
    const char *message = NULL;
    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(pieces) && !failed && !message; i++) {
        PyObject *item = PyList_GET_ITEM(pieces, i);
        if (!PyTuple_Check(item) || PyTuple_GET_SIZE(item) != 8 || !PyTuple_Check(PyTuple_GET_ITEM(item, 1))) {
            PyErr_SetString(PyExc_TypeError,
                            "a piece is (views, datas, validity, start, stop, views_at, data_at, validity_at)");
            failed = 1;
            break;
        }
        PyObject *short_item = PyTuple_GetSlice(item, 0, 5);
        if (!short_item) {
            failed = 1;
            break;
        }
        PyObject *one = PyList_New(1);
        if (!one) {
            Py_DECREF(short_item);
            failed = 1;
            break;
        }
        PyList_SET_ITEM(one, 0, short_item);
        piece_t p;
        if (get_piece(one, 0, &p) < 0) {
            Py_DECREF(one);
            failed = 1;
            break;
        }
        Py_ssize_t views_at = PyLong_AsSsize_t(PyTuple_GET_ITEM(item, 5));
        Py_ssize_t data_at = PyLong_AsSsize_t(PyTuple_GET_ITEM(item, 6));
        PyObject *validity_at_obj = PyTuple_GET_ITEM(item, 7);
        Py_ssize_t validity_at = validity_at_obj == Py_None ? -2 : PyLong_AsSsize_t(validity_at_obj);
        Py_ssize_t rows = p.stop - p.start;
        if (PyErr_Occurred()) {
            failed = 1;
        } else if (views_at < 0 || data_at < 0 || views_at > vout.len || vout.len - views_at < rows * VIEW_BYTES ||
                   data_at > dout.len) {
            message = "output buffer too short";
        } else if (validity_at != -2 &&
                   (!p.has_validity || !has_bout || validity_at < 0 || validity_at > bout.len ||
                    bout.len - validity_at < (rows + 7) / 8)) {
            message = "validity buffer missing or too short";
        }
        Py_ssize_t ndatas = PyTuple_GET_SIZE(p.datas);
        Py_buffer *datas = NULL;
        char *have = NULL;
        if (!failed && !message) {
            datas = PyMem_Calloc(ndatas ? (size_t)ndatas : 1, sizeof(Py_buffer));
            have = PyMem_Calloc(ndatas ? (size_t)ndatas : 1, 1);
            if (!datas || !have) {
                PyErr_NoMemory();
                failed = 1;
            }
        }
        if (!failed && !message && validity_at != -2)
            copy_bits((unsigned char *)bout.buf + validity_at, (const unsigned char *)p.validity.buf, p.start, rows);
        Py_ssize_t row = p.start, written = data_at;
        const char *v = (const char *)p.views.buf;
        const unsigned char *bits = p.has_validity ? (const unsigned char *)p.validity.buf : NULL;
        char *vo = (char *)vout.buf + views_at;
        char *d = (char *)dout.buf;
        while (!failed && !message) {
            Py_ssize_t need = -1;
            Py_BEGIN_ALLOW_THREADS
            for (; row < p.stop; row++) {
                char *to = vo + (row - p.start) * VIEW_BYTES;
                if (!is_valid(bits, row)) {
                    memset(to, 0, VIEW_BYTES);
                    continue;
                }
                const char *view = v + row * VIEW_BYTES;
                int32_t n = read_i32(view);
                if (n < 0) { message = "negative view length"; break; }
                if (n <= INLINE_BYTES) {
                    memcpy(to, view, VIEW_BYTES);
                    continue;
                }
                int32_t at_index = read_i32(view + 8), at = read_i32(view + 12);
                if (at_index < 0 || at_index >= ndatas || at < 0) { message = "a view points outside its data buffers"; break; }
                if (!have[at_index]) { need = at_index; break; }
                if ((Py_ssize_t)at + n > datas[at_index].len) { message = "a view points outside its data buffers"; break; }
                if (dout.len - written < n || written - data_at + n > INT32_MAX) { message = "data buffer too short"; break; }
                memcpy(d + written, (const char *)datas[at_index].buf + at, (size_t)n);
                int32_t zero = 0, offset = (int32_t)(written - data_at);
                memcpy(to, view, 8); /* the length and the first 4 bytes */
                memcpy(to + 8, &zero, 4);
                memcpy(to + 12, &offset, 4);
                written += n;
            }
            Py_END_ALLOW_THREADS
            if (message || need < 0) break;
            if (PyObject_GetBuffer(PyTuple_GET_ITEM(p.datas, need), &datas[need], PyBUF_SIMPLE) < 0) {
                failed = 1;
                break;
            }
            have[need] = 1;
        }
        if (have)
            for (Py_ssize_t k = 0; k < ndatas; k++)
                if (have[k]) PyBuffer_Release(&datas[k]);
        PyMem_Free(have);
        PyMem_Free(datas);
        release_piece(&p);
        Py_DECREF(one);
    }
    if (has_bout) PyBuffer_Release(&bout);
    PyBuffer_Release(&dout);
    PyBuffer_Release(&vout);
    if (failed) return NULL;
    if (message) {
        PyErr_SetString(PyExc_ValueError, message);
        return NULL;
    }
    Py_RETURN_NONE;
}

/* copy_fixed_chunks(pieces, values_out, validity_out) -> None
 *
 * Each piece (values, width, validity, start, stop, values_at, validity_at)
 * of a fixed-width column copied: the bytes of its rows from byte
 * values_at, and when validity_at is not None its validity bits from byte
 * validity_at, from bit 0. */
static PyObject *py_copy_fixed_chunks(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    (void)self;
    if (nargs != 3 || !PyList_Check(args[0])) {
        PyErr_SetString(PyExc_TypeError, "copy_fixed_chunks takes pieces and two buffers");
        return NULL;
    }
    PyObject *pieces = args[0];
    Py_buffer vout, bout;
    if (PyObject_GetBuffer(args[1], &vout, PyBUF_WRITABLE) < 0) return NULL;
    int has_bout = 0;
    if (args[2] != Py_None) {
        if (PyObject_GetBuffer(args[2], &bout, PyBUF_WRITABLE) < 0) {
            PyBuffer_Release(&vout);
            return NULL;
        }
        has_bout = 1;
    }
    const char *message = NULL;
    int failed = 0;
    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(pieces) && !failed && !message; i++) {
        PyObject *item = PyList_GET_ITEM(pieces, i);
        if (!PyTuple_Check(item) || PyTuple_GET_SIZE(item) != 7) {
            PyErr_SetString(PyExc_TypeError,
                            "a piece is (values, width, validity, start, stop, values_at, validity_at)");
            failed = 1;
            break;
        }
        Py_ssize_t width = PyLong_AsSsize_t(PyTuple_GET_ITEM(item, 1));
        Py_ssize_t start = PyLong_AsSsize_t(PyTuple_GET_ITEM(item, 3));
        Py_ssize_t stop = PyLong_AsSsize_t(PyTuple_GET_ITEM(item, 4));
        Py_ssize_t values_at = PyLong_AsSsize_t(PyTuple_GET_ITEM(item, 5));
        PyObject *validity_at_obj = PyTuple_GET_ITEM(item, 6);
        Py_ssize_t validity_at = validity_at_obj == Py_None ? -2 : PyLong_AsSsize_t(validity_at_obj);
        if (PyErr_Occurred()) {
            failed = 1;
            break;
        }
        Py_ssize_t rows = stop - start;
        if (width <= 0 || start < 0 || rows < 0 || values_at < 0 || values_at > vout.len ||
            vout.len - values_at < rows * width) {
            message = "bad piece or output buffer too short";
            break;
        }
        Py_buffer values;
        if (PyObject_GetBuffer(PyTuple_GET_ITEM(item, 0), &values, PyBUF_SIMPLE) < 0) {
            failed = 1;
            break;
        }
        if (values.len < stop * width) {
            PyBuffer_Release(&values);
            message = "values buffer too short";
            break;
        }
        Py_BEGIN_ALLOW_THREADS
        memcpy((char *)vout.buf + values_at, (const char *)values.buf + start * width, (size_t)(rows * width));
        Py_END_ALLOW_THREADS
        PyBuffer_Release(&values);
        if (validity_at != -2) {
            Py_buffer validity;
            PyObject *validity_obj = PyTuple_GET_ITEM(item, 2);
            if (validity_obj == Py_None || !has_bout || validity_at < 0 || validity_at > bout.len ||
                bout.len - validity_at < (rows + 7) / 8) {
                message = "validity buffer missing or too short";
                break;
            }
            if (PyObject_GetBuffer(validity_obj, &validity, PyBUF_SIMPLE) < 0) {
                failed = 1;
                break;
            }
            if (validity.len < (stop + 7) / 8) {
                PyBuffer_Release(&validity);
                message = "validity buffer too short";
                break;
            }
            copy_bits((unsigned char *)bout.buf + validity_at, (const unsigned char *)validity.buf, start, rows);
            PyBuffer_Release(&validity);
        }
    }
    if (has_bout) PyBuffer_Release(&bout);
    PyBuffer_Release(&vout);
    if (failed) return NULL;
    if (message) {
        PyErr_SetString(PyExc_ValueError, message);
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyMethodDef cviews_methods[] = {
    {"view_lengths", (PyCFunction)(void (*)(void))py_view_lengths, METH_FASTCALL,
     "view_lengths(views, validity, start, stop, out) -> rows written"},
    {"view_values", (PyCFunction)(void (*)(void))py_view_values, METH_FASTCALL,
     "view_values(views, datas, validity, start, skip, stop, out) -> (row, skip, written)"},
    {"copy_pieces", (PyCFunction)(void (*)(void))py_copy_pieces, METH_FASTCALL,
     "copy_pieces(pieces, index, skip, out) -> (index, skip, written)"},
    {"view_data_sizes", (PyCFunction)(void (*)(void))py_view_data_sizes, METH_FASTCALL,
     "view_data_sizes(pieces) -> [bytes]"},
    {"copy_view_chunks", (PyCFunction)(void (*)(void))py_copy_view_chunks, METH_FASTCALL,
     "copy_view_chunks(pieces, views_out, data_out, validity_out) -> None"},
    {"copy_fixed_chunks", (PyCFunction)(void (*)(void))py_copy_fixed_chunks, METH_FASTCALL,
     "copy_fixed_chunks(pieces, values_out, validity_out) -> None"},
    {"view_lengths_many", (PyCFunction)(void (*)(void))py_view_lengths_many, METH_FASTCALL,
     "view_lengths_many(pieces, index, row, out) -> (index, row, written)"},
    {"view_values_many", (PyCFunction)(void (*)(void))py_view_values_many, METH_FASTCALL,
     "view_values_many(pieces, index, row, skip, out) -> (index, row, skip, written)"},
    {NULL, NULL, 0, NULL}};

int cviews_add_functions(PyObject *m) { return PyModule_AddFunctions(m, cviews_methods); }
