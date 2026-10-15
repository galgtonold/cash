/*
 * cash._cwatch: "provably unchanged" signals for values cash hashes.
 *
 * The module never computes a digest. cash computes every digest in Python,
 * exactly as without this module; this module only answers, for a value
 * cash hashed before, "nothing its digest depends on has changed since",
 * and then hands back that earlier digest. Anything it cannot prove is
 * answered "unknown", and cash hashes as it always did.
 *
 * How it proves it
 *   arm(value) walks the value once (in C, no Python code runs) and records
 *   what pickling it reads:
 *     - every exact dict: watched with a dict watcher (an event on it marks
 *       the value dirty);
 *     - every exact list and set: a snapshot of its item pointers, compared
 *       when the value is checked (there are no list watchers);
 *     - every instance of a plain class (a class cash's Python side
 *       accepted, `set_type_verdict`): its type, its __dict__ object (itself
 *       a watched dict) and its slot values;
 *     - every class and function the pickle names: the type's version tag
 *       (with a type watcher), and the names pickle looks it up by
 *       (sys.modules[module], module.__dict__[name]) as watched dict keys;
 *     - copyreg.dispatch_table[type] and copyreg's extension registry as
 *       watched keys;
 *     - tuples, frozensets and atoms (None, bool, int, float, str, bytes)
 *       are immutable: walked through, not recorded.
 *   A value holding anything else (an array, a C object, a class with its
 *   own pickling, ...) is not armed.
 *
 * The table
 *   - roots: one per armed value, keyed by address. A root holds the digest
 *     cash computed per consumer kind, with the root version it was computed
 *     at (per-consumer versions), its snapshot records, its watched nodes,
 *     and what keeps its identity provable (the value is a dict whose
 *     DEALLOCATED event drops the root; or a weak reference; or an
 *     "anchor": a namespace slot that has held it since it was armed).
 *   - watched nodes: 16 bytes (address, root, chain link), indexed by
 *     address in one open-addressing table; one node per (object, root).
 *     A dict event marks each root holding the dict dirty, stopping at a
 *     root that is already dirty (linear). Keyed nodes (24 bytes) carry a
 *     key: only an event on that key (or a clear) counts.
 *   - snapshot records: words in one array per root, in walk order
 *     (preorder), so a check stops at the first changed container before
 *     reading anything that container held.
 *
 * Memory policy: values are armed only on their second check (`check`
 * reports first sight), only above `min_bytes` of estimated content, and
 * the whole table stays under `budget` bytes, evicting the roots with the
 * least hashing time saved per byte (aged by how long ago they were
 * checked). A root that keeps changing between checks is dropped and its
 * address skipped for a growing while.
 *
 * Threads: built for the GIL. On a free-threaded build the module imports
 * but reports USABLE = False (phase 1: the fast path is off there).
 * Re-entrance (a finalizer calling back into cash while this module runs
 * Python's allocator) is refused with BUSY; callbacks only set flags.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>
#include <stdint.h>
#include <string.h>

#define CWATCH_API_VERSION 1
#define CWATCH_VERSION "0.1.0"

#if PY_VERSION_HEX < 0x030C0000
#error "cash._cwatch needs CPython 3.12 or later (dict/type/function watchers)"
#endif

#ifdef Py_GIL_DISABLED
#define CWATCH_USABLE 0
#else
#define CWATCH_USABLE 1
#endif

#if CWATCH_USABLE

/* ------------------------------------------------------------ constants */
#define NONE32 0xFFFFFFFFu
#define TOMB32 0xFFFFFFFEu
#define KBIT 0x80000000u     /* node index: keyed pool */
#define DEADBIT 0x80000000u  /* node root field: object gone */
#define NKINDS 4

/* check() answers besides a digest */
enum { ST_FIRST = 0, ST_ARM = 1, ST_SKIP = 2, ST_BUSY = 3, ST_OFF = 4 };
/* arm() answers */
enum {
    A_ARMED = 0, A_ALREADY = 1, A_NEED_TYPE = 2, A_INELIGIBLE = 3, A_TOO_SMALL = 4,
    A_NO_ANCHOR = 5, A_OVER_BUDGET = 6, A_BUSY = 7, A_ERROR = 8
};
/* snapshot record kinds */
enum { R_LIST = 1, R_SET = 2, R_INST = 3, R_FUNC = 4, R_TYPE = 5, R_VIEW = 6 };

/* ------------------------------------------------------------ globals */
typedef struct { uintptr_t addr; uint32_t root; uint32_t next; } WNode;
typedef struct { WNode n; PyObject *key; /* borrowed: the root holds it in refs */ } KNode;

typedef struct {
    uintptr_t addr;
    PyObject *wref;              /* weak reference to the value, or NULL */
    uint32_t *nodes; uint32_t nnodes, capnodes;
    uintptr_t *snap; size_t nsnap, capsnap;
    PyObject **refs; uint32_t nrefs, caprefs;
    PyObject *dg[NKINDS]; uint32_t dgver[NKINDS];
    uint32_t version;
    uint8_t used, dirty, gone, anchored;
    uint32_t streak;             /* re-arms since the last reuse */
    uint64_t last_used;
    double hash_s;               /* seconds one hash of it took */
    uint64_t reuses;
    size_t bytes;                /* table bytes it holds */
    size_t content;              /* estimated content bytes */
} Root;

static WNode *wn = NULL; static uint32_t nwn = 0, capwn = 0, wfree = NONE32, wlive = 0;
static KNode *kn = NULL; static uint32_t nkn = 0, capkn = 0, kfree = NONE32, klive = 0;
static uint32_t *ht = NULL; static size_t ht_cap = 0, ht_used = 0, ht_tomb = 0;

static Root *roots = NULL; static uint32_t nroots = 0, caproots = 0, rfree = NONE32, rlive = 0;
static uint32_t *rht = NULL; static size_t rht_cap = 0, rht_used = 0, rht_tomb = 0;

static int dict_wid = -1, type_wid = -1, func_wid = -1;
static int busy = 0;
static int any_gone = 0;
static uint64_t tick = 0;

/* settings */
static size_t budget = (size_t)256 << 20;
static size_t min_bytes = 64 << 10;
static int enabled = 1;

/* references handed in by configure() */
static PyObject *dispatch_table = NULL;    /* copyreg.dispatch_table */
static PyObject *ext_registry = NULL;      /* copyreg._extension_registry */
static PyObject *cached_property_type = NULL;
static PyObject *views_of = NULL;   /* obj -> frozenset of the view attributes its key leaves out, or None */

/* interned names */
static PyObject *s_module, *s_getstate, *s_reduce, *s_reduce_ex, *s_dot;

/* counters */
static unsigned long long c_checks, c_hits, c_first, c_arms, c_commits, c_events, c_dirty_marks,
    c_evicted, c_dropped_streak, c_too_small, c_ineligible, c_no_anchor, c_over_budget, c_busy,
    c_verify_fail, c_type_events, c_func_events, c_swept;

/* ------------------------------------------------------------ hashing */
static inline size_t hmix(uintptr_t p) {
    uint64_t x = (uint64_t)p;
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33;
    return (size_t)x;
}

static inline WNode *NODE(uint32_t i) { return (i & KBIT) ? &kn[i & ~KBIT].n : &wn[i]; }

/* ------------------------------------------------------------ node index */
static long ht_slot(uintptr_t addr) {
    if (!ht_cap) return -1;
    size_t m = ht_cap - 1, i = hmix(addr) & m;
    for (;;) {
        uint32_t v = ht[i];
        if (v == NONE32) return -1;
        if (v != TOMB32 && NODE(v)->addr == addr) return (long)i;
        i = (i + 1) & m;
    }
}

static int ht_grow(void) {
    size_t oc = ht_cap, nc = oc ? oc : 1024;
    while ((ht_used + 1) * 2 > nc) nc *= 2;
    if (nc == oc && ht_tomb == 0) nc *= 2;
    uint32_t *old = ht;
    uint32_t *nh = malloc(nc * sizeof *nh);
    if (!nh) { PyErr_NoMemory(); return -1; }
    memset(nh, 0xFF, nc * sizeof *nh);
    for (size_t j = 0; j < oc; j++) {
        uint32_t v = old[j];
        if (v == NONE32 || v == TOMB32) continue;
        size_t m = nc - 1, i = hmix(NODE(v)->addr) & m;
        while (nh[i] != NONE32) i = (i + 1) & m;
        nh[i] = v;
    }
    free(old); ht = nh; ht_cap = nc; ht_tomb = 0;
    return 0;
}

static int ht_add(uint32_t idx) {
    WNode *n = NODE(idx);
    long s = ht_slot(n->addr);
    if (s >= 0) { n->next = ht[s]; ht[s] = idx; return 0; }
    if ((ht_used + ht_tomb + 1) * 2 > ht_cap) { if (ht_grow() < 0) return -1; }
    size_t m = ht_cap - 1, i = hmix(n->addr) & m;
    while (ht[i] != NONE32 && ht[i] != TOMB32) i = (i + 1) & m;
    if (ht[i] == TOMB32) ht_tomb--;
    ht[i] = idx; n->next = NONE32; ht_used++;
    return 0;
}

/* unlink idx from its chain; returns 1 when no live node is left for the address */
static int ht_remove(uint32_t idx) {
    WNode *n = NODE(idx);
    long s = ht_slot(n->addr);
    if (s < 0) return 1;
    uint32_t *link = &ht[s];
    while (*link != NONE32 && *link != idx) link = &NODE(*link)->next;
    if (*link == idx) *link = n->next;
    if (ht[s] == NONE32) { ht[s] = TOMB32; ht_used--; ht_tomb++; return 1; }
    for (uint32_t j = ht[s]; j != NONE32; j = NODE(j)->next)
        if (!(NODE(j)->root & DEADBIT)) return 0;
    return 1;
}

static int64_t node_new(uintptr_t addr, uint32_t root, PyObject *key) {
    uint32_t idx;
    if (key) {
        if (kfree != NONE32) { idx = kfree; kfree = kn[idx].n.next; }
        else {
            if (nkn == capkn) {
                uint32_t nc = capkn ? capkn * 2 : 256;
                KNode *p = realloc(kn, (size_t)nc * sizeof *p);
                if (!p) { PyErr_NoMemory(); return -1; }
                kn = p; capkn = nc;
            }
            idx = nkn++;
        }
        kn[idx].key = key; kn[idx].n.addr = addr; kn[idx].n.root = root; kn[idx].n.next = NONE32;
        klive++;
        idx |= KBIT;
    } else {
        if (wfree != NONE32) { idx = wfree; wfree = wn[idx].next; }
        else {
            if (nwn == capwn) {
                uint32_t nc = capwn ? capwn * 2 : 4096;
                if (nc >= KBIT) { PyErr_NoMemory(); return -1; }
                WNode *p = realloc(wn, (size_t)nc * sizeof *p);
                if (!p) { PyErr_NoMemory(); return -1; }
                wn = p; capwn = nc;
            }
            idx = nwn++;
        }
        wn[idx].addr = addr; wn[idx].root = root; wn[idx].next = NONE32;
        wlive++;
    }
    if (ht_add(idx) < 0) return -1;
    return (int64_t)idx;
}

static void node_release(uint32_t idx) {
    if (idx & KBIT) {
        uint32_t k = idx & ~KBIT;
        kn[k].key = NULL; kn[k].n.addr = 0; kn[k].n.next = kfree; kfree = k; klive--;
    } else {
        wn[idx].addr = 0; wn[idx].next = wfree; wfree = idx; wlive--;
    }
}

/* ------------------------------------------------------------ roots index */
static long rht_find(uintptr_t addr) {
    if (!rht_cap) return -1;
    size_t m = rht_cap - 1, i = hmix(addr) & m;
    for (;;) {
        uint32_t v = rht[i];
        if (v == NONE32) return -1;
        if (v != TOMB32 && roots[v].addr == addr) return (long)v;
        i = (i + 1) & m;
    }
}

static int rht_grow(void) {
    size_t oc = rht_cap, nc = oc ? oc : 256;
    while ((rht_used + 1) * 2 > nc) nc *= 2;
    if (nc == oc && rht_tomb == 0) nc *= 2;
    uint32_t *nh = malloc(nc * sizeof *nh);
    if (!nh) { PyErr_NoMemory(); return -1; }
    memset(nh, 0xFF, nc * sizeof *nh);
    for (size_t j = 0; j < oc; j++) {
        uint32_t v = rht[j];
        if (v == NONE32 || v == TOMB32) continue;
        size_t m = nc - 1, i = hmix(roots[v].addr) & m;
        while (nh[i] != NONE32) i = (i + 1) & m;
        nh[i] = v;
    }
    free(rht); rht = nh; rht_cap = nc; rht_tomb = 0;
    return 0;
}

static int rht_add(uint32_t r) {
    if ((rht_used + rht_tomb + 1) * 2 > rht_cap) { if (rht_grow() < 0) return -1; }
    size_t m = rht_cap - 1, i = hmix(roots[r].addr) & m;
    while (rht[i] != NONE32 && rht[i] != TOMB32) i = (i + 1) & m;
    if (rht[i] == TOMB32) rht_tomb--;
    rht[i] = r; rht_used++;
    return 0;
}

static void rht_remove(uint32_t r) {
    if (!rht_cap) return;
    size_t m = rht_cap - 1, i = hmix(roots[r].addr) & m;
    for (;;) {
        uint32_t v = rht[i];
        if (v == NONE32) return;
        if (v == r) { rht[i] = TOMB32; rht_used--; rht_tomb++; return; }
        i = (i + 1) & m;
    }
}

static size_t root_bytes(Root *R) {
    return sizeof(Root) + (size_t)R->capnodes * 4 + R->capsnap * sizeof(uintptr_t) +
           (size_t)R->caprefs * sizeof(PyObject *);
}

static size_t table_bytes_total = 0;   /* sum of roots' bytes */

static size_t index_bytes(void) {
    return ht_cap * 4 + rht_cap * 4 + (size_t)capwn * sizeof(WNode) + (size_t)capkn * sizeof(KNode) +
           (size_t)caproots * sizeof(Root);
}

/* free everything a root holds except its slot; unwatches what nobody else watches */
static void root_clear(Root *R) {
    for (uint32_t j = 0; j < R->nnodes; j++) {
        uint32_t idx = R->nodes[j];
        WNode *n = NODE(idx);
        int dead = (n->root & DEADBIT) != 0;
        int last = ht_remove(idx);
        if (last && !dead && !(idx & KBIT)) {
            /* a dict or a type: (functions have one global watcher) */
            PyObject *o = (PyObject *)n->addr;
            if (PyDict_Check(o)) { if (PyDict_Unwatch(dict_wid, o) < 0) PyErr_Clear(); }
        } else if (last && !dead && (idx & KBIT)) {
            PyObject *o = (PyObject *)n->addr;
            if (PyDict_Unwatch(dict_wid, o) < 0) PyErr_Clear();
        }
        node_release(idx);
    }
    R->nnodes = 0;
    R->nsnap = 0;
    for (uint32_t j = 0; j < R->nrefs; j++) Py_XDECREF(R->refs[j]);
    R->nrefs = 0;
    for (int k = 0; k < NKINDS; k++) { Py_CLEAR(R->dg[k]); R->dgver[k] = 0; }
    Py_CLEAR(R->wref);
    R->anchored = 0;
}

static void root_free(uint32_t r) {
    Root *R = &roots[r];
    table_bytes_total -= R->bytes;
    root_clear(R);
    free(R->snap); free(R->nodes); free(R->refs);
    rht_remove(r);
    memset(R, 0, sizeof *R);
    R->addr = 0; R->used = 0;
    R->version = 0;
    R->streak = rfree;   /* free-list link */
    rfree = r; rlive--;
}

static long root_new(uintptr_t addr) {
    uint32_t r;
    if (rfree != NONE32) { r = rfree; rfree = roots[r].streak; }
    else {
        if (nroots == caproots) {
            uint32_t nc = caproots ? caproots * 2 : 64;
            Root *p = realloc(roots, (size_t)nc * sizeof *p);
            if (!p) { PyErr_NoMemory(); return -1; }
            roots = p; caproots = nc;
        }
        r = nroots++;
    }
    Root *R = &roots[r];
    memset(R, 0, sizeof *R);
    R->addr = addr; R->used = 1;
    if (rht_add(r) < 0) { R->used = 0; R->streak = rfree; rfree = r; return -1; }
    rlive++;
    R->bytes = root_bytes(R);
    table_bytes_total += R->bytes;
    return (long)r;
}

static inline void mark_dirty(uint32_t r) {
    if (!roots[r].dirty) { roots[r].dirty = 1; c_dirty_marks++; }
}

/* ------------------------------------------------------------ callbacks */
static int str_equal(PyObject *a, PyObject *b) {
    if (a == b) return 1;
    if (!PyUnicode_CheckExact(a) || !PyUnicode_CheckExact(b)) return 0;
    Py_ssize_t n = PyUnicode_GET_LENGTH(a);
    if (n != PyUnicode_GET_LENGTH(b) || PyUnicode_KIND(a) != PyUnicode_KIND(b)) return 0;
    return memcmp(PyUnicode_DATA(a), PyUnicode_DATA(b), (size_t)n * PyUnicode_KIND(a)) == 0;
}

static int on_dict(PyDict_WatchEvent ev, PyObject *d, PyObject *key, PyObject *nv) {
    (void)nv;
    c_events++;
    long s = ht_slot((uintptr_t)d);
    if (s < 0) return 0;
    for (uint32_t i = ht[s]; i != NONE32; i = NODE(i)->next) {
        WNode *n = NODE(i);
        if (n->root & DEADBIT) continue;
        uint32_t r = n->root;
        if ((i & KBIT) && key != NULL && ev != PyDict_EVENT_DEALLOCATED) {
            PyObject *k = kn[i & ~KBIT].key;
            if (!str_equal(k, key)) continue;
        }
        mark_dirty(r);
        if (ev == PyDict_EVENT_DEALLOCATED) {
            n->root |= DEADBIT;
            if (roots[r].addr == (uintptr_t)d || (i & KBIT)) { roots[r].gone = 1; any_gone = 1; }
        } else if ((i & KBIT) && roots[r].anchored && roots[r].nnodes && roots[r].nodes[0] == i) {
            /* the namespace slot that proved the value's identity: rebound or deleted */
            roots[r].gone = 1; any_gone = 1;
        }
    }
    return 0;
}

static int on_type(PyTypeObject *t) {
    c_type_events++;
    long s = ht_slot((uintptr_t)t);
    if (s < 0) return 0;
    for (uint32_t i = ht[s]; i != NONE32; i = NODE(i)->next) {
        WNode *n = NODE(i);
        if (n->root & DEADBIT) continue;
        mark_dirty(n->root);
    }
    return 0;
}

static int on_func(PyFunction_WatchEvent ev, PyFunctionObject *f, PyObject *nv) {
    (void)nv;
    if (ev == PyFunction_EVENT_CREATE) return 0;
    long s = ht_slot((uintptr_t)f);
    if (s < 0) return 0;
    c_func_events++;
    for (uint32_t i = ht[s]; i != NONE32; i = NODE(i)->next) {
        WNode *n = NODE(i);
        if (n->root & DEADBIT) continue;
        mark_dirty(n->root);
        if (ev == PyFunction_EVENT_DESTROY) n->root |= DEADBIT;
    }
    return 0;
}

static int ensure_watchers(void) {
    if (dict_wid < 0) { dict_wid = PyDict_AddWatcher(on_dict); if (dict_wid < 0) return -1; }
    if (type_wid < 0) { type_wid = PyType_AddWatcher(on_type); if (type_wid < 0) return -1; }
    if (func_wid < 0) { func_wid = PyFunction_AddWatcher(on_func); if (func_wid < 0) return -1; }
    return 0;
}

/* ------------------------------------------------------------ type verdicts */
typedef struct {
    PyTypeObject *t;      /* borrowed: identity with tag */
    unsigned int tag;
    int ok;
    Py_ssize_t nslots;
    Py_ssize_t *offsets;
    int views;            /* cash's keyed pickle leaves this class's view attributes out */
} TypeInfo;

static TypeInfo *tinfo = NULL; static size_t tinfo_cap = 0, tinfo_used = 0;

static TypeInfo *tinfo_find(PyTypeObject *t) {
    if (!tinfo_cap) return NULL;
    size_t m = tinfo_cap - 1, i = hmix((uintptr_t)t) & m;
    for (;;) {
        if (tinfo[i].t == NULL) return NULL;
        if (tinfo[i].t == t) return &tinfo[i];
        i = (i + 1) & m;
    }
}

static TypeInfo *tinfo_put(PyTypeObject *t) {
    TypeInfo *e = tinfo_find(t);
    if (e) { free(e->offsets); e->offsets = NULL; e->nslots = 0; e->views = 0; return e; }
    if ((tinfo_used + 1) * 2 > tinfo_cap) {
        size_t nc = tinfo_cap ? tinfo_cap * 2 : 256;
        TypeInfo *nt = calloc(nc, sizeof *nt);
        if (!nt) { PyErr_NoMemory(); return NULL; }
        for (size_t j = 0; j < tinfo_cap; j++) {
            if (!tinfo[j].t) continue;
            size_t m = nc - 1, i = hmix((uintptr_t)tinfo[j].t) & m;
            while (nt[i].t) i = (i + 1) & m;
            nt[i] = tinfo[j];
        }
        free(tinfo); tinfo = nt; tinfo_cap = nc;
    }
    size_t m = tinfo_cap - 1, i = hmix((uintptr_t)t) & m;
    while (tinfo[i].t) i = (i + 1) & m;
    tinfo[i].t = t; tinfo_used++;
    return &tinfo[i];
}

/* the verdict for t, valid only while its version tag is the one it was given at */
static TypeInfo *verdict_of(PyTypeObject *t) {
    TypeInfo *e = tinfo_find(t);
    if (!e) return NULL;
    if (e->ok && (t->tp_version_tag == 0 || t->tp_version_tag != e->tag)) return NULL;
    if (!e->ok && e->tag != t->tp_version_tag) return NULL;
    return e;
}

/* ------------------------------------------------------------ walk state */
typedef struct { uintptr_t *a; size_t cap, used; } AddrSet;

static int aset_add(AddrSet *S, uintptr_t p) {   /* 1 added, 0 present, -1 error */
    if ((S->used + 1) * 2 > S->cap) {
        size_t nc = S->cap ? S->cap * 2 : 1024;
        uintptr_t *na = calloc(nc, sizeof *na);
        if (!na) { PyErr_NoMemory(); return -1; }
        for (size_t j = 0; j < S->cap; j++) if (S->a[j]) {
            size_t m = nc - 1, i = hmix(S->a[j]) & m;
            while (na[i]) i = (i + 1) & m;
            na[i] = S->a[j];
        }
        free(S->a); S->a = na; S->cap = nc;
    }
    size_t m = S->cap - 1, i = hmix(p) & m;
    while (S->a[i]) { if (S->a[i] == p) return 0; i = (i + 1) & m; }
    S->a[i] = p; S->used++;
    return 1;
}

static void aset_clear(AddrSet *S) {
    if (S->cap > (1u << 16)) { free(S->a); S->a = NULL; S->cap = 0; }
    else if (S->a) memset(S->a, 0, S->cap * sizeof *S->a);
    S->used = 0;
}

typedef struct { PyObject **v; size_t n, cap; } ObjVec;
static int ovec_push(ObjVec *V, PyObject *o) {   /* steals nothing: increfs */
    if (V->n == V->cap) {
        size_t nc = V->cap ? V->cap * 2 : 1024;
        PyObject **p = realloc(V->v, nc * sizeof *p);
        if (!p) { PyErr_NoMemory(); return -1; }
        V->v = p; V->cap = nc;
    }
    Py_INCREF(o); V->v[V->n++] = o;
    return 0;
}

static AddrSet visited, typeseen, insttypes;
static ObjVec stack_, keep;

typedef struct {
    Root *R; uint32_t r;
    size_t content;
    int status;            /* A_* when stopping */
    PyObject *detail;      /* new ref: the type to classify or the reason */
    int saw_instances;
    int saw_globals;
} Walk;

static int root_push_node(Walk *W, uintptr_t addr, PyObject *key) {
    Root *R = W->R;
    if (R->nnodes == R->capnodes) {
        uint32_t nc = R->capnodes ? R->capnodes * 2 : 16;
        uint32_t *p = realloc(R->nodes, (size_t)nc * 4);
        if (!p) { PyErr_NoMemory(); return -1; }
        R->nodes = p; R->capnodes = nc;
    }
    int64_t idx = node_new(addr, W->r, key);
    if (idx < 0) return -1;
    R->nodes[R->nnodes++] = (uint32_t)idx;
    return 0;
}

static int snap_reserve(Root *R, size_t n) {
    if (R->nsnap + n > R->capsnap) {
        size_t nc = R->capsnap ? R->capsnap : 64;
        while (nc < R->nsnap + n) nc *= 2;
        uintptr_t *p = realloc(R->snap, nc * sizeof *p);
        if (!p) { PyErr_NoMemory(); return -1; }
        R->snap = p; R->capsnap = nc;
    }
    return 0;
}

static int root_ref(Root *R, PyObject *o) {
    if (R->nrefs == R->caprefs) {
        uint32_t nc = R->caprefs ? R->caprefs * 2 : 8;
        PyObject **p = realloc(R->refs, (size_t)nc * sizeof *p);
        if (!p) { PyErr_NoMemory(); return -1; }
        R->refs = p; R->caprefs = nc;
    }
    Py_INCREF(o); R->refs[R->nrefs++] = o;
    return 0;
}

static void stop(Walk *W, int status, PyObject *detail) {
    W->status = status;
    Py_XINCREF(detail);
    Py_XSETREF(W->detail, detail);
}

static int push(PyObject *o) { return ovec_push(&stack_, o); }

/* a heap class with the plain metaclass that cash's Python side accepted */
static int type_record(Walk *W, PyTypeObject *t);

/* pickle names obj (a class or a function) by module and qualified name:
   record what that lookup reads. 0 ok, 1 not provable (stop set), -1 error */
static int global_ref(Walk *W, PyObject *obj, PyObject *modname, PyObject *qualname) {
    Root *R = W->R;
    if (!modname || !qualname || !PyUnicode_CheckExact(modname) || !PyUnicode_CheckExact(qualname)) {
        stop(W, A_INELIGIBLE, obj); return 1;
    }
    PyObject *modules = PyImport_GetModuleDict();
    if (!modules || !PyDict_CheckExact(modules)) { stop(W, A_INELIGIBLE, obj); return 1; }
    PyObject *mod = PyDict_GetItemWithError(modules, modname);   /* borrowed */
    if (!mod) { if (PyErr_Occurred()) return -1; stop(W, A_INELIGIBLE, obj); return 1; }
    if (!PyModule_CheckExact(mod)) { stop(W, A_INELIGIBLE, obj); return 1; }
    PyObject *md = PyModule_GetDict(mod);
    if (!md || !PyDict_CheckExact(md)) { stop(W, A_INELIGIBLE, obj); return 1; }
    PyObject *parts = PyUnicode_Split(qualname, s_dot, -1);
    if (!parts) return -1;
    Py_ssize_t np = PyList_GET_SIZE(parts);
    int rc = 0;
    PyObject *cur = NULL;
    for (Py_ssize_t i = 0; i < np && rc == 0; i++) {
        PyObject *comp = PyList_GET_ITEM(parts, i);
        Py_ssize_t cl = PyUnicode_GET_LENGTH(comp);
        if (cl == 0 || PyUnicode_READ_CHAR(comp, 0) == '<' ||
            (cl >= 2 && PyUnicode_READ_CHAR(comp, 0) == '_' && PyUnicode_READ_CHAR(comp, 1) == '_')) {
            stop(W, A_INELIGIBLE, obj); rc = 1; break;
        }
        if (i == 0) {
            if (root_ref(R, modname) < 0 || root_ref(R, comp) < 0) { rc = -1; break; }
            if (root_push_node(W, (uintptr_t)modules, modname) < 0) { rc = -1; break; }
            if (root_push_node(W, (uintptr_t)md, comp) < 0) { rc = -1; break; }
            if (PyDict_Watch(dict_wid, modules) < 0 || PyDict_Watch(dict_wid, md) < 0) { rc = -1; break; }
            cur = PyDict_GetItemWithError(md, comp);
            if (!cur) { if (PyErr_Occurred()) rc = -1; else { stop(W, A_INELIGIBLE, obj); rc = 1; } break; }
        } else {
            if (!PyType_Check(cur) || Py_TYPE(cur) != &PyType_Type ||
                !(((PyTypeObject *)cur)->tp_flags & Py_TPFLAGS_HEAPTYPE)) {
                stop(W, A_INELIGIBLE, obj); rc = 1; break;
            }
            int t = type_record(W, (PyTypeObject *)cur);
            if (t) { rc = t; break; }
            cur = _PyType_Lookup((PyTypeObject *)cur, comp);   /* borrowed, runs no code */
            if (!cur) { stop(W, A_INELIGIBLE, obj); rc = 1; break; }
        }
    }
    Py_DECREF(parts);
    if (rc) return rc;
    if (cur != obj) { stop(W, A_INELIGIBLE, obj); return 1; }
    if (!W->saw_globals && ext_registry) {
        /* copyreg's extension codes: pickle asks them for every global it writes */
        if (root_push_node(W, (uintptr_t)ext_registry, NULL) < 0) return -1;
        if (PyDict_Watch(dict_wid, ext_registry) < 0) return -1;
    }
    W->saw_globals = 1;
    return 0;
}

/* a class pickle reads (an instance's type, or a class named on the way to a
   global): its version tag (moves on any change to it or a base), watched;
   its qualified name. Once per root. */
static int type_record(Walk *W, PyTypeObject *t) {
    Root *R = W->R;
    int a = aset_add(&typeseen, (uintptr_t)t);
    if (a < 0) return -1;
    if (a == 0) return 0;
    if (!(t->tp_flags & Py_TPFLAGS_HEAPTYPE) || Py_TYPE(t) != &PyType_Type) {
        stop(W, A_INELIGIBLE, (PyObject *)t); return 1;
    }
    if (t->tp_version_tag == 0 && !PyUnstable_Type_AssignVersionTag(t)) {
        stop(W, A_INELIGIBLE, (PyObject *)t); return 1;
    }
    if (PyType_Watch(type_wid, (PyObject *)t) < 0) return -1;
    PyObject *qn = ((PyHeapTypeObject *)t)->ht_qualname;
    if (root_ref(R, (PyObject *)t) < 0 || root_ref(R, qn) < 0) return -1;
    if (root_push_node(W, (uintptr_t)t, NULL) < 0) return -1;
    if (snap_reserve(R, 4) < 0) return -1;
    R->snap[R->nsnap++] = R_TYPE;
    R->snap[R->nsnap++] = (uintptr_t)t;
    R->snap[R->nsnap++] = (uintptr_t)t->tp_version_tag;
    R->snap[R->nsnap++] = (uintptr_t)qn;
    return 0;
}

/* a class as a value (pickled by name) */
static int class_value(Walk *W, PyTypeObject *t) {
    if (!(t->tp_flags & Py_TPFLAGS_HEAPTYPE) || Py_TYPE(t) != &PyType_Type) {
        stop(W, A_INELIGIBLE, (PyObject *)t); return 1;
    }
    int r = type_record(W, t);
    if (r) return r;
    PyObject *mod = PyDict_GetItemWithError(t->tp_dict, s_module);
    if (!mod && PyErr_Occurred()) return -1;
    return global_ref(W, (PyObject *)t, mod, ((PyHeapTypeObject *)t)->ht_qualname);
}

static int func_value(Walk *W, PyFunctionObject *f) {
    Root *R = W->R;
    PyObject *qn = f->func_qualname, *mn = f->func_module;
    if (!qn || !mn) { stop(W, A_INELIGIBLE, (PyObject *)f); return 1; }
    if (root_ref(R, qn) < 0 || root_ref(R, mn) < 0) return -1;
    if (root_push_node(W, (uintptr_t)f, NULL) < 0) return -1;
    if (snap_reserve(R, 4) < 0) return -1;
    R->snap[R->nsnap++] = R_FUNC;
    R->snap[R->nsnap++] = (uintptr_t)f;
    R->snap[R->nsnap++] = (uintptr_t)qn;
    R->snap[R->nsnap++] = (uintptr_t)mn;
    return global_ref(W, (PyObject *)f, mn, qn);
}

static inline int is_atom(PyObject *o, PyTypeObject *t) {
    return o == Py_None || t == &PyLong_Type || t == &PyUnicode_Type || t == &PyFloat_Type ||
           t == &PyBytes_Type || t == &PyBool_Type;
}

static size_t atom_size(PyObject *o, PyTypeObject *t) {
    if (t == &PyUnicode_Type) return 40 + (size_t)PyUnicode_GET_LENGTH(o) * PyUnicode_KIND(o);
    if (t == &PyBytes_Type) return 33 + (size_t)PyBytes_GET_SIZE(o);
    return 24;
}

static PyObject *set_dummy = NULL;

/* a view the keyed pickle leaves out of its owner's state (`views_of`):
   what decides that it is one -- its class and the pointers it holds in
   its slots and dict -- is recorded, not what it points at (its owner's
   own values, walked there). Python vetted its class (`views_of`). */
static int view_record(Walk *W, PyObject *v) {
    Root *R = W->R;
    PyTypeObject *vt = Py_TYPE(v);
    if (!(vt->tp_flags & Py_TPFLAGS_HEAPTYPE)) { stop(W, A_INELIGIBLE, v); return 1; }
    if (vt->tp_version_tag == 0 && !PyUnstable_Type_AssignVersionTag(vt)) { stop(W, A_INELIGIBLE, v); return 1; }
    int a = aset_add(&typeseen, (uintptr_t)vt);
    if (a < 0) return -1;
    if (a == 1) {
        if (PyType_Watch(type_wid, (PyObject *)vt) < 0) return -1;
        if (root_ref(R, (PyObject *)vt) < 0) return -1;
        if (root_push_node(W, (uintptr_t)vt, NULL) < 0) return -1;
        if (snap_reserve(R, 4) < 0) return -1;
        R->snap[R->nsnap++] = R_TYPE;
        R->snap[R->nsnap++] = (uintptr_t)vt;
        R->snap[R->nsnap++] = (uintptr_t)vt->tp_version_tag;
        R->snap[R->nsnap++] = (uintptr_t)((PyHeapTypeObject *)vt)->ht_qualname;
        if (root_ref(R, ((PyHeapTypeObject *)vt)->ht_qualname) < 0) return -1;
    }
    /* every object slot of every class in its MRO */
    PyObject *mro = vt->tp_mro;
    if (!mro || !PyTuple_Check(mro)) { stop(W, A_INELIGIBLE, v); return 1; }
    Py_ssize_t offs[64]; size_t n = 0;
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(mro); i++) {
        PyTypeObject *k = (PyTypeObject *)PyTuple_GET_ITEM(mro, i);
        if (k == &PyBaseObject_Type) continue;
        if (!(k->tp_flags & Py_TPFLAGS_HEAPTYPE)) { stop(W, A_INELIGIBLE, v); return 1; }
        PyMemberDef *m = k->tp_members;   /* a heap class: its __slots__ (and offsets of __dict__, __weakref__) */
        for (; m && m->name; m++) {
            if (m->type != Py_T_OBJECT_EX) continue;
            if (m->offset <= 0 || m->offset + (Py_ssize_t)sizeof(PyObject *) > vt->tp_basicsize) { stop(W, A_INELIGIBLE, v); return 1; }
            int dup = 0;
            for (size_t j = 0; j < n; j++) if (offs[j] == m->offset) dup = 1;
            if (dup) continue;
            if (n == 64) { stop(W, A_INELIGIBLE, v); return 1; }
            offs[n++] = m->offset;
        }
    }
    PyObject *d = NULL;
    if (vt->tp_dictoffset != 0 || (vt->tp_flags & Py_TPFLAGS_MANAGED_DICT)) {
        d = PyObject_GenericGetDict(v, NULL);
        if (!d) return -1;
        if (!PyDict_CheckExact(d)) { Py_DECREF(d); stop(W, A_INELIGIBLE, v); return 1; }
        /* any change to it: a node of its own, its values not walked */
        if (root_push_node(W, (uintptr_t)d, NULL) < 0 || PyDict_Watch(dict_wid, d) < 0 ||
            ovec_push(&keep, d) < 0) { Py_DECREF(d); return -1; }
        Py_DECREF(d);   /* kept alive by keep for the walk, by v after it */
    }
    if (snap_reserve(R, 4 + 2 * n) < 0) return -1;
    R->snap[R->nsnap++] = R_VIEW | ((uintptr_t)n << 8);
    R->snap[R->nsnap++] = (uintptr_t)v;
    R->snap[R->nsnap++] = (uintptr_t)vt;
    R->snap[R->nsnap++] = (uintptr_t)d;
    for (size_t j = 0; j < n; j++) {
        R->snap[R->nsnap++] = (uintptr_t)offs[j];
        R->snap[R->nsnap++] = (uintptr_t)*(PyObject **)((char *)v + offs[j]);
    }
    W->content += (size_t)vt->tp_basicsize;
    return 0;
}

/* an instance's dict when its key leaves the views in *views* out: watched,
   its other values walked, each view only recorded (`view_record`) */
static int dict_with_views(Walk *W, PyObject *d, PyObject *views) {
    int added = aset_add(&visited, (uintptr_t)d);
    if (added < 0) return -1;
    if (added == 0) return 0;       /* walked already, all of it as content */
    if (ovec_push(&keep, d) < 0) return -1;
    W->content += 64 + (size_t)PyDict_GET_SIZE(d) * 24;
    if (root_push_node(W, (uintptr_t)d, NULL) < 0) return -1;
    if (PyDict_Watch(dict_wid, d) < 0) return -1;
    Py_ssize_t pos = 0; PyObject *k, *v;
    while (PyDict_Next(d, &pos, &k, &v)) {
        PyTypeObject *kt = Py_TYPE(k), *vt = Py_TYPE(v);
        if (is_atom(k, kt)) W->content += atom_size(k, kt); else if (push(k) < 0) return -1;
        int is_view = PyUnicode_CheckExact(k) ? PySet_Contains(views, k) : 0;
        if (is_view < 0) return -1;
        if (is_view) {
            int r = view_record(W, v);
            if (r) return r;
            continue;
        }
        if (is_atom(v, vt)) W->content += atom_size(v, vt); else if (push(v) < 0) return -1;
    }
    return 0;
}

/* walk one container popped from the stack: 0 ok, 1 stop, -1 error */
static int visit(Walk *W, PyObject *o) {
    Root *R = W->R;
    PyTypeObject *t = Py_TYPE(o);
    if (t == &PyDict_Type) {
        W->content += 64 + (size_t)PyDict_GET_SIZE(o) * 24;
        if (root_push_node(W, (uintptr_t)o, NULL) < 0) return -1;
        if (PyDict_Watch(dict_wid, o) < 0) return -1;
        Py_ssize_t pos = 0; PyObject *k, *v;
        while (PyDict_Next(o, &pos, &k, &v)) {
            PyTypeObject *kt = Py_TYPE(k), *vt = Py_TYPE(v);
            if (is_atom(k, kt)) W->content += atom_size(k, kt); else if (push(k) < 0) return -1;
            if (is_atom(v, vt)) W->content += atom_size(v, vt); else if (push(v) < 0) return -1;
        }
        return 0;
    }
    if (t == &PyList_Type) {
        Py_ssize_t n = PyList_GET_SIZE(o);
        W->content += 56 + (size_t)n * 8;
        if (snap_reserve(R, 2 + (size_t)n) < 0) return -1;
        R->snap[R->nsnap++] = R_LIST | ((uintptr_t)n << 8);
        R->snap[R->nsnap++] = (uintptr_t)o;
        PyObject **items = ((PyListObject *)o)->ob_item;
        if (n) memcpy(&R->snap[R->nsnap], items, (size_t)n * sizeof(PyObject *));
        R->nsnap += (size_t)n;
        for (Py_ssize_t i = 0; i < n; i++) {
            PyObject *x = ((PyListObject *)o)->ob_item[i];
            PyTypeObject *xt = Py_TYPE(x);
            if (is_atom(x, xt)) W->content += atom_size(x, xt); else if (push(x) < 0) return -1;
        }
        return 0;
    }
    if (t == &PyTuple_Type || t == &PyFrozenSet_Type) {
        if (t == &PyTuple_Type) {
            Py_ssize_t n = PyTuple_GET_SIZE(o);
            W->content += 40 + (size_t)n * 8;
            for (Py_ssize_t i = 0; i < n; i++) {
                PyObject *x = PyTuple_GET_ITEM(o, i);
                PyTypeObject *xt = Py_TYPE(x);
                if (is_atom(x, xt)) W->content += atom_size(x, xt); else if (push(x) < 0) return -1;
            }
            return 0;
        }
        /* frozenset: fall through to the table walk without a record */
    }
    if (t == &PySet_Type || t == &PyFrozenSet_Type) {
        PySetObject *so = (PySetObject *)o;
        Py_ssize_t n = so->used;
        W->content += 200 + (size_t)n * 16;
        int record = t == &PySet_Type;
        if (record) {
            if (snap_reserve(R, 2 + (size_t)n) < 0) return -1;
            R->snap[R->nsnap++] = R_SET | ((uintptr_t)n << 8);
            R->snap[R->nsnap++] = (uintptr_t)o;
        }
        Py_ssize_t seen = 0;
        for (Py_ssize_t i = 0; i <= so->mask && seen < n; i++) {
            PyObject *x = so->table[i].key;
            if (!x || x == set_dummy) continue;
            seen++;
            if (record) R->snap[R->nsnap++] = (uintptr_t)x;
            PyTypeObject *xt = Py_TYPE(x);
            if (is_atom(x, xt)) W->content += atom_size(x, xt); else if (push(x) < 0) return -1;
        }
        if (seen != n) { stop(W, A_INELIGIBLE, o); return 1; }
        return 0;
    }
    if (t == &PyFunction_Type) return func_value(W, (PyFunctionObject *)o);
    if (PyType_Check(o)) return class_value(W, (PyTypeObject *)o);

    /* an instance of a class cash's Python side vetted */
    TypeInfo *ti = verdict_of(t);
    if (!ti) { stop(W, A_NEED_TYPE, (PyObject *)t); return 1; }
    if (!ti->ok) { stop(W, A_INELIGIBLE, (PyObject *)t); return 1; }
    W->saw_instances = 1;
    int a = aset_add(&insttypes, (uintptr_t)t);
    if (a < 0) return -1;
    if (a == 1) {
        /* first instance of t in this value: the class (its version tag and
           how pickle finds it by name) and copyreg's reducer table at t */
        if (t->tp_version_tag != ti->tag) { stop(W, A_NEED_TYPE, (PyObject *)t); return 1; }
        int tr = type_record(W, t);
        if (tr) return tr;
        PyObject *mod = PyDict_GetItemWithError(t->tp_dict, s_module);
        if (!mod && PyErr_Occurred()) return -1;
        int g = global_ref(W, (PyObject *)t, mod, ((PyHeapTypeObject *)t)->ht_qualname);
        if (g) return g;
        if (dispatch_table) {
            if (root_push_node(W, (uintptr_t)dispatch_table, (PyObject *)t) < 0) return -1;
            if (PyDict_Watch(dict_wid, dispatch_table) < 0) return -1;
            PyObject *reg = PyDict_GetItemWithError(dispatch_table, (PyObject *)t);
            if (reg) { stop(W, A_INELIGIBLE, (PyObject *)t); return 1; }
            if (PyErr_Occurred()) return -1;
        }
    }
    PyObject *d = NULL;
    if (t->tp_dictoffset != 0 || (t->tp_flags & Py_TPFLAGS_MANAGED_DICT)) {
        d = PyObject_GenericGetDict(o, NULL);    /* materialises it, once */
        if (!d) return -1;
        if (!PyDict_CheckExact(d)) { Py_DECREF(d); stop(W, A_INELIGIBLE, o); return 1; }
        /* an instance attribute pickle would call in place of the class's */
        PyObject *names[3] = {s_getstate, s_reduce, s_reduce_ex};
        for (int i = 0; i < 3; i++) {
            PyObject *x = PyDict_GetItemWithError(d, names[i]);
            if (x) { Py_DECREF(d); stop(W, A_INELIGIBLE, o); return 1; }
            if (PyErr_Occurred()) { Py_DECREF(d); return -1; }
        }
    }
    Py_ssize_t ns = ti->nslots;
    W->content += (size_t)t->tp_basicsize;
    if (snap_reserve(R, 4 + (size_t)ns) < 0) { Py_XDECREF(d); return -1; }
    R->snap[R->nsnap++] = R_INST | ((uintptr_t)ns << 8);
    R->snap[R->nsnap++] = (uintptr_t)o;
    R->snap[R->nsnap++] = (uintptr_t)t;
    R->snap[R->nsnap++] = (uintptr_t)d;
    for (Py_ssize_t i = 0; i < ns; i++) {
        PyObject *x = *(PyObject **)((char *)o + ti->offsets[i]);
        R->snap[R->nsnap++] = (uintptr_t)x;
        if (x) {
            PyTypeObject *xt = Py_TYPE(x);
            if (is_atom(x, xt)) W->content += atom_size(x, xt);
            else if (push(x) < 0) { Py_XDECREF(d); return -1; }
        }
    }
    if (d && ti->views && views_of) {
        /* runs Python: `ti` may move, nothing below reads it */
        PyObject *views = PyObject_CallOneArg(views_of, o);
        if (!views) { Py_DECREF(d); return -1; }
        if (!PyAnySet_CheckExact(views)) { Py_DECREF(views); Py_DECREF(d); stop(W, A_INELIGIBLE, o); return 1; }
        if (PySet_GET_SIZE(views) > 0) {
            int r = dict_with_views(W, d, views);
            Py_DECREF(views);
            Py_DECREF(d);
            return r;
        }
        Py_DECREF(views);
    }
    if (d) {
        int r = push(d);
        Py_DECREF(d);
        if (r < 0) return -1;
    }
    return 0;
}

/* ------------------------------------------------------------ verify */
static PyObject *weak_target(PyObject *wref) {   /* borrowed or NULL */
#if PY_VERSION_HEX >= 0x030D0000
    PyObject *o = NULL;
    if (PyWeakref_GetRef(wref, &o) < 0) { PyErr_Clear(); return NULL; }
    if (!o) return NULL;
    Py_DECREF(o);   /* the caller holds the value itself */
    return o;
#else
    PyObject *o = PyWeakref_GET_OBJECT(wref);
    return o == Py_None ? NULL : o;
#endif
}

/* does every record still read as it did? Preorder: a container is compared
   before anything it holds is read. Returns 1 same, 0 changed, -1 error. */
static int verify(Root *R) {
    uintptr_t *s = R->snap, *end = R->snap + R->nsnap;
    while (s < end) {
        uintptr_t h = s[0];
        int kind = (int)(h & 0xFF);
        size_t n = (size_t)(h >> 8);
        PyObject *o = (PyObject *)s[1];
        switch (kind) {
        case R_LIST:
            if (Py_TYPE(o) != &PyList_Type || (size_t)PyList_GET_SIZE(o) != n) return 0;
            if (n && memcmp(((PyListObject *)o)->ob_item, &s[2], n * sizeof(PyObject *)) != 0) return 0;
            s += 2 + n;
            break;
        case R_SET: {
            if (Py_TYPE(o) != &PySet_Type) return 0;
            PySetObject *so = (PySetObject *)o;
            if ((size_t)so->used != n) return 0;
            size_t j = 0;
            for (Py_ssize_t i = 0; i <= so->mask && j < n; i++) {
                PyObject *x = so->table[i].key;
                if (!x || x == set_dummy) continue;
                if ((uintptr_t)x != s[2 + j]) return 0;
                j++;
            }
            if (j != n) return 0;
            s += 2 + n;
            break;
        }
        case R_INST: {
            PyTypeObject *t = (PyTypeObject *)s[2];
            if (Py_TYPE(o) != t) return 0;
            PyObject *d = NULL;
            if (s[3]) {
                d = PyObject_GenericGetDict(o, NULL);
                if (!d) { PyErr_Clear(); return 0; }
                int same = (uintptr_t)d == s[3];
                Py_DECREF(d);
                if (!same) return 0;
            } else if (t->tp_dictoffset != 0 || (t->tp_flags & Py_TPFLAGS_MANAGED_DICT)) {
                return 0;
            }
            TypeInfo *ti = verdict_of(t);
            if (!ti || !ti->ok || (size_t)ti->nslots != n) return 0;
            for (size_t i = 0; i < n; i++)
                if (*(uintptr_t *)((char *)o + ti->offsets[i]) != s[4 + i]) return 0;
            s += 4 + n;
            break;
        }
        case R_VIEW: {
            PyTypeObject *t = (PyTypeObject *)s[2];
            if (Py_TYPE(o) != t) return 0;
            if (s[3]) {
                PyObject *d = PyObject_GenericGetDict(o, NULL);
                if (!d) { PyErr_Clear(); return 0; }
                int same = (uintptr_t)d == s[3];
                Py_DECREF(d);
                if (!same) return 0;
            } else if (t->tp_dictoffset != 0 || (t->tp_flags & Py_TPFLAGS_MANAGED_DICT)) {
                return 0;
            }
            for (size_t i = 0; i < n; i++)
                if (*(uintptr_t *)((char *)o + (Py_ssize_t)s[4 + 2 * i]) != s[5 + 2 * i]) return 0;
            s += 4 + 2 * n;
            break;
        }
        case R_FUNC: {
            PyFunctionObject *f = (PyFunctionObject *)o;
            if ((uintptr_t)f->func_qualname != s[2] || (uintptr_t)f->func_module != s[3]) return 0;
            s += 4;
            break;
        }
        case R_TYPE: {
            PyTypeObject *t = (PyTypeObject *)o;
            if ((uintptr_t)t->tp_version_tag != s[2] || t->tp_version_tag == 0) return 0;
            if ((uintptr_t)((PyHeapTypeObject *)t)->ht_qualname != s[3]) return 0;
            s += 4;
            break;
        }
        default:
            return 0;
        }
    }
    return 1;
}

/* is the root's value still the object it was armed for? 1 yes, 0 no */
static int same_object(Root *R, PyObject *o) {
    if (R->gone) return 0;
    if (R->wref) return weak_target(R->wref) == o;
    return 1;   /* a dict root (dealloc drops it) or an anchored one (rebinding drops it) */
}

/* ------------------------------------------------------------ sightings */
typedef struct { uintptr_t addr; uint64_t until; float secs; uint8_t state; } Sight;
#define SIGHT_BITS 13
static Sight sights[1 << SIGHT_BITS];
enum { SG_NONE = 0, SG_SEEN = 1, SG_SKIP = 2 };

static Sight *sight_of(uintptr_t a) { return &sights[hmix(a) & ((1u << SIGHT_BITS) - 1)]; }

static void skip_for(uintptr_t a, uint64_t ticks) {
    Sight *g = sight_of(a);
    g->addr = a; g->state = SG_SKIP; g->until = tick + ticks;
}

/* ------------------------------------------------------------ housekeeping */
static void sweep(void) {
    for (uint32_t r = 0; r < nroots; r++) {
        Root *R = &roots[r];
        if (!R->used) continue;
        int drop = R->gone;
        if (!drop && R->wref && weak_target(R->wref) == NULL) drop = 1;
        if (drop) { root_free(r); c_swept++; }
    }
    any_gone = 0;
}

static double score(Root *R) {
    double age = (double)(tick - R->last_used);
    double bytes = (double)(R->bytes ? R->bytes : 1);
    return (R->hash_s * (1.0 + (double)R->reuses)) / bytes / (1.0 + age / 10000.0);
}

static size_t total_bytes(void) { return table_bytes_total + index_bytes(); }

/* make room for `need` bytes, evicting roots worth less than `worth`;
   1 when it fits */
static int make_room(size_t need, double worth, uint32_t except) {
    while (total_bytes() + need > budget) {
        uint32_t worst = NONE32; double ws = 0;
        for (uint32_t r = 0; r < nroots; r++) {
            if (!roots[r].used || r == except) continue;
            double sc = score(&roots[r]);
            if (worst == NONE32 || sc < ws) { worst = r; ws = sc; }
        }
        if (worst == NONE32 || ws > worth) return 0;
        root_free(worst); c_evicted++;
    }
    return 1;
}

static void housekeeping(void) {
    if (any_gone || (tick & 1023) == 0) sweep();
}

/* ------------------------------------------------------------ Python API */
static int kind_arg(PyObject *k, int *out) {
    long v = PyLong_AsLong(k);
    if (v == -1 && PyErr_Occurred()) return -1;
    if (v < 0 || v >= NKINDS) { PyErr_SetString(PyExc_ValueError, "kind out of range"); return -1; }
    *out = (int)v;
    return 0;
}

/* check(value, kind) -> the digest (any object) when value is provably
   unchanged since that digest was committed; else an int: FIRST (first
   sight: hash, then call seen()), ARM (arm it, hash, commit), SKIP (just
   hash), BUSY, OFF. */
static PyObject *py_check(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 2) { PyErr_SetString(PyExc_TypeError, "check(value, kind)"); return NULL; }
    PyObject *o = args[0];
    int kind; if (kind_arg(args[1], &kind) < 0) return NULL;
    if (!enabled) return PyLong_FromLong(ST_OFF);
    if (busy) { c_busy++; return PyLong_FromLong(ST_BUSY); }
    busy = 1;
    tick++; c_checks++;
    PyObject *res = NULL;
    housekeeping();
    uintptr_t a = (uintptr_t)o;
    long r = rht_find(a);
    if (r >= 0) {
        Root *R = &roots[r];
        R->last_used = tick;
        if (!same_object(R, o)) { root_free((uint32_t)r); r = -1; }
        else if (!R->dirty) {
            int v = verify(R);
            if (v < 0) { busy = 0; return NULL; }
            if (v == 0) { mark_dirty((uint32_t)r); c_verify_fail++; }
        }
        if (r >= 0) {
            R = &roots[r];
            if (!R->dirty) {
                if (R->dg[kind] && R->dgver[kind] == R->version) {
                    c_hits++; R->reuses++; R->streak = 0;
                    res = Py_NewRef(R->dg[kind]);
                } else {
                    res = PyLong_FromLong(ST_ARM);   /* armed, but not for this kind yet */
                }
            } else if (R->streak >= 3) {
                /* changed every time it was looked at: tracking costs more than it saves */
                uint32_t st = R->streak;
                root_free((uint32_t)r); c_dropped_streak++;
                skip_for(a, (uint64_t)64 << (st < 14 ? st : 14));
                res = PyLong_FromLong(ST_SKIP);
            } else {
                res = PyLong_FromLong(ST_ARM);
            }
            busy = 0;
            return res;
        }
    }
    /* not tracked */
    TypeInfo *ti = NULL;
    PyTypeObject *t = Py_TYPE(o);
    if (t != &PyDict_Type && t != &PyList_Type && t != &PySet_Type && t != &PyTuple_Type &&
        t != &PyFrozenSet_Type) {
        ti = verdict_of(t);
        if (ti && !ti->ok) { busy = 0; return PyLong_FromLong(ST_SKIP); }
    }
    Sight *g = sight_of(a);
    if (g->addr == a && g->state == SG_SKIP) {
        busy = 0;
        if (tick < g->until) return PyLong_FromLong(ST_SKIP);
        g->state = SG_NONE;
        return PyLong_FromLong(ST_ARM);
    }
    if (g->addr == a && g->state == SG_SEEN) { busy = 0; return PyLong_FromLong(ST_ARM); }
    c_first++;
    busy = 0;
    return PyLong_FromLong(ST_FIRST);
}

/* seen(value, seconds): first sight hashed in `seconds` */
static PyObject *py_seen(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 2) { PyErr_SetString(PyExc_TypeError, "seen(value, seconds)"); return NULL; }
    double secs = PyFloat_AsDouble(args[1]);
    if (secs == -1.0 && PyErr_Occurred()) return NULL;
    uintptr_t a = (uintptr_t)args[0];
    Sight *g = sight_of(a);
    if (g->state == SG_SKIP && g->addr != a && tick < g->until) Py_RETURN_NONE;  /* keep a live skip */
    g->addr = a; g->state = SG_SEEN; g->secs = (float)secs; g->until = 0;
    Py_RETURN_NONE;
}

static int find_anchor(PyObject *o, PyObject *anchors, PyObject **ns_out, PyObject **key_out) {
    if (!anchors || anchors == Py_None) return 0;
    if (!PyTuple_Check(anchors)) { PyErr_SetString(PyExc_TypeError, "anchors must be a tuple"); return -1; }
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(anchors); i++) {
        PyObject *ns = PyTuple_GET_ITEM(anchors, i);
        if (!PyDict_Check(ns)) continue;
        Py_ssize_t pos = 0; PyObject *k, *v;
        while (PyDict_Next(ns, &pos, &k, &v)) {
            if (v == o && PyUnicode_CheckExact(k)) { *ns_out = ns; *key_out = k; return 1; }
        }
    }
    return 0;
}

static PyObject *arm_result(int st, PyObject *detail) {
    PyObject *res = Py_BuildValue("(iO)", st, detail ? detail : Py_None);
    return res;
}

/* arm(value, anchors) -> (status, detail). anchors: a tuple of namespace
   dicts one of which may hold value (needed for a value with no other proof
   of identity: a list, a set, a tuple). */
static PyObject *py_arm(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs < 1 || nargs > 2) { PyErr_SetString(PyExc_TypeError, "arm(value, anchors=None)"); return NULL; }
    PyObject *o = args[0];
    PyObject *anchors = nargs > 1 ? args[1] : NULL;
    if (!enabled) return arm_result(A_INELIGIBLE, NULL);
    if (busy) { c_busy++; return arm_result(A_BUSY, NULL); }
    if (ensure_watchers() < 0) return NULL;
    busy = 1;
    uintptr_t a = (uintptr_t)o;
    PyTypeObject *t = Py_TYPE(o);
    if (is_atom(o, t)) { busy = 0; return arm_result(A_INELIGIBLE, NULL); }

    long r = rht_find(a);
    if (r >= 0) {
        Root *R = &roots[r];
        if (!same_object(R, o)) { root_free((uint32_t)r); r = -1; }
        else if (!R->dirty) {
            int v = verify(R);
            if (v < 0) { busy = 0; return NULL; }
            if (v == 1) { busy = 0; return arm_result(A_ALREADY, NULL); }
            mark_dirty((uint32_t)r);
        }
    }
    /* identity: what proves later that the object at this address is this one */
    PyObject *wref = NULL, *ans = NULL, *akey = NULL;
    if (t != &PyDict_Type) {
        wref = PyWeakref_NewRef(o, NULL);
        if (!wref) {
            if (!PyErr_ExceptionMatches(PyExc_TypeError)) { busy = 0; return NULL; }
            PyErr_Clear();
            int f = find_anchor(o, anchors, &ans, &akey);
            if (f < 0) { busy = 0; return NULL; }
            if (!f) {
                if (r >= 0) root_free((uint32_t)r);
                c_no_anchor++; skip_for(a, 256);
                busy = 0; return arm_result(A_NO_ANCHOR, NULL);
            }
        }
    }
    Root *R;
    if (r >= 0) {
        R = &roots[r];
        table_bytes_total -= R->bytes;
        uint32_t streak = R->streak; double hs = R->hash_s; uint64_t reuses = R->reuses; uint32_t ver = R->version;
        root_clear(R);
        R->streak = streak + 1; R->hash_s = hs; R->reuses = reuses; R->version = ver;
        R->bytes = root_bytes(R); table_bytes_total += R->bytes;
    } else {
        r = root_new(a);
        if (r < 0) { Py_XDECREF(wref); busy = 0; return NULL; }
        R = &roots[r];
        Sight *g = sight_of(a);
        if (g->addr == a && g->state == SG_SEEN) R->hash_s = g->secs;
        if (g->addr == a) g->state = SG_NONE;
    }
    R->version++;
    R->dirty = 0; R->gone = 0;
    R->last_used = ++tick;
    R->wref = wref;

    Walk W = {R, (uint32_t)r, 0, A_ARMED, NULL, 0, 0};
    int rc = 0;
    if (ans) {
        /* the namespace slot holding it: rebinding or deleting drops the root */
        rc = root_ref(R, akey);
        if (!rc) rc = root_push_node(&W, (uintptr_t)ans, akey);
        if (!rc) rc = PyDict_Watch(dict_wid, ans);
        R = &roots[r];
        R->anchored = 1;
    }
    aset_clear(&visited); aset_clear(&typeseen); aset_clear(&insttypes);
    stack_.n = 0;
    if (!rc) rc = push(o);
    while (!rc && stack_.n) {
        PyObject *x = stack_.v[--stack_.n];   /* owned: moved to keep */
        int added = aset_add(&visited, (uintptr_t)x);
        if (added <= 0) { Py_DECREF(x); if (added < 0) rc = -1; continue; }
        if (ovec_push(&keep, x) < 0) { Py_DECREF(x); rc = -1; break; }
        Py_DECREF(x);
        rc = visit(&W, x);
        R = &roots[r];
    }
    /* drop what the walk held */
    while (stack_.n) { Py_DECREF(stack_.v[--stack_.n]); }
    size_t nkeep = keep.n; keep.n = 0;
    PyObject **kv = keep.v;
    keep.v = NULL; keep.cap = 0;
    for (size_t i = 0; i < nkeep; i++) Py_DECREF(kv[i]);
    free(kv);
    R = &roots[r];

    PyObject *detail = W.detail;
    int st = W.status;
    if (rc < 0) {
        root_free((uint32_t)r);
        Py_XDECREF(detail);
        busy = 0;
        return NULL;
    }
    if (rc == 1 || st != A_ARMED) {
        root_free((uint32_t)r);
        if (st == A_INELIGIBLE) { c_ineligible++; skip_for(a, 4096); }
        PyObject *res = arm_result(st, detail);
        Py_XDECREF(detail);
        busy = 0;
        return res;
    }
    R->content = W.content;
    if (W.content < min_bytes) {
        root_free((uint32_t)r);
        c_too_small++; skip_for(a, 4096);
        busy = 0;
        return arm_result(A_TOO_SMALL, NULL);
    }
    table_bytes_total -= R->bytes;
    R->bytes = root_bytes(R);
    table_bytes_total += R->bytes;
    if (total_bytes() > budget) {
        double worth = R->hash_s > 0 ? R->hash_s / (double)R->bytes : 0.0;
        if (!make_room(0, worth, (uint32_t)r)) {
            root_free((uint32_t)r);
            c_over_budget++; skip_for(a, 4096);
            busy = 0;
            return arm_result(A_OVER_BUDGET, NULL);
        }
    }
    c_arms++;
    busy = 0;
    return arm_result(A_ARMED, NULL);
}

/* commit(value, kind, digest, seconds) -> bool: keep digest for value when
   nothing changed since arm() */
static PyObject *py_commit(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 4) { PyErr_SetString(PyExc_TypeError, "commit(value, kind, digest, seconds)"); return NULL; }
    PyObject *o = args[0];
    int kind; if (kind_arg(args[1], &kind) < 0) return NULL;
    double secs = PyFloat_AsDouble(args[3]);
    if (secs == -1.0 && PyErr_Occurred()) return NULL;
    if (busy) { c_busy++; Py_RETURN_FALSE; }
    busy = 1;
    long r = rht_find((uintptr_t)o);
    if (r < 0) { busy = 0; Py_RETURN_FALSE; }
    Root *R = &roots[r];
    if (R->dirty || !same_object(R, o)) { busy = 0; Py_RETURN_FALSE; }
    int v = verify(R);
    if (v < 0) { busy = 0; return NULL; }
    R = &roots[r];
    if (v == 0) { mark_dirty((uint32_t)r); busy = 0; Py_RETURN_FALSE; }
    Py_XSETREF(R->dg[kind], Py_NewRef(args[2]));
    R->dgver[kind] = R->version;
    if (secs > 0) R->hash_s = R->hash_s > 0 ? (R->hash_s + secs) / 2 : secs;
    c_commits++;
    busy = 0;
    Py_RETURN_TRUE;
}

/* peek(value, kind) -> the digest check() would hand back, or None; no
   sighting is counted and nothing is armed */
static PyObject *py_peek(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 2) { PyErr_SetString(PyExc_TypeError, "peek(value, kind)"); return NULL; }
    PyObject *o = args[0];
    int kind; if (kind_arg(args[1], &kind) < 0) return NULL;
    if (!enabled || busy) Py_RETURN_NONE;
    long r = rht_find((uintptr_t)o);
    if (r < 0) Py_RETURN_NONE;
    busy = 1;
    Root *R = &roots[r];
    PyObject *res = Py_None;
    if (!R->dirty && same_object(R, o) && R->dg[kind] && R->dgver[kind] == R->version) {
        int v = verify(R);
        if (v < 0) { busy = 0; return NULL; }
        R = &roots[r];
        if (v == 1) res = R->dg[kind];
        else { mark_dirty((uint32_t)r); c_verify_fail++; }
    }
    busy = 0;
    return Py_NewRef(res);
}

static PyObject *py_forget(PyObject *self, PyObject *o) {
    if (busy) Py_RETURN_FALSE;
    busy = 1;
    long r = rht_find((uintptr_t)o);
    if (r >= 0) root_free((uint32_t)r);
    busy = 0;
    return PyBool_FromLong(r >= 0);
}

static PyObject *py_version(PyObject *self, PyObject *o) {
    long r = rht_find((uintptr_t)o);
    if (r < 0) Py_RETURN_NONE;
    Root *R = &roots[r];
    if (R->dirty || !same_object(R, o)) Py_RETURN_NONE;
    return PyLong_FromUnsignedLong(R->version);
}

/* set_type_verdict(type, ok, slot_descriptors): cash's Python side says
   whether instances of type pickle from their __dict__ and slots alone */
static PyObject *py_set_type_verdict(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 3 && nargs != 4) { PyErr_SetString(PyExc_TypeError, "set_type_verdict(type, ok, slots, views=False)"); return NULL; }
    if (busy) { PyErr_SetString(PyExc_RuntimeError, "cash._cwatch is busy"); return NULL; }
    int views = nargs == 4 ? PyObject_IsTrue(args[3]) : 0;
    if (views < 0) return NULL;
    if (!PyType_Check(args[0])) { PyErr_SetString(PyExc_TypeError, "not a type"); return NULL; }
    PyTypeObject *t = (PyTypeObject *)args[0];
    int ok = PyObject_IsTrue(args[1]);
    if (ok < 0) return NULL;
    PyObject *slots = args[2];
    Py_ssize_t ns = 0;
    Py_ssize_t *offs = NULL;
    if (ok) {
        if (!(t->tp_flags & Py_TPFLAGS_HEAPTYPE) || Py_TYPE(t) != &PyType_Type || t->tp_itemsize != 0) ok = 0;
    }
    if (ok) {
        if (!PyTuple_Check(slots)) { PyErr_SetString(PyExc_TypeError, "slots must be a tuple"); return NULL; }
        ns = PyTuple_GET_SIZE(slots);
        offs = ns ? malloc((size_t)ns * sizeof *offs) : NULL;
        if (ns && !offs) return PyErr_NoMemory();
        for (Py_ssize_t i = 0; i < ns; i++) {
            PyObject *dsc = PyTuple_GET_ITEM(slots, i);
            if (Py_TYPE(dsc) != &PyMemberDescr_Type) { free(offs); PyErr_SetString(PyExc_TypeError, "not a member descriptor"); return NULL; }
            PyMemberDef *m = ((PyMemberDescrObject *)dsc)->d_member;
            if (m->type != Py_T_OBJECT_EX || m->offset <= 0 ||
                m->offset + (Py_ssize_t)sizeof(PyObject *) > t->tp_basicsize ||
                !PyType_IsSubtype(t, ((PyMemberDescrObject *)dsc)->d_common.d_type)) {
                free(offs); ok = 0; ns = 0; offs = NULL; break;
            }
            offs[i] = m->offset;
        }
    }
    if (ok && t->tp_version_tag == 0 && !PyUnstable_Type_AssignVersionTag(t)) { free(offs); offs = NULL; ns = 0; ok = 0; }
    if (ok) {
        if (ensure_watchers() < 0) { free(offs); return NULL; }
        if (PyType_Watch(type_wid, (PyObject *)t) < 0) { free(offs); return NULL; }
    }
    TypeInfo *e = tinfo_put(t);
    if (!e) { free(offs); return NULL; }
    e->tag = t->tp_version_tag; e->ok = ok; e->nslots = ns; e->offsets = offs; e->views = ok && views;
    return PyBool_FromLong(ok);
}

static PyObject *py_type_verdict(PyObject *self, PyObject *t) {
    if (!PyType_Check(t)) Py_RETURN_NONE;
    TypeInfo *e = verdict_of((PyTypeObject *)t);
    if (!e) Py_RETURN_NONE;
    return PyBool_FromLong(e->ok);
}

static PyObject *py_configure(PyObject *self, PyObject *args, PyObject *kw) {
    static char *names[] = {"budget", "min_bytes", "enabled", "dispatch_table", "extension_registry",
                            "cached_property", "views_of", NULL};
    Py_ssize_t b = -1, mb = -1; int en = -1;
    PyObject *dt = NULL, *er = NULL, *cp = NULL, *vo = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "|nnpOOOO", names, &b, &mb, &en, &dt, &er, &cp, &vo)) return NULL;
    if (busy) { PyErr_SetString(PyExc_RuntimeError, "cash._cwatch is busy"); return NULL; }
    if (b >= 0) budget = (size_t)b;
    if (mb >= 0) min_bytes = (size_t)mb;
    if (en >= 0) enabled = en;
    if (dt && dt != Py_None) { if (!PyDict_Check(dt)) { PyErr_SetString(PyExc_TypeError, "dispatch_table"); return NULL; } Py_XSETREF(dispatch_table, Py_NewRef(dt)); }
    if (er && er != Py_None) { if (!PyDict_Check(er)) { PyErr_SetString(PyExc_TypeError, "extension_registry"); return NULL; } Py_XSETREF(ext_registry, Py_NewRef(er)); }
    if (cp && cp != Py_None) Py_XSETREF(cached_property_type, Py_NewRef(cp));
    if (vo && vo != Py_None) {
        if (!PyCallable_Check(vo)) { PyErr_SetString(PyExc_TypeError, "views_of"); return NULL; }
        Py_XSETREF(views_of, Py_NewRef(vo));
    }
    Py_RETURN_NONE;
}

static PyObject *py_reset(PyObject *self, PyObject *noargs) {
    if (busy) { PyErr_SetString(PyExc_RuntimeError, "cash._cwatch is busy"); return NULL; }
    busy = 1;
    for (uint32_t r = 0; r < nroots; r++) if (roots[r].used) root_free(r);
    free(roots); roots = NULL; nroots = caproots = 0; rfree = NONE32; rlive = 0;
    free(rht); rht = NULL; rht_cap = rht_used = rht_tomb = 0;
    free(wn); wn = NULL; nwn = capwn = 0; wfree = NONE32; wlive = 0;
    free(kn); kn = NULL; nkn = capkn = 0; kfree = NONE32; klive = 0;
    free(ht); ht = NULL; ht_cap = ht_used = ht_tomb = 0;
    for (size_t i = 0; i < tinfo_cap; i++) free(tinfo[i].offsets);
    free(tinfo); tinfo = NULL; tinfo_cap = tinfo_used = 0;
    memset(sights, 0, sizeof sights);
    table_bytes_total = 0; any_gone = 0;
    c_checks = c_hits = c_first = c_arms = c_commits = c_events = c_dirty_marks = c_evicted =
        c_dropped_streak = c_too_small = c_ineligible = c_no_anchor = c_over_budget = c_busy =
            c_verify_fail = c_type_events = c_func_events = c_swept = 0;
    busy = 0;
    Py_RETURN_NONE;
}

static PyObject *py_stats(PyObject *self, PyObject *noargs) {
    size_t snapw = 0, nodes = 0;
    for (uint32_t r = 0; r < nroots; r++) if (roots[r].used) { snapw += roots[r].nsnap; nodes += roots[r].nnodes; }
    return Py_BuildValue(
        "{sIsnsnsnsnsnsnsnsKsKsKsKsKsKsKsKsKsKsKsKsKsKsKsKsKsK}",
        "roots", rlive, "nodes", (Py_ssize_t)wlive, "keyed_nodes", (Py_ssize_t)klive,
        "snapshot_words", (Py_ssize_t)snapw, "table_bytes", (Py_ssize_t)total_bytes(),
        "budget", (Py_ssize_t)budget, "min_bytes", (Py_ssize_t)min_bytes,
        "type_verdicts", (Py_ssize_t)tinfo_used,
        "checks", c_checks, "hits", c_hits, "first_sights", c_first, "arms", c_arms, "commits", c_commits,
        "dict_events", c_events, "dirty_marks", c_dirty_marks, "evicted", c_evicted,
        "dropped_changing", c_dropped_streak, "too_small", c_too_small, "ineligible", c_ineligible,
        "no_anchor", c_no_anchor, "over_budget", c_over_budget, "busy", c_busy,
        "verify_changed", c_verify_fail, "type_events", c_type_events, "func_events", c_func_events,
        "swept", c_swept);
}

/* info(value): a root's state, for tests */
static PyObject *py_info(PyObject *self, PyObject *o) {
    long r = rht_find((uintptr_t)o);
    if (r < 0) Py_RETURN_NONE;
    Root *R = &roots[r];
    PyObject *kinds = PyList_New(0);
    if (!kinds) return NULL;
    for (int k = 0; k < NKINDS; k++) {
        if (R->dg[k] && R->dgver[k] == R->version) {
            PyObject *x = PyLong_FromLong(k);
            if (!x || PyList_Append(kinds, x) < 0) { Py_XDECREF(x); Py_DECREF(kinds); return NULL; }
            Py_DECREF(x);
        }
    }
    return Py_BuildValue("{sIsisisisisIsnsnsnsNsd}", "version", R->version, "dirty", (int)R->dirty,
                         "gone", (int)R->gone, "anchored", (int)R->anchored, "weak", R->wref != NULL,
                         "nodes", R->nnodes, "snapshot_words", (Py_ssize_t)R->nsnap,
                         "table_bytes", (Py_ssize_t)R->bytes, "content_bytes", (Py_ssize_t)R->content,
                         "kinds", kinds, "hash_seconds", R->hash_s);
}

static PyMethodDef methods[] = {
    {"check", (PyCFunction)(void (*)(void))py_check, METH_FASTCALL, "check(value, kind) -> digest | status int"},
    {"seen", (PyCFunction)(void (*)(void))py_seen, METH_FASTCALL, "seen(value, seconds)"},
    {"arm", (PyCFunction)(void (*)(void))py_arm, METH_FASTCALL, "arm(value, anchors=None) -> (status, detail)"},
    {"commit", (PyCFunction)(void (*)(void))py_commit, METH_FASTCALL, "commit(value, kind, digest, seconds) -> bool"},
    {"peek", (PyCFunction)(void (*)(void))py_peek, METH_FASTCALL, "peek(value, kind) -> digest | None"},
    {"forget", py_forget, METH_O, "stop tracking value"},
    {"version", py_version, METH_O, "the version of a clean tracked value, or None"},
    {"set_type_verdict", (PyCFunction)(void (*)(void))py_set_type_verdict, METH_FASTCALL, ""},
    {"type_verdict", py_type_verdict, METH_O, ""},
    {"configure", (PyCFunction)(void (*)(void))py_configure, METH_VARARGS | METH_KEYWORDS, ""},
    {"reset", py_reset, METH_NOARGS, "drop every root and verdict"},
    {"stats", py_stats, METH_NOARGS, ""},
    {"info", py_info, METH_O, ""},
    {NULL, NULL, 0, NULL}};

static int find_set_dummy(void) {
    /* the marker a set leaves where it deleted a key (not exported on every version) */
    PyObject *s = PySet_New(NULL);
    if (!s) return -1;
    PyObject *k = PyLong_FromLong(12345);
    if (!k || PySet_Add(s, k) < 0 || PySet_Discard(s, k) < 0) { Py_XDECREF(k); Py_DECREF(s); return -1; }
    Py_DECREF(k);
    PySetObject *so = (PySetObject *)s;
    for (Py_ssize_t i = 0; i <= so->mask; i++) {
        if (so->table[i].key) { set_dummy = so->table[i].key; break; }
    }
    Py_DECREF(s);
    if (!set_dummy) { PyErr_SetString(PyExc_RuntimeError, "cash._cwatch: no set dummy"); return -1; }
    return 0;
}

#else  /* !CWATCH_USABLE */
static PyMethodDef methods[] = {{NULL, NULL, 0, NULL}};
#endif

static int exec_module(PyObject *m) {
    if (PyModule_AddIntConstant(m, "API_VERSION", CWATCH_API_VERSION) < 0) return -1;
    if (PyModule_AddStringConstant(m, "__version__", CWATCH_VERSION) < 0) return -1;
#if CWATCH_USABLE
    if (PyModule_AddObjectRef(m, "USABLE", Py_True) < 0) return -1;
    if (PyModule_AddObjectRef(m, "UNUSABLE_REASON", Py_None) < 0) return -1;
    if (!s_module) {
        s_module = PyUnicode_InternFromString("__module__");
        s_getstate = PyUnicode_InternFromString("__getstate__");
        s_reduce = PyUnicode_InternFromString("__reduce__");
        s_reduce_ex = PyUnicode_InternFromString("__reduce_ex__");
        s_dot = PyUnicode_InternFromString(".");
        if (!s_module || !s_getstate || !s_reduce || !s_reduce_ex || !s_dot) return -1;
        if (find_set_dummy() < 0) return -1;
    }
    struct { const char *n; int v; } consts[] = {
        {"FIRST", ST_FIRST}, {"ARM", ST_ARM}, {"SKIP", ST_SKIP}, {"BUSY", ST_BUSY}, {"OFF", ST_OFF},
        {"ARMED", A_ARMED}, {"ALREADY", A_ALREADY}, {"NEED_TYPE", A_NEED_TYPE},
        {"INELIGIBLE", A_INELIGIBLE}, {"TOO_SMALL", A_TOO_SMALL}, {"NO_ANCHOR", A_NO_ANCHOR},
        {"OVER_BUDGET", A_OVER_BUDGET}, {"ARM_BUSY", A_BUSY}, {"KINDS", NKINDS}};
    for (size_t i = 0; i < sizeof consts / sizeof consts[0]; i++)
        if (PyModule_AddIntConstant(m, consts[i].n, consts[i].v) < 0) return -1;
#else
    if (PyModule_AddObjectRef(m, "USABLE", Py_False) < 0) return -1;
    if (PyModule_AddStringConstant(m, "UNUSABLE_REASON",
                                   "free-threaded Python: the watcher fast path is off in this version") < 0)
        return -1;
#endif
    return 0;
}

static PyModuleDef_Slot slots[] = {
    {Py_mod_exec, exec_module},
    {Py_mod_multiple_interpreters, Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED},
#ifdef Py_GIL_DISABLED
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
    {0, NULL}};

static struct PyModuleDef moddef = {
    PyModuleDef_HEAD_INIT, "cash._cwatch",
    "Watcher signals that a value cash hashed before is provably unchanged.", 0, methods, slots,
    NULL, NULL, NULL};

PyMODINIT_FUNC PyInit__cwatch(void) { return PyModuleDef_Init(&moddef); }
