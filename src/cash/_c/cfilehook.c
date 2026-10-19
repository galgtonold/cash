/* cash._cwatch, part 4: the native file-read hook (cfilehook.c).
 *
 * On by default (cash.tracking.native_reads; CASH_NATIVE_FILE_HOOK=0 turns it
 * off). Python's audit
 * hook sees every open made through Python; a C library that opens a file
 * itself (libarrow, libsqlite3, libstdc++'s fstream, a Rust crate) raises no
 * event. This hook rebinds the slots through which already-loaded shared
 * libraries call the C library's (and on Windows the kernel's) open family,
 * so those opens are seen too. Two back ends share everything else:
 *
 * Linux/glibc on x86-64 and aarch64: GOT rebinding of
 *
 *   open open64 openat openat64 __open_2 __open64_2 __openat_2 __openat64_2
 *   creat creat64 fopen fopen64 freopen freopen64 opendir fdopendir
 *
 * The slots are found from each object's dynamic section (JUMP_SLOT, GLOB_DAT
 * and absolute relocations with no addend against an undefined symbol).
 * Never hooked: the main program and libpython (Python's own opens are the
 * audit hook's), the dynamic loader and libc itself, sanitizer runtimes,
 * OpenSSL (whose opens are CA bundles and openssl.cnf: machine state), and
 * this module. A slot is rebound only when it holds the very function
 * dlsym(RTLD_DEFAULT) returns for the name, or is still unresolved (a lazy
 * PLT slot pointing into its own object); a slot bound to anything else (an
 * interposer) is left alone and counted. dlopen is NOT hooked: glibc's
 * dlopen resolves $ORIGIN/RUNPATH from its caller's address, which a hook
 * would change. Objects loaded later are picked up by filehook_rescan(),
 * which each window open runs (cheap: dl_iterate_phdr's load counter).
 *
 * Windows on x64: Import Address Table patching in every loaded module
 * (EnumProcessModules), of
 *
 *   CreateFileW CreateFileA CreateFile2            (kernel32 / KernelBase)
 *   _open _wopen _sopen _wsopen _sopen_s _wsopen_s fopen _wfopen fopen_s
 *   _wfopen_s _fsopen _wfsopen freopen _wfreopen   (ucrtbase, msvcrt)
 *
 * A slot is matched by the address it holds, not by its name: whatever
 * import route a module took (kernel32, an api-ms-win-* forwarder, an
 * ordinal, a resolved delay-load slot) ends at the same function. The UCRT
 * and the old msvcrt.dll (MinGW builds) each get their own hooks, since a
 * FILE* of one must go back to the same one. Never hooked: the main program,
 * the python3xx DLL, modules under the Windows directory (the system's own
 * DLLs, and Defender's scanner, which injects itself from ProgramData), the
 * CRT itself (ucrtbase, vcruntime, msvcrt, api-ms-win-*), OpenSSL as on Linux
 * and this module. New modules are noticed through LdrRegisterDllNotification (a
 * counter the next window open compares). Calls through an address fetched
 * with GetProcAddress (ctypes, a library's own dynamic loading, sqlite's
 * table of system calls) bypass the IAT, as dlsym'd pointers bypass the GOT.
 *
 * Recording happens only while at least one window is open (a tracker is
 * running): the hook then calls the real function, saves errno (and the
 * thread's last error on Windows), notes the path (absolute: a relative one
 * is joined to the cwd or to the dirfd's /proc/self/fd link; GetFullPathNameW
 * on Windows), restores them and returns the real result. With no window open
 * the hook is one relaxed load and a tail call. Events live in one buffer
 * under a mutex; a window collects every event recorded while it was open,
 * from any thread (a library's own I/O threads have no Python context to
 * attribute them by). The cost of that is over-attribution: concurrent
 * trackers each get every native open made meanwhile, which can only add
 * dependencies, never drop one. A buffer that fills up marks the windows that
 * lost events (the tracker then records a dependency that is never fresh).
 *
 * Not seen: direct syscalls (Go runtimes, io_uring, a library issuing
 * syscall(SYS_openat) itself, NtCreateFile), libraries linked statically
 * against the C library (/MT on Windows), opens inside the C library itself
 * (getaddrinfo's /etc/hosts, iconv modules), stat/access probes, directory
 * listings on Windows (FindFirstFile), and opens by a library loaded and used
 * within one window (its constructors and first calls run before the next
 * rescan).
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <Python.h>
#include "cwalk.h"

#if defined(__linux__) && defined(__GLIBC__) && (defined(__x86_64__) || defined(__aarch64__))
#define FH_LINUX 1
#else
#define FH_LINUX 0
#endif
#if defined(_WIN32) && defined(_M_X64) && !defined(_M_ARM64EC)
#define FH_WINDOWS 1
#else
#define FH_WINDOWS 0
#endif
#define FH_SUPPORTED (FH_LINUX || FH_WINDOWS)

#if FH_SUPPORTED
#include <errno.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if FH_LINUX
#include <dirent.h>
#include <dlfcn.h>
#include <elf.h>
#include <fcntl.h>
#include <limits.h>
#include <link.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__x86_64__)
#define R_JUMP_SLOT R_X86_64_JUMP_SLOT
#define R_GLOB_DAT R_X86_64_GLOB_DAT
#define R_ABS64 R_X86_64_64
#else
#define R_JUMP_SLOT R_AARCH64_JUMP_SLOT
#define R_GLOB_DAT R_AARCH64_GLOB_DAT
#define R_ABS64 R_AARCH64_ABS64
#endif

typedef pthread_mutex_t fh_mutex;
#define FH_MUTEX_INIT PTHREAD_MUTEX_INITIALIZER
#define fh_lock(m) pthread_mutex_lock(m)
#define fh_unlock(m) pthread_mutex_unlock(m)
#define FH_TLS __thread
typedef int fh_flag;
typedef uint64_t fh_counter;
#define FLAG_GET(p) __atomic_load_n((p), __ATOMIC_RELAXED)
#define FLAG_SET(p, v) __atomic_store_n((p), (v), __ATOMIC_RELAXED)
#define COUNTER_INC(p) __atomic_fetch_add((p), 1, __ATOMIC_RELAXED)
#define COUNTER_GET(p) __atomic_load_n((p), __ATOMIC_RELAXED)

#else /* FH_WINDOWS */
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#include <locale.h>
#include <psapi.h>
#include <share.h>
#include <wchar.h>

typedef SRWLOCK fh_mutex;
#define FH_MUTEX_INIT SRWLOCK_INIT
#define fh_lock(m) AcquireSRWLockExclusive(m)
#define fh_unlock(m) ReleaseSRWLockExclusive(m)
#define FH_TLS __declspec(thread)
typedef volatile LONG fh_flag;
typedef volatile LONG64 fh_counter;
/* Aligned loads and stores of these are atomic on x64. */
#define FLAG_GET(p) (*(p))
#define FLAG_SET(p, v) InterlockedExchange((p), (v))
#define COUNTER_INC(p) InterlockedIncrement64(p)
#define COUNTER_GET(p) (*(p))
#endif

/* Event kinds, as Python sees them (native_reads.py). */
enum { K_READ = 1, K_ABSENT = 2, K_CREATED = 3, K_WRITE = 4, K_UNRESOLVED = 5 };

/* ---- event buffer ------------------------------------------------------ */

typedef struct {
    uint64_t seq;
    int kind;
    uint32_t hash;
    char *path;
} fh_event;

#define FH_MAX_EVENTS 65536
static fh_mutex ev_mutex = FH_MUTEX_INIT;
static fh_event *events;    /* in seq order */
static size_t n_events, cap_events;
static uint32_t *ev_index;  /* open-addressing set of event indexes + 1, for dedupe */
static size_t index_cap;    /* power of two, >= 2 * cap_events */
static uint64_t next_seq = 1;
static uint64_t last_lost_seq; /* 0: nothing lost */
static uint64_t *windows;      /* start seqs of the open windows */
static size_t n_windows, cap_windows;
static uint64_t newest_start;  /* max of windows[] */
static fh_flag armed;          /* n_windows > 0, read without the lock */
static FH_TLS int in_hook;     /* this thread is inside cash's recording */
static uint64_t st_recorded, st_deduped, st_lost;
static fh_counter st_calls;

#define ARMED() (FLAG_GET(&armed) && !in_hook)

static int grow(void **arr, size_t *cap, size_t need, size_t item) {
    if (need <= *cap) return 0;
    size_t c = *cap ? *cap * 2 : 16;
    while (c < need) c *= 2;
    void *p = realloc(*arr, c * item);
    if (!p) return -1;
    *arr = p;
    *cap = c;
    return 0;
}

static uint32_t path_hash(const char *s, int kind) {
    uint32_t h = 2166136261u ^ (uint32_t)kind;
    for (; *s; s++) h = (h ^ (unsigned char)*s) * 16777619u;
    return h;
}

static void index_rebuild(void) {
    if (!ev_index) return;
    memset(ev_index, 0, index_cap * sizeof *ev_index);
    for (size_t i = 0; i < n_events; i++) {
        size_t j = events[i].hash & (index_cap - 1);
        while (ev_index[j]) j = (j + 1) & (index_cap - 1);
        ev_index[j] = (uint32_t)(i + 1);
    }
}

/* Record one event; takes ownership of *path*. Called with in_hook set. */
static void record(int kind, char *path) {
    uint32_t h = path_hash(path, kind);
    fh_lock(&ev_mutex);
    if (!armed) goto drop; /* the last window closed meanwhile */
    st_recorded++;
    /* The same (kind, path) recorded after the newest window opened is in
     * every window that would get this one: one is enough. */
    if (ev_index) {
        size_t j = h & (index_cap - 1);
        while (ev_index[j]) {
            fh_event *e = &events[ev_index[j] - 1];
            if (e->hash == h && e->kind == kind && e->seq >= newest_start && strcmp(e->path, path) == 0) {
                st_deduped++;
                goto drop;
            }
            j = (j + 1) & (index_cap - 1);
        }
    }
    if (n_events >= FH_MAX_EVENTS) goto lost;
    if (n_events == cap_events) {
        size_t c = cap_events ? cap_events * 2 : 256;
        if (c > FH_MAX_EVENTS) c = FH_MAX_EVENTS;
        fh_event *p = realloc(events, c * sizeof *events);
        if (!p) goto lost;
        events = p;
        cap_events = c;
        uint32_t *ix = calloc(c * 2, sizeof *ix);
        if (!ix) goto lost;
        free(ev_index);
        ev_index = ix;
        index_cap = c * 2;
        index_rebuild();
    }
    events[n_events] = (fh_event){next_seq++, kind, h, path};
    {
        size_t j = h & (index_cap - 1);
        while (ev_index[j]) j = (j + 1) & (index_cap - 1);
        ev_index[j] = (uint32_t)(n_events + 1);
    }
    n_events++;
    fh_unlock(&ev_mutex);
    return;
lost:
    st_lost++;
    last_lost_seq = next_seq++;
drop:
    fh_unlock(&ev_mutex);
    free(path);
}

static void clear_events_locked(void) {
    for (size_t i = 0; i < n_events; i++) free(events[i].path);
    n_events = 0;
    index_rebuild();
}

/* ---- hooked slots, shared state ---------------------------------------- */

typedef struct {
    void **slot;
    void *old;
    int sym;
    uintptr_t base;
} fh_patch;

static fh_mutex hook_mutex = FH_MUTEX_INIT;
static int enabled;
/* Set by fh_hook_new for one scan: objects hooked before are skipped. */
static int scan_only_new;
static fh_patch *patches;
static size_t n_patches, cap_patches;
static uintptr_t own_base;
static size_t foreign_slots;

#if FH_LINUX
/* ======================================================================== */
/* Linux: GOT rebinding                                                     */
/* ======================================================================== */

enum {
    S_OPEN, S_OPEN64, S_OPENAT, S_OPENAT64, S_OPEN_2, S_OPEN64_2, S_OPENAT_2, S_OPENAT64_2,
    S_CREAT, S_CREAT64, S_FOPEN, S_FOPEN64, S_FREOPEN, S_FREOPEN64, S_OPENDIR, S_FDOPENDIR, S_COUNT
};
static const char *const sym_names[S_COUNT] = {
    "open", "open64", "openat", "openat64", "__open_2", "__open64_2", "__openat_2", "__openat64_2",
    "creat", "creat64", "fopen", "fopen64", "freopen", "freopen64", "opendir", "fdopendir"};
static void *real_fn[S_COUNT];
static void *hook_fn[S_COUNT];

typedef int (*open_t)(const char *, int, ...);
typedef int (*openat_t)(int, const char *, int, ...);
typedef int (*open2_t)(const char *, int);
typedef int (*openat2_t)(int, const char *, int);
typedef int (*creat_t)(const char *, mode_t);
typedef FILE *(*fopen_t)(const char *, const char *);
typedef FILE *(*freopen_t)(const char *, const char *, FILE *);
typedef DIR *(*opendir_t)(const char *);
typedef DIR *(*fdopendir_t)(int);

typedef struct {
    uintptr_t base;
    const void *phdr;
    char *name;
    int patched; /* slots rebound in it */
} fh_object;

static fh_object *objects;
static size_t n_objects, cap_objects;
static unsigned long long seen_adds = 0, seen_subs = 0;
static long page_size;
static int atfork_done;

static const char plat_how[] = "Linux ELF GOT rebinding";

/* *path* made absolute, malloc'd, or NULL when it cannot be. */
static char *absolute(int dirfd, const char *path) {
    if (path[0] == '/') return strdup(path);
    char base[PATH_MAX];
    if (dirfd == AT_FDCWD) {
        if (!getcwd(base, sizeof base)) return NULL;
    } else {
        char link[64];
        snprintf(link, sizeof link, "/proc/self/fd/%d", dirfd);
        ssize_t n = readlink(link, base, sizeof base - 1);
        if (n <= 0) return NULL;
        base[n] = '\0';
        if (base[0] != '/') return NULL; /* "anon_inode:..." */
    }
    size_t lb = strlen(base), lp = strlen(path);
    char *out = malloc(lb + lp + 2);
    if (!out) return NULL;
    memcpy(out, base, lb);
    size_t k = lb;
    if (k == 0 || out[k - 1] != '/') out[k++] = '/';
    memcpy(out + k, path, lp + 1);
    return out;
}

static int pseudo_fs(const char *p) {
    return strncmp(p, "/proc/", 6) == 0 || strncmp(p, "/sys/", 5) == 0 || strncmp(p, "/dev/", 5) == 0;
}

static void note(int dirfd, const char *path, int kind) {
    if (!path || !*path || kind == 0) return;
    if (pseudo_fs(path)) return; /* machine state, never a dependency (is_pseudo_fs) */
    char *abs = absolute(dirfd, path);
    if (!abs) {
        abs = strdup(path);
        if (!abs) return;
        kind = K_UNRESOLVED;
    } else if (pseudo_fs(abs)) {
        free(abs);
        return;
    }
    record(kind, abs);
}

/* Was *path* there before an open that may create it? */
static int existed(int dirfd, const char *path) {
    struct stat st;
    return path && fstatat(dirfd, path, &st, 0) == 0;
}

/* What an open with *flags* that returned *ok* (errno *err*) means. *before*:
 * the file existed before, for an O_CREAT open that neither truncates nor
 * insists on making it (-1 when not asked). */
static int classify_flags(int flags, int ok, int err, int before) {
    if (flags & O_PATH) return 0;
    if ((flags & O_TMPFILE) == O_TMPFILE) return 0;
    int acc = flags & O_ACCMODE;
    if (!ok) {
        if (err == ENOENT || err == ENOTDIR) return (flags & O_CREAT) ? 0 : K_ABSENT;
        return acc == O_RDONLY ? K_READ : 0;
    }
    if (acc == O_RDONLY) return K_READ;
    if ((flags & O_TRUNC) || ((flags & O_CREAT) && (flags & O_EXCL))) return K_CREATED;
    if (flags & O_CREAT) {
        if (!before) return K_CREATED;
        return acc == O_RDWR ? K_READ : K_WRITE;
    }
    return acc == O_RDWR ? K_READ : K_WRITE;
}

static int classify_mode(const char *mode, int ok, int err, int before) {
    if (!mode) return 0;
    int plus = strchr(mode, '+') != NULL;
    switch (mode[0]) {
    case 'r':
        if (!ok) return (err == ENOENT || err == ENOTDIR) ? K_ABSENT : K_READ;
        return K_READ;
    case 'w':
        return ok ? K_CREATED : 0;
    case 'a':
        if (!ok) return 0;
        if (!before) return K_CREATED;
        return plus ? K_READ : K_WRITE;
    default:
        return 0;
    }
}

static int needs_mode(int flags) { return (flags & O_CREAT) || (flags & O_TMPFILE) == O_TMPFILE; }

static int asks_before(int flags) {
    return (flags & O_CREAT) && !(flags & O_TRUNC) && !(flags & O_EXCL);
}

static int open_common(int sym, int dirfd, const char *path, int flags, mode_t mode, int has_mode) {
    if (!ARMED()) {
        switch (sym) {
        case S_OPEN: case S_OPEN64: return ((open_t)real_fn[sym])(path, flags, mode);
        case S_OPENAT: case S_OPENAT64: return ((openat_t)real_fn[sym])(dirfd, path, flags, mode);
        case S_OPEN_2: case S_OPEN64_2: return ((open2_t)real_fn[sym])(path, flags);
        case S_OPENAT_2: case S_OPENAT64_2: return ((openat2_t)real_fn[sym])(dirfd, path, flags);
        default: return ((creat_t)real_fn[sym])(path, mode);
        }
    }
    (void)has_mode;
    in_hook = 1;
    int before = asks_before(flags) ? existed(dirfd, path) : -1;
    int fd;
    switch (sym) {
    case S_OPEN: case S_OPEN64: fd = ((open_t)real_fn[sym])(path, flags, mode); break;
    case S_OPENAT: case S_OPENAT64: fd = ((openat_t)real_fn[sym])(dirfd, path, flags, mode); break;
    case S_OPEN_2: case S_OPEN64_2: fd = ((open2_t)real_fn[sym])(path, flags); break;
    case S_OPENAT_2: case S_OPENAT64_2: fd = ((openat2_t)real_fn[sym])(dirfd, path, flags); break;
    default: fd = ((creat_t)real_fn[sym])(path, mode); break;
    }
    int err = errno;
    COUNTER_INC(&st_calls);
    int kind = classify_flags(flags, fd >= 0, err, before);
    if (kind == K_READ && fd >= 0) {
        /* A directory opened as a descriptor (sqlite's fsync of the folder,
         * an openat base) is not a read of it; a listing comes through
         * opendir/fdopendir. */
        struct stat st;
        if (fstat(fd, &st) == 0 && S_ISDIR(st.st_mode)) kind = 0;
    }
    note(dirfd, path, kind);
    in_hook = 0;
    errno = err;
    return fd;
}

#define MODE_ARG(flags)                                \
    mode_t mode = 0;                                   \
    if (needs_mode(flags)) {                           \
        va_list ap;                                    \
        va_start(ap, flags);                           \
        mode = (mode_t)va_arg(ap, int);                \
        va_end(ap);                                    \
    }

static int h_open(const char *p, int flags, ...) { MODE_ARG(flags); return open_common(S_OPEN, AT_FDCWD, p, flags, mode, 1); }
static int h_open64(const char *p, int flags, ...) { MODE_ARG(flags); return open_common(S_OPEN64, AT_FDCWD, p, flags, mode, 1); }
static int h_openat(int d, const char *p, int flags, ...) { MODE_ARG(flags); return open_common(S_OPENAT, d, p, flags, mode, 1); }
static int h_openat64(int d, const char *p, int flags, ...) { MODE_ARG(flags); return open_common(S_OPENAT64, d, p, flags, mode, 1); }
static int h_open_2(const char *p, int flags) { return open_common(S_OPEN_2, AT_FDCWD, p, flags, 0, 0); }
static int h_open64_2(const char *p, int flags) { return open_common(S_OPEN64_2, AT_FDCWD, p, flags, 0, 0); }
static int h_openat_2(int d, const char *p, int flags) { return open_common(S_OPENAT_2, d, p, flags, 0, 0); }
static int h_openat64_2(int d, const char *p, int flags) { return open_common(S_OPENAT64_2, d, p, flags, 0, 0); }
static int h_creat(const char *p, mode_t m) { return open_common(S_CREAT, AT_FDCWD, p, O_CREAT | O_WRONLY | O_TRUNC, m, 1); }
static int h_creat64(const char *p, mode_t m) { return open_common(S_CREAT64, AT_FDCWD, p, O_CREAT | O_WRONLY | O_TRUNC, m, 1); }

static FILE *fopen_common(int sym, const char *path, const char *mode, FILE *stream) {
    if (!ARMED()) {
        if (sym == S_FREOPEN || sym == S_FREOPEN64) return ((freopen_t)real_fn[sym])(path, mode, stream);
        return ((fopen_t)real_fn[sym])(path, mode);
    }
    in_hook = 1;
    int before = (path && mode && mode[0] == 'a') ? existed(AT_FDCWD, path) : -1;
    FILE *f;
    if (sym == S_FREOPEN || sym == S_FREOPEN64) f = ((freopen_t)real_fn[sym])(path, mode, stream);
    else f = ((fopen_t)real_fn[sym])(path, mode);
    int err = errno;
    COUNTER_INC(&st_calls);
    note(AT_FDCWD, path, classify_mode(mode, f != NULL, err, before));
    in_hook = 0;
    errno = err;
    return f;
}

static FILE *h_fopen(const char *p, const char *m) { return fopen_common(S_FOPEN, p, m, NULL); }
static FILE *h_fopen64(const char *p, const char *m) { return fopen_common(S_FOPEN64, p, m, NULL); }
static FILE *h_freopen(const char *p, const char *m, FILE *s) { return fopen_common(S_FREOPEN, p, m, s); }
static FILE *h_freopen64(const char *p, const char *m, FILE *s) { return fopen_common(S_FREOPEN64, p, m, s); }

static DIR *h_opendir(const char *p) {
    if (!ARMED()) return ((opendir_t)real_fn[S_OPENDIR])(p);
    in_hook = 1;
    DIR *d = ((opendir_t)real_fn[S_OPENDIR])(p);
    int err = errno;
    COUNTER_INC(&st_calls);
    note(AT_FDCWD, p, d ? K_READ : (err == ENOENT || err == ENOTDIR) ? K_ABSENT : 0);
    in_hook = 0;
    errno = err;
    return d;
}

static DIR *h_fdopendir(int fd) {
    if (!ARMED()) return ((fdopendir_t)real_fn[S_FDOPENDIR])(fd);
    in_hook = 1;
    DIR *d = ((fdopendir_t)real_fn[S_FDOPENDIR])(fd);
    int err = errno;
    COUNTER_INC(&st_calls);
    if (d) note(fd, ".", K_READ); /* the listed folder: what fd names */
    in_hook = 0;
    errno = err;
    return d;
}

/* ---- GOT rebinding ----------------------------------------------------- */

static int skipped_object(const char *name, uintptr_t base) {
    if (base == own_base) return 1;
    if (!name || !*name) return 1; /* the main program: Python's own opens */
    const char *b = strrchr(name, '/');
    b = b ? b + 1 : name;
    static const char *const prefixes[] = {
        "linux-vdso", "linux-gate", "ld-linux", "ld64.so", "ld-2.", "libc.so", "libc-", "libpthread",
        "libdl.", "libdl-", "librt.", "libm.so", "libm-", "libmvec", "libpython", "libasan", "libubsan",
        "liblsan", "libtsan", "libhwasan", "libgcc_s", "libresolv", "libnss_", "libutil", "libmemusage",
        "libSegFault", "libpcprofile",
        /* OpenSSL: its opens are CA bundles and openssl.cnf, machine state like /etc/hosts */
        "libcrypto", "libssl"};
    for (size_t i = 0; i < sizeof prefixes / sizeof prefixes[0]; i++)
        if (strncmp(b, prefixes[i], strlen(prefixes[i])) == 0) return 1;
    return 0;
}

static fh_object *find_object(uintptr_t base, const void *phdr) {
    for (size_t i = 0; i < n_objects; i++)
        if (objects[i].base == base && objects[i].phdr == phdr) return &objects[i];
    return NULL;
}

static uintptr_t dyn_ptr(uintptr_t v, uintptr_t base) {
    /* glibc relocates these entries of l_ld in place; another loader may not. */
    return v < base ? v + base : v;
}

/* The protection of the page holding *addr*, from the object's segments. */
static int page_prot(struct dl_phdr_info *info, uintptr_t addr) {
    uintptr_t page = addr & ~(uintptr_t)(page_size - 1);
    for (int i = 0; i < info->dlpi_phnum; i++) {
        const ElfW(Phdr) *ph = &info->dlpi_phdr[i];
        if (ph->p_type != PT_GNU_RELRO) continue;
        uintptr_t s = (info->dlpi_addr + ph->p_vaddr) & ~(uintptr_t)(page_size - 1);
        uintptr_t e = (info->dlpi_addr + ph->p_vaddr + ph->p_memsz) & ~(uintptr_t)(page_size - 1);
        if (page >= s && page < e) return PROT_READ;
    }
    for (int i = 0; i < info->dlpi_phnum; i++) {
        const ElfW(Phdr) *ph = &info->dlpi_phdr[i];
        if (ph->p_type != PT_LOAD) continue;
        uintptr_t s = info->dlpi_addr + ph->p_vaddr, e = s + ph->p_memsz;
        if (addr >= s && addr < e) {
            int p = 0;
            if (ph->p_flags & PF_R) p |= PROT_READ;
            if (ph->p_flags & PF_W) p |= PROT_WRITE;
            if (ph->p_flags & PF_X) p |= PROT_EXEC;
            return p;
        }
    }
    return -1;
}

static int in_object(struct dl_phdr_info *info, uintptr_t addr) {
    for (int i = 0; i < info->dlpi_phnum; i++) {
        const ElfW(Phdr) *ph = &info->dlpi_phdr[i];
        if (ph->p_type != PT_LOAD) continue;
        uintptr_t s = info->dlpi_addr + ph->p_vaddr;
        if (addr >= s && addr < s + ph->p_memsz) return 1;
    }
    return 0;
}

static int write_slot(struct dl_phdr_info *info, void **slot, void *value) {
    int prot = page_prot(info, (uintptr_t)slot);
    if (prot < 0) return -1;
    void *page = (void *)((uintptr_t)slot & ~(uintptr_t)(page_size - 1));
    if (!(prot & PROT_WRITE)) {
        if (mprotect(page, (size_t)page_size, prot | PROT_WRITE) != 0) return -1;
    }
    __atomic_store_n(slot, value, __ATOMIC_RELEASE);
    if (!(prot & PROT_WRITE)) mprotect(page, (size_t)page_size, prot);
    return 0;
}

static int sym_of(const char *name) {
    if (name[0] != 'o' && name[0] != '_' && name[0] != 'c' && name[0] != 'f') return -1;
    for (int i = 0; i < S_COUNT; i++)
        if (strcmp(name, sym_names[i]) == 0) return i;
    return -1;
}

static void patch_relocs(struct dl_phdr_info *info, fh_object *obj, const ElfW(Rela) *rel, size_t count,
                         const ElfW(Sym) *symtab, const char *strtab) {
    for (size_t i = 0; i < count; i++) {
        unsigned type = (unsigned)ELF64_R_TYPE(rel[i].r_info);
        size_t si = ELF64_R_SYM(rel[i].r_info);
        if (si == 0 || (type != R_JUMP_SLOT && type != R_GLOB_DAT && type != R_ABS64)) continue;
        if (type == R_ABS64 && rel[i].r_addend != 0) continue;
        const ElfW(Sym) *s = &symtab[si];
        if (s->st_shndx != SHN_UNDEF) continue; /* the object's own definition */
        int sym = sym_of(strtab + s->st_name);
        if (sym < 0 || !real_fn[sym]) continue;
        void **slot = (void **)(info->dlpi_addr + rel[i].r_offset);
        void *cur = __atomic_load_n(slot, __ATOMIC_ACQUIRE);
        if (cur == hook_fn[sym]) continue;
        if (cur != real_fn[sym] && !(type == R_JUMP_SLOT && in_object(info, (uintptr_t)cur))) {
            foreign_slots++;
            continue;
        }
        if (grow((void **)&patches, &cap_patches, n_patches + 1, sizeof *patches) < 0) return;
        if (write_slot(info, slot, hook_fn[sym]) < 0) continue;
        patches[n_patches++] = (fh_patch){slot, cur, sym, info->dlpi_addr};
        obj->patched++;
    }
}

static void hook_object(struct dl_phdr_info *info, fh_object *obj) {
    const ElfW(Dyn) *dyn = NULL;
    for (int i = 0; i < info->dlpi_phnum; i++)
        if (info->dlpi_phdr[i].p_type == PT_DYNAMIC)
            dyn = (const ElfW(Dyn) *)(info->dlpi_addr + info->dlpi_phdr[i].p_vaddr);
    if (!dyn) return;
    uintptr_t base = info->dlpi_addr;
    const ElfW(Sym) *symtab = NULL;
    const char *strtab = NULL;
    const ElfW(Rela) *rela = NULL, *jmprel = NULL;
    size_t relasz = 0, pltrelsz = 0, relaent = sizeof(ElfW(Rela));
    long pltrel = DT_RELA;
    for (; dyn->d_tag != DT_NULL; dyn++) {
        switch (dyn->d_tag) {
        case DT_SYMTAB: symtab = (const ElfW(Sym) *)dyn_ptr(dyn->d_un.d_ptr, base); break;
        case DT_STRTAB: strtab = (const char *)dyn_ptr(dyn->d_un.d_ptr, base); break;
        case DT_RELA: rela = (const ElfW(Rela) *)dyn_ptr(dyn->d_un.d_ptr, base); break;
        case DT_RELASZ: relasz = dyn->d_un.d_val; break;
        case DT_RELAENT: relaent = dyn->d_un.d_val; break;
        case DT_JMPREL: jmprel = (const ElfW(Rela) *)dyn_ptr(dyn->d_un.d_ptr, base); break;
        case DT_PLTRELSZ: pltrelsz = dyn->d_un.d_val; break;
        case DT_PLTREL: pltrel = (long)dyn->d_un.d_val; break;
        default: break;
        }
    }
    if (!symtab || !strtab || relaent != sizeof(ElfW(Rela))) return;
    if (jmprel && pltrel == DT_RELA) patch_relocs(info, obj, jmprel, pltrelsz / sizeof(ElfW(Rela)), symtab, strtab);
    if (rela) patch_relocs(info, obj, rela, relasz / sizeof(ElfW(Rela)), symtab, strtab);
}

static int scan_cb(struct dl_phdr_info *info, size_t size, void *data) {
    (void)data;
    if (size < offsetof(struct dl_phdr_info, dlpi_subs) + sizeof info->dlpi_subs) return 1;
    if (skipped_object(info->dlpi_name, info->dlpi_addr)) return 0;
    fh_object *obj = find_object(info->dlpi_addr, info->dlpi_phdr);
    if (!obj) {
        if (grow((void **)&objects, &cap_objects, n_objects + 1, sizeof *objects) < 0) return 1;
        obj = &objects[n_objects++];
        *obj = (fh_object){info->dlpi_addr, info->dlpi_phdr, strdup(info->dlpi_name), 0};
    } else if (scan_only_new) {
        return 0;
    }
    hook_object(info, obj); /* also re-binds a slot something reset */
    return 0;
}

static int counter_cb(struct dl_phdr_info *info, size_t size, void *data) {
    if (size < offsetof(struct dl_phdr_info, dlpi_subs) + sizeof info->dlpi_subs) return 1;
    unsigned long long *c = data;
    c[0] = info->dlpi_adds;
    c[1] = info->dlpi_subs;
    return 1; /* the counters are the same on every entry */
}

typedef struct { uintptr_t base; int found; struct dl_phdr_info info; } find_req;

static int still_loaded_cb(struct dl_phdr_info *info, size_t size, void *data) {
    (void)size;
    find_req *r = data;
    if (info->dlpi_addr == r->base) {
        r->found = 1;
        r->info = *info;
        return 1;
    }
    return 0;
}

static void drop_unloaded_objects(void) {
    size_t k = 0;
    for (size_t i = 0; i < n_objects; i++) {
        find_req r = {objects[i].base, 0, {0}};
        dl_iterate_phdr(still_loaded_cb, &r);
        if (r.found && r.info.dlpi_phdr == objects[i].phdr) {
            objects[k++] = objects[i];
        } else {
            free(objects[i].name);
            size_t m = 0; /* its patches went with it */
            for (size_t j = 0; j < n_patches; j++)
                if (patches[j].base != objects[i].base) patches[m++] = patches[j];
            n_patches = m;
        }
    }
    n_objects = k;
}

/* Scan for objects loaded since the last scan. Call with hook_mutex held. */
static long rescan_locked(int force) {
    unsigned long long c[2] = {0, 0};
    dl_iterate_phdr(counter_cb, c);
    if (!force && c[0] == seen_adds && c[1] == seen_subs) return 0;
    size_t before = n_patches;
    if (c[1] != seen_subs) drop_unloaded_objects();
    dl_iterate_phdr(scan_cb, NULL);
    seen_adds = c[0];
    seen_subs = c[1];
    return (long)(n_patches - before);
}

static void unhook_locked(void) {
    for (size_t i = 0; i < n_patches; i++) {
        find_req r = {patches[i].base, 0, {0}};
        dl_iterate_phdr(still_loaded_cb, &r);
        if (!r.found) continue;
        void *cur = __atomic_load_n(patches[i].slot, __ATOMIC_ACQUIRE);
        if (cur == hook_fn[patches[i].sym]) write_slot(&r.info, patches[i].slot, patches[i].old);
    }
    n_patches = 0;
    for (size_t i = 0; i < n_objects; i++) free(objects[i].name);
    n_objects = 0;
    seen_adds = seen_subs = 0;
    foreign_slots = 0;
}

static void atfork_prepare(void) {
    pthread_mutex_lock(&hook_mutex);
    pthread_mutex_lock(&ev_mutex);
}
static void atfork_parent(void) {
    pthread_mutex_unlock(&ev_mutex);
    pthread_mutex_unlock(&hook_mutex);
}
static void atfork_child(void) {
    /* The parent's windows never close here: start the child with none. */
    n_windows = 0;
    newest_start = 0;
    FLAG_SET(&armed, 0);
    clear_events_locked();
    pthread_mutex_unlock(&ev_mutex);
    pthread_mutex_unlock(&hook_mutex);
}

/* Find the real functions and this module; call with hook_mutex held. */
static void prepare_locked(void) {
    if (!page_size) page_size = sysconf(_SC_PAGESIZE);
    if (!own_base) {
        Dl_info di;
        if (dladdr((void *)prepare_locked, &di)) own_base = (uintptr_t)di.dli_fbase;
    }
    if (!atfork_done && pthread_atfork(atfork_prepare, atfork_parent, atfork_child) == 0) atfork_done = 1;
    for (int i = 0; i < S_COUNT; i++) real_fn[i] = dlsym(RTLD_DEFAULT, sym_names[i]);
}

static void init_hooks(void) {
    hook_fn[S_OPEN] = (void *)h_open;
    hook_fn[S_OPEN64] = (void *)h_open64;
    hook_fn[S_OPENAT] = (void *)h_openat;
    hook_fn[S_OPENAT64] = (void *)h_openat64;
    hook_fn[S_OPEN_2] = (void *)h_open_2;
    hook_fn[S_OPEN64_2] = (void *)h_open64_2;
    hook_fn[S_OPENAT_2] = (void *)h_openat_2;
    hook_fn[S_OPENAT64_2] = (void *)h_openat64_2;
    hook_fn[S_CREAT] = (void *)h_creat;
    hook_fn[S_CREAT64] = (void *)h_creat64;
    hook_fn[S_FOPEN] = (void *)h_fopen;
    hook_fn[S_FOPEN64] = (void *)h_fopen64;
    hook_fn[S_FREOPEN] = (void *)h_freopen;
    hook_fn[S_FREOPEN64] = (void *)h_freopen64;
    hook_fn[S_OPENDIR] = (void *)h_opendir;
    hook_fn[S_FDOPENDIR] = (void *)h_fdopendir;
}

#else /* FH_WINDOWS */
/* ======================================================================== */
/* Windows: Import Address Table patching                                   */
/* ======================================================================== */

#ifdef _MSC_VER
#pragma comment(lib, "psapi.lib")
#endif

/* The CRT open family, in the same order for each CRT; odd entries are the
 * wide-character twins of the one before. */
enum {
    C_OPEN, C_WOPEN, C_SOPEN, C_WSOPEN, C_SOPEN_S, C_WSOPEN_S, C_FOPEN, C_WFOPEN, C_FOPEN_S, C_WFOPEN_S,
    C_FSOPEN, C_WFSOPEN, C_FREOPEN, C_WFREOPEN, C_COUNT
};
static const char *const crt_names[C_COUNT] = {
    "_open", "_wopen", "_sopen", "_wsopen", "_sopen_s", "_wsopen_s", "fopen", "_wfopen", "fopen_s",
    "_wfopen_s", "_fsopen", "_wfsopen", "freopen", "_wfreopen"};

/* The CreateFile family (kernel32 / KernelBase), then the UCRT's (ucrtbase)
 * and msvcrt.dll's CRT families. */
enum { S_CREATEFILEW, S_CREATEFILEA, S_CREATEFILE2, S_UCRT, S_MSVCRT = S_UCRT + C_COUNT, S_COUNT = S_MSVCRT + C_COUNT };
enum { P_KERNEL, P_UCRT, P_MSVCRT };
static const char *const kernel_names[S_UCRT] = {"CreateFileW", "CreateFileA", "CreateFile2"};

static void *real_fn[S_COUNT]; /* what a hook calls */
static void *alt_fn[S_COUNT];  /* another address a slot may hold for it (kernel32's export) */
static void *hook_fn[S_COUNT];

static int family_of(int sym) { return sym < S_UCRT ? P_KERNEL : sym < S_MSVCRT ? P_UCRT : P_MSVCRT; }
static int crt_of(int sym) { return (sym - S_UCRT) % C_COUNT; }
static const char *name_of(int sym) { return sym < S_UCRT ? kernel_names[sym] : crt_names[crt_of(sym)]; }

typedef HANDLE(WINAPI *createfilew_t)(LPCWSTR, DWORD, DWORD, LPSECURITY_ATTRIBUTES, DWORD, DWORD, HANDLE);
typedef HANDLE(WINAPI *createfilea_t)(LPCSTR, DWORD, DWORD, LPSECURITY_ATTRIBUTES, DWORD, DWORD, HANDLE);
typedef HANDLE(WINAPI *createfile2_t)(LPCWSTR, DWORD, DWORD, DWORD, void *);
/* Paths are const char * or const wchar_t *; on x64 both are one register. */
typedef int(__cdecl *crt_open_t)(const void *, int, ...);
typedef int(__cdecl *crt_sopen_t)(const void *, int, int, ...);
typedef errno_t(__cdecl *crt_sopen_s_t)(int *, const void *, int, int, int);
typedef FILE *(__cdecl *crt_fopen_t)(const void *, const void *);
typedef errno_t(__cdecl *crt_fopen_s_t)(FILE **, const void *, const void *);
typedef FILE *(__cdecl *crt_fsopen_t)(const void *, const void *, int);
typedef FILE *(__cdecl *crt_freopen_t)(const void *, const void *, FILE *);
typedef int *(__cdecl *crt_errno_t)(void);

static crt_errno_t msvcrt_errno; /* msvcrt.dll's own errno, which ours is not */

typedef struct {
    uintptr_t base;
    DWORD size, stamp; /* SizeOfImage and TimeDateStamp: the same module still there */
    char *name;
    int patched; /* slots rebound in it */
} fh_object;

static fh_object *objects;
static size_t n_objects, cap_objects;
static uintptr_t own_end;
static wchar_t windir[MAX_PATH + 2]; /* "C:\Windows\" */
static size_t windir_len;

/* Address -> sym, sorted by address, for matching a slot by what it holds. */
typedef struct { uintptr_t addr; int sym; } fh_addr;
static fh_addr addrs[2 * S_COUNT];
static size_t n_addrs;

/* Loader notifications bump these; a window open compares them. */
static volatile LONG load_gen, unload_gen;
static LONG seen_load = -1, seen_unload = -1;
static int notify_ok;
static void *notify_cookie;
static DWORD seen_module_bytes; /* without notifications: the module list's size */

static const char plat_how[] = "Windows PE import-table patching";

/* -- paths ------------------------------------------------------------------ */

/* The code page a narrow path is in, for the C library of *family*. */
static UINT narrow_cp(int family) {
    if (family == P_UCRT && ___lc_codepage_func() == CP_UTF8) return CP_UTF8;
    return AreFileApisANSI() ? CP_ACP : CP_OEMCP;
}

/* *path* (narrow in *cp* or wide) as a malloc'd wide string. */
static wchar_t *widen(const void *path, int wide, UINT cp) {
    if (!path) return NULL;
    if (wide) {
        size_t n = wcslen((const wchar_t *)path) + 1;
        wchar_t *w = malloc(n * sizeof *w);
        if (w) memcpy(w, path, n * sizeof *w);
        return w;
    }
    int n = MultiByteToWideChar(cp, 0, (const char *)path, -1, NULL, 0);
    if (n <= 0) return NULL;
    wchar_t *w = malloc((size_t)n * sizeof *w);
    if (w && MultiByteToWideChar(cp, 0, (const char *)path, -1, w, n) <= 0) {
        free(w);
        return NULL;
    }
    return w;
}

/* *path* made absolute (GetFullPathNameW), malloc'd, or NULL. */
static wchar_t *full_path(const void *path, int wide, UINT cp) {
    wchar_t *w = widen(path, wide, cp);
    if (!w || !*w) {
        free(w);
        return NULL;
    }
    wchar_t buf[MAX_PATH];
    DWORD n = GetFullPathNameW(w, MAX_PATH, buf, NULL);
    wchar_t *out = NULL;
    if (n > 0 && n < MAX_PATH) {
        out = malloc((n + 1) * sizeof *out);
        if (out) memcpy(out, buf, (n + 1) * sizeof *out);
    } else if (n >= MAX_PATH) {
        out = malloc((n + 1) * sizeof *out);
        if (out && GetFullPathNameW(w, n + 1, out, NULL) - 1 >= n) {
            free(out);
            out = NULL;
        }
    }
    free(w);
    return out;
}

/* A device or pipe ("\\.\pipe\x", "\\.\NUL", "\\?\GLOBALROOT\..."): never a file. */
static int device_path(const wchar_t *p) {
    return wcsncmp(p, L"\\\\.\\", 4) == 0 || _wcsnicmp(p, L"\\\\?\\GLOBALROOT", 14) == 0;
}

/* *w* as UTF-8 that keeps lone surrogates (what os.fsdecode reads back). */
static char *wtf8(const wchar_t *w) {
    int unc = 0;
    if (wcsncmp(w, L"\\\\?\\UNC\\", 8) == 0) {
        w += 8; /* "\\?\UNC\server\share" -> "\\server\share" */
        unc = 1;
    } else if (wcsncmp(w, L"\\\\?\\", 4) == 0 && w[4] && w[5] == L':') {
        w += 4; /* "\\?\C:\x" -> "C:\x" */
    }
    size_t n = wcslen(w);
    char *out = malloc(3 * n + 5);
    if (!out) return NULL;
    char *o = out;
    if (unc) {
        *o++ = '\\';
        *o++ = '\\';
    }
    for (size_t i = 0; i < n; i++) {
        uint32_t c = w[i];
        if (c >= 0xD800 && c <= 0xDBFF && i + 1 < n && w[i + 1] >= 0xDC00 && w[i + 1] <= 0xDFFF) {
            c = 0x10000 + ((c - 0xD800) << 10) + (w[++i] - 0xDC00);
        }
        if (c < 0x80) {
            *o++ = (char)c;
        } else if (c < 0x800) {
            *o++ = (char)(0xC0 | (c >> 6));
            *o++ = (char)(0x80 | (c & 0x3F));
        } else if (c < 0x10000) {
            *o++ = (char)(0xE0 | (c >> 12));
            *o++ = (char)(0x80 | ((c >> 6) & 0x3F));
            *o++ = (char)(0x80 | (c & 0x3F));
        } else {
            *o++ = (char)(0xF0 | (c >> 18));
            *o++ = (char)(0x80 | ((c >> 12) & 0x3F));
            *o++ = (char)(0x80 | ((c >> 6) & 0x3F));
            *o++ = (char)(0x80 | (c & 0x3F));
        }
    }
    *o = '\0';
    return out;
}

/* Record *kind* for the open of *raw*, whose absolute form is *full* (NULL
 * when it could not be made); frees *full*. */
static void note_w(wchar_t *full, const void *raw, int wide, UINT cp, int kind) {
    if (kind == 0 || !raw) {
        free(full);
        return;
    }
    char *path;
    if (full) {
        if (device_path(full)) {
            free(full);
            return;
        }
        path = wtf8(full);
        free(full);
    } else {
        wchar_t *w = widen(raw, wide, cp);
        if (!w || !*w || device_path(w)) {
            free(w);
            return;
        }
        path = wtf8(w);
        free(w);
        kind = K_UNRESOLVED;
    }
    if (path) record(kind, path);
}

static DWORD attributes(const wchar_t *full) { return full ? GetFileAttributesW(full) : INVALID_FILE_ATTRIBUTES; }

static int is_dir(const wchar_t *full) {
    DWORD a = attributes(full);
    return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY);
}

/* -- CreateFile ------------------------------------------------------------- */

#define ACCESS_READ (GENERIC_READ | GENERIC_ALL | FILE_READ_DATA | MAXIMUM_ALLOWED)
#define ACCESS_WRITE (GENERIC_WRITE | GENERIC_ALL | FILE_WRITE_DATA | FILE_APPEND_DATA)

/* What a CreateFile with *access* and *disposition* that returned *h* (last
 * error *le*) means. */
static int classify_access(DWORD access, DWORD disposition, DWORD flags, HANDLE h, DWORD le, const wchar_t *full) {
    int rd = (access & ACCESS_READ) != 0, wr = (access & ACCESS_WRITE) != 0;
    int creates = disposition == CREATE_NEW || disposition == CREATE_ALWAYS || disposition == OPEN_ALWAYS;
    if (flags & FILE_FLAG_DELETE_ON_CLOSE) return 0; /* a scratch file */
    if (h == INVALID_HANDLE_VALUE) {
        if (le == ERROR_FILE_NOT_FOUND || le == ERROR_PATH_NOT_FOUND) return creates ? 0 : K_ABSENT;
        if (!(rd && !wr)) return 0;
        return is_dir(full) ? 0 : K_READ;
    }
    if (GetFileType(h) != FILE_TYPE_DISK) return 0; /* a pipe, a console */
    int kind;
    if (!wr) {
        kind = rd ? K_READ : 0; /* attributes only: a stat, not a read */
    } else if (disposition == CREATE_ALWAYS || disposition == TRUNCATE_EXISTING || disposition == CREATE_NEW) {
        kind = K_CREATED;
    } else if (disposition == OPEN_ALWAYS && le != ERROR_ALREADY_EXISTS) {
        kind = K_CREATED;
    } else {
        kind = rd ? K_READ : K_WRITE;
    }
    if (kind == K_READ) {
        /* A directory opened as a handle (FILE_FLAG_BACKUP_SEMANTICS) is not
         * a read of it. */
        BY_HANDLE_FILE_INFORMATION bhi;
        if (GetFileInformationByHandle(h, &bhi) ? (bhi.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0
                                                : is_dir(full))
            kind = 0;
    }
    return kind;
}

static HANDLE create_common(int sym, const void *path, DWORD access, DWORD share, LPSECURITY_ATTRIBUTES sa,
                            DWORD disposition, DWORD flags, HANDLE tmpl, void *ext) {
    HANDLE h;
#define CALL_REAL()                                                                                             \
    (sym == S_CREATEFILEW   ? ((createfilew_t)real_fn[sym])(path, access, share, sa, disposition, flags, tmpl) \
     : sym == S_CREATEFILEA ? ((createfilea_t)real_fn[sym])(path, access, share, sa, disposition, flags, tmpl) \
                            : ((createfile2_t)real_fn[sym])(path, access, share, disposition, ext))
    if (!ARMED()) return CALL_REAL();
    in_hook = 1;
    int saved_errno = errno;
    h = CALL_REAL();
#undef CALL_REAL
    DWORD le = GetLastError();
    COUNTER_INC(&st_calls);
    int wide = sym != S_CREATEFILEA;
    UINT cp = wide ? 0 : narrow_cp(P_KERNEL);
    if (sym == S_CREATEFILE2 && ext) flags = ((CREATEFILE2_EXTENDED_PARAMETERS *)ext)->dwFileFlags;
    wchar_t *full = full_path(path, wide, cp);
    note_w(full, path, wide, cp, classify_access(access, disposition, flags, h, le, full));
    errno = saved_errno;
    in_hook = 0;
    SetLastError(le);
    return h;
}

static HANDLE WINAPI h_CreateFileW(LPCWSTR p, DWORD a, DWORD s, LPSECURITY_ATTRIBUTES sa, DWORD d, DWORD f, HANDLE t) {
    return create_common(S_CREATEFILEW, p, a, s, sa, d, f, t, NULL);
}
static HANDLE WINAPI h_CreateFileA(LPCSTR p, DWORD a, DWORD s, LPSECURITY_ATTRIBUTES sa, DWORD d, DWORD f, HANDLE t) {
    return create_common(S_CREATEFILEA, p, a, s, sa, d, f, t, NULL);
}
static HANDLE WINAPI h_CreateFile2(LPCWSTR p, DWORD a, DWORD s, DWORD d, void *ext) {
    return create_common(S_CREATEFILE2, p, a, s, NULL, d, 0, NULL, ext);
}

/* -- the C library ---------------------------------------------------------- */

static int crt_errno(int family) {
    if (family == P_MSVCRT) return msvcrt_errno ? *msvcrt_errno() : 0;
    return errno;
}

/* As classify_flags on Linux, with the CRT's _O_ flags. */
static int classify_oflag(int oflag, int ok, int err, int before, const wchar_t *full) {
    if (oflag & _O_TEMPORARY) return 0; /* deleted on close: a scratch file */
    int acc = oflag & (_O_RDONLY | _O_WRONLY | _O_RDWR);
    if (!ok) {
        if (err == ENOENT) return (oflag & _O_CREAT) ? 0 : K_ABSENT;
        if (acc != _O_RDONLY || is_dir(full)) return 0; /* a folder fails with EACCES */
        return K_READ;
    }
    if (acc == _O_RDONLY) return K_READ;
    if ((oflag & _O_TRUNC) || ((oflag & _O_CREAT) && (oflag & _O_EXCL))) return K_CREATED;
    if (oflag & _O_CREAT) {
        if (!before) return K_CREATED;
        return acc == _O_RDWR ? K_READ : K_WRITE;
    }
    return acc == _O_RDWR ? K_READ : K_WRITE;
}

/* The first letter of an fopen mode ('r', 'w', 'a') and whether it has '+'. */
static int mode_letter(const void *mode, int wide, int *plus) {
    *plus = 0;
    if (!mode) return 0;
    int first = 0;
    for (size_t i = 0;; i++) {
        int c = wide ? ((const wchar_t *)mode)[i] : ((const unsigned char *)mode)[i];
        if (c == 0 || c == ',') break; /* ",ccs=UTF-8" */
        if (c == ' ' && !first) continue;
        if (!first) first = c;
        if (c == '+') *plus = 1;
    }
    return first;
}

static int classify_fmode(int letter, int plus, int ok, int err, int before, const wchar_t *full) {
    switch (letter) {
    case 'r':
        if (!ok) {
            if (err == ENOENT) return K_ABSENT;
            return is_dir(full) ? 0 : K_READ;
        }
        return K_READ;
    case 'w':
        return ok ? K_CREATED : 0;
    case 'a':
        if (!ok) return 0;
        if (!before) return K_CREATED;
        return plus ? K_READ : K_WRITE;
    default:
        return 0;
    }
}

/* _open, _wopen, _sopen, _wsopen, _sopen_s, _wsopen_s. For the _s pair the
 * result is the errno_t and the descriptor goes to *pfh*. */
static int crt_open_common(int sym, int *pfh, const void *path, int oflag, int shflag, int pmode) {
    int c = crt_of(sym), family = family_of(sym), wide = c & 1;
#define CALL_REAL()                                                                     \
    (c == C_OPEN || c == C_WOPEN     ? ((crt_open_t)real_fn[sym])(path, oflag, pmode)    \
     : c == C_SOPEN || c == C_WSOPEN ? ((crt_sopen_t)real_fn[sym])(path, oflag, shflag, pmode) \
                                     : (int)((crt_sopen_s_t)real_fn[sym])(pfh, path, oflag, shflag, pmode))
    if (!ARMED()) return CALL_REAL();
    in_hook = 1;
    int saved_errno = errno;
    UINT cp = wide ? 0 : narrow_cp(family);
    wchar_t *full = full_path(path, wide, cp);
    int before = ((oflag & _O_CREAT) && !(oflag & (_O_TRUNC | _O_EXCL))) ? attributes(full) != INVALID_FILE_ATTRIBUTES : -1;
    errno = saved_errno;
    int r = CALL_REAL();
#undef CALL_REAL
    DWORD le = GetLastError();
    int err = crt_errno(family);
    int ok, why;
    if (c == C_SOPEN_S || c == C_WSOPEN_S) {
        ok = r == 0;
        why = r;
    } else {
        ok = r >= 0;
        why = err;
    }
    if (family == P_UCRT) saved_errno = err;
    COUNTER_INC(&st_calls);
    note_w(full, path, wide, cp, classify_oflag(oflag, ok, why, before, full));
    errno = saved_errno;
    in_hook = 0;
    SetLastError(le);
    return r;
}

/* fopen, _wfopen, fopen_s, _wfopen_s, _fsopen, _wfsopen, freopen, _wfreopen.
 * For the _s pair the result is the errno_t (in *ret*) and the FILE* goes to
 * *pf*. */
static FILE *crt_fopen_common(int sym, FILE **pf, errno_t *ret, const void *path, const void *mode, int shflag,
                              FILE *stream) {
    int c = crt_of(sym), family = family_of(sym), wide = c & 1;
    FILE *f = NULL;
#define CALL_REAL()                                                                  \
    do {                                                                             \
        if (c == C_FOPEN_S || c == C_WFOPEN_S) {                                     \
            *ret = ((crt_fopen_s_t)real_fn[sym])(pf, path, mode);                    \
        } else if (c == C_FSOPEN || c == C_WFSOPEN) {                                \
            f = ((crt_fsopen_t)real_fn[sym])(path, mode, shflag);                    \
        } else if (c == C_FREOPEN || c == C_WFREOPEN) {                              \
            f = ((crt_freopen_t)real_fn[sym])(path, mode, stream);                   \
        } else {                                                                     \
            f = ((crt_fopen_t)real_fn[sym])(path, mode);                             \
        }                                                                            \
    } while (0)
    if (!ARMED()) {
        CALL_REAL();
        return f;
    }
    in_hook = 1;
    int saved_errno = errno;
    UINT cp = wide ? 0 : narrow_cp(family);
    int plus, letter = mode_letter(mode, wide, &plus);
    wchar_t *full = path ? full_path(path, wide, cp) : NULL;
    int before = letter == 'a' ? attributes(full) != INVALID_FILE_ATTRIBUTES : -1;
    errno = saved_errno;
    CALL_REAL();
#undef CALL_REAL
    DWORD le = GetLastError();
    int err = crt_errno(family);
    int ok, why;
    if (c == C_FOPEN_S || c == C_WFOPEN_S) {
        ok = *ret == 0;
        why = *ret;
    } else {
        ok = f != NULL;
        why = err;
    }
    if (family == P_UCRT) saved_errno = err;
    COUNTER_INC(&st_calls);
    note_w(full, path, wide, cp, classify_fmode(letter, plus, ok, why, before, full));
    errno = saved_errno;
    in_hook = 0;
    SetLastError(le);
    return f;
}

/* One set of CRT hooks per C library (*B*: its first sym). The optional
 * permission argument is read only when _O_CREAT says it was passed. */
#define PMODE(last, oflag)                 \
    int pm = 0;                            \
    if ((oflag) & _O_CREAT) {              \
        va_list ap;                        \
        va_start(ap, last);                \
        pm = va_arg(ap, int);              \
        va_end(ap);                        \
    }

#define CRT_HOOKS(F, B)                                                                                          \
    static int __cdecl F##_open(const char *p, int fl, ...) {                                                   \
        PMODE(fl, fl);                                                                                           \
        return crt_open_common(B + C_OPEN, NULL, p, fl, _SH_DENYNO, pm);                                         \
    }                                                                                                            \
    static int __cdecl F##_wopen(const wchar_t *p, int fl, ...) {                                               \
        PMODE(fl, fl);                                                                                           \
        return crt_open_common(B + C_WOPEN, NULL, p, fl, _SH_DENYNO, pm);                                        \
    }                                                                                                            \
    static int __cdecl F##_sopen(const char *p, int fl, int sh, ...) {                                          \
        PMODE(sh, fl);                                                                                           \
        return crt_open_common(B + C_SOPEN, NULL, p, fl, sh, pm);                                                \
    }                                                                                                            \
    static int __cdecl F##_wsopen(const wchar_t *p, int fl, int sh, ...) {                                      \
        PMODE(sh, fl);                                                                                           \
        return crt_open_common(B + C_WSOPEN, NULL, p, fl, sh, pm);                                               \
    }                                                                                                            \
    static errno_t __cdecl F##_sopen_s(int *pfh, const char *p, int fl, int sh, int pm) {                       \
        return crt_open_common(B + C_SOPEN_S, pfh, p, fl, sh, pm);                                               \
    }                                                                                                            \
    static errno_t __cdecl F##_wsopen_s(int *pfh, const wchar_t *p, int fl, int sh, int pm) {                   \
        return crt_open_common(B + C_WSOPEN_S, pfh, p, fl, sh, pm);                                              \
    }                                                                                                            \
    static FILE *__cdecl F##_fopen(const char *p, const char *m) {                                              \
        return crt_fopen_common(B + C_FOPEN, NULL, NULL, p, m, 0, NULL);                                         \
    }                                                                                                            \
    static FILE *__cdecl F##_wfopen(const wchar_t *p, const wchar_t *m) {                                       \
        return crt_fopen_common(B + C_WFOPEN, NULL, NULL, p, m, 0, NULL);                                        \
    }                                                                                                            \
    static errno_t __cdecl F##_fopen_s(FILE **pf, const char *p, const char *m) {                               \
        errno_t r = 0;                                                                                           \
        crt_fopen_common(B + C_FOPEN_S, pf, &r, p, m, 0, NULL);                                                  \
        return r;                                                                                                \
    }                                                                                                            \
    static errno_t __cdecl F##_wfopen_s(FILE **pf, const wchar_t *p, const wchar_t *m) {                        \
        errno_t r = 0;                                                                                           \
        crt_fopen_common(B + C_WFOPEN_S, pf, &r, p, m, 0, NULL);                                                 \
        return r;                                                                                                \
    }                                                                                                            \
    static FILE *__cdecl F##_fsopen(const char *p, const char *m, int sh) {                                     \
        return crt_fopen_common(B + C_FSOPEN, NULL, NULL, p, m, sh, NULL);                                       \
    }                                                                                                            \
    static FILE *__cdecl F##_wfsopen(const wchar_t *p, const wchar_t *m, int sh) {                              \
        return crt_fopen_common(B + C_WFSOPEN, NULL, NULL, p, m, sh, NULL);                                      \
    }                                                                                                            \
    static FILE *__cdecl F##_freopen(const char *p, const char *m, FILE *s) {                                   \
        return crt_fopen_common(B + C_FREOPEN, NULL, NULL, p, m, 0, s);                                          \
    }                                                                                                            \
    static FILE *__cdecl F##_wfreopen(const wchar_t *p, const wchar_t *m, FILE *s) {                            \
        return crt_fopen_common(B + C_WFREOPEN, NULL, NULL, p, m, 0, s);                                         \
    }                                                                                                            \
    static void F##_register(void) {                                                                            \
        hook_fn[B + C_OPEN] = (void *)F##_open;                                                                  \
        hook_fn[B + C_WOPEN] = (void *)F##_wopen;                                                                \
        hook_fn[B + C_SOPEN] = (void *)F##_sopen;                                                                \
        hook_fn[B + C_WSOPEN] = (void *)F##_wsopen;                                                              \
        hook_fn[B + C_SOPEN_S] = (void *)F##_sopen_s;                                                            \
        hook_fn[B + C_WSOPEN_S] = (void *)F##_wsopen_s;                                                          \
        hook_fn[B + C_FOPEN] = (void *)F##_fopen;                                                                \
        hook_fn[B + C_WFOPEN] = (void *)F##_wfopen;                                                              \
        hook_fn[B + C_FOPEN_S] = (void *)F##_fopen_s;                                                            \
        hook_fn[B + C_WFOPEN_S] = (void *)F##_wfopen_s;                                                          \
        hook_fn[B + C_FSOPEN] = (void *)F##_fsopen;                                                              \
        hook_fn[B + C_WFSOPEN] = (void *)F##_wfsopen;                                                            \
        hook_fn[B + C_FREOPEN] = (void *)F##_freopen;                                                            \
        hook_fn[B + C_WFREOPEN] = (void *)F##_wfreopen;                                                          \
    }

CRT_HOOKS(h_ucrt, S_UCRT)
CRT_HOOKS(h_msvcrt, S_MSVCRT)

/* -- IAT patching ----------------------------------------------------------- */

static int addr_cmp(const void *a, const void *b) {
    uintptr_t x = ((const fh_addr *)a)->addr, y = ((const fh_addr *)b)->addr;
    return x < y ? -1 : x > y;
}

/* The sym whose real function *p* is, or -1. */
static int sym_of_addr(void *p) {
    size_t lo = 0, hi = n_addrs;
    uintptr_t a = (uintptr_t)p;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (addrs[mid].addr < a) lo = mid + 1;
        else hi = mid;
    }
    return lo < n_addrs && addrs[lo].addr == a ? addrs[lo].sym : -1;
}

/* Look up the real functions of the modules loaded so far (msvcrt.dll may
 * come later than the first scan). */
static void resolve_reals(void) {
    HMODULE kb = GetModuleHandleW(L"kernelbase.dll"), k32 = GetModuleHandleW(L"kernel32.dll");
    HMODULE ucrt = GetModuleHandleW(L"ucrtbase.dll"), ms = GetModuleHandleW(L"msvcrt.dll");
    for (int s = 0; s < S_COUNT; s++) {
        if (real_fn[s]) continue;
        int fam = family_of(s);
        const char *n = name_of(s);
        if (fam == P_KERNEL) {
            void *a = kb ? (void *)GetProcAddress(kb, n) : NULL;
            void *b = k32 ? (void *)GetProcAddress(k32, n) : NULL;
            real_fn[s] = a ? a : b;
            alt_fn[s] = (a && b && a != b) ? b : NULL;
        } else {
            HMODULE m = fam == P_UCRT ? ucrt : ms;
            real_fn[s] = m ? (void *)GetProcAddress(m, n) : NULL;
        }
    }
    if (ms && !msvcrt_errno) msvcrt_errno = (crt_errno_t)(void *)GetProcAddress(ms, "_errno");
    n_addrs = 0;
    for (int s = 0; s < S_COUNT; s++) {
        if (real_fn[s]) addrs[n_addrs++] = (fh_addr){(uintptr_t)real_fn[s], s};
        if (alt_fn[s]) addrs[n_addrs++] = (fh_addr){(uintptr_t)alt_fn[s], s};
    }
    qsort(addrs, n_addrs, sizeof *addrs, addr_cmp);
    /* Two names for one function (an alias export): keep the first. */
    size_t k = 0;
    for (size_t i = 0; i < n_addrs; i++)
        if (k == 0 || addrs[k - 1].addr != addrs[i].addr) addrs[k++] = addrs[i];
    n_addrs = k;
}

static const IMAGE_NT_HEADERS64 *nt_headers(uintptr_t base) {
    const IMAGE_DOS_HEADER *dos = (const IMAGE_DOS_HEADER *)base;
    if (dos->e_magic != IMAGE_DOS_SIGNATURE) return NULL;
    const IMAGE_NT_HEADERS64 *nt = (const IMAGE_NT_HEADERS64 *)(base + (uintptr_t)dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE || nt->OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC) return NULL;
    return nt;
}

static int write_slot(void **slot, void *value) {
    MEMORY_BASIC_INFORMATION mbi;
    if (!VirtualQuery(slot, &mbi, sizeof mbi)) return -1;
    DWORD writable = PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
    if (mbi.Protect & writable) {
        InterlockedExchangePointer(slot, value);
        return 0;
    }
    DWORD old, tmp;
    if (!VirtualProtect(slot, sizeof *slot, PAGE_READWRITE, &old)) return -1;
    InterlockedExchangePointer(slot, value);
    VirtualProtect(slot, sizeof *slot, old, &tmp);
    return 0;
}

/* The C library a module imports from, by the DLL's name; -1 for any other. */
static int provider_of(const char *dll) {
    if (_strnicmp(dll, "kernel32", 8) == 0 || _strnicmp(dll, "kernelbase", 10) == 0 ||
        _strnicmp(dll, "api-ms-win-core-file-", 21) == 0)
        return P_KERNEL;
    if (_strnicmp(dll, "ucrtbase", 8) == 0 || _strnicmp(dll, "api-ms-win-crt-", 15) == 0) return P_UCRT;
    if (_stricmp(dll, "msvcrt.dll") == 0) return P_MSVCRT;
    return -1;
}

static int named_sym(int family, const char *name) {
    for (int s = 0; s < S_COUNT; s++)
        if (family_of(s) == family && strcmp(name, name_of(s)) == 0) return s;
    return -1;
}

/* Rebind every slot of one import table that holds a real function. *names*
 * (the import name table, or NULL) only serves to count slots bound to
 * something else. */
static void patch_iat(uintptr_t base, fh_object *obj, IMAGE_THUNK_DATA64 *iat, const IMAGE_THUNK_DATA64 *names,
                      int family) {
    for (size_t i = 0; iat[i].u1.Function; i++) {
        void **slot = (void **)&iat[i].u1.Function;
        void *cur = *(void *volatile *)slot;
        int sym = sym_of_addr(cur);
        if (sym < 0) {
            if ((uintptr_t)cur >= own_base && (uintptr_t)cur < own_end) continue; /* ours already */
            if (family >= 0 && names && !IMAGE_SNAP_BY_ORDINAL64(names[i].u1.Ordinal)) {
                const IMAGE_IMPORT_BY_NAME *ibn = (const IMAGE_IMPORT_BY_NAME *)(base + (uintptr_t)names[i].u1.AddressOfData);
                if (named_sym(family, (const char *)ibn->Name) >= 0) foreign_slots++;
            }
            continue;
        }
        if (grow((void **)&patches, &cap_patches, n_patches + 1, sizeof *patches) < 0) return;
        if (write_slot(slot, hook_fn[sym]) < 0) continue;
        patches[n_patches++] = (fh_patch){slot, cur, sym, base};
        obj->patched++;
    }
}

static void hook_module(uintptr_t base, fh_object *obj) {
    const IMAGE_NT_HEADERS64 *nt = nt_headers(base);
    if (!nt) return;
    const IMAGE_DATA_DIRECTORY *dir = &nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (dir->VirtualAddress && dir->Size) {
        const IMAGE_IMPORT_DESCRIPTOR *d = (const IMAGE_IMPORT_DESCRIPTOR *)(base + dir->VirtualAddress);
        for (; d->Name && d->FirstThunk; d++) {
            const IMAGE_THUNK_DATA64 *names =
                d->OriginalFirstThunk ? (const IMAGE_THUNK_DATA64 *)(base + d->OriginalFirstThunk) : NULL;
            patch_iat(base, obj, (IMAGE_THUNK_DATA64 *)(base + d->FirstThunk), names,
                      provider_of((const char *)(base + d->Name)));
        }
    }
    /* Delay-loaded imports: the slots already resolved (an unresolved one
     * holds the module's own loader thunk, which matches nothing). */
    dir = &nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT];
    if (dir->VirtualAddress && dir->Size) {
        const IMAGE_DELAYLOAD_DESCRIPTOR *d = (const IMAGE_DELAYLOAD_DESCRIPTOR *)(base + dir->VirtualAddress);
        for (; d->DllNameRVA && d->ImportAddressTableRVA; d++) {
            if (!d->Attributes.RvaBased) continue; /* the old VA-based form: not on x64 */
            patch_iat(base, obj, (IMAGE_THUNK_DATA64 *)(base + d->ImportAddressTableRVA), NULL, -1);
        }
    }
}

/* Lower-case basename of *path*. */
static const wchar_t *basename_w(const wchar_t *path) {
    const wchar_t *b = wcsrchr(path, L'\\');
    const wchar_t *s = wcsrchr(path, L'/');
    if (s > b) b = s;
    return b ? b + 1 : path;
}

static int skipped_module(uintptr_t base, const wchar_t *path) {
    if (base == own_base) return 1;
    if (base == (uintptr_t)GetModuleHandleW(NULL)) return 1; /* the main program: Python's own opens */
    if (wcsncmp(path, L"\\\\?\\", 4) == 0) path += 4;
    if (windir_len && _wcsnicmp(path, windir, windir_len) == 0) return 1; /* the system's own DLLs */
    /* Defender's in-process scanner (MpOav.dll), which lives outside it. */
    static const wchar_t defender[] = L"\\Microsoft\\Windows Defender\\";
    for (const wchar_t *q = path; (q = wcschr(q, L'\\')) != NULL; q++)
        if (_wcsnicmp(q, defender, wcslen(defender)) == 0) return 1;
    const wchar_t *b = basename_w(path);
    static const wchar_t *const prefixes[] = {L"python3", L"ucrtbase", L"vcruntime", L"api-ms-win-", L"ext-ms-",
                                              L"msvcrt.", L"kernel32.", L"kernelbase.",
                                              /* OpenSSL: CA bundles and openssl.cnf, machine state */
                                              L"libcrypto", L"libssl"};
    for (size_t i = 0; i < sizeof prefixes / sizeof prefixes[0]; i++)
        if (_wcsnicmp(b, prefixes[i], wcslen(prefixes[i])) == 0) return 1;
    return 0;
}

static fh_object *find_object(uintptr_t base, DWORD size, DWORD stamp) {
    for (size_t i = 0; i < n_objects; i++)
        if (objects[i].base == base && objects[i].size == size && objects[i].stamp == stamp) return &objects[i];
    return NULL;
}

/* Is the module *o* describes still loaded, the same one? */
static int still_loaded(const fh_object *o) {
    HMODULE h;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            (LPCWSTR)o->base, &h) ||
        (uintptr_t)h != o->base)
        return 0;
    const IMAGE_NT_HEADERS64 *nt = nt_headers(o->base);
    return nt && nt->OptionalHeader.SizeOfImage == o->size && nt->FileHeader.TimeDateStamp == o->stamp;
}

static void drop_unloaded_objects(void) {
    size_t k = 0;
    for (size_t i = 0; i < n_objects; i++) {
        if (still_loaded(&objects[i])) {
            objects[k++] = objects[i];
        } else {
            free(objects[i].name);
            size_t m = 0; /* its patches went with it */
            for (size_t j = 0; j < n_patches; j++)
                if (patches[j].base != objects[i].base) patches[m++] = patches[j];
            n_patches = m;
        }
    }
    n_objects = k;
}

/* The loaded modules, malloc'd; *n* of them. */
static HMODULE *list_modules(size_t *n, DWORD *bytes) {
    DWORD cap = 512 * sizeof(HMODULE), need = 0;
    for (int tries = 0; tries < 4; tries++) {
        HMODULE *mods = malloc(cap);
        if (!mods) return NULL;
        if (!EnumProcessModules(GetCurrentProcess(), mods, cap, &need)) {
            free(mods);
            return NULL;
        }
        if (need <= cap) {
            *n = need / sizeof(HMODULE);
            if (bytes) *bytes = need;
            return mods;
        }
        free(mods);
        cap = need + 64 * sizeof(HMODULE);
    }
    return NULL;
}

/* Scan for modules loaded since the last scan. Call with hook_mutex held. */
static long rescan_locked(int force) {
    LONG lg = load_gen, ug = unload_gen;
    if (!force && notify_ok && lg == seen_load && ug == seen_unload) return 0;
    size_t n = 0;
    DWORD bytes = 0;
    HMODULE *mods = list_modules(&n, &bytes);
    if (!mods) return 0;
    if (!force && !notify_ok && bytes == seen_module_bytes) {
        free(mods);
        return 0;
    }
    size_t before = n_patches;
    if (ug != seen_unload || !notify_ok) drop_unloaded_objects();
    /* Seen before scanning: a module loaded meanwhile bumps the counter again. */
    seen_load = lg;
    seen_unload = ug;
    seen_module_bytes = bytes;
    resolve_reals();
    foreign_slots = 0;
    wchar_t *path = malloc(32768 * sizeof *path);
    for (size_t i = 0; path && i < n; i++) {
        HMODULE pin; /* held while it is read, so it cannot unload under us */
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, (LPCWSTR)mods[i], &pin)) continue;
        if (pin != mods[i]) {
            FreeLibrary(pin);
            continue;
        }
        uintptr_t base = (uintptr_t)mods[i];
        DWORD len = GetModuleFileNameW(mods[i], path, 32768);
        const IMAGE_NT_HEADERS64 *nt = nt_headers(base);
        if (len > 0 && len < 32768 && nt && !skipped_module(base, path)) {
            DWORD size = nt->OptionalHeader.SizeOfImage, stamp = nt->FileHeader.TimeDateStamp;
            fh_object *obj = find_object(base, size, stamp);
            int known = obj != NULL;
            if (!obj && grow((void **)&objects, &cap_objects, n_objects + 1, sizeof *objects) == 0) {
                obj = &objects[n_objects++];
                *obj = (fh_object){base, size, stamp, wtf8(path), 0};
            }
            if (obj && !(known && scan_only_new)) hook_module(base, obj); /* also re-binds a slot something reset */
        }
        FreeLibrary(pin);
    }
    free(path);
    free(mods);
    return (long)(n_patches - before);
}

static void unhook_locked(void) {
    for (size_t i = 0; i < n_patches; i++) {
        HMODULE pin;
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, (LPCWSTR)patches[i].slot, &pin)) continue;
        if ((uintptr_t)pin == patches[i].base && *(void *volatile *)patches[i].slot == hook_fn[patches[i].sym])
            write_slot(patches[i].slot, patches[i].old);
        FreeLibrary(pin);
    }
    n_patches = 0;
    for (size_t i = 0; i < n_objects; i++) free(objects[i].name);
    n_objects = 0;
    seen_load = seen_unload = -1;
    seen_module_bytes = 0;
    foreign_slots = 0;
}

/* LdrRegisterDllNotification (ntdll; no import library has it). */
typedef VOID(CALLBACK *ldr_notify_fn)(ULONG reason, const void *data, PVOID ctx);
typedef LONG(NTAPI *ldr_register_t)(ULONG flags, ldr_notify_fn fn, PVOID ctx, PVOID *cookie);

static VOID CALLBACK on_dll(ULONG reason, const void *data, PVOID ctx) {
    (void)data;
    (void)ctx;
    /* Under the loader lock: count, nothing more. */
    if (reason == 1) InterlockedIncrement(&load_gen);
    else InterlockedIncrement(&unload_gen);
}

/* Find the real functions and this module; call with hook_mutex held. */
static void prepare_locked(void) {
    if (!own_base) {
        HMODULE me;
        if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               (LPCWSTR)(void *)prepare_locked, &me)) {
            own_base = (uintptr_t)me;
            const IMAGE_NT_HEADERS64 *nt = nt_headers(own_base);
            own_end = own_base + (nt ? nt->OptionalHeader.SizeOfImage : 0);
        }
    }
    if (!windir_len) {
        UINT n = GetSystemWindowsDirectoryW(windir, MAX_PATH);
        if (n > 0 && n < MAX_PATH) {
            if (windir[n - 1] != L'\\') windir[n++] = L'\\';
            windir[n] = L'\0';
            windir_len = n;
        }
    }
    if (!notify_cookie) {
        ldr_register_t reg = (ldr_register_t)(void *)GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "LdrRegisterDllNotification");
        notify_ok = reg && reg(0, on_dll, NULL, &notify_cookie) == 0 && notify_cookie;
    }
    resolve_reals();
}

static void init_hooks(void) {
    hook_fn[S_CREATEFILEW] = (void *)h_CreateFileW;
    hook_fn[S_CREATEFILEA] = (void *)h_CreateFileA;
    hook_fn[S_CREATEFILE2] = (void *)h_CreateFile2;
    h_ucrt_register();
    h_msvcrt_register();
}

#endif /* FH_WINDOWS */

/* ---- Python API -------------------------------------------------------- */

static PyObject *fh_supported(PyObject *self, PyObject *unused) {
    (void)self; (void)unused;
    return Py_BuildValue("(Os)", Py_True, plat_how);
}

static PyObject *fh_enable(PyObject *self, PyObject *unused) {
    (void)self; (void)unused;
    long added;
    Py_BEGIN_ALLOW_THREADS
    fh_lock(&hook_mutex);
    prepare_locked();
    enabled = 1;
    added = rescan_locked(1);
    fh_unlock(&hook_mutex);
    Py_END_ALLOW_THREADS
    return PyLong_FromLong(added);
}

static PyObject *fh_disable(PyObject *self, PyObject *unused) {
    (void)self; (void)unused;
    Py_BEGIN_ALLOW_THREADS
    fh_lock(&hook_mutex);
    unhook_locked();
    enabled = 0;
    fh_unlock(&hook_mutex);
    in_hook = 1;
    fh_lock(&ev_mutex);
    n_windows = 0;
    newest_start = 0;
    FLAG_SET(&armed, 0);
    clear_events_locked();
    fh_unlock(&ev_mutex);
    in_hook = 0;
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

static PyObject *fh_rescan(PyObject *self, PyObject *unused) {
    (void)self; (void)unused;
    long added = 0;
    Py_BEGIN_ALLOW_THREADS
    fh_lock(&hook_mutex);
    if (enabled) added = rescan_locked(0);
    fh_unlock(&hook_mutex);
    Py_END_ALLOW_THREADS
    return PyLong_FromLong(added);
}

/* Hook only the objects loaded since the last scan, leaving the slots of the
   ones already hooked unread: called after each extension module an import
   inside a tracked block loads, where a full pass per module (pandas loads
   dozens) cost a tenth of a second and more. */
static PyObject *fh_hook_new(PyObject *self, PyObject *unused) {
    (void)self; (void)unused;
    long added = 0;
    Py_BEGIN_ALLOW_THREADS
    fh_lock(&hook_mutex);
    if (enabled) {
        scan_only_new = 1;
        added = rescan_locked(0);
        scan_only_new = 0;
    }
    fh_unlock(&hook_mutex);
    Py_END_ALLOW_THREADS
    return PyLong_FromLong(added);
}

static PyObject *fh_window_open(PyObject *self, PyObject *unused) {
    (void)self; (void)unused;
    uint64_t start = 0;
    int ok = 1;
    Py_BEGIN_ALLOW_THREADS
    fh_lock(&hook_mutex);
    if (enabled) rescan_locked(0);
    fh_unlock(&hook_mutex);
    in_hook = 1;
    fh_lock(&ev_mutex);
    if (grow((void **)&windows, &cap_windows, n_windows + 1, sizeof *windows) < 0) {
        ok = 0;
    } else {
        start = next_seq;
        windows[n_windows++] = start;
        newest_start = start;
        FLAG_SET(&armed, 1);
    }
    fh_unlock(&ev_mutex);
    in_hook = 0;
    Py_END_ALLOW_THREADS
    if (!ok) return PyErr_NoMemory();
    return PyLong_FromUnsignedLongLong(start);
}

typedef struct { int kind; char *path; } fh_out;

/* Close the window opened at *start*: ``(events, lost)``, events being
 * ``[(kind, path bytes)]`` in the order they happened. */
static PyObject *fh_window_close(PyObject *self, PyObject *arg) {
    (void)self;
    unsigned long long start = PyLong_AsUnsignedLongLong(arg);
    if (start == (unsigned long long)-1 && PyErr_Occurred()) return NULL;
    fh_out *out = NULL;
    size_t n_out = 0;
    int lost = 0, oom = 0;
    Py_BEGIN_ALLOW_THREADS
    in_hook = 1;
    fh_lock(&ev_mutex);
    for (size_t i = 0; i < n_windows; i++) {
        if (windows[i] == start) {
            windows[i] = windows[--n_windows];
            break;
        }
    }
    lost = last_lost_seq >= start;
    size_t first = n_events;
    for (size_t i = 0; i < n_events; i++)
        if (events[i].seq >= start) { first = i; break; }
    if (first < n_events) {
        out = malloc((n_events - first) * sizeof *out);
        if (!out) oom = 1;
        for (size_t i = first; out && i < n_events; i++) {
            out[n_out].kind = events[i].kind;
            out[n_out].path = strdup(events[i].path);
            if (!out[n_out].path) { oom = 1; break; }
            n_out++;
        }
    }
    /* Keep only what an open window still needs. */
    uint64_t oldest = UINT64_MAX;
    newest_start = 0;
    for (size_t i = 0; i < n_windows; i++) {
        if (windows[i] < oldest) oldest = windows[i];
        if (windows[i] > newest_start) newest_start = windows[i];
    }
    size_t drop = 0;
    while (drop < n_events && events[drop].seq < oldest) free(events[drop++].path);
    if (drop) {
        memmove(events, events + drop, (n_events - drop) * sizeof *events);
        n_events -= drop;
        index_rebuild();
    }
    if (!n_windows) FLAG_SET(&armed, 0);
    fh_unlock(&ev_mutex);
    in_hook = 0;
    Py_END_ALLOW_THREADS
    PyObject *list = oom ? NULL : PyList_New((Py_ssize_t)n_out);
    for (size_t i = 0; i < n_out; i++) {
        if (list) {
            PyObject *t = Py_BuildValue("(iy)", out[i].kind, out[i].path);
            if (!t) Py_CLEAR(list);
            else PyList_SET_ITEM(list, (Py_ssize_t)i, t);
        }
        free(out[i].path);
    }
    free(out);
    if (oom && !PyErr_Occurred()) return PyErr_NoMemory();
    if (!list) return NULL;
    return Py_BuildValue("(NO)", list, lost ? Py_True : Py_False);
}

static PyObject *fh_stats(PyObject *self, PyObject *unused) {
    (void)self; (void)unused;
    PyObject *libs = PyList_New(0);
    if (!libs) return NULL;
    size_t slots, foreign, nobj;
    int en;
    fh_lock(&hook_mutex);
    en = enabled;
    slots = n_patches;
    foreign = foreign_slots;
    nobj = n_objects;
    for (size_t i = 0; i < n_objects; i++) {
        if (!objects[i].patched) continue;
        PyObject *s = PyUnicode_DecodeFSDefault(objects[i].name ? objects[i].name : "");
        if (!s || PyList_Append(libs, s) < 0) {
            Py_XDECREF(s);
            fh_unlock(&hook_mutex);
            Py_DECREF(libs);
            return NULL;
        }
        Py_DECREF(s);
    }
    fh_unlock(&hook_mutex);
    in_hook = 1;
    fh_lock(&ev_mutex);
    unsigned long long rec = st_recorded, dd = st_deduped, lo = st_lost, buf = n_events, win = n_windows;
    fh_unlock(&ev_mutex);
    in_hook = 0;
    unsigned long long calls = (unsigned long long)COUNTER_GET(&st_calls);
    return Py_BuildValue("{s:O,s:n,s:n,s:n,s:N,s:K,s:K,s:K,s:K,s:K,s:K}", "enabled", en ? Py_True : Py_False,
                         "slots", (Py_ssize_t)slots, "foreign_slots", (Py_ssize_t)foreign, "objects_scanned",
                         (Py_ssize_t)nobj, "hooked_objects", libs, "calls_seen", calls, "recorded", rec,
                         "deduped", dd, "lost", lo, "buffered", buf, "windows", win);
}

static PyMethodDef fh_methods[] = {
    {"filehook_supported", fh_supported, METH_NOARGS, "(True, how) or (False, why not)."},
    {"filehook_enable", fh_enable, METH_NOARGS, "Rebind the open family in loaded objects; slots rebound."},
    {"filehook_disable", fh_disable, METH_NOARGS, "Put every rebound slot back and forget all events."},
    {"filehook_rescan", fh_rescan, METH_NOARGS, "Hook objects loaded since the last scan; slots rebound."},
    {"filehook_hook_new", fh_hook_new, METH_NOARGS, "Hook only objects not seen before; slots rebound."},
    {"filehook_window_open", fh_window_open, METH_NOARGS, "Start recording native opens; the window's id."},
    {"filehook_window_close", fh_window_close, METH_O, "(events, lost) of the window; closes it."},
    {"filehook_stats", fh_stats, METH_NOARGS, "Counters and the hooked objects."},
    {NULL, NULL, 0, NULL}};

int cfilehook_add_functions(PyObject *m) {
    init_hooks();
    if (PyModule_AddFunctions(m, fh_methods) < 0) return -1;
    struct { const char *n; int v; } consts[] = {{"FH_READ", K_READ}, {"FH_ABSENT", K_ABSENT},
        {"FH_CREATED", K_CREATED}, {"FH_WRITE", K_WRITE}, {"FH_UNRESOLVED", K_UNRESOLVED}};
    for (size_t i = 0; i < sizeof consts / sizeof consts[0]; i++)
        if (PyModule_AddIntConstant(m, consts[i].n, consts[i].v) < 0) return -1;
    return 0;
}

#else /* !FH_SUPPORTED */

static PyObject *fh_supported(PyObject *self, PyObject *unused) {
    (void)self; (void)unused;
    return Py_BuildValue("(Os)", Py_False,
                         "not supported here (the native file hook is Linux/glibc on x86-64 and aarch64, "
                         "and Windows on x64)");
}

static PyMethodDef fh_methods[] = {
    {"filehook_supported", fh_supported, METH_NOARGS, "(True, how) or (False, why not)."},
    {NULL, NULL, 0, NULL}};

int cfilehook_add_functions(PyObject *m) { return PyModule_AddFunctions(m, fh_methods); }

#endif
