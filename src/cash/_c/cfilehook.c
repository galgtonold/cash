/* cash._cwatch, part 4: the native file-read hook (cfilehook.c).
 *
 * Opt-in (cash.tracking.native_reads, CASH_NATIVE_FILE_HOOK=1). Python's audit
 * hook sees every open made through Python; a C library that opens a file
 * itself (libarrow, libsqlite3, libstdc++'s fstream, a Rust crate) raises no
 * event. This hook rebinds the GOT slots through which already-loaded shared
 * objects call libc's open family, so those opens are seen too:
 *
 *   open open64 openat openat64 __open_2 __open64_2 __openat_2 __openat64_2
 *   creat creat64 fopen fopen64 freopen freopen64 opendir fdopendir
 *
 * Linux/glibc on x86-64 and aarch64 only: the slots are found from each
 * object's dynamic section (JUMP_SLOT, GLOB_DAT and absolute relocations with
 * no addend against an undefined symbol). Elsewhere filehook_supported()
 * answers False and why.
 *
 * Never hooked: the main program and libpython (Python's own opens are the
 * audit hook's), the dynamic loader and libc itself, sanitizer runtimes, and
 * this module. A slot is rebound only when it holds the very function
 * dlsym(RTLD_DEFAULT) returns for the name, or is still unresolved (a lazy
 * PLT slot pointing into its own object); a slot bound to anything else (an
 * interposer) is left alone and counted. dlopen is NOT hooked: glibc's
 * dlopen resolves $ORIGIN/RUNPATH from its caller's address, which a hook
 * would change. Objects loaded later are picked up by filehook_rescan(),
 * which each window open runs (cheap: dl_iterate_phdr's load counter).
 *
 * Recording happens only while at least one window is open (a tracker is
 * running): the hook then calls the real function, saves errno, notes the
 * path (absolute: a relative one is joined to the cwd or to the dirfd's
 * /proc/self/fd link), restores errno and returns the real result. With no
 * window open the hook is one relaxed load and a tail call. Events live in
 * one buffer under a mutex; a window collects every event recorded while it
 * was open, from any thread (a library's own I/O threads have no Python
 * context to attribute them by). The cost of that is over-attribution:
 * concurrent trackers each get every native open made meanwhile, which can
 * only add dependencies, never drop one. A buffer that fills up marks the
 * windows that lost events (the tracker then records a dependency that is
 * never fresh).
 *
 * Not seen: direct syscalls (Go runtimes, io_uring, a library issuing
 * syscall(SYS_openat) itself), libraries linked statically against libc,
 * opens inside libc (getaddrinfo's /etc/hosts, iconv modules), stat/access
 * probes, and opens by a library loaded and used within one window (its
 * constructors and first calls run before the next rescan).
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <Python.h>
#include "cwalk.h"

#if defined(__linux__) && defined(__GLIBC__) && (defined(__x86_64__) || defined(__aarch64__))
#define FH_SUPPORTED 1
#else
#define FH_SUPPORTED 0
#endif

#if FH_SUPPORTED
#include <dirent.h>
#include <dlfcn.h>
#include <elf.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <link.h>
#include <pthread.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

/* Event kinds, as Python sees them (native_reads.py). */
enum { K_READ = 1, K_ABSENT = 2, K_CREATED = 3, K_WRITE = 4, K_UNRESOLVED = 5 };

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

/* ---- event buffer ------------------------------------------------------ */

typedef struct {
    uint64_t seq;
    int kind;
    uint32_t hash;
    char *path;
} fh_event;

#define FH_MAX_EVENTS 65536
static pthread_mutex_t ev_mutex = PTHREAD_MUTEX_INITIALIZER;
static fh_event *events;    /* in seq order */
static size_t n_events, cap_events;
static uint32_t *ev_index;  /* open-addressing set of event indexes + 1, for dedupe */
static size_t index_cap;    /* power of two, >= 2 * cap_events */
static uint64_t next_seq = 1;
static uint64_t last_lost_seq; /* 0: nothing lost */
static uint64_t *windows;      /* start seqs of the open windows */
static size_t n_windows, cap_windows;
static uint64_t newest_start;  /* max of windows[] */
static int armed;              /* n_windows > 0, read without the lock */
static __thread int in_hook;   /* this thread is inside cash's recording */
static uint64_t st_recorded, st_deduped, st_lost, st_calls;

/* ---- hooked slots ------------------------------------------------------ */

typedef struct {
    void **slot;
    void *old;
    int sym;
    uintptr_t base;
} fh_patch;

typedef struct {
    uintptr_t base;
    const void *phdr;
    char *name;
    int patched; /* slots rebound in it */
} fh_object;

static pthread_mutex_t hook_mutex = PTHREAD_MUTEX_INITIALIZER;
static int enabled;
static fh_patch *patches;
static size_t n_patches, cap_patches;
static fh_object *objects;
static size_t n_objects, cap_objects;
static unsigned long long seen_adds = 0, seen_subs = 0;
static uintptr_t own_base;
static size_t foreign_slots;
static long page_size;
static int atfork_done;

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
    pthread_mutex_lock(&ev_mutex);
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
    pthread_mutex_unlock(&ev_mutex);
    return;
lost:
    st_lost++;
    last_lost_seq = next_seq++;
drop:
    pthread_mutex_unlock(&ev_mutex);
    free(path);
}

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

#define ARMED() (__atomic_load_n(&armed, __ATOMIC_RELAXED) && !in_hook)

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
    __atomic_fetch_add(&st_calls, 1, __ATOMIC_RELAXED);
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
    __atomic_fetch_add(&st_calls, 1, __ATOMIC_RELAXED);
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
    __atomic_fetch_add(&st_calls, 1, __ATOMIC_RELAXED);
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
    __atomic_fetch_add(&st_calls, 1, __ATOMIC_RELAXED);
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
        "libSegFault", "libpcprofile"};
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

static void clear_events_locked(void) {
    for (size_t i = 0; i < n_events; i++) free(events[i].path);
    n_events = 0;
    index_rebuild();
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
    __atomic_store_n(&armed, 0, __ATOMIC_RELAXED);
    clear_events_locked();
    pthread_mutex_unlock(&ev_mutex);
    pthread_mutex_unlock(&hook_mutex);
}

/* ---- Python API -------------------------------------------------------- */

static PyObject *fh_supported(PyObject *self, PyObject *unused) {
    (void)self; (void)unused;
    return Py_BuildValue("(Os)", Py_True, "Linux ELF GOT rebinding");
}

static PyObject *fh_enable(PyObject *self, PyObject *unused) {
    (void)self; (void)unused;
    long added;
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&hook_mutex);
    if (!page_size) page_size = sysconf(_SC_PAGESIZE);
    if (!own_base) {
        Dl_info di;
        if (dladdr((void *)fh_enable, &di)) own_base = (uintptr_t)di.dli_fbase;
    }
    if (!atfork_done && pthread_atfork(atfork_prepare, atfork_parent, atfork_child) == 0) atfork_done = 1;
    for (int i = 0; i < S_COUNT; i++) real_fn[i] = dlsym(RTLD_DEFAULT, sym_names[i]);
    enabled = 1;
    added = rescan_locked(1);
    pthread_mutex_unlock(&hook_mutex);
    Py_END_ALLOW_THREADS
    return PyLong_FromLong(added);
}

static PyObject *fh_disable(PyObject *self, PyObject *unused) {
    (void)self; (void)unused;
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&hook_mutex);
    unhook_locked();
    enabled = 0;
    pthread_mutex_unlock(&hook_mutex);
    in_hook = 1;
    pthread_mutex_lock(&ev_mutex);
    n_windows = 0;
    newest_start = 0;
    __atomic_store_n(&armed, 0, __ATOMIC_RELAXED);
    clear_events_locked();
    pthread_mutex_unlock(&ev_mutex);
    in_hook = 0;
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

static PyObject *fh_rescan(PyObject *self, PyObject *unused) {
    (void)self; (void)unused;
    long added = 0;
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&hook_mutex);
    if (enabled) added = rescan_locked(0);
    pthread_mutex_unlock(&hook_mutex);
    Py_END_ALLOW_THREADS
    return PyLong_FromLong(added);
}

static PyObject *fh_window_open(PyObject *self, PyObject *unused) {
    (void)self; (void)unused;
    uint64_t start = 0;
    int ok = 1;
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&hook_mutex);
    if (enabled) rescan_locked(0);
    pthread_mutex_unlock(&hook_mutex);
    in_hook = 1;
    pthread_mutex_lock(&ev_mutex);
    if (grow((void **)&windows, &cap_windows, n_windows + 1, sizeof *windows) < 0) {
        ok = 0;
    } else {
        start = next_seq;
        windows[n_windows++] = start;
        newest_start = start;
        __atomic_store_n(&armed, 1, __ATOMIC_RELAXED);
    }
    pthread_mutex_unlock(&ev_mutex);
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
    pthread_mutex_lock(&ev_mutex);
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
    if (!n_windows) __atomic_store_n(&armed, 0, __ATOMIC_RELAXED);
    pthread_mutex_unlock(&ev_mutex);
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
    pthread_mutex_lock(&hook_mutex);
    en = enabled;
    slots = n_patches;
    foreign = foreign_slots;
    nobj = n_objects;
    for (size_t i = 0; i < n_objects; i++) {
        if (!objects[i].patched) continue;
        PyObject *s = PyUnicode_DecodeFSDefault(objects[i].name ? objects[i].name : "");
        if (!s || PyList_Append(libs, s) < 0) {
            Py_XDECREF(s);
            pthread_mutex_unlock(&hook_mutex);
            Py_DECREF(libs);
            return NULL;
        }
        Py_DECREF(s);
    }
    pthread_mutex_unlock(&hook_mutex);
    in_hook = 1;
    pthread_mutex_lock(&ev_mutex);
    unsigned long long rec = st_recorded, dd = st_deduped, lo = st_lost, buf = n_events, win = n_windows;
    pthread_mutex_unlock(&ev_mutex);
    in_hook = 0;
    unsigned long long calls = __atomic_load_n(&st_calls, __ATOMIC_RELAXED);
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
    {"filehook_window_open", fh_window_open, METH_NOARGS, "Start recording native opens; the window's id."},
    {"filehook_window_close", fh_window_close, METH_O, "(events, lost) of the window; closes it."},
    {"filehook_stats", fh_stats, METH_NOARGS, "Counters and the hooked objects."},
    {NULL, NULL, 0, NULL}};

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
                         "not supported here (the native file hook is Linux/glibc on x86-64 and aarch64 only)");
}

static PyMethodDef fh_methods[] = {
    {"filehook_supported", fh_supported, METH_NOARGS, "(True, how) or (False, why not)."},
    {NULL, NULL, 0, NULL}};

int cfilehook_add_functions(PyObject *m) { return PyModule_AddFunctions(m, fh_methods); }

#endif
