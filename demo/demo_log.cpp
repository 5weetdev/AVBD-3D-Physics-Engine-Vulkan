/*
 * Copyright (c) 2026 Chris Giles
 *
 * Permission to use, copy, modify, distribute and sell this software
 * and its documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies.
 * Chris Giles makes no representations about the suitability
 * of this software for any purpose.
 * It is provided "as is" without express or implied warranty.
 */

#include "demo_log.h"

#include <SDL2/SDL.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <dbghelp.h>
#include <fcntl.h>
#include <io.h>
#else
#include <fcntl.h>
#include <unistd.h>
#if defined(__has_include)
#if __has_include(<execinfo.h>)
#include <execinfo.h>
#define DEMO_LOG_HAVE_EXECINFO 1
#endif
#endif
#endif

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

namespace
{
// The stderr copy uses the C runtime's descriptor calls, which Windows spells with an underscore.
#ifdef _WIN32
int pipeOpen(int fds[2])
{
    return _pipe(fds, 1 << 16, _O_BINARY | _O_NOINHERIT);
}
int fdRead(int fd, void *buf, unsigned n)
{
    return _read(fd, buf, n);
}
int fdWrite(int fd, const void *buf, unsigned n)
{
    return _write(fd, buf, n);
}
int fdDup(int fd)
{
    return _dup(fd);
}
int fdDup2(int from, int to)
{
    return _dup2(from, to);
}
int fdClose(int fd)
{
    return _close(fd);
}
#else
int pipeOpen(int fds[2])
{
    if (pipe(fds) != 0)
        return -1;
    fcntl(fds[0], F_SETFD, FD_CLOEXEC);
    fcntl(fds[1], F_SETFD, FD_CLOEXEC);
    return 0;
}
int fdRead(int fd, void *buf, unsigned n)
{
    return (int)read(fd, buf, n);
}
int fdWrite(int fd, const void *buf, unsigned n)
{
    return (int)write(fd, buf, n);
}
int fdDup(int fd)
{
    return dup(fd);
}
int fdDup2(int from, int to)
{
    return dup2(from, to) < 0 ? -1 : 0;
}
int fdClose(int fd)
{
    return close(fd);
}
#endif

FILE *g_file = nullptr;
char g_path[1024] = {};
std::recursive_timed_mutex g_mutex;
std::atomic<int> g_frame{-1};
std::atomic<const char *> g_activity{"startup"};
std::chrono::steady_clock::time_point g_t0;

// stderr copy: fd 2 is pointed at a pipe, and a reader thread writes what comes out of
// it both to the original stderr and into the file.
int g_origErr = -1;
int g_pipeRead = -1;
std::thread g_tee;
std::atomic<bool> g_teeDrained{false};
const char kDrainSentinel = '\x01'; // never printed; marks "everything before me is copied"

double elapsed()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - g_t0).count();
}

// Caller holds g_mutex.
void writePrefix()
{
    fprintf(g_file, "[f %6d %9.3fs] ", g_frame.load(), elapsed());
}

void teeLoop()
{
    char buf[4096];
    bool lineStart = true;
    for (;;)
    {
        int n = fdRead(g_pipeRead, buf, sizeof(buf));
        if (n <= 0)
            break;
        bool drained = false;
        int w = 0;
        for (int i = 0; i < n; i++)
        {
            if (buf[i] == kDrainSentinel)
                drained = true;
            else
                buf[w++] = buf[i];
        }
        n = w;
        if (g_origErr >= 0 && n > 0)
            fdWrite(g_origErr, buf, (unsigned)n);
        {
            std::lock_guard<std::recursive_timed_mutex> lock(g_mutex);
            if (g_file)
            {
                for (int i = 0; i < n; i++)
                {
                    if (lineStart)
                    {
                        writePrefix();
                        fputs("stderr: ", g_file);
                    }
                    fputc(buf[i], g_file);
                    lineStart = buf[i] == '\n';
                }
                fflush(g_file);
            }
        }
        if (drained)
            g_teeDrained = true;
    }
}

void startTee()
{
    fflush(stderr);
    int fds[2];
    if (pipeOpen(fds) != 0)
        return;
    g_origErr = fdDup(2);
    if (g_origErr < 0 || fdDup2(fds[1], 2) != 0)
    {
        fdClose(fds[0]);
        fdClose(fds[1]);
        if (g_origErr >= 0)
            fdClose(g_origErr);
        g_origErr = -1;
        return;
    }
    fdClose(fds[1]); // fd 2 now holds the write end; restoring fd 2 closes it -> EOF
    setvbuf(stderr, nullptr, _IONBF, 0);
    g_pipeRead = fds[0];
    g_tee = std::thread(teeLoop);
}

void stopTee()
{
    if (g_pipeRead < 0)
        return;
    fflush(stderr);
    fdDup2(g_origErr, 2);
    if (g_tee.joinable())
        g_tee.join();
    fdClose(g_pipeRead);
    fdClose(g_origErr);
    g_pipeRead = -1;
    g_origErr = -1;
}

// Waits (bounded) for the reader to have copied everything already written to stderr:
// a sentinel byte goes down the pipe behind it, and the pipe is FIFO. (Peeking the pipe
// instead deadlocks: synchronous I/O on one handle is serialised, so a peek waits for
// the reader's blocked ReadFile, which waits for data that never comes.)
void drainTee()
{
    if (g_pipeRead < 0)
        return;
    fflush(stderr);
    g_teeDrained = false;
    const char sentinel = kDrainSentinel;
    if (fdWrite(2, &sentinel, 1) != 1)
        return;
    for (int i = 0; i < 100 && !g_teeDrained; i++)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

#ifdef _WIN32
// "module+0xoffset  function file:line" for one return address. Symbols come from the
// PDB beside the exe when there is one; the module offset is there either way.
void initSymbols()
{
    static bool done = false;
    if (done)
        return;
    done = true;
    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
    SymInitialize(GetCurrentProcess(), nullptr, TRUE);
}

void describeAddress(void *addr, char *out, size_t outSize)
{
    initSymbols();
    HMODULE mod = nullptr;
    char modName[MAX_PATH] = "?";
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCSTR)addr, &mod))
    {
        char full[MAX_PATH];
        if (GetModuleFileNameA(mod, full, MAX_PATH))
        {
            const char *base = strrchr(full, '\\');
            snprintf(modName, sizeof(modName), "%s", base ? base + 1 : full);
        }
    }
    const unsigned long long off = (unsigned long long)((const char *)addr - (const char *)mod);
    int n = snprintf(out, outSize, "%s+0x%llx", modName, off);

    alignas(SYMBOL_INFO) char symBuf[sizeof(SYMBOL_INFO) + 256];
    SYMBOL_INFO *sym = (SYMBOL_INFO *)symBuf;
    memset(symBuf, 0, sizeof(symBuf));
    sym->SizeOfStruct = sizeof(SYMBOL_INFO);
    sym->MaxNameLen = 255;
    DWORD64 disp = 0;
    const HANDLE proc = GetCurrentProcess();
    if (n > 0 && (size_t)n < outSize && SymFromAddr(proc, (DWORD64)addr, &disp, sym))
    {
        n += snprintf(out + n, outSize - n, "  %s", sym->Name);
        IMAGEHLP_LINE64 line = {};
        line.SizeOfStruct = sizeof(line);
        DWORD lineDisp = 0;
        if (n > 0 && (size_t)n < outSize && SymGetLineFromAddr64(proc, (DWORD64)addr, &lineDisp, &line))
        {
            const char *file = strrchr(line.FileName, '\\');
            snprintf(out + n, outSize - n, "  %s:%lu", file ? file + 1 : line.FileName, line.LineNumber);
        }
    }
}

void writeBacktrace()
{
    void *frames[48];
    const USHORT count = CaptureStackBackTrace(0, 48, frames, nullptr);
    fprintf(g_file, "backtrace (%u frames):\n", (unsigned)count);
    for (USHORT i = 0; i < count; i++)
    {
        char desc[512];
        describeAddress(frames[i], desc, sizeof(desc));
        fprintf(g_file, "  #%-2u %s\n", (unsigned)i, desc);
    }
}

LONG WINAPI onUnhandledException(EXCEPTION_POINTERS *ep)
{
    char where[512];
    describeAddress(ep->ExceptionRecord->ExceptionAddress, where, sizeof(where));
    char msg[640];
    snprintf(msg, sizeof(msg), "unhandled exception 0x%08lX at %s", (unsigned long)ep->ExceptionRecord->ExceptionCode,
             where);
    fprintf(stderr, "avbd3d_demo: %s\n", msg);
    demolog::fatal(msg);
    return EXCEPTION_CONTINUE_SEARCH;
}
#else
void writeBacktrace()
{
#ifdef DEMO_LOG_HAVE_EXECINFO
    void *frames[48];
    const int count = backtrace(frames, 48);
    char **names = backtrace_symbols(frames, count);
    fprintf(g_file, "backtrace (%d frames):
", count);
    for (int i = 0; i < count; i++)
        fprintf(g_file, "  #%-2d %s
", i, names ? names[i] : "?");
    free(names);
#else
    fprintf(g_file, "backtrace: not available on this platform
");
#endif
}
#endif
} // namespace

namespace demolog
{
bool open(const char *commandLine)
{
    if (g_file)
        return true;
    g_t0 = std::chrono::steady_clock::now();

    char *base = SDL_GetBasePath();
    const time_t now = time(nullptr);
    struct tm lt;
#ifdef _WIN32
    localtime_s(&lt, &now);
#else
    localtime_r(&now, &lt);
#endif
    char stamp[32];
    strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", &lt);
    snprintf(g_path, sizeof(g_path), "%savbd3d_demo_%s.log", base ? base : "", stamp);
    SDL_free(base);

    FILE *f = nullptr;
    if (!(f = std::fopen(g_path, "w")))
    {
        fprintf(stderr, "avbd3d_demo: could not create log %s\n", g_path);
        g_path[0] = '\0';
        return false;
    }
    {
        std::lock_guard<std::recursive_timed_mutex> lock(g_mutex);
        g_file = f;
        char when[64];
        strftime(when, sizeof(when), "%Y-%m-%d %H:%M:%S", &lt);
        fprintf(g_file, "avbd3d_demo log, opened %s\n", when);
        fprintf(g_file, "built %s %s\n", __DATE__, __TIME__);
        fprintf(g_file, "command line: %s\n", commandLine ? commandLine : "");
        fflush(g_file);
    }
    startTee();
    return true;
}

void close()
{
    if (!g_file)
        return;
    line("log closed");
    stopTee();
    std::lock_guard<std::recursive_timed_mutex> lock(g_mutex);
    fclose(g_file);
    g_file = nullptr;
}

bool active()
{
    return g_file != nullptr;
}
const char *path()
{
    return g_path;
}
void setFrame(int frame)
{
    g_frame = frame;
}
void setActivity(const char *what)
{
    g_activity = what;
}

void line(const char *fmt, ...)
{
    if (!g_file)
        return;
    std::lock_guard<std::recursive_timed_mutex> lock(g_mutex);
    if (!g_file)
        return;
    writePrefix();
    va_list ap;
    va_start(ap, fmt);
    vfprintf(g_file, fmt, ap);
    va_end(ap);
    const size_t len = strlen(fmt);
    if (len == 0 || fmt[len - 1] != '\n')
        fputc('\n', g_file);
    fflush(g_file);
}

void fatal(const char *what)
{
    if (!g_file)
        return;
    drainTee();
    // Bounded: the fault may have hit while this thread held the lock.
    const bool locked = g_mutex.try_lock_for(std::chrono::milliseconds(500));
    if (g_file)
    {
        writePrefix();
        fprintf(g_file, "FATAL: %s\n", what ? what : "?");
        const char *act = g_activity.load();
        fprintf(g_file, "while: %s, frame %d\n", act ? act : "?", g_frame.load());
        writeBacktrace();
        fflush(g_file);
    }
    if (locked)
        g_mutex.unlock();
}

void installCrashHandler()
{
#ifdef _WIN32
    SetUnhandledExceptionFilter(onUnhandledException);
#endif
    // Elsewhere the SIGABRT handler in main.cpp calls fatal(); there is no native exception filter.
}
} // namespace demolog
