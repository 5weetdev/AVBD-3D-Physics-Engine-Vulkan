#pragma once

// The demo's optional session log: what the user did, what the solver was set to, and
// how the world looked, written to a file beside the executable so a crash can be read
// back after the process is gone.
//
// While the log is open, everything written to stderr (VK_CHECK, the step's own abort
// messages, validation output) is also copied into it, and a fatal error -- an abort() or
// an unhandled native exception -- appends what was running, the frame number and a
// return-address backtrace before the process dies. Every line is flushed as written.

namespace demolog
{
// Opens avbd3d_demo_YYYYMMDD_HHMMSS.log beside the executable (a new file each time, so a
// crash log is never overwritten by the next run) and starts the stderr copy.
// commandLine goes into the header. Returns false if the file could not be created.
bool open(const char *commandLine);
void close();
bool active();
const char *path(); // "" when no log has been opened this run

// Prefix for the lines that follow, and what a fatal error reports as running.
void setFrame(int frame);
void setActivity(const char *what); // a string literal; kept by pointer

void line(const char *fmt, ...);

// For a SIGABRT handler: drains the stderr copy, writes the fatal record and closes.
// Safe to call when no log is open.
void fatal(const char *what);

// Installs the unhandled-exception filter that calls fatal() with the exception code
// and address. Done once at startup, whether or not a log is open.
void installCrashHandler();
} // namespace demolog
