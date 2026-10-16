#########################################################################
# AdaptiveNDE Project                                                   #
# ROS-Scanning Package                                                  #
#########################################################################
# Copyright 2023-2024 University of Dayton                              #
#                                                                       #
# Redistribution and use in source and binary forms, with or without    #
# modification, are permitted provided that the following conditions    #
# are met:                                                              #
#                                                                       #
# 1. Redistributions of source code must retain the above copyright     #
#    notice, this list of conditions and the following disclaimer.      #
# 2. Redistributions in binary form must reproduce the above copyright  #
#    notice, this list of conditions and the following disclaimer in    #
#    the documentation and/or other materials provided with the         #
#    distribution.                                                      #
# 3. Neither the name of the copyright holder nor the names of its      #
#    contributors may be used to endorse or promote products derived    #
#    from this software without specific prior written permission.      #
#                                                                       #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS   #
# “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT     #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR #
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT  #
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,#
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT      #
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, #
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY #
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT   #
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE #
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  #
#########################################################################
# This material is based in part upon work supported by the U.S. Air    #
# Force under Contract FA8650-19-D-5230.                                #
#                                                                       #
# DISTRIBUTION A. Approved for public release: distribution unlimited.  #
# File:  capture_stall_script_test.py
# Authors: Tyler Lesthaeghe
# Created: 2026-10-15
# Description:
#   tools/capture_stall.sh has to work the one time it is ever run.
# Derived From:
#
#########################################################################
# Revision Log
#
# 2026-10-15:  Initial Commit
#
#########################################################################
# Development Notes
#
#
#########################################################################
"""tools/capture_stall.sh runs once, at a cell, on a process about to be killed.

It gets one attempt. There is no iterating on a diagnostic that is only ever
reached during the fault it exists to describe, so what is tested here is that
it runs to completion against a genuinely stalled interpreter and that the
files a person is told to read in come out with the discriminating content in
them.

THE DISCRIMINATOR is threads.txt: one line per thread with the kernel's wait
channel. A staging worker in `io_schedule` or `folio_wait_bit` is waiting on
storage; one in `futex_do_wait` is waiting on another thread. Those are
different faults, and the 2026-10-15 stall left no way to tell them apart. The
stand-in process below parks a thread named `stager` on an Event, so the right
answer is a futex -- the point is that the column is populated and attributed
to the right thread, not which value it holds.

Nothing here attaches gdb (--no-gdb): ptrace is denied under several of the
sandboxes this suite runs in, and the gdb step is deliberately last in the
script for exactly that reason.

    python3 capture_stall_script_test.py
"""
import os
import subprocess
import sys
import tempfile
import textwrap
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, os.pardir, os.pardir, "tools", "capture_stall.sh")
SCRIPT = os.path.normpath(SCRIPT)
FAILURES = []

STANDIN = textwrap.dedent('''
    import os, sys, threading, time

    def inner_wait(ev):
        ev.wait(600)

    def stage_one_trace(ev):
        inner_wait(ev)

    ev = threading.Event()
    t = threading.Thread(target=stage_one_trace, args=(ev,), name="stager")
    t.daemon = True
    t.start()
    sys.stdout.write("%d\\n" % os.getpid())
    sys.stdout.flush()
    time.sleep(600)
''')


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        FAILURES.append(name)
        print("  FAIL %s %s" % (name, detail))


def _capture():
    """Run the script against a stalled interpreter. -> (outdir, stdout)"""
    d = tempfile.mkdtemp(prefix="stallcap-")
    src = os.path.join(d, "standin.py")
    with open(src, "w") as f:
        f.write(STANDIN)
    proc = subprocess.Popen([sys.executable, "-I", src],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            start_new_session=True)
    try:
        pid = int(proc.stdout.readline().strip())
        out = os.path.join(d, "capture")
        r = subprocess.run(["bash", SCRIPT, str(pid), "-o", out, "--no-gdb"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=300)
        # ASKED BEFORE THE HARNESS KILLS IT. The script must leave the process
        # it sampled running and runnable -- the gdb step suspends it, and a
        # diagnostic that leaves a scan stopped is worse than no diagnostic.
        alive = False
        if proc.poll() is None and os.path.exists("/proc/%d/stat" % pid):
            with open("/proc/%d/stat" % pid) as f:
                alive = f.read().rsplit(")", 1)[-1].split()[0] != "T"
        return out, r.stdout.decode("utf-8", "replace"), r.returncode, pid, alive
    finally:
        proc.kill()
        proc.wait(timeout=10)


def _read(outdir, name):
    path = os.path.join(outdir, name)
    if not os.path.exists(path):
        return None
    with open(path, errors="replace") as f:
        return f.read()


def main():
    print("the script is executable and parses")
    check("it exists", os.path.exists(SCRIPT), SCRIPT)
    check("it is executable", os.access(SCRIPT, os.X_OK), SCRIPT)
    r = subprocess.run(["bash", "-n", SCRIPT], stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    check("bash accepts it", r.returncode == 0, r.stdout.decode())

    r = subprocess.run(["bash", SCRIPT, "--help"], stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, timeout=60)
    helptext = r.stdout.decode("utf-8", "replace")
    check("--help explains itself without running anything",
          "WHY THIS EXISTS" in helptext and "USAGE" in helptext,
          helptext[:120])
    check("--help does not print the licence header",
          "REDISTRIBUTION" not in helptext.upper(), helptext[:120])

    print("\nit refuses a pid that is not there, rather than writing nothing")
    r = subprocess.run(["bash", SCRIPT, "999999"], stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, timeout=60)
    check("a dead pid is an error", r.returncode != 0, r.stdout.decode()[:200])

    print("\nagainst a stalled interpreter")
    outdir, text, rc, pid, alive = _capture()
    check("it exits 0", rc == 0, text[-400:])
    check("and says where it put everything", "collected:" in text,
          text[-200:])

    for name in ("about.txt", "threads.txt", "pressure.txt", "cpu.txt",
                 "storage.txt", "files.txt", "python.txt", "scanners.txt",
                 "native.txt", "capture.log"):
        check("%s was written" % name, _read(outdir, name) is not None)

    threads = _read(outdir, "threads.txt") or ""
    check("threads.txt names the stalled thread",
          "stager" in threads, threads[:400])
    check("and gives it a wait channel, which is the whole point",
          any(line.split()[-1] == "stager" and len(line.split()) >= 5
              and line.split()[3] not in ("", "0")
              for line in threads.splitlines() if "stager" in line),
          "\n".join(l for l in threads.splitlines() if "stager" in l))
    check("it explains how to read the wait channel, in the file",
          "io_schedule" in threads and "futex" in threads, threads[:300])

    pressure = _read(outdir, "pressure.txt") or ""
    check("pressure.txt carries the io figure that settles storage-vs-lock",
          "/proc/pressure/io" in pressure and "avg10" in pressure,
          pressure[:200])
    check("and this process's own blocked-io accounting",
          "delayacct blkio ticks" in pressure, pressure[:200])

    about = _read(outdir, "about.txt") or ""
    check("about.txt records which pid was sampled",
          str(pid) in about, about[:300])

    python = _read(outdir, "python.txt") or ""
    check("python.txt either has the stacks or says why not",
          ("stage_one_trace" in python) or ("py-spy" in python),
          python[:200])

    scanners = _read(outdir, "scanners.txt") or ""
    check("scanners.txt is a clean report, not a grep error",
          "Trailing backslash" not in scanners and "Usage" not in scanners,
          scanners[:200])
    check("and it says that no match is not the same as nothing running",
          "not that nothing was" in scanners, scanners[:200])

    log = _read(outdir, "capture.log") or ""
    check("capture.log ends with what to read first",
          "read in this order" in log, log[-300:])

    check("the stalled process was left running and not suspended", alive)

    print("\n%s" % ("ALL OK" if not FAILURES else "%d FAILED: %s"
                    % (len(FAILURES), ", ".join(FAILURES))))
    return 1 if FAILURES else 0


def test_capture_stall_script():
    assert main() == 0, "%d check(s) failed: %s" % (len(FAILURES),
                                                    "; ".join(FAILURES))


if __name__ == "__main__":
    sys.exit(main())
