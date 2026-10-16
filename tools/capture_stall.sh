#!/bin/bash
#########################################################################
# AdaptiveNDE Project                                                   #
# ROS-Scanning Package                                                  #
#########################################################################
# File:  capture_stall.sh
# Authors: Tyler Lesthaeghe
# Created: 2026-10-15
# Description:
#   Gather everything about a stalled scan, in one pass, from another
#   terminal, while it is still stalled.
#
#########################################################################
#
# WHY THIS EXISTS. A scan stalled at the cell on 2026-10-15 at pass 31 of 74.
# The evidence it left behind was one line -- "37 trace(s) queued, 2845 staged,
# 0 error(s)" -- and that line cannot distinguish the three things it could have
# been: a lock nobody will release, a globalrev that will never complete, or a
# disk read that has not come back. By the time anyone could have asked, the
# process was gone. Ctrl-C does not work during that stall either (the GUI loop
# owns dataguzzler-python's real main thread, so SIGINT lands in Qt's C++ event
# loop with the GIL released and no handler ever runs), so there is no graceful
# way to make the process tell you anything. This asks the kernel instead.
#
# ORDER MATTERS. Everything cheap and non-invasive is collected FIRST, because
# the one step that could go wrong -- attaching gdb -- SUSPENDS the process it
# attaches to, and a gdb that is killed leaves it suspended. Run with --no-gdb
# to skip that step entirely. If it is interrupted anyway, this script's trap
# sends SIGCONT, and so does:
#
#     kill -CONT <pid>
#
# WHAT TO LOOK AT FIRST, in the output:
#
#   threads.txt      every thread's kernel wait channel. This is the
#                    discriminator. `io_schedule`, `wait_on_page_bit`,
#                    `folio_wait_bit` = waiting on STORAGE. `futex_wait` =
#                    waiting on a lock held by another thread. `do_sys_poll`,
#                    `ep_poll` = idle, waiting for work. A staging worker in
#                    io_schedule is a disk problem, not a deadlock.
#   pressure.txt     /proc/pressure/io. `some avg10` near 100 means nearly
#                    everything on this machine was blocked on storage for the
#                    last ten seconds, which settles the question on its own.
#   python.txt       which Python line every thread is on (needs py-spy).
#   storage.txt      PCIe AER counters and NVMe health. The cell PC was
#                    throwing ~11 corrected PCIe receiver errors a MINUTE on
#                    its NVMe on 2026-10-15, continuously.
#   cpu.txt          who else was running. An antivirus on-demand scan was.
#
# USAGE
#     ROS-Scanning/tools/capture_stall.sh [pid] [-o outdir] [--no-gdb] [--gdb]
#
# With no pid it finds the dataguzzler-python process itself and refuses if
# there is more than one. Nothing here writes to the stalled process, kills
# anything, or touches the robot.
#########################################################################
set -u

PID=""
OUTDIR=""
WANT_GDB=auto

while [ $# -gt 0 ]; do
    case "$1" in
        --no-gdb) WANT_GDB=no ;;
        --gdb)    WANT_GDB=yes ;;
        -o)       shift; OUTDIR="${1:-}" ;;
        -h|--help)
            # From the WHY paragraph to the end of the block comment, which is
            # the part a person needs; the licence header above it is not.
            sed -n '/^# WHY THIS EXISTS/,/^####*$/p' "$0" \
                | sed -e '$d' -e 's/^#\( \|$\)//'
            exit 0 ;;
        -*) echo "unknown option $1" >&2; exit 2 ;;
        *)  PID="$1" ;;
    esac
    shift
done

# ---------------------------------------------------------------- find the pid
if [ -z "$PID" ]; then
    # -f, because the interpreter is python3 and the name to match is the
    # script. pgrep's own pattern is matched against the full command line.
    # -f because the interpreter is python3 and the name to match is the
    # script; then drop the shells, because `bash -lc 'dataguzzler-python ...'`
    # and the `script` wrapper both match the pattern and neither is the
    # interpreter whose threads are wanted.
    mapfile -t FOUND < <(
        pgrep -f '[d]ataguzzler-python' 2>/dev/null | while read -r p; do
            case "$(cat /proc/$p/comm 2>/dev/null)" in
                bash|sh|dash|zsh|script|timeout|sudo|su) ;;
                "") ;;
                *) echo "$p" ;;
            esac
        done)
    if [ "${#FOUND[@]}" -eq 0 ]; then
        echo "no dataguzzler-python process found. Pass the pid:" >&2
        echo "    $0 <pid>" >&2
        exit 1
    elif [ "${#FOUND[@]}" -gt 1 ]; then
        echo "more than one dataguzzler-python process; say which:" >&2
        for p in "${FOUND[@]}"; do
            echo "    $p  $(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)" >&2
        done
        exit 1
    fi
    PID="${FOUND[0]}"
fi

if [ ! -d "/proc/$PID" ]; then
    echo "no process $PID" >&2
    exit 1
fi

STAMP="$(date +%Y-%m-%d_%H%M%S)"
if [ -z "$OUTDIR" ]; then
    BASE="${TARI_STALL_DIR:-${SCANDATA_DIR:-/data/scandata}}"
    [ -w "$BASE" ] || BASE="$HOME"
    OUTDIR="$BASE/stall_$STAMP"
fi
mkdir -p "$OUTDIR" || exit 1

say() { echo "$@" | tee -a "$OUTDIR/capture.log"; }
have() { command -v "$1" >/dev/null 2>&1; }
# Run a command into a file, recording the command line and never failing the
# script. A diagnostic that aborts the collection is worse than a missing file.
grab() {
    local out="$1"; shift
    {
        echo "### $*"
        timeout 30 "$@" 2>&1
        echo "### exit $?"
        echo
    } >> "$OUTDIR/$out"
}

say "capture_stall.sh -> $OUTDIR"
say "pid $PID: $(tr '\0' ' ' < /proc/$PID/cmdline 2>/dev/null)"

# ======================================================================== cheap
{
    echo "captured      $(date -Is)"
    echo "host          $(uname -n)"
    echo "kernel        $(uname -srm)"
    echo "uptime        $(uptime)"
    echo "pid           $PID"
    echo "cmdline       $(tr '\0' ' ' < /proc/$PID/cmdline 2>/dev/null)"
    echo "started       $(ps -o lstart= -p "$PID" 2>/dev/null)"
    echo "cwd           $(readlink -f /proc/$PID/cwd 2>/dev/null)"
    echo "gdb           $WANT_GDB"
} > "$OUTDIR/about.txt"

# THE DISCRIMINATOR. One line per thread: what the kernel says it is waiting
# on. See the header for how to read it.
{
    echo "# tid  state  %cpu  wchan  name"
    echo "#"
    echo "# state: R running  S interruptible sleep  D UNINTERRUPTIBLE (storage)"
    echo "# wchan: io_schedule/wait_on_page_bit/folio_wait_bit = STORAGE"
    echo "#        futex_wait/rwsem/mutex_lock = waiting on another thread"
    echo "#        do_sys_poll/ep_poll/futex_wait_queue = idle"
    echo
    for t in /proc/$PID/task/*; do
        tid="$(basename "$t")"
        st="$(awk '{print $3}' "$t/stat" 2>/dev/null)"
        ut="$(awk '{print $14+$15}' "$t/stat" 2>/dev/null)"
        wc="$(cat "$t/wchan" 2>/dev/null)"
        nm="$(cat "$t/comm" 2>/dev/null)"
        printf "%-8s %-3s %-10s %-28s %s\n" "$tid" "${st:-?}" "${ut:-?}" "${wc:-0}" "${nm:-?}"
    done
    echo
    echo "### kernel stacks (needs root; blank without it)"
    for t in /proc/$PID/task/*; do
        echo "--- $(basename "$t") $(cat "$t/comm" 2>/dev/null)"
        cat "$t/stack" 2>&1 | head -20
    done
} > "$OUTDIR/threads.txt"

# Pressure Stall Information: the one number that says "this machine was
# waiting on its disk" without any interpretation.
{
    for f in /proc/pressure/cpu /proc/pressure/io /proc/pressure/memory; do
        echo "### $f"; cat "$f" 2>&1; echo
    done
    echo "### /proc/loadavg"; cat /proc/loadavg
    echo
    echo "### this process"
    echo "--- /proc/$PID/io"; cat /proc/$PID/io 2>&1
    echo "--- delayacct blkio ticks (field 42 of stat)"
    awk '{print $42}' /proc/$PID/stat 2>/dev/null
    echo "--- /proc/$PID/status"; cat /proc/$PID/status 2>&1
    echo "--- /proc/$PID/smaps_rollup"; cat /proc/$PID/smaps_rollup 2>&1
} > "$OUTDIR/pressure.txt"

grab cpu.txt uptime
if have top; then grab cpu.txt top -b -n1 -o %CPU -w 200; fi
grab cpu.txt ps -eo pid,ppid,stat,pcpu,pmem,rss,wchan:24,lstart,args --sort=-pcpu
if have vmstat; then grab cpu.txt vmstat 1 5; fi

# WHO ELSE IS RUNNING. An antivirus on-demand scan was, on 2026-10-15, and it
# was both burning CPU and reading the same disk the trace store is on.
{
    # One line, deliberately: a backslash-newline inside single quotes is a
    # literal backslash, and grep then refuses the whole pattern.
    scanners='clamd|clamscan|freshclam|savscan|sophos|mcafee|nails|trellix|cbagent|cbdaemon|crowdstrike|falcon|defender|mdatp|wdavdaemon|esets|avast|sentinel|tanium|rkhunter|tracker-(miner|extract)|updatedb|mlocate|plocate|aide|osqueryd'
    echo "### processes matching known scanners and indexers"
    echo "### pattern: $scanners"
    ps -eo pid,pcpu,pmem,etime,args 2>/dev/null \
      | grep -Ei -- "$scanners" | grep -v '[g]rep -Ei'
    echo "### (nothing above means nothing MATCHED, not that nothing was"
    echo "###  running: read cpu.txt for everything, sorted by CPU)"
} > "$OUTDIR/scanners.txt" 2>&1

# STORAGE. Which device holds the data, how healthy its link is, and what the
# kernel has been saying about it.
DATA_DIR="${SCANDATA_DIR:-/data/scandata}"
[ -d "$DATA_DIR" ] || DATA_DIR="$(readlink -f /proc/$PID/cwd 2>/dev/null)"
{
    echo "### the filesystem the traces are written to: $DATA_DIR"
    df -h "$DATA_DIR" 2>&1
    findmnt -T "$DATA_DIR" 2>&1
    echo
    echo "### PCIe corrected/fatal error counters (per device)"
    for d in /sys/bus/pci/devices/*/aer_dev_correctable; do
        [ -r "$d" ] || continue
        dev="$(dirname "$d")"
        # Only the devices that have actually counted something.
        if grep -qvE ' 0$' "$d" 2>/dev/null; then
            echo "--- ${dev##*/}  $(cat "$dev/class" 2>/dev/null)"
            for f in aer_dev_correctable aer_dev_nonfatal aer_dev_fatal \
                     current_link_speed current_link_width \
                     max_link_speed max_link_width; do
                [ -r "$dev/$f" ] && echo "    $f: $(tr '\n' ' ' < "$dev/$f")"
            done
        fi
    done
    echo
    echo "### block devices"
    lsblk -o NAME,SIZE,MODEL,SERIAL,TRAN,ROTA,MOUNTPOINT 2>&1
} > "$OUTDIR/storage.txt"

if have iostat; then grab storage.txt iostat -x 1 5; fi
if have nvme; then
    grab storage.txt nvme list
    for n in /dev/nvme[0-9]*n[0-9]*; do
        [ -e "$n" ] || continue
        grab storage.txt nvme smart-log "$n"
        grab storage.txt nvme error-log "$n"
    done
fi
if have smartctl; then
    for n in /dev/nvme[0-9]*n[0-9]* /dev/sd?; do
        [ -e "$n" ] || continue
        grab storage.txt smartctl -a "$n"
    done
fi

# The kernel log. -T for wall-clock times, so it lines up with the scan log.
if have dmesg; then
    { echo "### dmesg -T (tail)"; dmesg -T 2>&1 | tail -400; } > "$OUTDIR/dmesg.txt"
    { echo "### dmesg -T, errors only"
      dmesg -TL=never 2>/dev/null | grep -Ei \
        'error|fail|timeout|reset|aer|nvme|xhci|hung|blocked|oom|throttl' \
        | tail -200; } >> "$OUTDIR/dmesg.txt"
fi
grab dmesg.txt journalctl -k -n 400 --no-pager

# WHAT THE PROCESS HAS OPEN. The trace store is file-backed, so the files it
# holds and the sizes they have reached are part of the picture.
{
    echo "### open files"
    ls -l /proc/$PID/fd 2>&1 | head -200
    echo
    echo "### the trace run directory"
    ls -lrt --time-style=full-iso "$DATA_DIR" 2>&1 | tail -20
    newest="$(ls -dt "$DATA_DIR"/ec_scan_* 2>/dev/null | head -1)"
    if [ -n "$newest" ]; then
        echo "--- $newest"
        ls -lrt --time-style=full-iso "$newest" 2>&1 | tail -20
    fi
} > "$OUTDIR/files.txt"

# ROS, if it is reachable. Optional on purpose: this script has to work from a
# shell with nothing sourced.
if have ros2; then
    grab ros.txt ros2 node list
    grab ros.txt ros2 topic list
    grab ros.txt timeout 8 ros2 topic hz /joint_states
    grab ros.txt timeout 8 ros2 topic echo --once \
        /speed_scaling_state_broadcaster/speed_scaling
    grab ros.txt timeout 8 ros2 topic echo --once \
        /io_and_status_controller/robot_program_running
else
    echo "ros2 is not on this shell's PATH, so the ROS graph was not sampled." \
        > "$OUTDIR/ros.txt"
fi

# ============================================================ Python, then gdb
# py-spy reads the interpreter's own structures out of another process WITHOUT
# stopping it (--nonblocking) and needs no cooperation from the code that is
# stuck. If it is here, it is the single most useful thing in this directory.
if have py-spy; then
    grab python.txt py-spy dump --pid "$PID" --locals --nonblocking
else
    {
        echo "py-spy is not installed, so there are no Python stacks here."
        echo
        echo "It is the best tool for this and it needs no debugger:"
        echo "    sudo apt-get install -y python3-pip && pip3 install py-spy"
        echo "or   pip3 install --user py-spy"
        echo
        echo "The process itself also prints the staging worker's own stack in"
        echo "its stall note (see triggered_acquisition.dpi), so the scan log"
        echo "has the one stack that matters even without this."
    } > "$OUTDIR/python.txt"
fi

# gdb LAST, and only with the trap armed. Attaching stops the process; the trap
# and the explicit SIGCONT below are what start it again.
if [ "$WANT_GDB" = "no" ]; then
    echo "skipped on request (--no-gdb)" > "$OUTDIR/native.txt"
elif ! have gdb; then
    { echo "gdb is not installed, so there are no native stacks here."
      echo "    sudo apt-get install -y gdb"
      echo "Native stacks are what show WHICH C++ call has not returned --"
      echo "a spatialnde2 globalrev wait and an uninterruptible page fault look"
      echo "identical from Python."; } > "$OUTDIR/native.txt"
else
    say "attaching gdb -- the process is SUSPENDED until this finishes"
    trap 'kill -CONT '"$PID"' 2>/dev/null; echo; echo "resumed pid '"$PID"'"' \
        EXIT INT TERM
    # Every one of these flags was paid for once:
    #   -nx                     the operator's ~/.gdbinit is not part of this
    #   debuginfod enabled off  it goes to the network and blocks for minutes
    #   pagination/height/width the "---Type <return>---" prompt is a hang
    #   timeout                 a gdb that cannot attach must not be forever
    #   2>&1 into the file      the reason it failed is on stderr
    timeout 180 gdb -p "$PID" -batch -nx \
        -iex 'set debuginfod enabled off' \
        -iex 'set pagination off' \
        -iex 'set height 0' \
        -iex 'set width 0' \
        -iex 'set confirm off' \
        -ex 'info threads' \
        -ex 'thread apply all bt' \
        > "$OUTDIR/native.txt" 2>&1
    echo "### exit $?" >> "$OUTDIR/native.txt"
    kill -CONT "$PID" 2>/dev/null
    trap - EXIT INT TERM
    say "gdb done, pid $PID resumed"
fi

# ===================================================================== wrap up
{
    echo
    echo "read in this order:"
    echo "  threads.txt   is any thread in state D or in io_schedule?  -> storage"
    echo "  pressure.txt  /proc/pressure/io some avg10 near 100?       -> storage"
    echo "  python.txt    which Python line is the staging worker on?"
    echo "  native.txt    which C++ call has not returned?"
    echo "  storage.txt   AER counters moving, link below max width?   -> the link"
    echo "  scanners.txt  was an on-demand antivirus scan running?"
} >> "$OUTDIR/capture.log"

TAR="$OUTDIR.tar.gz"
if tar -czf "$TAR" -C "$(dirname "$OUTDIR")" "$(basename "$OUTDIR")" 2>/dev/null; then
    say "collected: $TAR"
else
    say "collected: $OUTDIR  (could not write the tarball)"
fi
sed -n '/read in this order/,$p' "$OUTDIR/capture.log"
exit 0
