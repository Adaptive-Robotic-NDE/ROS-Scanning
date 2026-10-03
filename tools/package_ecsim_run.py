#!/usr/bin/env python3
"""Build the ecsim-run zip for the table-generation machines, and verify it.

    python3 package_ecsim_run.py [--ecsim ~/ecsim] [--out ~/ecsim-run.zip]

WHY THIS IS A SCRIPT AND NOT A SEQUENCE OF COMMANDS.

The first two of these packages were assembled by hand and then PATCHED when
something was missing, rather than rebuilt. The result drifted from its source
without saying so: one shipped an ecsim predating the CuPy backend, so the GPU
option it documented could only fail, and the only visible symptom was an
unrelated missing script. Staging is exactly where a check passes while the
artefact is wrong.

So this always stages from scratch, and then verifies the contents OF THE ZIP
-- not of the staging directory -- against the ecsim working tree, file by
file. It refuses to produce a zip it cannot vouch for.
"""
from __future__ import annotations

import argparse
import filecmp
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROS_SCANNING = os.path.dirname(HERE)

# Never shipped: the git history, the dev venv, and the 305 MB read-only clone
# of the reference repository.
EXCLUDE = (".git", ".venv", "reference", "__pycache__", ".pytest_cache",
           ".ipynb_checkpoints", "data/measured")

FROM_ECSIM = ("ecsim", "scripts", "tests", "docs", "notebooks",
              "pyproject.toml", "README.md", "LICENSE")

FROM_OURS = (("tools/generate_ec_table.py", "tools"),
             ("tools/make_ec_test_plate.py", "tools"),
             ("docs/ec-probe-model-spec.md", "docs"),
             ("docs/RUN-ON-WINDOWS.md", "."))


def _ignore(_dir, names):
    return [n for n in names if n in EXCLUDE]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ecsim", default=os.path.expanduser("~/ecsim"))
    ap.add_argument("--out", default=os.path.expanduser("~/ecsim-run.zip"))
    args = ap.parse_args(argv)

    head = subprocess.run(["git", "-C", args.ecsim, "log", "--oneline", "-1"],
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", args.ecsim, "status", "--porcelain"],
                           capture_output=True, text=True).stdout
    dirty = [l for l in dirty.splitlines() if not l.startswith("??")]
    print("ecsim at %s" % head)
    if dirty:
        print("  WARNING: %d uncommitted change(s); packaging the WORKING TREE"
              % len(dirty))

    tmp = tempfile.mkdtemp(prefix="ecsim-run-")
    root = os.path.join(tmp, "ecsim-run")
    os.makedirs(os.path.join(root, "tools"))

    for name in FROM_ECSIM:
        src = os.path.join(args.ecsim, name)
        if not os.path.exists(src):
            print("  skipping absent %s" % name)
            continue
        dst = os.path.join(root, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst, ignore=_ignore)
        else:
            shutil.copy2(src, dst)

    for rel, into in FROM_OURS:
        src = os.path.join(ROS_SCANNING, rel)
        if not os.path.exists(src):
            sys.exit("missing %s" % src)
        dst_dir = os.path.join(root, into) if into != "." else root
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src, dst_dir)

    if os.path.exists(args.out):
        os.remove(args.out)
    shutil.make_archive(args.out[:-4], "zip", tmp, "ecsim-run")

    # ---- verify the ZIP, not the staging directory -------------------------
    tracked = subprocess.run(
        ["git", "-C", args.ecsim, "ls-files", "ecsim", "scripts"],
        capture_output=True, text=True).stdout.split()
    tracked = [t for t in tracked if t.endswith(".py")]

    bad = []
    with zipfile.ZipFile(args.out) as z:
        names = set(z.namelist())
        check = tempfile.mkdtemp(prefix="ecsim-verify-")
        for rel in tracked:
            member = "ecsim-run/" + rel
            if member not in names:
                bad.append("MISSING %s" % rel)
                continue
            z.extract(member, check)
            if not filecmp.cmp(os.path.join(args.ecsim, rel),
                               os.path.join(check, member), shallow=False):
                bad.append("DIFFERS %s" % rel)
        for rel, _into in FROM_OURS:
            leaf = os.path.basename(rel)
            if not any(n.endswith("/" + leaf) for n in names):
                bad.append("MISSING %s" % leaf)
        shutil.rmtree(check, ignore_errors=True)
        n_files = len(names)
    shutil.rmtree(tmp, ignore_errors=True)

    if bad:
        for b in bad:
            print("  %s" % b)
        os.remove(args.out)
        sys.exit("REFUSING to ship a package that does not match its source")

    print("verified %d tracked ecsim file(s) byte-identical inside the zip"
          % len(tracked))
    print("%s  %.1f MB, %d files" % (args.out, os.path.getsize(args.out)/1e6,
                                     n_files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
