"""Delete old remote branches (Philipp asked 2026-11-19).

Run from the cash repo folder:
    git fetch origin claude/project-thread-99weak && git show FETCH_HEAD:branch_cleanup.py | python -

Deletes (a) claude/ and cext/ branches fully merged into claude/batch19 and
unchanged for a day, and (b) unmerged ones unchanged for a week. Every
unmerged branch is backed up in the project's
archive/unmerged-branches-2026-11-19.bundle.
"""

import subprocess
import time


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


KEEP = {"main", "claude/batch19", "claude/batch19-next", "claude/project-thread-99weak"}
DAY = 86400


def branches(flag, min_age):
    now = time.time()
    found = []
    out = git("branch", "-r", flag, "origin/claude/batch19", "--format=%(refname:short) %(committerdate:unix)")
    for line in out.splitlines():
        ref, ts = line.rsplit(" ", 1)
        if not ref.startswith("origin/"):
            continue
        name = ref[len("origin/"):]
        if name == "HEAD" or name in KEEP or not name.startswith(("claude/", "cext/")):
            continue
        if now - int(ts) >= min_age:
            found.append(name)
    return found


git("fetch", "--prune", "origin")
merged = branches("--merged", DAY)
old_unmerged = branches("--no-merged", 7 * DAY)
todo = merged + old_unmerged
print(f"merged: {len(merged)}, old unmerged: {len(old_unmerged)}, deleting {len(todo)}")
done = 0
for i in range(0, len(todo), 40):
    chunk = todo[i:i + 40]
    result = subprocess.run(["git", "push", "origin", "--delete", *chunk], capture_output=True, text=True)
    if result.returncode == 0:
        done += len(chunk)
    else:
        print("a chunk failed:", result.stderr[-500:])
git("fetch", "--prune", "origin")
left = [line for line in git("branch", "-r").splitlines() if "->" not in line]
print(f"deleted {done}; {len(left)} remote branches left")
