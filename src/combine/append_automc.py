from __future__ import annotations

import os
import sys

indir = sys.argv[1]
print(indir)
for f in os.listdir(indir):
    file = f"{indir}/{f}"
    if os.path.isdir(file):
        continue
    with open(f"{indir}/{f}") as rf:
        if "autoMCStats" in rf.read():
            continue

    with open(f"{indir}/{f}", "a") as af:
        af.write("* autoMCStats 0")
