from __future__ import annotations

import argparse

import uproot
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--source", default="MC")
parser.add_argument("-y", "--year", default="")
parser.add_argument("-i", "--indir", default=None)
parser.add_argument("--process", default="")
parser.add_argument("--check-xrd", default=False)
parser.add_argument("--by-group", action="store_true")
args = parser.parse_args()

infile = f"filesets/{args.source}_{args.year}.yaml"

with open(infile, "r") as stream:
    fileset = yaml.safe_load(stream)
    for sample, files in fileset.items():
        print(sample)
        for f in files:
            try:
                uproot.open(f)
            except Exception:
                print(f"Failed to open {f}")
