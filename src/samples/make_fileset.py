from __future__ import annotations

import argparse
import os
from os.path import join

import numpy as np


def get_sample_info(f):
    return np.genfromtxt(
        f,
        delimiter=",",
        names=True,
        comments="#",
        dtype=np.dtype(
            [
                ("f0", "<U9"),
                ("f1", "<U128"),
                ("f2", "<U128"),
                ("f3", "<U250"),
                ("f4", "<f16"),
                ("f5", "<f8"),
            ]
        ),
    )


parser = argparse.ArgumentParser()
parser.add_argument("-s", "--source", default="MC")
parser.add_argument("-y", "--year", default="")
parser.add_argument("-i", "--indir", default=None)
parser.add_argument("--process", default="")
parser.add_argument("--check-xrd", default=False)
parser.add_argument("--by-group", action="store_true")
args = parser.parse_args()


def make_yaml(source, year, all_samples, sample_info, target_group=None):
    outfile = f"{source}_{year}.yaml"
    if target_group is not None:
        outfile = f"{target_group}_UL_{year}.yaml"
    outfile = open(join("filesets", outfile), "w+")
    for i in range(len(sample_info)):
        name = sample_info["name"][i]
        group = sample_info["group"][i]
        if (target_group is not None) and (target_group != group):
            continue
        name_str = (
            name.replace("TuneCP5", "").replace("_postVFP", "").replace("_preVFP", "")
        )
        name_str = name_str.replace("LL_M-50", "LLM-50").replace("LLM50", "LLM-50")
        print(f"...processing {group}: {name_str}")
        samples = [s for s in all_samples if name == s]
        if len(samples) != 1:
            print(f"only found {samples}")
            break
        sample_dir = join(base_dir, samples[0])
        files = os.listdir(sample_dir)
        files = [f for f in files if ".root" in f]
        sample_dir = join("root://cmseos.fnal.gov/", sample_dir)
        files = [join(sample_dir, f) for f in files]
        outfile.write(f"{name_str}_{args.year}:\n")
        for f in files:
            outfile.write(f" - root://cmseos.fnal.gov/{f}\n")
    outfile.close()


year = args.year
year_str = year if "2016" not in year else "2016"
base_dir = "/eos/uscms/store/group/lpcsusyhiggs/" f"ntuples/AZh/nAODv9/{year_str}"
all_samples = os.listdir(base_dir)
print(all_samples)

# open sample file
sample_info = f"{args.source}_{args.year}.csv"
sample_info = get_sample_info(sample_info)
groups = np.unique(sample_info["group"])

if args.by_group:
    for group in groups:
        make_yaml(
            args.source,
            args.year,
            all_samples,
            sample_info,
            target_group=group,
        )
else:
    make_yaml(
        args.source,
        args.year,
        all_samples,
        sample_info,
    )
