from __future__ import annotations

import argparse
import logging
import os
import sys
from math import ceil
from os.path import join
from string import Template

import numpy as np

from azh_analysis.utils.sample_utils import get_fileset

sys.path.append(os.path.abspath(join(os.path.dirname(__file__), os.path.pardir)))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("")
    add_arg = parser.add_argument
    add_arg("-y", "--year", default="2018")
    add_arg("-s", "--source", default="MC_UL")
    add_arg("--submit", default=False)
    add_arg("--label", default="test")
    add_arg("--test-mode", action="store_true")
    add_arg("--script", default="../run_analysis.py")
    add_arg("-v", "--verbose", action="store_true")
    add_arg("--show-config", action="store_true")
    add_arg("--files-per-job", default=10)
    return parser.parse_args()


def write_template(templ_file: str, out_file: str, templ_args: dict):
    """Write to ``out_file`` based on template from ``templ_file`` using ``templ_args``"""

    with open(templ_file, "r") as f:
        templ = Template(f.read())

    with open(out_file, "w") as f:
        f.write(templ.substitute(templ_args))


def main(args):
    try:
        proxy = os.environ["X509_USER_PROXY"]
        logging.info(f"Using proxy: {proxy}")
    except Exception:
        logging.info("No valid proxy. Exiting.")

    indir = os.path.abspath(os.path.pardir)
    username = os.environ["USER"]
    local_dir = "condor"
    homedir = f"/store/user/{username}/azh_output/"
    outdir = join(homedir, "results")
    logging.info(f"Condor working directory: {outdir}")
    os.system(f"mkdir -p /eos/uscms/{outdir}")

    logdir = join(local_dir, "logs")
    os.system(f"mkdir -p {logdir}")

    year, source = args.year, args.source
    logging.info(f"Using parameters source={source}, year={year}")
    name = f"{source}_{year}"

    sample_base = join(indir, "sample_lists")
    logging.info(f"Attempting to locate sample_info: {sample_base}")
    fileset_base = join(indir, "sample_lists/sample_yamls")
    logging.info(f"Attempting to locate fileset: {fileset_base}")
    fileset = get_fileset(join(fileset_base, name + ".yaml"))
    logging.info(f"{fileset}")

    for sample, files in fileset.items():
        good_files = []
        for f in files:
            if f.split(".")[-1] == "root":
                good_files.append(f)
                fileset[sample] = good_files
    logging.info(f"{fileset.keys()}")

    jdl_templ = join(indir, "condor/submit.templ.jdl")
    sh_templ = join(indir, "condor/submit.templ.sh")

    n_submit = 0
    for sample, files in fileset.items():
        n_jobs = ceil(len(files) / args.files_per_job)
        subsamples = np.array_split(np.arange(len(files)), n_jobs)
        for j, subsample in enumerate(subsamples):
            if args.test_mode and j == 2:
                break

            eosoutput_dir = f"root://cmseos.fnal.gov/{outdir}/{year}/{sample}/"
            os.system(f"mkdir -p /eos/uscms/{outdir}/{args.year}/{sample}/")
            local_condor = join(local_dir, f"{sample}_{j}.jdl")
            jdl_args = {"dir": local_dir, "prefix": sample, "jobid": j, "proxy": proxy}
            write_template(jdl_templ, local_condor, jdl_args)
            local_sh = join(local_dir, f"{sample}_{j}.sh")

            sh_args = {
                "script": args.script,
                "year": year,
                "source": source,
                "label": args.label,
                "sample": sample,
                "start_idx": min(subsample),
                "end_idx": max(subsample) + 1,
                "eosoutpkl": f"{eosoutput_dir}/pickles/out_{j}.pkl",
                "eosoutparquet": f"{eosoutput_dir}/parquet/out_{j}.parquet",
                "eosoutroot": f"{eosoutput_dir}/root/nano_skim_{j}.root",
            }

            write_template(sh_templ, local_sh, sh_args)
            os.system(f"chmod u+x {local_sh}")

            if os.path.exists(f"{local_condor}.log"):
                os.system(f"rm {local_condor}.log")

            if args.submit:
                os.system(f"Condor_submit {local_condor}")
            n_submit += 1


if __name__ == "__main__":
    args = parse_args()

    # setup logging
    log_format = "%(asctime)s %(levelname)s %(message)s"
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format=log_format)
    logging.info("Initializing")

    main(args)
