from __future__ import annotations

import argparse
import logging
import time
from os.path import join

import numpy as np
from coffea import processor, util
from coffea.nanoevents import NanoAODSchema
from distributed import Client
from lpcjobqueue import LPCCondorCluster

from azh_analysis.processors.btag_eff_processor import bTagEffProcessor
from azh_analysis.utils.sample import get_fileset, get_sample_info


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("prepare.py")
    add_arg = parser.add_argument
    add_arg("-y", "--year", default=2018)
    add_arg("-s", "--source", default="MC")
    add_arg("-v", "--verbose", action="store_true")
    add_arg("--add-signal", action="store_true")
    add_arg("--show-config", action="store_true")
    add_arg("--interactive", action="store_true")
    add_arg("--min-workers", default=10)
    add_arg("--max-workers", default=100)
    add_arg("--test-mode", action="store_true")
    return parser.parse_args()


# parse the command line
args = parse_args()

# setup logging
log_format = "%(asctime)s %(levelname)s %(message)s"
log_level = logging.DEBUG if args.verbose else logging.INFO
logging.basicConfig(level=log_level, format=log_format)
logging.info("Initializing")

# relevant parameters
year = args.year
csv_indir = "samples"
yaml_indir = "samples/filesets"
source, year = args.source, args.year
fileset = get_fileset(join(yaml_indir, f"{source}_{year}.yaml"))
sample_info = get_sample_info(join(csv_indir, f"{source}_{year}.csv"))
if args.add_signal:
    signal_yaml = f"signal_UL_{year[:4]}.yaml"
    fileset.update(get_fileset(join(yaml_indir, signal_yaml)))
    signal_csv = join(csv_indir, f"signal_UL_{year[:4]}.csv")
    sample_info = np.append(sample_info, get_sample_info(signal_csv))

fileset = {k: v for k, v in fileset.items()}
for f, l in fileset.items():
    print(f, len(l), "\n")
if args.test_mode:
    fileset = {k: v[:1] for k, v in fileset.items()}

# start timer, initiate cluster, ship over files
tic = time.time()
infiles = ["azh_analysis"]

cluster = LPCCondorCluster(
    ship_env=False,
    transfer_input_files=infiles,
    scheduler_options={"dashboard_address": ":8787"},
    shared_temp_directory="/tmp",
)

# scale the number of workers
cluster.adapt(
    minimum=1 if args.test_mode else args.min_workers, maximum=args.max_workers
)

# initiate client, wait for workers
client = Client(cluster)
logging.info("Waiting for at least one worker...")
client.wait_for_workers(1)

exe_args = {
    "client": client,
    "savemetrics": True,
    "schema": NanoAODSchema,
}

# instantiate processor module
processor_instance = bTagEffProcessor()
hists, metrics = processor.run_uproot_job(
    fileset,
    treename="Events",
    processor_instance=processor_instance,
    executor=processor.dask_executor,
    executor_args=exe_args,
    # maxchunks=20,
    chunksize=100000,
)

# measure, report summary statistics
elapsed = time.time() - tic
logging.info(f"Output: {hists}")
logging.info(f"Metrics: {metrics}")
logging.info(f"Finished in {elapsed:.1f}s")
logging.info(f"Events/s: {metrics['entries'] / elapsed:.0f}")

# dump output
outdir = "/srv"
outfile = f"{source}_{year}_btag_effs.coffea"
util.save(hists, outfile)
