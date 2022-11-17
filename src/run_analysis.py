from __future__ import annotations

import argparse
import logging
import time
from os.path import join

from coffea import processor, util
from coffea.lumi_tools import LumiMask
from coffea.nanoevents import NanoAODSchema

from azh_analysis.processors.analysis_processor import AnalysisProcessor
from azh_analysis.utils.btag import get_btag_SFs, get_btag_tables
from azh_analysis.utils.corrections import (
    dyjets_stitch_weights,
    get_electron_ID_weights,
    get_electron_trigger_SFs,
    get_fake_rates,
    get_muon_ID_weights,
    get_muon_trigger_SFs,
    get_tau_ID_weights,
)
from azh_analysis.utils.pileup import get_pileup_tables
from azh_analysis.utils.sample import get_fileset, get_nevts_dict, get_sample_info


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("run_analysis.py")
    add_arg = parser.add_argument
    add_arg("-y", "--year", default="2018")
    add_arg("-s", "--source", default=None)
    add_arg("--process", default="")
    add_arg("--start-idx", default=-1)
    add_arg("--end-idx", default=-1)
    add_arg("--outdir", default=".")
    add_arg("--outfile", default="")
    add_arg("--test-mode", action="store_true")
    add_arg("-v", "--verbose", default=False)
    add_arg("--show-config", action="store_true")
    add_arg("--interactive", action="store_true")
    add_arg("--min-workers", type=int, default=50)
    add_arg("--max-workers", type=int, default=300)
    add_arg("--mass", type=str, default="")
    return parser.parse_args()


# parse the command line
args = parse_args()

# setup logging
log_format = "%(asctime)s %(levelname)s %(message)s"
log_level = logging.DEBUG if args.verbose else logging.INFO
logging.basicConfig(level=log_level, format=log_format)
logging.info("Initializing")

# relevant parameters
year, source = args.year, args.source

# load up golden jsons
golden_json_dir = "samples/data_certification"
golden_jsons = {
    "2018": join(golden_json_dir, "data_cert_2018.json"),
    "2017": join(golden_json_dir, "data_cert_2017.json"),
    "2016postVFP": join(golden_json_dir, "data_cert_2016.json"),
    "2016preVFP": join(golden_json_dir, "data_cert_2016.json"),
}
lumi_masks = {year: LumiMask(golden_json) for year, golden_json in golden_jsons.items()}

# load up fake rates
fr_base = f"corrections/fake_rates/UL_{year}"
fake_rates = get_fake_rates(fr_base, year)
logging.info(f"Using fake rates\n{fr_base}")

# load up electron / muon / tau IDs
eID_base = f"corrections/electron_ID/UL_{year}"
eID_file = join(
    eID_base, f"Electron_RunUL{year}_IdIso_AZh_IsoLt0p15_IdFall17MVA90noIsov2.root"
)
eIDs = get_electron_ID_weights(eID_file)
logging.info(f"Using eID_SFs:\n{eID_file}")

mID_base = f"corrections/muon_ID/UL_{year}"
mID_file = join(mID_base, f"Muon_RunUL{year}_IdIso_AZh_IsoLt0p15_IdLoose.root")
mIDs = get_muon_ID_weights(mID_file)
logging.info(f"Using mID_SFs:\n{mID_file}")

tID_base = f"corrections/tau_ID/UL_{year}"
tID_file = join(tID_base, "tau.corr.json")
tIDs = get_tau_ID_weights(tID_file)
logging.info(f"Using tID_SFs:\n{tID_file}")

# load up electron / muon trigger SFs
e_trigs = {
    "2016preVFP": "Ele25_EtaLt2p1",
    "2016postVFP": "Ele25_EtaLt2p1",
    "2017": "Ele35",
    "2018": "Ele35",
}
e_trig_base = f"corrections/electron_trigger/UL_{year}"
e_trig_file = join(e_trig_base, f"Electron_RunUL{year}_{e_trigs[year]}.root")
e_trig_SFs = get_electron_trigger_SFs(e_trig_file)

m_trigs = {
    "2016preVFP": "IsoMu24orIsoTkMu24",
    "2016postVFP": "IsoMu24orIsoTkMu24",
    "2017": "IsoMu27",
    "2018": "IsoMu27",
}
m_trig_base = f"corrections/muon_trigger/UL_{year}"
m_trig_file = join(m_trig_base, f"Muon_RunUL{year}_{m_trigs[year]}.root")
m_trig_SFs = get_muon_trigger_SFs(m_trig_file)

# load up btagging tables
btag_root = "corrections/btag/"
btag_tables = get_btag_tables(btag_root, "2018", UL=True)
btag_SFs = get_btag_SFs(btag_root, "2018", UL=True)

# load up non-signal MC csv / yaml files
fset_string = f"{source}_{year}"
sample_info = get_sample_info(join("samples", fset_string + ".csv"))
fileset = get_fileset(join("samples/filesets", fset_string + ".yaml"))
pileup_tables = get_pileup_tables(
    fileset.keys(), year, UL=True, pileup_dir="corrections/pileup"
)

# load up signal MC csv / yaml files
if args.test_mode:
    fileset = {k: v[:1] for k, v in fileset.items()}
if len(args.process) > 0:
    fileset = {k: v for k, v in fileset.items() if k == args.process}
elif "signal" in args.source:
    fileset = {k: v for k, v in fileset.items() if args.mass in k}

# only run over root files
for sample, files in fileset.items():
    good_files = []
    for f in files:
        if f.split(".")[-1] == "root":
            good_files.append(f)
    fileset[sample] = good_files
logging.info(f"running on\n {fileset.keys()}")

# extract the sum_of_weights from the ntuples
nevts_dict, dyjets_weights = None, None
if "MC" in source:
    nevts_dict = get_nevts_dict(fileset, year)
    print("fileset keys", fileset.keys())
    if f"DYJetsToLLM-50_{year}" in fileset.keys():
        dyjets_weights = dyjets_stitch_weights(sample_info, nevts_dict, year)

logging.info(f"Successfully built sum_of_weights dict:\n {nevts_dict}")
logging.info(f"Successfully built dyjets stitch weights:\n {dyjets_weights}")

# start timer, initiate cluster, ship over files
tic = time.time()

# instantiate processor module
proc_instance = AnalysisProcessor(
    sample_info=sample_info,
    fileset=fileset,
    pileup_tables=pileup_tables,
    lumi_masks=lumi_masks,
    nevts_dict=nevts_dict,
    eleID_SFs=eIDs,
    muID_SFs=mIDs,
    tauID_SFs=tIDs,
    e_trig_SFs=e_trig_SFs,
    m_trig_SFs=m_trig_SFs,
    fake_rates=fake_rates,
    dyjets_weights=dyjets_weights,
    btag_eff_tables=btag_tables[0],
    btag_SFs=btag_SFs,
    btag_pt_bins=btag_tables[1],
    btag_eta_bins=btag_tables[2],
    run_fastmtt=True,
)

futures_run = processor.Runner(
    executor=processor.FuturesExecutor(compression=None, workers=1),
    schema=NanoAODSchema,
)

out = futures_run(
    fileset,
    "Events",
    processor_instance=proc_instance,
)

logging.info(f"Output: {out}")

# measure, report summary statistics
elapsed = time.time() - tic
logging.info(f"Finished in {elapsed:.1f}s")

# dump output
outfile = time.strftime("%m-%d") + ".coffea"
namestring = f"{source}_{year}"
if len(args.outfile) > 0:
    namestring = args.outfile
util.save(out, join(args.outdir, f"{namestring}_{outfile}"))
