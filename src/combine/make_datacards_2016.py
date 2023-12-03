from __future__ import annotations

import argparse

import CombineHarvester.CombineTools.ch as ch
import uproot

parser = argparse.ArgumentParser(description="Description of your program")
parser.add_argument("-m", "--mass", required=True)
parser.add_argument("-b", "--btag", required=True)
parser.add_argument("-p", "--process", required=True)
parser.add_argument("--per-category", action="store_true")
args = vars(parser.parse_args())

year, mass, btag_label = "2016", args["mass"], args["btag"]
per_category = args["per_category"]

mc_bkgd = [  # TTW, TT, ggHtt, VFBHtt, WHtt, ggHWW, VBFHWW
    "ggZZ",
    "ZZ",
    "TTZ",
    "VVV",
    "WZ",
    "ZHtt",
    "TTHtt",
    "WHWW",
    "ZHWW",
    "ggZHWW",
    "ggHZZ",
]
reducible = ["reducible"]

process = args["process"]
sig = [process]

cb = ch.CombineHarvester()
cats = [
    (1, "eeem"),
    (2, "eeet"),
    (3, "eemt"),
    (4, "eett"),
    (5, "mmem"),
    (6, "mmet"),
    (7, "mmmt"),
    (8, "mmtt"),
]

cb.AddObservations(["*"], ["azh"], [], [btag_label], cats)
cb.AddProcesses([mass], ["azh"], [year], [btag_label], [sig], cats, True)
cb.AddProcesses(["*"], ["azh"], [year], [btag_label], reducible, cats, False)
cb.AddProcesses(["*"], ["azh"], [year], [btag_label], mc_bkgd, cats, False)


# luminosity
cb.cp().signals().AddSyst(cb, "CMS_lumi_13TeV_2016", "lnN", ch.SystMap()(1.01))
cb.cp().process(mc_bkgd).AddSyst(cb, "CMS_lumi_13TeV_2016", "lnN", ch.SystMap()(1.01))
cb.cp().signals().AddSyst(cb, "CMS_lumi_13TeV_correlated", "lnN", ch.SystMap()(1.006))
cb.cp().process(mc_bkgd).AddSyst(
    cb, "CMS_lumi_13TeV_correlated", "lnN", ch.SystMap()(1.006)
)

# Higgs tau tau PU alphas
cb.cp().signals().AddSyst(cb, "BR_htt_PU_alphas", "lnN", ch.SystMap()(1.062))
cb.cp().process(["ggHtt", "VBFHtt", "WHtt", "ZHtt", "TTHtt"]).AddSyst(
    cb, "BR_htt_PU_alphas", "lnN", ch.SystMap()(1.062)
)
cb.cp().signals().AddSyst(cb, "BR_htt_PU_mq", "lnN", ch.SystMap()(1.099))
cb.cp().process(["ggHtt", "VBFHtt", "WHtt", "ZHtt", "TTHtt"]).AddSyst(
    cb, "BR_htt_PU_mq", "lnN", ch.SystMap()(1.099)
)
cb.cp().signals().AddSyst(cb, "BR_htt_THU", "lnN", ch.SystMap()(1.017))
cb.cp().process(["ggHtt", "VBFHtt", "WHtt", "ZHtt", "TTHtt"]).AddSyst(
    cb, "BR_htt_THU", "lnN", ch.SystMap()(1.017)
)

# Higgs WW PU alphas
cb.cp().process(["ggHWW", "VBFHWW", "WHWW", "ZHWW", "ggZHWW"]).AddSyst(
    cb, "BR_hww_PU_alphas", "lnN", ch.SystMap()(1.066)
)
cb.cp().process(["ggHWW", "VBFHWW", "WHWW", "ZHWW", "ggZHWW"]).AddSyst(
    cb, "BR_hww_PU_mq", "lnN", ch.SystMap()(1.099)
)
cb.cp().process(["ggHWW", "VBFHWW", "WHWW", "ZHWW", "ggZHWW"]).AddSyst(
    cb, "BR_hww_THU", "lnN", ch.SystMap()(1.099)
)

# CMS_NNLO_ggZZ
cb.cp().process(["ggZZ"]).AddSyst(cb, "CMS_NNLO_ggZZ", "lnN", ch.SystMap()(1.1))

# CMS electron efficiencies
syst_map = ch.SystMap("bin_id")([1, 2], 1.06)([3, 4], 1.04)([5, 6], 1.02)([7, 8], 1.0)
cb.cp().process(mc_bkgd).AddSyst(cb, "CMS_eff_e", "lnN", syst_map)
cb.cp().signals().AddSyst(cb, "CMS_eff_e", "lnN", syst_map)

# CMS muon efficiencies
syst_map = ch.SystMap("bin_id")([5, 7], 1.06)([6, 8], 1.04)([1, 3], 1.02)([2, 4], 1.0)
cb.cp().process(mc_bkgd).AddSyst(cb, "CMS_eff_m", "lnN", syst_map)
cb.cp().signals().AddSyst(cb, "CMS_eff_m", "lnN", syst_map)

# refs:
# https://twiki.cern.ch/twiki/bin/viewauth/CMS/StandardModelCrossSectionsat13TeV

# cross sections
cb.cp().process(["ggZZ"]).AddSyst(cb, "CMS_xsec_ggZZ", "lnN", ch.SystMap()(1.1))
cb.cp().process(["TT"]).AddSyst(cb, "CMS_xsec_top", "lnN", ch.SystMap()(1.06))
cb.cp().process(["TTW"]).AddSyst(cb, "CMS_xsec_ttW", "lnN", ch.SystMap()(1.25))
cb.cp().process(["TTZ"]).AddSyst(cb, "CMS_xsec_ttZ", "lnN", ch.SystMap()(1.25))

cb.cp().process(["ZZ", "WZ"]).AddSyst(cb, "CMS_xsec_vv", "lnN", ch.SystMap()(1.048))
cb.cp().process(["VVV"]).AddSyst(cb, "CMS_xsec_vvv", "lnN", ch.SystMap()(1.25))

# QCD scale VH
cb.cp().process(
    [
        "WHtt",
        "WHWW",
    ]
).AddSyst(cb, "QCDscale_VH", "lnN", ch.SystMap()(1.008))
cb.cp().process(["ZHtt", "ZHWW", "ggZHWW"]).AddSyst(
    cb, "QCDscale_VH", "lnN", ch.SystMap()(1.009)
)
cb.cp().process(["ggHtt", "ggHWW", "ggHZZ"]).AddSyst(
    cb, "QCDscale_ggh", "lnN", ch.SystMap()(1.039)
)
cb.cp().process(["VBFHtt", "VBFHWW"]).AddSyst(
    cb, "QCDscale_qqh", "lnN", ch.SystMap()(1.005)
)
cb.cp().process(["TTHtt"]).AddSyst(cb, "QCDscale_tth", "lnN", ch.SystMap()(1.08))

# pdf Higgs
cb.cp().process(["WHtt", "WHWW"]).AddSyst(
    cb, "pdf_Higgs_VH", "lnN", ch.SystMap()(1.018)
)
cb.cp().process(["ZHtt", "ZHWW", "ggZHWW"]).AddSyst(
    cb, "pdf_Higgs_VH", "lnN", ch.SystMap()(1.013)
)
cb.cp().process(["ggHtt", "ggHWW", "ggHZZ"]).AddSyst(
    cb, "pdf_Higgs_gg", "lnN", ch.SystMap()(1.032)
)
cb.cp().process(["VBFHtt", "VBFHWW"]).AddSyst(
    cb, "pdf_Higgs_qqbar", "lnN", ch.SystMap()(1.021)
)
cb.cp().process(["TTHtt"]).AddSyst(cb, "pdf_Higgs_ttH", "lnN", ch.SystMap()(1.036))

# add shape systematics
bkgd = mc_bkgd + sig
bkgd_mod = [b for b in bkgd if "ggHWW" not in b]
bkgd_tauID = [b for b in bkgd]
cb.cp().process(reducible).AddSyst(cb, "closure1", "shape", ch.SystMap()(1.00))
cb.cp().process(reducible).AddSyst(cb, "closure2", "shape", ch.SystMap()(1.00))
cb.cp().process(reducible).AddSyst(cb, "closure3", "shape", ch.SystMap()(1.00))
cb.cp().process(reducible).AddSyst(cb, "closure4", "shape", ch.SystMap()(1.00))
cb.cp().process(bkgd_tauID).AddSyst(cb, "tauID0", "shape", ch.SystMap()(1.00))
cb.cp().process(bkgd_tauID).AddSyst(cb, "tauID1", "shape", ch.SystMap()(1.00))
cb.cp().process(bkgd_tauID).AddSyst(cb, "tauID10", "shape", ch.SystMap()(1.00))
cb.cp().process(bkgd_tauID).AddSyst(cb, "tauID11", "shape", ch.SystMap()(1.00))
cb.cp().process([b for b in bkgd_mod if ("ggZHWW" not in b)]).AddSyst(
    cb, "tauES", "shape", ch.SystMap()(1.00)
)
cb.cp().process(
    [b for b in bkgd if (("ZHWW" not in b) and ("ggZHWW" not in b) and ("WZ" not in b))]
).AddSyst(cb, "unclMET", "shape", ch.SystMap()(1.00))
cb.cp().process(bkgd).AddSyst(cb, "pileup", "shape", ch.SystMap()(1.00))
cb.cp().process([b for b in bkgd if ("WZ" not in b)]).AddSyst(
    cb, "l1prefire", "shape", ch.SystMap()(1.00)
)
cb.cp().process(bkgd).AddSyst(cb, "eleES", "shape", ch.SystMap()(1.00))
cb.cp().process(bkgd).AddSyst(cb, "eleSmear", "shape", ch.SystMap()(1.00))
cb.cp().process(bkgd).AddSyst(cb, "muES", "shape", ch.SystMap()(1.00))
cb.cp().process(bkgd).AddSyst(cb, "efake", "shape", ch.SystMap()(1.00))
cb.cp().process(bkgd).AddSyst(cb, "mfake", "shape", ch.SystMap()(1.00))

# add bin uncertainty systematics
mc = uproot.open(
    "/uscms_data/d3/jdezoort/AZh_columnar/CMSSW_10_2_9/"
    + f"src/azh_coffea/src/notebooks/root_for_combine/MC_{btag_label}_{year}.root"
)
# for b in np.arange(15):
#    for i in cats:
#        cat, num = i[1], i[0]
#        #nums = [1, 2, 3, 4, 5, 6, 7, 8]
#        syst_map = ch.SystMap("bin_id")([num], 1.0)#([n for n in nums if n!=num], 0.0)
#        proc = mc_bkgd + reducible
#        keys = [k.decode('utf-8').strip(";1") for k in mc[cat].keys()]
#        #proc = [p for p in proc if f"{p}_bin{b}Up" in keys]
#        #print(cat, f"bin{b}", proc)
#        #cb.cp().process(proc).AddSyst(cb, f"bin{b}", "shape", syst_map)
#        for p in proc:
#            if f"{p}_{p}-{cat}-bin{b}Up" not in keys: continue
#            if "mmet" in cat and b==1 and "WZ" in p: continue
#            print(p, cat, b)
#            cb.cp().process([p]).AddSyst(cb, f"{p}-{cat}-bin{b}", "shape", syst_map)

# extract shapes
cb.cp().backgrounds().ExtractShapes(
    (
        "/uscms_data/d3/jdezoort/AZh_columnar/CMSSW_10_2_9/src/"
        + f"azh_coffea/src/notebooks/root_for_combine/MC_{btag_label}_{year}.root"
    ),
    "$BIN/$PROCESS",
    "$BIN/$PROCESS_$SYSTEMATIC",
)

cb.cp().signals().ExtractShapes(
    (
        "/uscms_data/d3/jdezoort/AZh_columnar/CMSSW_10_2_9/src/"
        + f"azh_coffea/src/notebooks/root_for_combine/signal_{mass}_{btag_label}_{year}.root"
    ),
    "$BIN/$PROCESS",
    "$BIN/$PROCESS_$SYSTEMATIC",
)

if per_category:
    writer = ch.CardWriter(
        "$TAG/$ANALYSIS_$ERA_$CHANNEL_$BIN_$MASS.txt",
        "$TAG/common/$ANALYSIS_$ERA_$CHANNEL_$BIN_$MASS.root",
    )
    writer.WriteCards(f"UL_{year}_{process}/", cb)
    print(f"wrote {mass}, {sig}")
else:
    writer = ch.CardWriter(
        "$TAG/$ANALYSIS_$ERA_$CHANNEL_$MASS.txt",
        "$TAG/common/$ANALYSIS_$ERA_$CHANNEL_$MASS.root",
    )
    writer.WriteCards(f"UL_{year}_{process}/", cb)
    print(f"wrote {mass}, {sig}")
