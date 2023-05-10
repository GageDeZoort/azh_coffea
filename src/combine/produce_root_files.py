from __future__ import annotations

import uproot
from coffea import util
from hist import Hist
from hist.axis import Variable


def get_empty_hist():
    return Hist(
        Variable(
            [
                200,
                220,
                240,
                260,
                280,
                300,
                320,
                340,
                360,
                380,
                400,
                450,
                550,
                700,
                1000,
                2400,
            ],
            name="mass",
        )
    )


mc = util.load("../output/MC_UL_2018_all_OS.coffea")
signal = util.load("../output/signal_UL_2018_all_OS.coffea")
data = util.load("../output/data_UL_2018_OS_ub.coffea")

group_labels = {
    "TT": [
        "TTToSemiLeptonic",
        "TTToHadronic",
        "TTTo2L2Nu",
    ],
    "TTZ": [
        "ttZJets",
    ],
    "TTW": [
        "TTWJetsToLNu",
    ],
    "ggZZ": [
        "GluGluToContinToZZTo2e2tau",
        "GluGluToContinToZZTo2mu2tau",
        "GluGluToContinToZZTo4e",
        "GluGluToContinToZZTo4mu",
        "GluGluToContinToZZTo4tau",
    ],
    "ZZ": [
        "ZZTo4L",
        "ZZTo2Q2Lmllmin4p0",
    ],
    "WZ": [
        "WZTo2Q2Lmllmin4p0",
        "WZTo3LNu",
    ],
    "VVV": [
        "WWW4F",
        "WWW4F_ext1",
        "WWZ4F",
        "WZZ",
        "WZZTuneCP5_ext1",
        "ZZZ",
        "ZZZTuneCP5_ext1",
    ],
    "ggHtt": [
        "GluGluHToTauTauM125",
    ],
    "VBFHtt": [
        "VBFHToTauTauM125",
    ],
    "WHtt": [
        "WminusHToTauTauM125",
        "WplusHToTauTauM125",
    ],
    "ZHtt": [
        "ZHToTauTauM125_ext1",
    ],
    "TTHtt": [
        "ttHToTauTauM125",
    ],
    "ggHWW": [
        "GluGluHToWWTo2L2NuM-125",
    ],
    "VBFHWW": [
        "VBFHToWWTo2L2NuM-125",
    ],
    "ggZHWW": [
        "GluGluZHHToWW",
    ],
    "ggHZZ": [
        "GluGluHToZZTo4LM125",
    ],
    "WHWW": [
        "HWminusJHToWW",
        "HWplusJHToWWTo2L2Nu",
    ],
    "ZHWW": [
        "HZJHToWW",
    ],
}

cats = ["eeet", "eemt", "eett", "eeem", "mmet", "mmmt", "mmtt", "mmem"]
systs = [
    "nom",
    "l1prefire_up",
    "l1prefire_down",
    "pileup_up",
    "pileup_down",
    "tauES_down",
    "tauES_up",
    "efake_down",
    "efake_up",
    "mfake_down",
    "mfake_up",
    "eleES_down",
    "eleES_up",
    "eleSmear_down",
    "eleSmear_up",
    "muES_down",
    "muES_up",
    "unclMET_down",
    "unclMET_up",
]

for b in [0, 1]:
    btag_label = "btag" if (b == 1) else "0btag"

    # fill MC output ROOT file
    mc_file = uproot.recreate(f"MC_{btag_label}_2018.root")
    for group, datasets in group_labels.items():
        mc_group = sum(v for k, v in mc["m4l"].items() if k in datasets)
        for cat in cats:
            for syst in systs:
                if "btag" in syst:
                    continue
                if (cat not in list(mc_group.axes[1])) or (
                    syst not in list(mc_group.axes[3])
                ):
                    group_hist = get_empty_hist()
                else:
                    group_hist = mc_group[::sum, cat, b, syst, "cons", :]

                if "nom" in syst:
                    fname = f"{cat}/{group}"
                    mc_file[fname] = group_hist.to_numpy()
                else:
                    shift = syst.split("_")[-1]
                    syst = syst.replace(f"_{shift}", "")
                    syst = syst + shift.capitalize()
                    fname = f"{cat}/{group}_{syst}"
                    mc_file[fname] = group_hist.to_numpy()

    data_group = sum(v for k, v in data["m4l"].items())

    # fill reducible into the MC file
    for cat in cats:
        if cat not in list(data_group.axes[1]):
            group_hist = get_empty_hist()
        else:
            group_hist = data_group["reducible", cat, b, ::sum, "cons", :]
        fname = f"{cat}/reducible"
        mc_file[fname] = group_hist.to_numpy()

    # now fill the fill data
    data_file = uproot.recreate(f"data_2018_{btag_label}.root")
    data_group = sum(v for k, v in data["m4l"].items())
    for cat in cats:
        if cat not in list(data_group.axes[1]):
            group_hist = get_empty_hist()
        else:
            group_hist = data_group["data", cat, b, ::sum, "cons", :]

        fname = f"{cat}/data"
        data_file[fname] = group_hist.to_numpy()

    for k, v in signal["m4l"].items():
        if (b == 0) and "GluGlu" not in k:
            continue
        if (b > 0) and "BBA" not in k:
            continue
        label = "ggA" if b == 0 else "bbA"
        k = label + f"_{k.split('TauM')[-1]}"
        file = uproot.recreate(f"{k}_2018.root")
        for cat in cats:
            for syst in systs:
                if "btag" in syst:
                    continue
                if (cat not in list(v.axes[1])) or (syst not in list(v.axes[3])):
                    group_hist = get_empty_hist()
                else:
                    group_hist = v[::sum, cat, b, syst, "cons", :]

                if "nom" in syst:
                    fname = f"{cat}/{k}"
                    file[fname] = group_hist.to_numpy()
                else:
                    shift = syst.split("_")[-1]
                    syst = syst.replace(f"_{shift}", "")
                    syst = syst + shift.capitalize()
                    fname = f"{cat}/{k}_{syst}"
                    file[fname] = group_hist.to_numpy()
