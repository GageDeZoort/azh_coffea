from __future__ import annotations

import os
from os.path import join

import numpy as np
import uproot
from coffea import util


def make_pileup_weights_file(
    root,
    year,
):
    if "UL" not in root:
        root = join(root, f"UL_{year}")
    print(root)
    # read MC nTrueInt distributions
    mc_pu = util.load(join(root, f"MC_UL_{year}_PU.coffea"))["pileup_mc"]
    samples = mc_pu.to_hist().axes["dataset"]
    mc_pu = sum([mc_pu[d, :].to_hist() for d in samples])
    mc_pu = mc_pu[::sum, :].values()
    mc_pu = (mc_pu / np.sum(mc_pu))[:-1]

    # draw data distribution
    for shift in ["up", "nom", "down"]:
        data_pu, data_bins = open_pileup_file(indir=root, year=year, shift=shift)
        data_pu = data_pu / np.sum(data_pu)
        weights = data_pu / mc_pu

        outfile = join(root, f"puweight{year}_{shift}.histo.root")
        print("Opening", outfile)
        with uproot.recreate(outfile) as f:
            h = np.histogram(
                np.arange(len(weights)),
                bins=np.arange(0, 100, 1),
                weights=np.nan_to_num(weights, posinf=0, neginf=0),
            )
            f["weight"] = h
            print("Writing", h)


def open_pileup_file(indir, year, UL=True, shift="nom"):
    if "UL" not in indir:
        indir = join(indir, f"UL_{year}")
    mb_xsec = {"up": "66000ub", "nom": "69200ub", "down": "72400ub"}
    xsec = mb_xsec[shift]
    if year == "2016postVFP":
        year = "2016-postVFP"
    if year == "2016preVFP":
        year = "2016-preVFP"
    file = f"PileupHistogram-goldenJSON-13tev-{year}-{xsec}-99bins.root"
    pileup_data = uproot.open(os.path.join(indir, file))
    key = pileup_data.keys()[0]
    return pileup_data[key].to_numpy()


# def get_pileup_table(pileup_MC, year, shift=None, UL=False):
#    pileup_data, bins = open_pileup_file(year, UL=UL, shift=shift)
#    integral_data = np.sum(pileup_data)
#    pileup_MC, bins = np.histogram(pileup_MC, bins, density=True)
#    ratios = (pileup_data / integral_data) / pileup_MC
#    ratios[np.isinf(ratios) | np.isnan(ratios)] = 0
#    return ratios


# def get_pileup_tables(names, year, shift=None, UL=False, pileup_dir=""):
#    pileup_data, bins = open_pileup_file(
#        year, UL=UL, shift=shift, pileup_dir=pileup_dir
#    )
#    integral_data = np.sum(pileup_data)
#    legacy_str = "UL" if UL else "Legacy"
#    pileup_mc_indir = join(pileup_dir, f"{legacy_str}_{year}")
#    pileup_mc_file = os.path.join(pileup_mc_indir, f"MC_{legacy_str}_{year}_PU.coffea")
#    pileup_mcs = util.load(pileup_mc_file)["pileup_mc"].values()
#    weight_dict = {}
#    for name in names:
#        pileup_mc = pileup_mcs[(name,)][:-1]
#        integral_mc = np.sum(pileup_mc)
#        weights = zero_division(pileup_data / integral_data, pileup_mc / integral_mc)
#        weights = np.nan_to_num(weights, neginf=0, posinf=0)
#        weight_dict[name] = weights

#    return weight_dict


# def get_pileup_weights(pileup_MC, bin_weights, bins):
#    pileup_MC = pileup_MC.to_numpy().astype(int)
#    bin_idx = np.digitize(pileup_MC, bins) - 1
#    # bin_idx[bin_idx<0] = 0
#    bin_idx[bin_idx > 98] = 98
#    bin_idx[bin_idx < 0] = 0
#    return bin_weights[bin_idx]
