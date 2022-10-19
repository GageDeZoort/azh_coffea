from __future__ import annotations

import os
from os.path import join

import correctionlib
import numpy as np
from coffea import util


def zero_division(a, b):
    a[b == 0] = 0
    return a / b


def open_btag_file(root, year, UL=True):
    indir = join(root, f"UL_{year}")
    file = f"MC_UL_{year}_btag_effs.coffea"
    return util.load(os.path.join(indir, file))


def get_btag_tables(root, year, UL=True):
    out = open_btag_file(root, year, UL)
    tables = {}
    for sample in out.keys():
        data = out[sample]
        pt_bins = data["pt_bins"].value
        unique_pt_bins = np.unique(pt_bins)
        eta_bins = data["eta_bins"].value
        unique_eta_bins = np.unique(eta_bins)
        nbjets = data["nbjets"].value
        nbtags = data["nbtags"].value
        table = {}
        for i, pt_bin in enumerate(unique_pt_bins):
            for j, eta_bin in enumerate(unique_eta_bins):
                in_bin = (pt_bins == pt_bin) & (eta_bins == eta_bin)
                num = sum(nbtags[in_bin])
                denom = sum(nbjets[in_bin])
                eff = num / denom if denom != 0 else 0
                table[(i, j)] = eff
        mpt = len(unique_pt_bins)
        for i, _ in enumerate(unique_eta_bins):
            table[(mpt, i)] = table[(mpt - 1, i)]
        meta = len(unique_eta_bins)
        for i, _ in enumerate(unique_pt_bins):
            table[(i, meta)] = table[(i, meta - 1)]
        tables[sample] = table
    return tables, unique_pt_bins, unique_eta_bins


def get_btag_effs(table, pt_bins, eta_bins, sample, pt, eta):
    lookup = table[sample]
    p = np.digitize(pt, bins=pt_bins)
    e = np.digitize(eta, bins=eta_bins)
    pe = list(zip(p, e))
    return [lookup[t] for t in pe]


def get_btag_SFs(root, year, UL=True):
    infile = join(root, f"UL_{year}") if UL else join(root, f"Legacy_{year}")
    infile = join(infile, "btagging.json.gz")
    corr = correctionlib.CorrectionSet.from_file(infile)
    return corr["deepJet_comb"]
