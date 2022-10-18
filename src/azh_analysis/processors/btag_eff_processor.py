from __future__ import annotations

import sys

import awkward as ak
import numpy as np
from coffea import hist, processor
from coffea.processor import column_accumulator as col_acc


class bTagEffProcessor(processor.ProcessorABC):
    def __init__(self):
        self._accumulator = processor.dict_accumulator({})

    @property
    def accumulator(self):
        return self._accumulator

    @staticmethod
    def accumulate(a, flatten=True):
        if flatten:
            flat = ak.to_numpy(ak.flatten(a, axis=None))
        else:
            flat = ak.to_numpy(a)
        return processor.column_accumulator(flat)

    def process(self, events):
        self.output = self.accumulator.identity()
        dataset = events.metadata["dataset"]
        year = dataset.split("_")[-1]
        jet = events.Jet
        bjet = ak.flatten(jet[(abs(jet.partonFlavour) == 5)])
        delta = {
            "2016preVFP": 0.2598,
            "2016postVFP": 0.2598,
            "2017": 0.3040,
            "2018": 0.2783,
        }
        btag = bjet.btagDeepFlavB > delta[year]
        pt, eta = bjet.pt, bjet.eta
        pt_bins = np.arange(0, 260, 20)
        eta_bins = [0, 0.5, 1, 1.5, 2, 2.5, 3]
        pt_bin_centers = []
        eta_bin_centers = []
        nums, denoms = [], []
        for i, _ in enumerate(pt_bins[:-1]):
            for j, _ in enumerate(eta_bins[:-1]):
                in_bin = (
                    (pt >= pt_bins[i])
                    & (pt < pt_bins[i + 1])
                    & (abs(eta) >= eta_bins[j])
                    & (abs(eta) < eta_bins[j + 1])
                )
                denom = sum(in_bin)
                if denom == 0:
                    continue
                pt_bin_centers.append((pt_bins[i + 1] + pt_bins[i]) / 2)
                eta_bin_centers.append((eta_bins[j + 1] + eta_bins[j]) / 2)
                nums.append(sum(btag[in_bin]))
                denoms.append(denom)

        if dataset not in self.output.keys():
            d = {
                "n": col_acc(np.array([])),
                "dataset": col_acc(np.array([])),
                "pt_bins": col_acc(np.array([])),
                "eta_bins": col_acc(np.array([])),
                "nbjets": col_acc(np.array([])),
                "nbtags": col_acc(np.array([])),
            }
            self.output[dataset] = processor.dict_accumulator(d)

        self.output[dataset]["pt_bins"] += self.accumulate(pt_bin_centers)
        self.output[dataset]["eta_bins"] += self.accumulate(eta_bin_centers)
        self.output[dataset]["nbjets"] += self.accumulate(denoms)
        self.output[dataset]["nbtags"] += self.accumulate(nums)
        return self.output

    def postprocess(self, accumulator):
        return accumulator
