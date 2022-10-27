from __future__ import annotations

from os.path import join

import awkward as ak
import correctionlib
import numpy as np
import uproot
from coffea.lookup_tools import extractor
from coffea.nanoevents.methods import vector

ak.behavior.update(vector.behavior)


def get_sample_weight(info, name, year):
    lumi = {"2016": 35.9, "2017": 41.5, "2018": 59.7}
    properties = info[info["name"] == name]
    nevts, xsec = properties["nevts"][0], properties["xsec"][0]
    sample_weight = lumi[year] * xsec / nevts
    return sample_weight


def make_evaluator(file):
    ext = extractor()
    ext.add_weight_sets([f"* * {file}"])
    ext.finalize()
    return ext.make_evaluator()


def get_fake_rates(base, year):
    fake_rates = {}
    ee_fr_file = f"JetEleFakeRate_Fall17MVAv2WP90_noIso_Iso0p15_UL{year}.root"
    mm_fr_file = f"JetMuFakeRate_Medium_Iso0p15_UL{year}.root"
    mt_fr_file = f"JetTauFakeRate_Medium_VLoose_Tight_UL{year}.root"
    et_fr_file = f"JetTauFakeRate_Medium_Tight_VLoose_UL{year}.root"
    tt_fr_file = f"JetTauFakeRate_Medium_VLoose_VLoose_UL{year}.root"

    fake_rate_files = {
        "ee": join(base, ee_fr_file),
        "mm": join(base, mm_fr_file),
        "mt": join(base, mt_fr_file),
        "et": join(base, et_fr_file),
        "tt": join(base, tt_fr_file),
    }

    for lep, fr_file in fake_rate_files.items():
        evaluator = make_evaluator(fr_file)
        if (lep == "ee") or (lep == "mm"):
            fake_rates[lep] = {
                "barrel": evaluator["POL2FitFR_Central_barrel"],
                "endcap": evaluator["POL2FitFR_Central_endcap"],
            }
        else:
            fake_rates[lep] = {
                "barrel": {
                    0: evaluator["POL2FitFR_Central_DM0"],
                    1: evaluator["POL2FitFR_Central_DM1"],
                    10: evaluator["POL2FitFR_Central_DM10"],
                    11: evaluator["POL2FitFR_Central_DM11"],
                },
                "endcap": {
                    0: evaluator["POL2FitFR_Central_DM0"],
                    1: evaluator["POL2FitFR_Central_DM1"],
                    10: evaluator["POL2FitFR_Central_DM10"],
                    11: evaluator["POL2FitFR_Central_DM11"],
                },
            }
    return fake_rates


class CustomWeights:
    def __init__(self, bins, weights):
        self.bins = bins
        self.weights = weights
        self.max_bin = np.argmax(bins)
        self.min_bin = np.argmin(bins)

    def apply(self, array):
        bin_idx = np.digitize(array, self.bins) - 1
        bin_idx[bin_idx < self.min_bin] = self.min_bin
        bin_idx[bin_idx > self.max_bin] = self.max_bin
        return self.weights[bin_idx]

    def __call__(self, array):
        return self.apply(array)

    def __repr__(self):
        return "CustomWeights()"

    def __str__(self):
        out = f"CustomWeights()\n - bins: {self.bins}\n - weights: {self.weights}"
        return out


def get_electron_ID_weights(infile):
    f = uproot.open(infile)
    eta_map = {"Lt1p0": 0, "1p0to1p48": 0, "1p48to1p65": 0, "1p65to2p1": 0, "Gt2p1": 0}
    for eta_range in eta_map.keys():
        mc_bins, mc_counts = f[f"ZMassEta{eta_range}_MC;1"].values()
        data_bins, data_counts = f[f"ZMassEta{eta_range}_Data;1"].values()
        ratio = np.nan_to_num(data_counts / mc_counts, 0, posinf=0, neginf=0)
        weights = CustomWeights(data_bins, ratio)
        eta_map[eta_range] = weights
    return eta_map


def get_muon_ID_weights(infile):
    f = uproot.open(infile)
    eta_map = {"Lt0p9": 0, "0p9to1p2": 0, "1p2to2p1": 0, "Gt2p1": 0}
    for eta_range in eta_map.keys():
        mc_bins, mc_counts = f[f"ZMassEta{eta_range}_MC;1"].values()
        data_bins, data_counts = f[f"ZMassEta{eta_range}_Data;1"].values()
        ratio = np.nan_to_num(data_counts / mc_counts, 0, posinf=0, neginf=0)
        weights = CustomWeights(data_bins, ratio)
        eta_map[eta_range] = weights
    return eta_map


def get_electron_trigger_SFs(infile):
    trigger_SFs = uproot.open(infile)
    eta_map = {"Lt1p0": 0, "1p0to1p48": 0, "1p48to1p65": 0, "1p65to2p1": 0, "Gt2p1": 0}
    for eta_range in eta_map.keys():
        mc_bins, mc_counts = trigger_SFs[f"ZMassEta{eta_range}_MC;1"].values()
        data_bins, data_counts = trigger_SFs[f"ZMassEta{eta_range}_Data;1"].values()
        ratio = np.nan_to_num(data_counts / mc_counts, 0, posinf=0, neginf=0)
        weights = CustomWeights(data_bins, ratio)
        eta_map[eta_range] = weights
    return eta_map


def get_muon_trigger_SFs(infile):
    trigger_SFs = uproot.open(infile)
    eta_map = {"Lt0p9": 0, "0p9to1p2": 0, "1p2to2p1": 0, "Gt2p1": 0}
    for eta_range in eta_map.keys():
        mc_bins, mc_counts = trigger_SFs[f"ZMassEta{eta_range}_MC;1"].values()
        data_bins, data_counts = trigger_SFs[f"ZMassEta{eta_range}_Data;1"].values()
        ratio = np.nan_to_num(data_counts / mc_counts, 0, posinf=0, neginf=0)
        weights = CustomWeights(data_bins, ratio)
        eta_map[eta_range] = weights
    return eta_map


def get_tau_ID_weights(infile):
    return correctionlib.CorrectionSet.from_file(infile)


def lepton_ID_weight(l, lep, SF_tool, is_data=False):
    eta_map = {
        "e": {
            "Lt1p0": [0, 1],
            "1p0to1p48": [1, 1.48],
            "1p48to1p65": [1.48, 1.65],
            "1p65to2p1": [1.65, 2.1],
            "Gt2p1": [2.1, 100],
        },
        "m": {
            "Lt0p9": [0, 0.9],
            "0p9to1p2": [0.9, 1.2],
            "1p2to2p1": [1.2, 2.1],
            "Gt2p1": [2.1, 100],
        },
    }
    eta_map = eta_map[lep]
    eta = ak.to_numpy(abs(l.eta))
    pt = ak.to_numpy(l.pt)
    weight = np.zeros(len(l), dtype=float)
    for key, eta_range in eta_map.items():
        mask = (eta > eta_range[0]) & (eta <= eta_range[1])
        if len(mask) == 0:
            continue
        weight += SF_tool[key](pt) * mask
    return weight


def tau_ID_weight(taus, SF_tool, cat, is_data=False, syst="nom", tight=True):
    corr_VSe = SF_tool["DeepTau2017v2p1VSe"]
    corr_VSmu = SF_tool["DeepTau2017v2p1VSmu"]
    corr_VSjet = SF_tool["DeepTau2017v2p1VSjet"]
    pt = ak.to_numpy(taus.pt)
    eta = ak.to_numpy(taus.eta)
    gen = ak.to_numpy(taus.genPartFlav)
    dm = ak.to_numpy(taus.decayMode)
    wp_vsJet = "VVVLoose" if not tight else "Medium"
    wp_vsEle = "Tight" if (tight and (cat[2:] == "et")) else "VLoose"
    wp_vsMu = "Tight" if (tight and (cat[2:] == "mt")) else "VLoose"
    tau_h_weight = corr_VSjet.evaluate(pt, dm, gen, wp_vsJet, syst, "pt")
    tau_ele_weight = corr_VSe.evaluate(eta, gen, wp_vsEle, syst)
    if not tight:
        tau_ele_weight = np.ones(len(pt), dtype=float)
    tau_mu_weight = corr_VSmu.evaluate(eta, gen, wp_vsMu, syst)
    if not tight:
        tau_mu_weight = np.ones(len(pt), dtype=float)
    return tau_h_weight * tau_ele_weight * tau_mu_weight


def lepton_trig_weight(w, pt, eta, SF_tool, lep=-1):
    pt, eta = ak.to_numpy(pt), ak.to_numpy(eta)
    eta_map = {
        "e": {
            "Lt1p0": [0, 1],
            "1p0to1p48": [1, 1.48],
            "1p48to1p65": [1.48, 1.65],
            "1p65to2p1": [1.65, 2.1],
            "Gt2p1": [2.1, 100],
        },
        "m": {
            "Lt0p9": [0, 0.9],
            "0p9to1p2": [0.9, 1.2],
            "1p2to2p1": [1.2, 2.1],
            "Gt2p1": [2.1, 100],
        },
    }
    eta_map = eta_map[lep]
    weight = np.zeros(len(w), dtype=float)
    for key, eta_range in eta_map.items():
        mask = (abs(eta) > eta_range[0]) & (abs(eta) <= eta_range[1])
        if len(mask) == 0:
            continue
        weight += SF_tool[key](pt) * mask
    weight[weight == 0] = 1
    return weight


def dyjets_stitch_weights(info, nevts_dict, year):
    lumis = {
        "2016preVFP": 35.9 * 1000,
        "2016postVFP": 35.9 * 1000,
        "2017": 41.5 * 1000,
        "2018": 59.7 * 1000,
    }
    lumi = lumis[year]
    # sort the nevts and xsec by the number of jets
    dyjets = info[info["group"] == "DY"]
    bins = np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5])
    xsec = np.sort(np.unique(dyjets["xsec"]))

    label = f"{year}"
    print(nevts_dict.keys())
    if "2016" in year:
        label = f"{year.split('16')[-1]}_{year}"
    ninc, xinc = nevts_dict[f"DYJetsToLLM-50_{label}"], xsec[4]
    n1, x1 = nevts_dict[f"DY1JetsToLLM-50_{label}"], xsec[3]
    n2, x2 = nevts_dict[f"DY2JetsToLLM-50_{label}"], xsec[2]
    n3, x3 = nevts_dict[f"DY3JetsToLLM-50_{label}"], xsec[1]
    n4, x4 = nevts_dict[f"DY4JetsToLLM-50_{label}"], xsec[0]
    n1_corr = ninc * x1 / xinc + n1
    n2_corr = ninc * x2 / xinc + n2
    n3_corr = ninc * x3 / xinc + n3
    n4_corr = ninc * x4 / xinc + n4
    w0 = lumi * xinc / ninc
    w1 = lumi * x1 / n1_corr
    w2 = lumi * x2 / n2_corr
    w3 = lumi * x3 / n3_corr
    w4 = lumi * x4 / n4_corr
    weights = np.array([w0, w1, w2, w3, w4, w0])
    return CustomWeights(bins, weights)


def apply_eleES(ele, eleES_shift="nom", eleSmear_shift="nom"):
    # decide ES weights by region of the detector
    in_barrel = abs(ele.eta) < 1.479
    in_crossover = (abs(ele.eta) > 1.479) & (abs(ele.eta) < 1.653)
    in_endcap = abs(ele.eta) > 1.653
    barrel_shifts = {"up": 1.03, "nom": 1.0, "down": 0.97}
    crossover_shifts = {"up": 1.04, "nom": 1.0, "down": 0.96}
    endcap_shifts = {"up": 1.05, "nom": 1.0, "down": 0.95}
    eleES_weights = (
        in_barrel * barrel_shifts[eleES_shift]
        + in_crossover * crossover_shifts[eleES_shift]
        + in_endcap * endcap_shifts[eleES_shift]
    )

    # get smearing weights
    if eleSmear_shift == "nom":
        shift = 0
    else:
        shift = ele.dEsigmaUp if (eleSmear_shift == "up") else ele.dEsigmaDown
    eleSmear_weights = shift + 1.0

    ele_p4 = ak.zip(
        {"pt": ele.pt, "eta": ele.eta, "phi": ele.phi, "mass": ele.mass},
        with_name="PtEtaPhiMLorentzVector",
    )

    # apply weights
    weights = eleES_weights * eleSmear_weights
    ele_p4_shift = weights * ele_p4
    ele_x_diff = (1 - weights) * ele.pt * np.cos(ele.phi)
    ele_y_diff = (1 - weights) * ele.pt * np.sin(ele.phi)
    return ele_p4_shift, {"x": ele_x_diff, "y": ele_y_diff}


def apply_muES(mu, syst="nom"):
    # grab weights corresponding to systematic shifts
    if syst == "nom":
        shifts = np.zeros(len(mu))
        return mu, {"x": shifts, "y": shifts}
    shifts = {"up": 1.01, "down": 0.99}
    weights = shifts[syst]
    mu_p4 = ak.zip(
        {"pt": mu.pt, "eta": mu.eta, "phi": mu.phi, "mass": mu.mass},
        with_name="PtEtaPhiMLorentzVector",
    )

    # apply weights
    mu_p4_shift = weights * mu_p4
    mu_x_diff = (1 - weights) * mu.pt * np.cos(mu.phi)
    mu_y_diff = (1 - weights) * mu.pt * np.sin(mu.phi)
    return mu_p4_shift, {"x": mu_x_diff, "y": mu_y_diff}


def apply_tauES(taus, SF_tool, tauES_shift="nom", efake_shift="nom", mfake_shift="nom"):
    # set up masks for use in the correctionlib tool
    corr = SF_tool["tau_energy_scale"]
    mask = (
        (taus.decayMode == 0)
        | (taus.decayMode == 1)
        | (taus.decayMode == 2)
        | (taus.decayMode == 10)
        | (taus.decayMode == 11)
    )
    tauES_mask = mask & (taus.genPartFlav == 5)
    efake_mask = mask & ((taus.genPartFlav == 1) | (taus.genPartFlav == 3))
    mfake_mask = mask & ((taus.genPartFlav == 2) | (taus.genPartFlav == 4))
    corrs = {
        "tauES_shift": {"mask": tauES_mask, "shift": tauES_shift},
        "efake_shift": {"mask": efake_mask, "shift": efake_shift},
        "mfake_shift": {"mask": mfake_mask, "shift": mfake_shift},
    }

    # initial four vectors
    tau_p4 = ak.zip(
        {"pt": taus.pt, "eta": taus.eta, "phi": taus.phi, "mass": taus.mass},
        with_name="PtEtaPhiMLorentzVector",
    )
    tau_x_diff, tau_y_diff = 0, 0

    # loop over different shifts
    for _, items in corrs.items():
        mask, shift = items["mask"], items["shift"]
        pt, eta = taus.pt[mask], taus.eta[mask]
        dm, gen = taus.decayMode[mask], taus.genPartFlav[mask]
        TES_new = corr.evaluate(
            ak.to_numpy(pt),
            ak.to_numpy(eta),
            ak.to_numpy(dm),
            ak.to_numpy(gen),
            "DeepTau2017v2p1",
            shift,
        )
        TES = np.ones(len(mask), dtype=float)
        TES[mask] = TES_new
        tau_x_diff = tau_x_diff + (1 - TES) * tau_p4.pt * np.cos(tau_p4.phi)
        tau_y_diff = tau_y_diff + (1 - TES) * tau_p4.pt * np.sin(tau_p4.phi)
        tau_p4 = TES * tau_p4

    return tau_p4, {"x": tau_x_diff, "y": tau_y_diff}


def apply_unclMET_shifts(met, shift="nom"):
    if shift == "nom":
        return met
    met_x = met.pt * np.cos(met.phi)
    met_y = met.py * np.sin(met.phi)
    f = 1 if shift == "up" else -1
    met_x = met_x + f * met.MetUnclustEnUpDeltaX
    met_y = met_y + f * met.MetUnclustEnUpDeltaY
    met_p4 = ak.zip({"x": met_x, "y": met_y, "z": 0, "t": 0}, with_name="LorentzVector")
    met["pt"] = met_p4.pt
    met["phi"] = met_p4.phi
    return met
