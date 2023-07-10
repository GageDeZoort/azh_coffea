from __future__ import annotations

from os.path import join

import awkward as ak
import correctionlib
import numpy as np
import uproot
from coffea.lookup_tools import extractor, rochester_lookup, txt_converters
from coffea.nanoevents.methods import vector

from azh_analysis.utils.btag import get_btag_effs
from azh_analysis.utils.parameters import get_lumis

ak.behavior.update(vector.behavior)


def get_sample_weight(info, name, year):
    lumi = get_lumis()
    properties = info[info["name"] == name]
    nevts, xsec = properties["nevts"][0], properties["xsec"][0]
    sample_weight = lumi[year] * xsec / nevts
    return sample_weight


def make_evaluator(file):
    ext = extractor()
    ext.add_weight_sets([f"* * {file}"])
    ext.finalize()
    return ext.make_evaluator()


def get_pileup_weights(base, year):
    pu_weights_dict = {}
    for shift in ["down", "nom", "up"]:
        ext = extractor()
        f = join(base, f"UL_{year}/puweight{year}_{shift}.histo.root")
        ext.add_weight_sets([f"weight weight {f}"])
        ext.finalize()
        pu_weights_dict[shift] = ext.make_evaluator()["weight"]
    return pu_weights_dict


def get_fake_rates(base, year, origin="_coffea"):
    fake_rates = {}
    ee_fr_file = f"JetEleFakeRate_Fall17MVAv2WP90_noIso_Iso0p15_UL{year}{origin}.root"
    mm_fr_file = f"JetMuFakeRate_Medium_Iso0p15_UL{year}{origin}.root"
    mt_fr_file = f"JetTauFakeRate_Medium_VLoose_Tight_UL{year}{origin}.root"
    et_fr_file = f"JetTauFakeRate_Medium_Tight_VLoose_UL{year}{origin}.root"
    tt_fr_file = f"JetTauFakeRate_Medium_VLoose_VLoose_UL{year}{origin}.root"

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


# def get_coffea_fake_rates(base, year):


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


def get_muon_ES_weights(base, year):
    filename = {
        "2018": "UL_2018/RoccoR2018UL.txt",
        "2017": "UL_2017/RoccoR2017UL.txt",
        "2016postVFP": "UL_2016postVFP/RoccoR2016bUL.txt",
        "2016preVFP": "UL_2016preVFP/RoccoR2016aUL.txt",
    }
    fname = filename[year]
    rochester_data = txt_converters.convert_rochester_file(
        join(base, fname), loaduncs=True
    )
    return rochester_lookup.rochester_lookup(rochester_data)


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


def tau_ID_weight_3l(taus, SF_tool, mode, syst="nom"):
    corr_VSe = SF_tool["DeepTau2017v2p1VSe"]
    corr_VSmu = SF_tool["DeepTau2017v2p1VSmu"]
    corr_VSjet = SF_tool["DeepTau2017v2p1VSjet"]
    pt = ak.to_numpy(taus.pt)
    eta = ak.to_numpy(taus.eta)
    gen = ak.to_numpy(taus.genPartFlav)
    dm = ak.to_numpy(taus.decayMode)
    wp_vsJet = "Medium"
    wp_vsEle = "Tight" if mode == "et" else "VLoose"
    wp_vsMu = "Tight" if mode == "mt" else "VLoose"
    tau_h_weight = corr_VSjet.evaluate(pt, dm, gen, wp_vsJet, syst, "pt")
    tau_ele_weight = corr_VSe.evaluate(eta, gen, wp_vsEle, syst)
    tau_mu_weight = corr_VSmu.evaluate(eta, gen, wp_vsMu, syst)
    tau_mu_weight = np.ones(len(pt), dtype=float)
    return tau_h_weight * tau_ele_weight * tau_mu_weight


def lepton_trig_weight(pt, eta, SF_tool, lep=-1):
    pt, eta = ak.to_numpy(pt), ak.to_numpy(eta)
    eta_map = {
        "e": {
            "Lt1p0": [0, 1],
            "1p0to1p48": [1.0001, 1.48],
            "1p48to1p65": [1.4801, 1.65],
            "1p65to2p1": [1.6501, 2.1],
            "Gt2p1": [2.101, 100],
        },
        "m": {
            "Lt0p9": [0, 0.9],
            "0p9to1p2": [0.9001, 1.2],
            "1p2to2p1": [1.2001, 2.1],
            "Gt2p1": [2.1001, 100],
        },
    }
    eta_map = eta_map[lep]
    weight = np.zeros(len(pt), dtype=float)
    for key, eta_range in eta_map.items():
        mask = (abs(eta) > eta_range[0]) & (abs(eta) <= eta_range[1])
        if len(mask) == 0:
            continue
        weight += SF_tool[key](pt) * mask
    weight[weight == 0] = 1
    return weight


def dyjets_stitch_weights(info, nevts_dict, year):
    lumi = get_lumis(as_picobarns=True)[year]
    # sort the nevts and xsec by the number of jets
    dyjets = info[info["group"] == "DY"]
    bins = np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5])
    xsec = np.sort(np.unique(dyjets["xsec"]))

    label = f"{year}"
    # if "2016" in year:
    #    label = f"{year.split('16')[-1]}_{year}"
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


def get_electron_ES_weights(root, year):
    return correctionlib.CorrectionSet.from_file(
        join(root, f"UL_{year}/EGM_ScaleUnc.json.gz")
    )["UL-EGM_ScaleUnc"]


def apply_eleES(
    ele, eleES_shifts, year, eleES_shift="nom", eleSmear_shift="nom", is_data=False
):
    if is_data:
        return ele, {"x": 0, "y": 0}
    # decide ES weights by region of the detector
    ele, num = ak.flatten(ele), ak.num(ele)

    shift = "scaleunc"
    if "up" in eleES_shift.lower():
        shift = "scaleup"
    if "down" in eleES_shift.lower():
        shift = "scaledown"
    eta = ele.eta
    eleES_weights = eleES_shifts.evaluate(year, shift, eta, 1)
    if ("up" not in shift) and ("down" not in shift):
        eleES_weights = np.ones(len(ele))

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
    ele["pt"] = ele_p4_shift["pt"]
    ele["eta"] = ele_p4_shift["eta"]
    ele["phi"] = ele_p4_shift["phi"]
    ele["mass"] = ele_p4_shift["mass"]

    return (
        ak.unflatten(ele, num),
        {
            "x": ak.sum(ak.unflatten(ele_x_diff, num), axis=1),
            "y": ak.sum(ak.unflatten(ele_y_diff, num), axis=1),
        },
    )


def apply_muES(mu, rochester, shift="nom", is_data=False):
    if is_data:
        weights = rochester.kScaleDT(mu.charge, mu.pt, mu.eta, mu.phi)
        weights = ak.flatten(weights)
    else:
        hasgen = ~np.isnan(ak.fill_none(mu.matched_gen.pt, np.nan))
        mc_rand = np.random.rand(*ak.to_numpy(ak.flatten(mu.pt)).shape)
        mc_rand = ak.unflatten(mc_rand, ak.num(mu.pt, axis=1))
        weights = np.array(ak.flatten(ak.ones_like(mu.pt)))

        mc_kspread = rochester.kSpreadMC(
            mu.charge[hasgen],
            mu.pt[hasgen],
            mu.eta[hasgen],
            mu.phi[hasgen],
            mu.matched_gen.pt[hasgen],
        )
        mc_ksmear = rochester.kSmearMC(
            mu.charge[~hasgen],
            mu.pt[~hasgen],
            mu.eta[~hasgen],
            mu.phi[~hasgen],
            mu.nTrackerLayers[~hasgen],
            mc_rand[~hasgen],
        )
        hasgen_flat = np.array(ak.flatten(hasgen))
        weights[hasgen_flat] = np.array(ak.flatten(mc_kspread))
        weights[~hasgen_flat] = np.array(ak.flatten(mc_ksmear))

    mu, num = ak.flatten(mu), ak.num(mu)
    mu_p4 = ak.zip(
        {"pt": mu.pt, "eta": mu.eta, "phi": mu.phi, "mass": mu.mass},
        with_name="PtEtaPhiMLorentzVector",
    )
    mu_p4_shift = weights * mu_p4
    mu_x_diff = (1 - weights) * mu.pt * np.cos(mu.phi)
    mu_y_diff = (1 - weights) * mu.pt * np.sin(mu.phi)
    mu["pt"] = mu_p4_shift["pt"]
    mu["eta"] = mu_p4_shift["eta"]
    mu["phi"] = mu_p4_shift["phi"]
    mu["mass"] = mu_p4_shift["mass"]
    return (
        ak.unflatten(mu, num),
        {
            "x": ak.sum(ak.unflatten(mu_x_diff, num), axis=1),
            "y": ak.sum(ak.unflatten(mu_y_diff, num), axis=1),
        },
    )


def apply_tauES(
    taus,
    SF_tool,
    tauES_shift="nom",
    efake_shift="nom",
    mfake_shift="nom",
    is_data=False,
):
    if is_data:
        return taus, {"x": 0, "y": 0}
    # set up masks for use in the correctionlib tool
    taus, num = ak.flatten(taus), ak.num(taus)
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

    taus["pt"] = tau_p4.pt
    taus["eta"] = tau_p4.eta
    taus["phi"] = tau_p4.phi
    taus["mass"] = tau_p4.mass
    return (
        ak.unflatten(taus, num),
        {
            "x": ak.sum(ak.unflatten(tau_x_diff, num), axis=1),
            "y": ak.sum(ak.unflatten(tau_y_diff, num), axis=1),
        },
    )


def shift_MET(met, diffs_list, is_data=False):
    if is_data:
        return met
    # met, num = ak.flatten(met), ak.num(met)
    met_x = met.pt * np.cos(met.phi)
    met_y = met.pt * np.sin(met.phi)
    for diffs in diffs_list:
        met_x = met_x + diffs["x"]
        met_y = met_y + diffs["y"]
    met_p4 = ak.zip({"x": met_x, "y": met_y, "z": 0, "t": 0}, with_name="LorentzVector")
    met["pt"] = met_p4.pt
    met["phi"] = met_p4.phi
    return met  # ak.unflatten(met, num)


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


def apply_btag_corrections(
    jets, btag_SFs, btag_eff_tables, btag_pt_bins, btag_eta_bins, dataset, shift
):
    """
    https://twiki.cern.ch/twiki/bin/view/CMS/BTagSFMethods
    """
    jets = jets[abs(jets.partonFlavour) == 5]
    flat_j, num_j = ak.flatten(jets), ak.num(jets)
    pt, eta = flat_j.pt, flat_j.eta
    delta = {
        "2016preVFP": 0.2598,
        "2016postVFP": 0.2598,
        "2017": 0.3040,
        "2018": 0.2783,
    }
    year = dataset.split("_")[-1]
    is_tagged = flat_j.btagDeepFlavB > delta[year]
    SFs = btag_SFs.evaluate(shift, "M", 5, abs(ak.to_numpy(eta)), ak.to_numpy(pt))
    btag_effs = np.array(
        get_btag_effs(
            btag_eff_tables,
            btag_pt_bins,
            btag_eta_bins,
            dataset,
            pt,
            abs(eta),
        )
    )
    w_is_tagged = is_tagged * btag_effs
    w_not_tagged = (1 - btag_effs) * ~is_tagged
    w_MC = w_is_tagged + w_not_tagged
    w_MC = ak.prod(ak.unflatten(w_MC, num_j), axis=1)
    w_is_tagged = btag_effs * is_tagged * SFs
    w_is_not_tagged = (1 - btag_effs * SFs) * ~is_tagged
    w_data = w_is_tagged + w_is_not_tagged
    w_data = ak.prod(ak.unflatten(w_data, num_j), axis=1)
    weight = w_data / w_MC
    return weight
