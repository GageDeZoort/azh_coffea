from __future__ import annotations

import awkward as ak
import numba as nb
import numpy as np

# from time import time


# in use
def filter_MET(events, selections, year, data=False):
    flags = events.Flag
    if (year == "2017") or (year == "2018"):
        MET_filter = (
            flags.goodVertices
            & flags.globalSuperTightHalo2016Filter
            & flags.HBHENoiseFilter
            & flags.HBHENoiseIsoFilter
            & flags.EcalDeadCellTriggerPrimitiveFilter
            & flags.BadPFMuonFilter
            & flags.eeBadScFilter
            & flags.ecalBadCalibFilter
        )
        # MET_filter = MET_filter & flags.BadPFMuonDzFilter

    elif "2016" in year:
        MET_filter = (
            flags.goodVertices
            & flags.globalSuperTightHalo2016Filter
            & flags.HBHENoiseFilter
            & flags.HBHENoiseIsoFilter
            & flags.EcalDeadCellTriggerPrimitiveFilter
            & flags.BadPFMuonFilter
            & flags.BadPFMuonDzFilter
            & flags.eeBadScFilter
        )
    else:
        raise Exception("Please enter a valid year.")
    selections.add("met_filter", MET_filter)


# in use
def filter_PV(events, selections):
    PV = events.PV
    pv_filter = (PV.ndof > 4) & (abs(PV.z) < 24) & (np.sqrt(PV.x**2 + PV.y**2) < 2)
    selections.add("pv_filter", pv_filter)


# in use
def get_baseline_electrons(electrons):
    baseline_e = electrons[
        (np.abs(electrons.dxy) < 0.045) & (np.abs(electrons.dz) < 0.2)
    ]

    baseline_e = baseline_e[(baseline_e.lostHits < 2)]
    baseline_e = baseline_e[(baseline_e.convVeto)]
    baseline_e = baseline_e[(baseline_e.pt > 10)]
    baseline_e = baseline_e[(np.abs(baseline_e.eta) < 2.5)]
    return baseline_e


# in use
def tight_electrons(electrons, relaxed=False):
    if relaxed:
        return electrons.pfRelIso03_all < 10**6
    return (electrons.mvaFall17V2noIso_WP90 > 0) & (electrons.pfRelIso03_all < 0.15)


# in use
def get_baseline_muons(muons):
    baseline_m = muons[((muons.isTracker) | (muons.isGlobal))]
    baseline_m = baseline_m[(np.abs(baseline_m.dxy) < 0.045)]
    baseline_m = baseline_m[(np.abs(baseline_m.dz) < 0.2)]
    baseline_m = baseline_m[(baseline_m.pt > 10)]
    baseline_m = baseline_m[(np.abs(baseline_m.eta) < 2.4)]
    return baseline_m


# in use
def tight_muons(muons, relaxed=False):
    if relaxed:
        return muons.pfRelIso04_all < 10**6
    return (muons.looseId | muons.mediumId | muons.tightId) & (
        muons.pfRelIso04_all < 0.15
    )


# in use
def get_baseline_taus(taus, loose=False):
    baseline_t = taus[(taus.pt > 20)]
    baseline_t = baseline_t[(np.abs(baseline_t.eta) < 2.3)]
    baseline_t = baseline_t[(np.abs(baseline_t.dz) < 0.2)]
    baseline_t = baseline_t[((baseline_t.decayMode != 5) & (baseline_t.decayMode != 6))]
    baseline_t = baseline_t[((baseline_t.idDeepTau2017v2p1VSjet & 1) > 0)]  # VVVLoose
    if loose:
        return baseline_t

    baseline_t = baseline_t[((baseline_t.idDeepTau2017v2p1VSmu & 1) > 0)]  # VLoose
    baseline_t = baseline_t[((baseline_t.idDeepTau2017v2p1VSe & 4) > 0)]  # VLoose
    return baseline_t


# in use
def tight_hadronic_taus(taus, cat, relaxed=False):
    if relaxed:
        return (taus.idDeepTau2017v2p1VSjet & 1) > 0
    vsJet_medium = (taus.idDeepTau2017v2p1VSjet & 16) > 0  # Medium
    if cat[2:] == "et":
        return vsJet_medium & ((taus.idDeepTau2017v2p1VSe & 32) > 0)  # Tight
    elif cat[2:] == "mt":
        return vsJet_medium & ((taus.idDeepTau2017v2p1VSmu & 8) > 0)  # Tight
    elif cat[2:] == "tt":
        return vsJet_medium


# in use
def append_tight_masks(lltt, cat, relaxed=False):
    t1, t2 = lltt["tt"]["t1"], lltt["tt"]["t2"]
    t1_mask, t2_mask = None, None
    if cat[2:] == "em":
        t1_mask = tight_electrons(t1, relaxed=relaxed)
        t2_mask = tight_muons(t2, relaxed=relaxed)
    if cat[2:] == "et":
        t1_mask = tight_electrons(t1, relaxed=relaxed)
        t2_mask = tight_hadronic_taus(t2, cat, relaxed=relaxed)
    if cat[2:] == "mt":
        t1_mask = tight_muons(t1, relaxed=relaxed)
        t2_mask = tight_hadronic_taus(t2, cat, relaxed=relaxed)
    if cat[2:] == "tt":
        t1_mask = tight_hadronic_taus(t1, cat, relaxed=relaxed)
        t2_mask = tight_hadronic_taus(t2, cat, relaxed=relaxed)
    lltt["t1_tight"] = t1_mask
    lltt["t2_tight"] = t2_mask
    return lltt


def tight_events(lltt, cat):
    l1, l2 = lltt["ll"]["l1"], lltt["ll"]["l2"]
    t1, t2 = lltt["tt"]["t1"], lltt["tt"]["t2"]
    l1_mask, l2_mask, t1_mask, t2_mask = 0, 0, 0, 0
    if cat[:2] == "ee":
        l1_mask = ak.flatten(tight_electrons(l1))
        l2_mask = ak.flatten(tight_electrons(l2))
    if cat[:2] == "mm":
        l1_mask = ak.flatten(tight_muons(l1))
        l2_mask = ak.flatten(tight_muons(l2))
    if cat[2:] == "em":
        t1_mask = ak.flatten(tight_electrons(t1))
        t2_mask = ak.flatten(tight_muons(t2))
    if cat[2:] == "et":
        t1_mask = ak.flatten(tight_electrons(t1))
        t2_mask = ak.flatten(tight_hadronic_taus(t2, cat))
    if cat[2:] == "mt":
        t1_mask = ak.flatten(tight_muons(t1))
        t2_mask = ak.flatten(tight_hadronic_taus(t2, cat))
    if cat[2:] == "tt":
        t1_mask = ak.flatten(tight_hadronic_taus(t1, cat))
        t2_mask = ak.flatten(tight_hadronic_taus(t2, cat))
    return (l1_mask, l2_mask, t1_mask, t2_mask)


def tight_events_denom(lltt, cat, mode=-1):
    l1, l2 = lltt["ll"]["l1"], lltt["ll"]["l2"]
    t1, t2 = lltt["tt"]["t1"], lltt["tt"]["t2"]
    mask = np.ones(len(l1), dtype=bool)

    # tighten Z decay leptons
    if cat[:2] == "ee":
        mask = mask & (
            ak.flatten(tight_electrons(l1)) & ak.flatten(tight_electrons(l2))
        )
    if cat[:2] == "mm":
        mask = mask & (ak.flatten(tight_muons(l1)) & ak.flatten(tight_muons(l2)))

    # tighten prompt di-tau candidates
    if (mode == "e") or (mode == "m"):
        mask = mask & ak.flatten(tight_hadronic_taus(t2, cat))
    if (mode == "lt") and cat[2] == "e":
        mask = mask & ak.flatten(tight_electrons(t1))
    if (mode == "lt") and cat[2] == "m":
        mask = mask & ak.flatten(tight_muons(t1))
    if mode == "tt":
        mask = mask & ak.flatten(tight_hadronic_taus(t2, cat))
    return mask


def get_baseline_jets(jet, year="2018"):
    baseline_j = jet[(jet.pt > 20)]

    # HEM recommendation
    if "2018" in year:
        baseline_j = baseline_j[
            ~(
                (baseline_j.eta > -3.2)
                & (baseline_j.eta < -1.3)
                & (baseline_j.phi > -1.57)
                & (baseline_j.phi < 0.87)
            )
        ]
    eta_per_year = {
        "2018": 2.5,
        "2017": 2.5,
        "2016postVFP": 2.4,
        "2016preVFP": 2.4,
        "2016": 2.4,
    }
    baseline_j = baseline_j[(np.abs(baseline_j.eta) < eta_per_year[year])]
    baseline_j = baseline_j[(baseline_j.jetId > 0)]
    return baseline_j


def get_baseline_bjets(baseline_j, year="2018"):
    delta = {
        "2016": 0.2598,
        "2016preVFP": 0.2598,
        "2016postVFP": 0.2598,
        "2017": 0.3040,
        "2018": 0.2783,
    }
    baseline_b = baseline_j[(baseline_j.btagDeepFlavB > delta[year])]
    # baseline_b = baseline_j[(baseline_j.btagDeepB > 0.4168)]
    return baseline_b


def get_ll(electrons, muons, cat):
    if cat[0] == "e":
        return ak.combinations(electrons, 2, axis=1, fields=["l1", "l2"])
    else:
        return ak.combinations(muons, 2, axis=1, fields=["l1", "l2"])


# in use
def get_tt(electrons, muons, taus, cat):
    if cat[2:] == "mt":
        return ak.cartesian({"t1": muons, "t2": taus}, axis=1)
    elif cat[2:] == "et":
        return ak.cartesian({"t1": electrons, "t2": taus}, axis=1)
    elif cat[2:] == "em":
        return ak.cartesian({"t1": electrons, "t2": muons}, axis=1)
    elif cat[2:] == "tt":
        return ak.combinations(taus, 2, axis=1, fields=["t1", "t2"])


def clean_duplicates(lltt, thld=0.05):
    l1, l2 = lltt["ll"]["l1"], lltt["ll"]["l2"]
    t1, t2 = lltt["tt"]["t1"], lltt["tt"]["t2"]
    dR_mask = (
        (l1.delta_r(l2) > thld)
        & (l1.delta_r(t1) > thld)
        & (l1.delta_r(t2) > thld)
        & (l2.delta_r(t1) > thld)
        & (l2.delta_r(t2) > thld)
        & (t1.delta_r(t2) > thld)
    )

    return lltt[dR_mask]


def match_Z_lepton(l_obs, l_gen, cat):
    ll_pdgid = 11 if cat[0] == "e" else 13
    dR = l_obs.delta_r(l_gen)
    dR = ak.fill_none(dR, False)
    matched = (dR < 0.2) & (abs(l_gen.pdgId) == ll_pdgid) & (l_gen.hasFlags("isPrompt"))
    return matched


def match_tau_e(t_obs, e_gen):
    dR = t_obs.delta_r(e_gen)
    dR = ak.fill_none(dR, False)
    matched = (
        (dR < 0.2)
        & (abs(e_gen.pdgId) == 11)
        & (e_gen.pt > 8)
        & (
            e_gen.hasFlags("isPromptTauDecayProduct")
            | e_gen.hasFlags("isDirectTauDecayProduct")
            | e_gen.hasFlags("isDirectPromptTauDecayProduct")
        )
    )
    return matched


def match_tau_m(t_obs, m_gen):
    dR = t_obs.delta_r(m_gen)
    dR = ak.fill_none(dR, False)
    matched = (
        (dR < 0.2)
        & (abs(m_gen.pdgId) == 13)
        & (m_gen.pt > 8)
        & (
            m_gen.hasFlags("isPromptTauDecayProduct")
            | m_gen.hasFlags("isDirectTauDecayProduct")
            | m_gen.hasFlags("isDirectPromptTauDecayProduct")
        )
    )
    return matched


def match_tau_h(t_obs, t_gen):
    dR = t_obs.delta_r(t_gen)
    dR = ak.fill_none(dR, False)
    matched = (dR < 0.2) & (t_gen.pt > 15)
    return matched


def tag_categories(ele, mu, tau, gen):
    Z_e = ele[
        (
            (ele.matched_gen.pt > 8)
            & (abs(ele.matched_gen.pdgId) == 11)
            & (ele.matched_gen.hasFlags("isPrompt"))
            & (ele.matched_gen.parent.pdgId == 23)
        )
    ]
    Z_m = mu[
        (
            (mu.matched_gen.pt > 8)
            & (abs(mu.matched_gen.pdgId) == 13)
            & (mu.matched_gen.hasFlags("isPrompt"))
            & (mu.matched_gen.parent.pdgId == 23)
        )
    ]
    tau_e = ele[
        (
            (abs(ele.matched_gen.pdgId) == 11)
            & (ele.matched_gen.pt > 8)
            & (
                ele.matched_gen.hasFlags("isPromptTauDecayProduct")
                | ele.matched_gen.hasFlags("isDirectTauDecayProduct")
                | ele.matched_gen.hasFlags("isDirectPromptTauDecayProduct")
            )
            & (abs(ele.matched_gen.parent.pdgId) == 15)
        )
    ]
    tau_m = mu[
        (
            (abs(mu.matched_gen.pdgId) == 13)
            & (mu.matched_gen.pt > 8)
            & (
                mu.matched_gen.hasFlags("isPromptTauDecayProduct")
                | mu.matched_gen.hasFlags("isDirectTauDecayProduct")
                | mu.matched_gen.hasFlags("isDirectPromptTauDecayProduct")
            )
            & (abs(mu.matched_gen.parent.pdgId) == 15)
        )
    ]
    tau_gen = ak.cartesian({"tau": tau, "gen": gen}, axis=1)
    dR_mask = tau_gen["tau"].delta_r(tau_gen["gen"]) < 0.2
    tau_h = tau_gen[(dR_mask & (tau_gen["gen"].pt > 15))]["tau"]

    Z_e_counts = ak.sum(~ak.is_none(Z_e, axis=1), axis=1)
    Z_m_counts = ak.sum(~ak.is_none(Z_m, axis=1), axis=1)
    tau_e_counts = ak.sum(~ak.is_none(tau_e, axis=1), axis=1)
    tau_m_counts = ak.sum(~ak.is_none(tau_m, axis=1), axis=1)
    tau_h_counts = ak.sum(~ak.is_none(tau_h, axis=1), axis=1)

    eemt = ak.sum((Z_e_counts == 2) & (tau_m_counts == 1) & (tau_h_counts == 1))
    eeet = ak.sum((Z_e_counts == 2) & (tau_e_counts == 1) & (tau_h_counts == 1))
    eett = ak.sum((Z_e_counts == 2) & (tau_h_counts == 2))
    eeem = ak.sum((Z_e_counts == 2) & (tau_e_counts == 1) & (tau_m_counts == 1))
    mmmt = ak.sum((Z_m_counts == 2) & (tau_m_counts == 1) & (tau_h_counts == 1))
    mmet = ak.sum((Z_m_counts == 2) & (tau_e_counts == 1) & (tau_h_counts == 1))
    mmtt = ak.sum((Z_m_counts == 2) & (tau_h_counts == 2))
    mmem = ak.sum((Z_m_counts == 2) & (tau_e_counts == 1) & (tau_m_counts == 1))
    return {
        "eemt": eemt,
        "eeet": eeet,
        "eett": eett,
        "eeem": eeem,
        "mmmt": mmmt,
        "mmet": mmet,
        "mmtt": mmtt,
        "mmem": mmem,
    }


# in use
def check_trigger_path(HLT, year, cat, sync=False):
    single_lep_trigs = {
        "ee": {
            "2018": ["Ele35_WPTight_Gsf"],
            "2017": ["Ele35_WPTight_Gsf"],
            "2016preVFP": ["Ele25_eta2p1_WPTight_Gsf"],
            "2016postVFP": ["Ele25_eta2p1_WPTight_Gsf"],
            "2016": ["Ele25_eta2p1_WPTight_Gsf"],
        },
        "mm": {
            "2018": ["IsoMu27"],
            "2017": ["IsoMu27"],
            "2016preVFP": ["IsoMu24", "IsoTkMu24"],
            "2016postVFP": ["IsoMu24", "IsoTkMu24"],
            "2016": ["IsoMu24", "IsoTkMu24"],
        },
    }
    trig_list = single_lep_trigs[cat[:2]][year]
    triggered = HLT[trig_list[0]]
    if len(trig_list) > 1:
        for trig in trig_list:
            triggered = triggered | HLT[trig]
    return triggered


# in use
@nb.njit(
    nb.int64[:](
        nb.types.int8,
        nb.int64[:],
        nb.int64[:],
        nb.int64[:],
        nb.int64[:],
        nb.int64[:],
    )
)
def lepton_count_veto_jitted(
    cat,
    offsets,
    i1_content,
    i2_content,
    i3_content,
    i4_content,
):
    nevts = len(offsets) - 1
    output = np.zeros(nevts, dtype=nb.int64)
    for i in range(nevts):
        start, stop = offsets[i], offsets[i + 1]
        e_ids, m_ids = [], []
        if cat > 4:
            m_ids.extend(i1_content[start:stop])
            m_ids.extend(i2_content[start:stop])
        if cat == 2:
            m_ids.extend(i3_content[start:stop])
        if cat == 4:
            m_ids.extend(i4_content[start:stop])
        if cat < 5:
            e_ids.extend(i1_content[start:stop])
            e_ids.extend(i2_content[start:stop])
        if (cat == 8) or (cat == 5):
            e_ids.extend(i3_content[start:stop])

        ne, nm = len(set(e_ids)), len(set(m_ids))
        if cat == 1:
            ec = ne == 3
            mc = nm == 0
        if cat == 2:
            ec = ne == 2
            mc = nm == 1
        if cat == 3:
            ec = ne == 2
            mc = nm == 0
        if cat == 4:
            ec = ne == 3
            mc = nm == 1
        if cat == 5:
            ec = ne == 1
            mc = nm == 2
        if cat == 6:
            ec = ne == 0
            mc = nm == 3
        if cat == 7:
            ec = ne == 0
            mc = nm == 2
        if cat == 8:
            ec = ne == 1
            mc = nm == 3
        output[i] = ec & mc
    return output


# in use
def lepton_count_veto(lltt, cat):
    i1, i2 = ak.values_astype(lltt.ll.l1.idx, int), ak.values_astype(
        lltt.ll.l2.idx, int
    )
    i3, i4 = ak.values_astype(lltt.tt.t1.idx, int), ak.values_astype(
        lltt.tt.t2.idx, int
    )
    i1.behavior = None
    i2.behavior = None
    i3.behavior = None
    i4.behavior = None
    offsets = np.array(i1.layout.offsets, dtype=np.int64)
    i1_content = np.array(i1.layout.content, dtype=np.int64)
    i2_content = np.array(i2.layout.content, dtype=np.int64)
    i3_content = np.array(i3.layout.content, dtype=np.int64)
    i4_content = np.array(i4.layout.content, dtype=np.int64)
    return np.array(
        lepton_count_veto_jitted(
            cat,
            offsets,
            i1_content,
            i2_content,
            i3_content,
            i4_content,
        ),
        dtype=bool,
    )


# in use
def get_lepton_count_veto_masks(baseline_e, baseline_m, baseline_t):
    baseline_e["idx"] = ak.local_index(baseline_e)
    baseline_m["idx"] = ak.local_index(baseline_m)
    baseline_t["idx"] = ak.local_index(baseline_t)
    cat_to_num = {
        "eeet": 1,
        "eemt": 2,
        "eett": 3,
        "eeem": 4,
        "mmet": 5,
        "mmmt": 6,
        "mmtt": 7,
        "mmem": 8,
    }
    vetoes = {}
    leps = {"e": baseline_e, "m": baseline_m, "t": baseline_t}
    for cat, num in cat_to_num.items():
        ll = ak.combinations(leps[cat[0]], 2, axis=1, fields=["l1", "l2"])
        if cat[2:] == "tt":
            tt = ak.combinations(leps["t"], 2, axis=1, fields=["t1", "t2"])
        else:
            tt = ak.cartesian({"t1": leps[cat[2]], "t2": leps[cat[3]]}, axis=1)
        lltt = ak.cartesian({"ll": ll, "tt": tt}, axis=1)
        lltt = dR_lltt(lltt, cat=cat)
        veto = lepton_count_veto(lltt, num)

        # present in the event
        if ("m" not in cat) or ("e" not in cat):
            check_lep = "e" if ("e" not in cat) else "m"
            llttl = ak.cartesian({"lltt": lltt, "l": leps[check_lep]}, axis=1)
            llttl = dr_clean(llttl, cat + check_lep)
            mask = ak.num(llttl) == 0
            vetoes[cat] = veto & mask
        else:
            vetoes[cat] = veto

    return vetoes


def dr_clean(llttl, cat):
    dR_select = {
        "ee": 0.3,
        "em": 0.3,
        "mm": 0.3,
        "me": 0.3,
        "et": 0.5,
        "te": 0.5,
        "mt": 0.5,
        "tm": 0.5,
        "tt": 0.5,
    }
    l1, l2 = llttl.lltt.ll.l1, llttl.lltt.ll.l2
    t1, t2 = llttl.lltt.tt.t1, llttl.lltt.tt.t2
    l = llttl.l
    dR_mask = (
        (l1.delta_r(l) > dR_select[cat[0] + cat[-1]])
        & (l2.delta_r(l) > dR_select[cat[1] + cat[-1]])
        & (t1.delta_r(l) > dR_select[cat[2] + cat[-1]])
        & (t2.delta_r(l) > dR_select[cat[3] + cat[-1]])
    )
    llttl = llttl.mask[(dR_mask)]
    return llttl[~ak.is_none(llttl, axis=1)]


def bjetveto(baseline_b):
    mask = ak.num(baseline_b) == 0
    return mask


def dR_ll(ll):
    l1, l2 = ll["l1"], ll["l2"]
    dR_mask = l1.delta_r(l2) > 0.3
    ll = ll[dR_mask]
    return ll[(~ak.is_none(ll, axis=1))]


def build_Z_cand(ll):
    ll_mass = (ll["l1"] + ll["l2"]).mass
    ll = ll[
        (ll["l1"].charge * ll["l2"].charge < 0) & ((ll_mass > 60) & (ll_mass < 120))
    ]
    return ll[(~ak.is_none(ll, axis=1))]


def closest_to_Z_mass(ll):
    mass_diffs = abs((ll["l1"] + ll["l2"]).mass - 91.118)
    min_mass_filter = ak.argmin(mass_diffs, axis=1, keepdims=True, mask_identity=False)
    min_mass_filter = min_mass_filter[min_mass_filter >= 0]
    ll = ll[min_mass_filter]
    return ll[(~ak.is_none(ll, axis=1))]


def tighten_Z_legs(ll, cat):
    l1, l2 = ll["l1"], ll["l2"]
    if cat[:2] == "ee":
        ll = ll[(tight_electrons(l1) & tight_electrons(l2))]
    elif cat[:2] == "mm":
        ll = ll[(tight_muons(l1) & tight_muons(l2))]
    return ll[~ak.is_none(ll, axis=1)]


def dR_lltt(lltt, cat):
    dR_select = {
        "ee": 0.3,
        "em": 0.3,
        "mm": 0.3,
        "me": 0.3,
        "et": 0.5,
        "mt": 0.5,
        "tt": 0.5,
    }
    l1, l2 = lltt["ll"]["l1"], lltt["ll"]["l2"]
    t1, t2 = lltt["tt"]["t1"], lltt["tt"]["t2"]
    dR_mask = (
        (l1.delta_r(l2) > dR_select[cat[0] + cat[1]])
        & (l1.delta_r(t1) > dR_select[cat[0] + cat[2]])
        & (l1.delta_r(t2) > dR_select[cat[0] + cat[3]])
        & (l2.delta_r(t1) > dR_select[cat[1] + cat[2]])
        & (l2.delta_r(t2) > dR_select[cat[1] + cat[3]])
        & (t1.delta_r(t2) > dR_select[cat[2] + cat[3]])
    )

    lltt = lltt.mask[(dR_mask)]
    return lltt[~ak.is_none(lltt, axis=1)]


def trigger_filter(ll, trig_obj, cat):
    if cat[:2] == "ee":
        pt_min = 36
    if cat[:2] == "mm":
        pt_min = 28

    lltrig = ak.cartesian({"ll": ll, "trobj": trig_obj}, axis=1)
    l1dR_matches = lltrig["ll"]["l1"].delta_r(lltrig["trobj"]) < 0.5
    l2dR_matches = lltrig["ll"]["l2"].delta_r(lltrig["trobj"]) < 0.5
    filter_bit = (lltrig["trobj"].filterBits & 2) > 0
    if cat[:2] == "mm":
        filter_bit = filter_bit | ((lltrig["trobj"].filterBits & 8) > 0)

    l1_matches = lltrig[l1dR_matches & (lltrig["ll"]["l1"].pt > pt_min) & filter_bit]
    # l1_dr = l1_matches['ll']['l1'].delta_r(l1_matches['trobj'])
    # l1_matches = l1_matches[ak.argmin(l1_dr, axis=1, keepdims=True)]
    l1_match_counts = ak.sum(~ak.is_none(l1_matches, axis=1), axis=1)

    l2_matches = lltrig[l2dR_matches & (lltrig["ll"]["l2"].pt > pt_min) & filter_bit]
    # l2_dr = l2_matches['ll']['l2'].delta_r(l2_matches['trobj'])
    # l2_matches = l2_matches[ak.argmin(l2_dr, axis=1, keepdims=True)]

    l2_match_counts = ak.sum(~ak.is_none(l2_matches, axis=1), axis=1)
    trig_match = ((l1_match_counts) > 0) | ((l2_match_counts) > 0)

    l1_match_pt = ak.max(l1_matches["ll"]["l1"].pt, axis=1)
    l2_match_pt = ak.max(l2_matches["ll"]["l2"].pt, axis=1)
    l1_match_eta = ak.max(l1_matches["ll"]["l1"].eta, axis=1)
    l2_match_eta = ak.max(l2_matches["ll"]["l2"].eta, axis=1)
    l1_match_pt = ak.fill_none(l1_match_pt, 0)
    l2_match_pt = ak.fill_none(l2_match_pt, 0)
    l1_match_eta = ak.fill_none(l1_match_eta, 0)
    l2_match_eta = ak.fill_none(l2_match_eta, 0)
    return trig_match, l1_match_pt, l1_match_eta, l2_match_pt, l2_match_eta


def tighten_ditau_legs(lltt, cat):
    t1, t2 = lltt["tt"]["t1"], lltt["tt"]["t2"]
    mask = 0
    if cat[2:] == "em":
        mask = tight_electrons(t1) & tight_muons(t2)
    if cat[2:] == "et":
        mask = tight_electrons(t1) & tight_hadronic_taus(t2, cat)
    if cat[2:] == "mt":
        mask = tight_muons(t1) & tight_hadronic_taus(t2, cat)
    if cat[2:] == "tt":
        mask = tight_hadronic_taus(t1, cat) & tight_hadronic_taus(t2, cat)
    lltt = lltt[mask]
    return lltt[~ak.is_none(lltt, axis=1)]


def highest_LT(lltt, cat, apply_LT_cut=False):
    t1, t2 = lltt["tt"]["t1"], lltt["tt"]["t2"]
    LT = t1.pt + t2.pt
    lltt = lltt[ak.argmax(LT, axis=1, keepdims=True)]
    if apply_LT_cut and "tt" in cat:
        print(cat, "applying LT cut")
        lltt = lltt[(lltt.tt.t1.pt + lltt.tt.t2.pt) > 60]
    return lltt[~ak.is_none(lltt, axis=1)]


def dR_llttj(llttj):
    dR_select = 0.5
    t1 = llttj["lltt"]["tt"]["t1"]
    t2 = llttj["lltt"]["tt"]["t2"]
    dR_t1_j = t1.delta_r(llttj["j"]) < dR_select
    dR_t2_j = t2.delta_r(llttj["j"]) < dR_select
    overlaps = ak.sum((dR_t1_j | dR_t2_j), axis=1)
    dR_mask = overlaps == 0
    return dR_mask


def is_prompt(lltt, cat):
    l1, l2 = lltt["ll"]["l1"], lltt["ll"]["l2"]
    t1, t2 = lltt["tt"]["t1"], lltt["tt"]["t2"]
    prompt_l1 = (l1.genPartFlav == 1) | (l1.genPartFlav == 15)
    prompt_l2 = (l2.genPartFlav == 1) | (l2.genPartFlav == 15)
    prompt_mask = prompt_l1 & prompt_l2

    if cat[2:] == "em":
        prompt_t1 = (t1.genPartFlav == 1) | (t1.genPartFlav == 15)
        prompt_t2 = (t2.genPartFlav == 1) | (t2.genPartFlav == 15)
        prompt_mask = prompt_mask & prompt_t1 & prompt_t2
    if cat[2:] == "et":
        prompt_t1 = (t1.genPartFlav == 1) | (t1.genPartFlav == 15)
        prompt_t2 = (t2.genPartFlav > 0) & (t2.genPartFlav < 6)
        prompt_mask = prompt_mask & prompt_t1 & prompt_t2
    if cat[2:] == "mt":
        prompt_t1 = (t1.genPartFlav == 1) | (t1.genPartFlav == 15)
        prompt_t2 = (t2.genPartFlav > 0) & (t2.genPartFlav < 6)
        prompt_mask = prompt_mask & prompt_t1 & prompt_t2
    if cat[2:] == "tt":
        prompt_t1 = (t1.genPartFlav > 0) & (t1.genPartFlav < 6)
        prompt_t2 = (t2.genPartFlav > 0) & (t2.genPartFlav < 6)
        prompt_mask = prompt_mask & prompt_t1 & prompt_t2

    # prompt_mask = prompt_mask.mask[ak.num(prompt_mask, axis=1)>0]
    return prompt_mask  # ak.fill_none(prompt_mask, False, axis=0)


def is_prompt_base(lltt, cat, mode=-1):
    l1, l2 = lltt["ll"]["l1"], lltt["ll"]["l2"]
    # t1, t2 = lltt["tt"]["t1"], lltt["tt"]["t2"]
    prompt_l1 = (l1.genPartFlav == 1) | (l1.genPartFlav == 15)
    prompt_l2 = (l2.genPartFlav == 1) | (l2.genPartFlav == 15)
    prompt_mask = prompt_l1 & prompt_l2

    if (mode == "e") or (mode == "m"):
        # prompt_t2 = (t2.genPartFlav > 0) & (t2.genPartFlav < 6)
        prompt_mask = prompt_mask  # & prompt_t2
    elif mode == "lt":
        # prompt_t1 = (t1.genPartFlav == 1) | (t1.genPartFlav == 15)
        prompt_mask = prompt_mask  # & prompt_t1
    elif mode == "tt":
        # prompt_t2 = (t2.genPartFlav > 0) & (t2.genPartFlav < 6)
        prompt_mask = prompt_mask  # & prompt_t2
    return prompt_mask


# in use
def is_prompt_lepton(lltt, cat, mode=-1):
    t1, t2 = lltt["tt"]["t1"], lltt["tt"]["t2"]
    if (mode == "e") or (mode == "m"):
        return (t1.genPartFlav == 1) | (t1.genPartFlav == 15)
    if mode == "lt":
        return (t2.genPartFlav > 0) & (t2.genPartFlav < 6)
    if mode == "tt":
        return (t1.genPartFlav > 0) & (t1.genPartFlav < 6)
    else:
        return -1


# in use
def is_tight_base(lltt, cat, mode=-1):
    l1, l2 = lltt["ll"]["l1"], lltt["ll"]["l2"]
    if cat[:2] == "ee":
        return tight_electrons(l1) & tight_electrons(l2)
    elif cat[:2] == "mm":
        return tight_muons(l1) & tight_muons(l2)


# in use
def is_tight_lepton(lltt, cat, mode=-1):
    t1, t2 = lltt["tt"]["t1"], lltt["tt"]["t2"]
    if mode == "e":
        return tight_electrons(t1)
    if mode == "m":
        return tight_muons(t1)
    if mode == "lt":
        return tight_hadronic_taus(t2, cat)
    if mode == "tt":
        return tight_hadronic_taus(t1, cat)


def higgsLT(lltt, cat):
    hLT_thld = {"et": 30, "mt": 40, "em": 20, "tt": 80}
    hLT = lltt["tt"]["t1"].pt + lltt["tt"]["t2"].pt
    hLT_mask = ak.flatten(hLT > hLT_thld[cat[2:]])
    return hLT_mask


def iso_ID(lltt, cat):
    t1 = lltt["tt"]["t1"]
    t2 = lltt["tt"]["t2"]
    if cat[2:] == "et":
        ID = (t1.mvaFall17V2noIso_WP80) & (t2.idDeepTau2017v2p1VSe > 60)
        iso = t1.pfRelIso03_all < 0.15
        iso_ID_mask = ak.flatten(iso & ID)
    if cat[2:] == "mt":
        ID = t2.idDeepTau2017v2p1VSe > 60
        iso = t1.pfRelIso04_all < 0.15
        iso_ID_mask = ak.flatten(iso & ID)
    if cat[2:] == "em":
        iso = (t1.pfRelIso03_all < 0.15) & (t2.pfRelIso04_all < 0.15)
        ID = t1.mvaFall17V2noIso_WP80
        iso_ID_mask = ak.flatten(iso & ID)
    if cat[2:] == "tt":
        iso_ID_mask = ak.Array(np.ones(len(lltt), dtype=bool))

    return iso_ID_mask


def calc_dphi(phi1, phi2):
    dphi = phi2 - phi1
    num, flat = ak.num(dphi), ak.to_numpy(ak.flatten(dphi))
    flat[flat > np.pi] -= 2 * np.pi
    flat[flat < -np.pi] += 2 * np.pi
    return ak.unflatten(flat, num)


def delta_r(a, b):
    return np.sqrt(calc_dphi(a.phi, b.phi) ** 2 + (a.eta - b.eta) ** 2)


def count_btags(lltt, bjets):
    bcounts = np.zeros(len(lltt))
    for i in range(4):
        lltt["bjet"] = bjets[:, i : i + 1]
        is_iso = (
            (delta_r(lltt.ll.l1, lltt.bjet) > 0.4)
            & (delta_r(lltt.ll.l2, lltt.bjet) > 0.4)
            & (delta_r(lltt.tt.t1, lltt.bjet) > 0.4)
            & (delta_r(lltt.tt.t2, lltt.bjet) > 0.4)
        )
        bcounts += ak.to_numpy(ak.sum(is_iso & True, 1))
    lltt["btags"] = bcounts
    return lltt
