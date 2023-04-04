from __future__ import annotations

import awkward as ak
import numba as nb
import numpy as np


def dr_3l(lll, cat):
    dR_select = {
        "ee": 0.3,
        "em": 0.3,
        "mm": 0.3,
        "me": 0.3,
        "et": 0.5,
        "mt": 0.5,
        "tt": 0.5,
    }
    l1, l2, l3 = lll["ll"]["l1"], lll["ll"]["l2"], lll["l"]
    dR_mask = (
        (l1.delta_r(l2) > dR_select[cat[0] + cat[1]])
        & (l1.delta_r(l3) > dR_select[cat[0] + cat[2]])
        & (l2.delta_r(l3) > dR_select[cat[1] + cat[2]])
    )
    lll = lll.mask[(dR_mask)]
    return lll[~ak.is_none(lll, axis=1)]


def dr_4l(llll, cat):
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
    l1, l2, l3, l = llll.lll.ll.l1, llll.lll.ll.l2, llll.lll.l, llll.l
    dR_mask = (
        (l1.delta_r(l) > dR_select[cat[0] + cat[-1]])
        & (l2.delta_r(l) > dR_select[cat[1] + cat[-1]])
        & (l3.delta_r(l) > dR_select[cat[2] + cat[-1]])
    )
    llll = llll.mask[(dR_mask)]
    return llll[~ak.is_none(llll, axis=1)]


# in use
@nb.njit(
    nb.int64[:](
        nb.types.int8,
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
):
    nevts = len(offsets) - 1
    output = np.zeros(nevts, dtype=nb.int64)
    for i in range(nevts):
        start, stop = offsets[i], offsets[i + 1]
        e_ids, m_ids = [], []
        if cat > 3:
            m_ids.extend(i1_content[start:stop])
            m_ids.extend(i2_content[start:stop])
        if cat == 2:
            m_ids.extend(i3_content[start:stop])
        if cat < 4:
            e_ids.extend(i1_content[start:stop])
            e_ids.extend(i2_content[start:stop])
        if cat == 4:
            e_ids.extend(i3_content[start:stop])

        ne, nm = len(set(e_ids)), len(set(m_ids))
        if cat == 1:
            ec = ne <= 3
            mc = nm == 0
        if cat == 2:
            ec = ne == 2
            mc = nm == 1
        if cat == 3:
            ec = ne == 2
            mc = nm == 0
        if cat == 4:
            ec = ne == 1
            mc = nm == 2
        if cat == 5:
            ec = ne == 0
            mc = nm == 3
        if cat == 6:
            ec = ne == 0
            mc = nm == 2
        output[i] = ec & mc
    return output


# in use
def lepton_count_veto(lll, cat):
    i1 = ak.values_astype(lll.ll.l1.idx, int)
    i2 = ak.values_astype(lll.ll.l2.idx, int)
    i3 = ak.values_astype(lll.l.idx, int)
    i1.behavior = None
    i2.behavior = None
    i3.behavior = None
    offsets = np.array(i1.layout.offsets, dtype=np.int64)
    i1_content = np.array(i1.layout.content, dtype=np.int64)
    i2_content = np.array(i2.layout.content, dtype=np.int64)
    i3_content = np.array(i3.layout.content, dtype=np.int64)
    return np.array(
        lepton_count_veto_jitted(
            cat,
            offsets,
            i1_content,
            i2_content,
            i3_content,
        ),
        dtype=bool,
    )


# in use
def get_lepton_count_veto_mask(cat, baseline_e, baseline_m, baseline_t):
    baseline_e["idx"] = ak.local_index(baseline_e)
    baseline_m["idx"] = ak.local_index(baseline_m)
    baseline_t["idx"] = ak.local_index(baseline_t)
    cat_to_num = {"eee": 1, "eem": 2, "eet": 3, "mme": 4, "mmm": 5, "mmt": 6}
    leps = {"e": baseline_e, "m": baseline_m, "t": baseline_t}

    ll = ak.combinations(leps[cat[0]], 2, axis=1, fields=["l1", "l2"])
    lll = ak.cartesian({"ll": ll, "l": leps[cat[2]]}, axis=1)
    lll = dr_3l(lll, cat=cat)
    # vetoes[cat] = lepton_count_veto(lll, cat_to_num[cat])
    veto = lepton_count_veto(lll, cat_to_num[cat])

    # if no electrons or muons in cat, make sure there aren't other ones
    # present in the event
    if ("m" not in cat) or ("e" not in cat):
        check_lep = "e" if ("e" not in cat) else "m"
        llll = ak.cartesian({"lll": lll, "l": leps[check_lep]}, axis=1)
        llll = dr_4l(llll, cat + check_lep)
        mask = ak.num(llll) == 0
        return veto & mask
    else:
        return veto


def lepton_count_veto_3l(e_counts, m_counts, t_counts, cat):
    correct_e_counts = {
        "eee": 3,
        "eem": 2,
        "mme": 1,
        "mmm": 0,
        "eet": 2,
        "mmt": 0,
    }
    correct_m_counts = {
        "eee": 0,
        "eem": 1,
        "mme": 2,
        "mmm": 3,
        "eet": 0,
        "mmt": 2,
    }
    correct_t_counts = {
        "eee": 100,
        "eem": 100,
        "mme": 100,
        "mmm": 100,
        "eet": 1,
        "mmt": 1,
    }
    mask = (
        (e_counts <= correct_e_counts[cat])
        & (m_counts <= correct_m_counts[cat])
        & (t_counts <= correct_t_counts[cat])
    )
    return mask


def additional_cuts(lll, mode):
    l1, l2, l, met = lll.ll.l1, lll.ll.l2, lll.l, lll.met
    if (mode == "e") or (mode == "m"):
        lll = lll.mask[((l1 + l2).mass > 81.2)]
    if mode == "e":
        lll = lll.mask[
            (
                ((l1 + l).mass > 5)
                & ((l2 + l).mass > 5)
                & (l1.pfRelIso03_all < 0.14)
                & (l2.pfRelIso03_all < 0.14)
                & (
                    ((transverse_mass(l, met) < 55) & (abs(l.eta) < 1.5))
                    | (abs(l.eta) >= 1.5)
                )
            )
        ]
    elif mode == "m":
        lll = lll.mask[
            (
                ((l1 + l2 + l).mass < 250)
                & (
                    ((transverse_mass(l, met) < 55) & (abs(l.eta) < 1.2))
                    | (abs(l.eta) >= 1.2)
                )
            )
        ]

    return lll[(~ak.is_none(lll, axis=1))]


def tight_hadronic_taus(taus, mode):
    vsJet_medium = (taus.idDeepTau2017v2p1VSjet & 16) > 0
    vsEle_vloose = (taus.idDeepTau2017v2p1VSe & 4) > 0
    vsMu_vloose = (taus.idDeepTau2017v2p1VSmu & 1) > 0
    vsEle_tight = (taus.idDeepTau2017v2p1VSe & 32) > 0
    vsMu_tight = (taus.idDeepTau2017v2p1VSmu & 8) > 0
    if mode == "e":
        return vsJet_medium & vsEle_tight
    elif mode == "m":
        return vsJet_medium & vsMu_tight
    elif mode == "et":
        return vsJet_medium & vsEle_tight & vsMu_vloose
    elif mode == "mt":
        return vsJet_medium & vsEle_vloose & vsMu_tight
    elif mode == "tt":
        return vsJet_medium & vsEle_vloose & vsMu_vloose


def is_prompt_lepton(lll, mode=-1):
    l = lll.l
    if (mode == "e") or (mode == "m"):
        return (l.genPartFlav == 1) | (l.genPartFlav == 15)
    if (mode == "et") or (mode == "mt") or (mode == "tt"):
        return (l.genPartFlav > 0) & (l.genPartFlav < 6)
    else:
        return -1


def transverse_mass(t, met):
    t_eT = np.sqrt(t.mass**2 + t.pt**2)
    t_px = t.pt * np.cos(t.phi)
    t_py = t.pt * np.sin(t.phi)
    eT_miss = met.pt
    ex_miss = met.pt * np.cos(met.phi)
    ey_miss = met.pt * np.sin(met.phi)
    mt = np.sqrt((t_eT + eT_miss) ** 2 - (t_px + ex_miss) ** 2 - (t_py + ey_miss) ** 2)
    return mt


def transverse_mass_cut(lltt, met, thld=40, leg="t1"):
    lep = lltt["tt"][leg]
    ET_lep = np.sqrt(lep.mass**2 + lep.pt**2)
    px_lep = lep.pt * np.cos(lep.phi)
    py_lep = lep.pt * np.sin(lep.phi)
    ET_miss = met.pt
    Ex_miss = met.pt * np.cos(met.phi)
    Ey_miss = met.pt * np.sin(met.phi)
    mT = np.sqrt(
        (ET_lep + ET_miss) ** 2 - (px_lep + Ex_miss) ** 2 - (py_lep + Ey_miss) ** 2
    )
    lltt = lltt[(mT < thld)]
    return lltt[(~ak.is_none(lltt, axis=1))]
