from __future__ import annotations

import awkward as ak
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
        "eee": 0,
        "eem": 0,
        "mme": 0,
        "mmm": 0,
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
