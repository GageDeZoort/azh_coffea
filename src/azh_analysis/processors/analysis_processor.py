from __future__ import annotations

import logging
import math

# import time
import warnings

import awkward as ak
import numba as nb
import numpy as np
import vector as vec
from coffea import analysis_tools, processor
from coffea.processor import column_accumulator as col_acc
from numba.core import types
from numba.typed import Dict

from azh_analysis.selections.preselections import (
    append_tight_masks,
    build_Z_cand,
    check_trigger_path,
    closest_to_Z_mass,
    count_btags,
    dR_ll,
    dR_lltt,
    filter_MET,
    filter_PV,
    get_baseline_bjets,
    get_baseline_electrons,
    get_baseline_jets,
    get_baseline_muons,
    get_baseline_taus,
    get_lepton_count_veto_masks,
    get_tt,
    highest_LT,
    is_prompt,
    tight_electrons,
    tight_muons,
    trigger_filter,
)
from azh_analysis.utils.corrections import (
    apply_btag_corrections,
    apply_eleES,
    apply_JER_shifts,
    apply_JES_shifts,
    apply_met_phi_SFs,
    apply_muES,
    apply_tauES,
    apply_unclMET_shifts,
    lepton_ID_weight,
    lepton_trig_weight,
    shift_MET,
    tau_ID_weight,
)
from azh_analysis.utils.histograms import make_analysis_hist_stack
from azh_analysis.utils.logging import init_logging
from azh_analysis.utils.parameters import get_categories, get_eras, get_lumis

warnings.filterwarnings("ignore")

float_array = types.float32[:]


def flat(a, axis=None):
    return ak.flatten(a, axis=axis)


def np_flat(a, axis=None):
    return ak.to_numpy(ak.flatten(a, axis=axis))


def is_inf(a):
    return np.sum(np.isnan(np_flat(a)) | np.isinf(np_flat(a)))


class AnalysisProcessor(processor.ProcessorABC):
    def __init__(
        self,
        source="",
        year="",
        sync=False,
        categories="all",
        collection_vars=None,
        global_vars=None,
        sample_info=None,
        fileset=None,
        sample_dir="../sample_lists/sample_yamls",
        exc1_path="sync/princeton_all.csv",
        exc2_path="sync/desy_all.csv",
        pileup_weights=None,
        lumi_masks=None,
        blind=True,
        nevts_dict=None,
        met_phi_SFs=None,
        fake_rates=None,
        eleID_SFs=None,
        muID_SFs=None,
        tauID_SFs=None,
        eleES_SFs=None,
        muES_SFs=None,
        dyjets_weights=None,
        e_trig_SFs=None,
        m_trig_SFs=None,
        btag_SFs=None,
        btag_eff_tables=None,
        btag_pt_bins=None,
        btag_eta_bins=None,
        run_fastmtt=False,
        fill_hists=True,
        verbose=False,
        A_mass="",
        systematic=None,
        same_sign=False,
        relaxed=False,
        mtt_corr_up=160,
        mtt_corr_down=90,
        m4l_cons_up=2000,
        m4l_cons_down=200,
        LT_cut=False,
    ):

        # initialize member variables
        init_logging(verbose=verbose)
        self.sync = sync
        self.info = sample_info
        self.fileset = fileset
        self.collection_vars = collection_vars
        self.global_vars = global_vars
        self.blind = blind

        # grab categories, eras, labels
        self.categories = get_categories()
        self.cat_to_num = {v: k for k, v in self.categories.items()}
        self.eras = get_eras()
        self.lumi = get_lumis(as_picobarns=True)

        # store inputs to the processor
        self.pu_weights = pileup_weights
        self.lumi_masks = lumi_masks
        self.nevts_dict = nevts_dict
        self.fake_rates = fake_rates
        self.met_phi_SFs = met_phi_SFs
        self.eleID_SFs = eleID_SFs
        self.muID_SFs = muID_SFs
        self.tauID_SFs = tauID_SFs
        self.eleES_SFs = eleES_SFs
        self.muES_SFs = muES_SFs
        self.e_trig_SFs = e_trig_SFs
        self.m_trig_SFs = m_trig_SFs
        self.btag_SFs = btag_SFs
        self.btag_eff_tables = btag_eff_tables
        self.btag_pt_bins = btag_pt_bins
        self.btag_eta_bins = btag_eta_bins
        self.dyjets_weights = dyjets_weights
        self.fastmtt = run_fastmtt
        self.fill_hists = fill_hists
        self.A_mass = A_mass
        self.same_sign = same_sign
        self.relaxed = relaxed
        self.mtt_corr_up = mtt_corr_up
        self.mtt_corr_down = mtt_corr_down
        self.m4l_cons_up = m4l_cons_up
        self.m4l_cons_down = m4l_cons_down
        self.LT_cut = LT_cut

        # systematics that affect event kinematics
        self.k_shifts = {
            "tauES": ["down", "up"],
            "efake": ["down", "up"],
            "mfake": ["down", "up"],
            "eleES": ["down", "up"],
            "eleSmear": ["down", "up"],
            "muES": ["down", "up"],
            "unclMET": ["down", "up"],
            "JES": ["down", "up"],
            "JER": ["down", "up"],
        }

        # systematics separated by kinematics vs. event weight
        self.systematic = systematic
        self.kin_syst_shifts = ["nom"]
        self.event_syst_shifts = ["nom"]
        if systematic is not None:
            if systematic in list(self.k_shifts.keys()):
                self.kin_syst_shifts = [
                    f"{systematic}_{i}" for i in self.k_shifts[systematic]
                ]
            elif systematic == "all":
                for k, v in self.k_shifts.items():
                    self.kin_syst_shifts.append(f"{k}_{v[0]}")
                    self.kin_syst_shifts.append(f"{k}_{v[1]}")
            else:
                raise Exception(f"Systematic {systematic} unaccounted for.")
        if (systematic is None) or ("all" in systematic):
            self.event_syst_shifts = [
                "nom",
                "l1prefire_up",
                "l1prefire_down",
                "pileup_up",
                "pileup_down",
                "tauID_0_up",
                "tauID_0_down",
                "tauID_1_up",
                "tauID_1_down",
                "tauID_10_up",
                "tauID_10_down",
                "tauID_11_up",
                "tauID_11_down",
                # "btag_down_uncorrelated",
                # "btag_down_correlated",
                # "btag_up_uncorrelated",
                # "btag_up_correlated",
            ]

        logging.info(f"Kinematic systematic shifts: {self.kin_syst_shifts}")
        logging.info(f"Event-level systematic shifts: {self.event_syst_shifts}")

    def process(self, events):
        logging.info(f"Processing {events.metadata['dataset']}")
        filename = events.metadata["filename"]

        # organize dataset, year, luminosity
        dataset = events.metadata["dataset"]
        year = dataset.split("_")[-1]
        name = dataset.replace(f"_{year}", "")
        names = np.array(
            [
                n.replace("TuneCP5", "")
                .replace("_postVFP", "")
                .replace("_preVFP", "")
                .replace("_preFVP", "")
                .replace("LL_M-50", "LLM-50")
                .replace("LLM50", "LLM-50")
                for n in self.info["name"]
            ]
        )
        properties = self.info[names == name]
        group = properties["group"][0]
        is_data = "data" in group
        nevts, xsec = properties["nevts"][0], properties["xsec"][0]
        output = make_analysis_hist_stack(self.fileset, year)

        # if running on ntuples, need the pre-skim sum_of_weights
        if self.nevts_dict is not None:
            nevts = self.nevts_dict[dataset]
        elif not is_data:
            logging.debug("WARNING: may be using wrong sum_of_weights!")

        # apply global event selections
        global_selections = analysis_tools.PackedSelection()
        filter_MET(events, global_selections, year, data=is_data)
        filter_PV(events, global_selections)
        if is_data:
            global_selections.add(
                "lumi_mask",
                self.lumi_masks[year](events.run, events.luminosityBlock),
            )
        global_mask = global_selections.all(*global_selections.names)
        events = events[global_mask]

        # necessary to apply met_phpi corrections
        events = events[events.MET.T1_pt < 6500]
        if len(events) == 0:
            return output

        # initial weights
        global_weights = analysis_tools.Weights(len(events), storeIndividual=True)
        ones = np.ones(len(events), dtype=float)

        # sample weights OR dyjets stitching weights
        if group == "DY":
            njets = ak.to_numpy(events.LHE.Njets)
            global_weights.add("dyjets_sample_weights", self.dyjets_weights(njets))
        else:  # otherwise weight by luminosity ratio
            logging.info(f"nevts: {nevts}")
            logging.info(f"xsec: {xsec}")
            sample_weight = 1 if is_data else self.lumi[year] * xsec / nevts
            logging.info(f"sample_weight: {sample_weight}")
            global_weights.add("sample_weight", ones * sample_weight)

        # pileup weights and gen weights
        if (self.pu_weights is not None) and not is_data:
            global_weights.add("gen_weight", events.genWeight)
            global_weights.add(
                "pileup",
                weight=self.pu_weights["nom"](events.Pileup.nTrueInt),
                weightUp=self.pu_weights["up"](events.Pileup.nTrueInt),
                weightDown=self.pu_weights["down"](events.Pileup.nTrueInt),
            )

        # L1 prefiring weights if available
        if not is_data:
            try:
                global_weights.add(
                    "l1prefire",
                    weight=events.L1PreFiringWeight.Nom,
                    weightUp=events.L1PreFiringWeight.Up,
                    weightDown=events.L1PreFiringWeight.Dn,
                )
                self.has_L1PreFiringWeight = True
            except Exception:
                logging.info(f"No prefiring weights in {dataset}.")
                self.has_L1PreFiringWeight = False

        # set up the event identifiers
        evtID, lumi_block, run = events.event, events.luminosityBlock, events.run

        # run the analysis over various systematic shifts
        for k_shift in self.kin_syst_shifts:
            up_or_down = k_shift.split("_")[-1]
            tauES_shift = up_or_down if ("tauES" in k_shift) else "nom"
            efake_shift = up_or_down if ("efake" in k_shift) else "nom"
            mfake_shift = up_or_down if ("mfake" in k_shift) else "nom"
            eleES_shift = up_or_down if ("eleES" in k_shift) else "nom"
            muES_shift = up_or_down if ("muES" in k_shift) else "nom"
            eleSmear_shift = up_or_down if ("eleSmear" in k_shift) else "nom"
            unclMET_shift = up_or_down if ("unclMET" in k_shift) else "nom"
            JES_shift = up_or_down if ("JES" in k_shift) else "nom"
            JER_shift = up_or_down if ("JER" in k_shift) else "nom"

            # grab baseline leptons, apply energy scale shifts
            baseline_e, e_shifts = apply_eleES(
                get_baseline_electrons(events.Electron),
                self.eleES_SFs,
                year=year,
                eleES_shift=eleES_shift,
                eleSmear_shift=eleSmear_shift,
                is_data=is_data,
            )
            baseline_m, m_shifts = apply_muES(
                get_baseline_muons(events.Muon),
                self.muES_SFs,
                shift=muES_shift,
                is_data=is_data,
            )
            baseline_t, t_shifts = apply_tauES(
                get_baseline_taus(events.Tau),
                self.tauID_SFs,
                tauES_shift=tauES_shift,
                efake_shift=efake_shift,
                mfake_shift=mfake_shift,
                is_data=is_data,
            )

            # grab the MET, shift it according to the energy scale shifts
            jets, MET = events.Jet, events.MET
            MET["pt"] = MET.T1_pt
            MET["phi"] = MET.T1_phi
            if is_data:
                MET = apply_met_phi_SFs(
                    self.met_phi_SFs,
                    MET,
                    events.PV.npvsGood,
                    events.run,
                    is_data,
                )
            else:
                MET = apply_met_phi_SFs(
                    self.met_phi_SFs, MET, events.PV.npvsGood, events.run
                )
            jets, MET = apply_JES_shifts(jets, MET, JES_shift)
            jets, MET = apply_JER_shifts(jets, MET, JER_shift)
            MET = apply_unclMET_shifts(MET, shift=unclMET_shift)
            MET = shift_MET(MET, [e_shifts, m_shifts, t_shifts], is_data=is_data)

            # seeds the lepton count veto
            tight_e = baseline_e[tight_electrons(baseline_e)]
            tight_m = baseline_m[tight_muons(baseline_m)]

            # grab the jets, count the number of b jets
            baseline_j = get_baseline_jets(jets)
            baseline_b = get_baseline_bjets(baseline_j)

            # build ll pairs
            candidates = {}
            for ll_pair in ["ee", "mm"]:
                if (ll_pair[:2] == "ee") and ("_Electrons" not in filename):
                    continue
                if (ll_pair[:2] == "mm") and ("_Muons" not in filename):
                    continue

                l = tight_e if (ll_pair == "ee") else tight_m
                ll = ak.combinations(l, 2, axis=1, fields=["l1", "l2"])
                ll = dR_ll(ll)
                ll = build_Z_cand(ll)
                ll = closest_to_Z_mass(ll)

                # apply trigger filter mask
                mask, tpt1, teta1, tpt2, teta2 = trigger_filter(
                    ll, events.TrigObj, ll_pair
                )
                mask = mask & check_trigger_path(events.HLT, year, ll_pair)
                ll = ak.fill_none(ll.mask[mask], [], axis=0)

                # build tt pairs, combine to form lltt candidates
                for cat in self.categories.values():
                    if cat[:2] != ll_pair:
                        continue

                    # build 4l final state
                    tt = get_tt(baseline_e, baseline_m, baseline_t, cat)
                    lltt = ak.cartesian({"ll": ll, "tt": tt}, axis=1)
                    lltt = dR_lltt(lltt, cat)
                    charge = lltt.tt.t1.charge * lltt.tt.t2.charge
                    lltt = lltt[charge > 0] if self.same_sign else lltt[charge < 0]
                    lltt = highest_LT(lltt, cat, apply_LT_cut=self.LT_cut)
                    mask = np.ones(len(lltt), dtype=bool)
                    lltt = ak.fill_none(lltt.mask[mask], [], axis=0)
                    if len(ak.flatten(lltt)) == 0:
                        continue

                    # determine which legs passed tight selections
                    relaxed = self.same_sign and self.relaxed
                    lltt = append_tight_masks(lltt, cat, relaxed=relaxed)
                    lltt["cat"] = self.cat_to_num[cat]
                    lltt["MET"] = MET

                    # determine weights
                    lltt["weight"] = np.ones(len(lltt))

                    # if it's MC, apply trigger weights and lepton ID
                    if not is_data:
                        # apply trigger scale factors
                        trig_SFs = (
                            self.e_trig_SFs if ll_pair == "ee" else self.m_trig_SFs
                        )
                        wt1 = lepton_trig_weight(tpt1, teta1, trig_SFs, lep=ll_pair[0])
                        wt2 = lepton_trig_weight(tpt2, teta2, trig_SFs, lep=ll_pair[0])
                        lltt["weight"] = lltt.weight * wt1 * wt2

                        # mask non-prompt and non-tight MC events
                        lltt = lltt[
                            is_prompt(lltt, cat) & lltt.t1_tight & lltt.t2_tight
                        ]
                        if len(ak.flatten(lltt)) == 0:
                            continue
                        # t0 = time.time()
                        lepton_IDs = self.apply_lepton_ID_SFs(lltt, cat)
                        lltt["weight"] = lltt.weight * lepton_IDs
                        lltt["tauID_nom"] = self.apply_tau_ID_SFs(lltt, cat)
                        for dm in [0, 1, 10, 11]:
                            lltt[f"tauID_{dm}_up"] = self.apply_tau_ID_SFs(
                                lltt,
                                cat,
                                shift="up",
                                dm_shift=dm,
                            )
                            lltt[f"tauID_{dm}_down"] = self.apply_tau_ID_SFs(
                                lltt,
                                cat,
                                shift="down",
                                dm_shift=dm,
                            )

                    # otherwise, if data apply the fake weights
                    else:
                        lltt["weight"] = lltt["weight"] * self.get_fake_weights(
                            lltt, cat
                        )

                    lltt["dR"] = lltt.tt.t1.delta_r(lltt.tt.t2)
                    print(lltt.dR)

                    # append the candidates
                    candidates[cat] = lltt

            if len(candidates) == 0:
                return output
            cands = ak.concatenate(list(candidates.values()), axis=1)
            cands["evtID"] = evtID
            cands["lumi_block"] = lumi_block
            cands["run"] = run
            # cands["btags"] = b_counts
            # cands["btruth"] = b_truth
            mask = ak.num(cands) == 1
            cands, jets, bjets = cands[mask], baseline_j[mask], baseline_b[mask]
            cands = ak.flatten(cands)

            if len(cands) == 0:
                continue

            # get lepton count veto masks
            # t0 = time.time()
            lepton_count_veto_masks = get_lepton_count_veto_masks(
                baseline_e[mask], baseline_m[mask], baseline_t[mask]
            )
            veto_mask = np.zeros(len(cands), dtype=bool)
            for cat_str, cat_num in self.cat_to_num.items():
                veto_mask = veto_mask | (
                    lepton_count_veto_masks[cat_str] & (cands.cat == cat_num)
                )
            cands = cands[veto_mask]
            jets = jets[veto_mask]
            bjets = bjets[veto_mask]

            # count btags
            cands = count_btags(cands, bjets)

            # for data, fill in categories of reducible/fake and tight/loose
            if is_data:
                is_tight = cands.t1_tight & cands.t2_tight
                for group_label, mask in [(group, is_tight), ("reducible", ~is_tight)]:
                    cands_group = cands[mask]
                    if len(cands_group) == 0:
                        continue
                    # t0 = time.time()
                    fastmtt_out = self.run_fastmtt(cands_group) if self.fastmtt else {}
                    # print(f"fastmtt time: {time.time() - t0}")
                    self.fill_histos(
                        output,
                        cands_group,
                        cands_group["weight"],
                        fastmtt_out,
                        group=group_label,
                        dataset=dataset,
                        name=name,
                        syst_shift="none",
                        blind=self.blind,
                        is_data=is_data,
                    )
                return output

            # if MC
            if not is_data:

                # calculate baseline global event weights
                global_weight = global_weights.weight()[mask][veto_mask]

                # t0 = time.time()
                # calculate the nominal btag event weights
                bshift_weight = apply_btag_corrections(
                    jets,
                    self.btag_SFs,
                    self.btag_eff_tables,
                    self.btag_pt_bins,
                    self.btag_eta_bins,
                    dataset,
                    shift="central",
                )

                # run fastmtt
                fastmtt_out = {}
                if self.fastmtt:
                    # t0 = time.time()
                    fastmtt_out = self.run_fastmtt(cands)
                    # print(f"FastMTT {time.time() - t0}")

                # loop over systematic shifts for the event weights
                for e_shift in self.event_syst_shifts:
                    if ("nom" not in k_shift) and ("nom" not in e_shift):
                        continue

                    up_or_down = e_shift.split("_")[-1]

                    # shift l1prefire or pileup weights
                    l1prefire_shift = up_or_down if ("l1prefire" in e_shift) else None
                    if "l1prefire" in e_shift and not self.has_L1PreFiringWeight:
                        continue
                    pileup_shift = up_or_down if ("pileup" in e_shift) else None
                    btag_shift = e_shift[5:] if ("btag" in e_shift) else None
                    tauID_shifts = {
                        "0": up_or_down if ("tauID_0_" in e_shift) else None,
                        "1": up_or_down if ("tauID_1_" in e_shift) else None,
                        "10": up_or_down if ("tauID_10" in e_shift) else None,
                        "11": up_or_down if ("tauID_11" in e_shift) else None,
                    }
                    tauID_dm = e_shift.split("_")[1] if ("tauID" in e_shift) else None

                    # apply shifts
                    w = ak.copy(cands["weight"])
                    if l1prefire_shift is not None:
                        gw = global_weights.weight(
                            modifier=f"l1prefire{l1prefire_shift.capitalize()}",
                        )[mask][veto_mask]
                        w = w * bshift_weight * gw * cands["tauID_nom"]
                    elif pileup_shift is not None:
                        gw = global_weights.weight(
                            modifier=f"pileup{pileup_shift.capitalize()}",
                        )[mask][veto_mask]
                        w = w * bshift_weight * gw * cands["tauID_nom"]
                    elif btag_shift is not None:
                        bw = apply_btag_corrections(
                            jets,
                            self.btag_SFs,
                            self.btag_eff_tables,
                            self.btag_pt_bins,
                            self.btag_eta_bins,
                            dataset,
                            shift=btag_shift,
                        )
                        w = w * bw * global_weight * cands["tauID_nom"]
                    elif tauID_dm is not None:
                        w = (
                            w
                            * bshift_weight
                            * global_weight
                            * cands[f"tauID_{tauID_dm}_{tauID_shifts[tauID_dm]}"]
                        )
                    else:
                        w = w * bshift_weight * global_weight * cands["tauID_nom"]

                    # label the systematics
                    syst_shift = "nom"
                    if e_shift != "nom":
                        syst_shift = e_shift
                    if k_shift != "nom":
                        syst_shift = k_shift

                    # t0 = time.time()
                    self.fill_histos(
                        output,
                        cands,
                        w,
                        fastmtt_out,
                        group=group,
                        dataset=dataset,
                        name=name,
                        syst_shift=syst_shift,
                        blind=(is_data and self.blind),
                    )
                    # print(f"Fill histograms: {time.time() - t0}")

        return output

    def get_fake_weights(self, lltt, cat):
        t1_tight_mask, t2_tight_mask = lltt.t1_tight, lltt.t2_tight
        t1_tight_mask, num = ak.flatten(t1_tight_mask), ak.num(t1_tight_mask)
        t2_tight_mask = ak.flatten(t2_tight_mask)
        if len(t1_tight_mask) == 0:
            return np.ones(len(lltt), dtype=float)

        # determine if in the barrel
        t1_barrel = ak.flatten(abs(lltt.tt.t1.eta) < 1.479)
        t2_barrel = ak.flatten(abs(lltt.tt.t2.eta) < 1.479)

        # t1 and t2 depend on the type of tau decay being considered
        t1_fake_barrel = t1_barrel & ~t1_tight_mask
        t1_fake_endcap = ~t1_barrel & ~t1_tight_mask
        t2_fake_barrel = t2_barrel & ~t2_tight_mask
        t2_fake_endcap = ~t2_barrel & ~t2_tight_mask
        t1_pt = ak.flatten(lltt.tt.t1.pt)
        t2_pt = ak.flatten(lltt.tt.t2.pt)

        # leptonic decays are easy to handle
        fr3 = np.ones(len(t1_pt)) * t1_tight_mask
        if (cat[2] == "e") or (cat[2] == "m"):
            ll_str = "ee" if cat[2] == "e" else "mm"
            t1_fr_barrel = self.fake_rates[ll_str]["barrel"]
            t1_fr_endcap = self.fake_rates[ll_str]["endcap"]
            fr3_barrel = t1_fr_barrel(t1_pt)
            fr3_endcap = t1_fr_endcap(t1_pt)
            fr3 = fr3 + ((fr3_barrel * t1_fake_barrel) + (fr3_endcap * t1_fake_endcap))

        # hadronic tau decays are not so easy
        elif cat[2] == "t":
            t1_fr_barrel = self.fake_rates["tt"]["barrel"]
            t1_fr_endcap = self.fake_rates["tt"]["endcap"]
            for dm in [0, 1, 10, 11]:
                t1_dm = ak.flatten(lltt.tt.t1.decayMode == dm)
                fr3_barrel = t1_fr_barrel[dm](t1_pt)
                fr3_endcap = t1_fr_endcap[dm](t1_pt)
                t1_fake_barrel_dm = t1_fake_barrel & t1_dm
                t1_fake_endcap_dm = t1_fake_endcap & t1_dm
                fr3 = fr3 + (
                    (fr3_barrel * t1_fake_barrel_dm) + (fr3_endcap * t1_fake_endcap_dm)
                )

        # ditto for the second di-tau leg
        fr4 = np.ones(len(t2_pt)) * t2_tight_mask
        if cat[3] == "m":
            t2_fr_barrel = self.fake_rates["mm"]["barrel"]
            t2_fr_endcap = self.fake_rates["mm"]["endcap"]
            fr4_barrel = t2_fr_barrel(t2_pt)
            fr4_endcap = t2_fr_endcap(t2_pt)
            fr4 = fr4 + ((fr4_barrel * t2_fake_barrel) + (fr4_endcap * t2_fake_endcap))

        elif cat[3] == "t":
            t2_fr_barrel = self.fake_rates[cat[2:]]["barrel"]
            t2_fr_endcap = self.fake_rates[cat[2:]]["endcap"]
            for dm in [0, 1, 10, 11]:
                t2_dm = ak.flatten(lltt["tt"]["t2"].decayMode == dm)
                fr4_barrel = t2_fr_barrel[dm](t2_pt)
                fr4_endcap = t2_fr_endcap[dm](t2_pt)
                t2_fake_barrel_dm = t2_fake_barrel & t2_dm
                t2_fake_endcap_dm = t2_fake_endcap & t2_dm
                fr4 = fr4 + (
                    (fr4_barrel * t2_fake_barrel_dm) + (fr4_endcap * t2_fake_endcap_dm)
                )

        fw1 = ak.nan_to_num(fr3 / (1 - fr3), nan=0, posinf=0, neginf=0)
        fw2 = ak.nan_to_num(fr4 / (1 - fr4), nan=0, posinf=0, neginf=0)
        ff = (~t1_tight_mask & ~t2_tight_mask) * fw1 * fw2
        pf = (t1_tight_mask & ~t2_tight_mask) * fw2
        fp = (~t1_tight_mask & t2_tight_mask) * fw1
        out = pf + fp - ff
        # make sure data gets a weight of 1
        out = out + (t1_tight_mask & t2_tight_mask)
        return ak.unflatten(out, num)

    def fill_histos(
        self,
        output,
        lltt,
        weight,
        fastmtt_out,
        dataset,
        name,
        group,
        blind=False,
        syst_shift=None,
        is_data=False,
    ):

        # fill the four-vectors
        label_dict = {
            ("ll", "l1"): "1",
            ("ll", "l2"): "2",
            ("tt", "t1"): "3",
            ("tt", "t2"): "4",
        }

        # force fastmtt output to be between mtt_down and mtt_up GeV
        mask = np_flat(
            (fastmtt_out["mtt_corr"] > self.mtt_corr_down)
            & (fastmtt_out["mtt_corr"] < self.mtt_corr_up)
        )

        # only plot outputs for m4l in the acceptible range
        m4l = fastmtt_out["m4l_cons"]
        mask = mask & (m4l >= self.m4l_cons_down) & (m4l <= self.m4l_cons_up)

        # sort out histogram categories
        signs = np_flat(lltt["tt"]["t1"].charge * lltt["tt"]["t2"].charge)[mask]
        btags = np_flat(lltt.btags > 0)[mask]
        cats = np_flat(lltt.cat)[mask]
        cats = np.array([self.categories[c] for c in cats])
        weight = np_flat(weight)[mask]
        weight = np.nan_to_num(weight, nan=0, posinf=0, neginf=0)
        evtID, lumi_block, run = lltt.evtID[mask], lltt.lumi_block[mask], lltt.run[mask]

        if is_data and "reducible" not in group:
            output["evtID"] = col_acc(np_flat(evtID))
            output["lumi_block"] = col_acc(np_flat(lumi_block))
            output["run"] = col_acc(np_flat(run))

        # fill the lltt leg four-vectors
        for leg, label in label_dict.items():
            p4 = lltt[leg[0]][leg[1]]
            output["pt"][name].fill(
                group=group,
                category=cats,
                # sign=signs,
                leg=label,
                btags=btags,
                syst_shift=syst_shift,
                pt=np_flat(p4.pt)[mask],
                weight=weight,
            )
            # hmask = (cats == "eeem") | (cats == "mmem")
            # print(cats[hmask])
            # print(f"leg {leg}", np_flat(p4.pt)[mask][hmask])

        # fill the mass of the dilepton system w/ various systematic shifts
        mll = np_flat((lltt["ll"]["l1"] + lltt["ll"]["l2"]).mass)
        output["mll"][name].fill(
            group=group,
            category=cats,
            # sign=signs,
            btags=btags,
            syst_shift=syst_shift,
            mll=mll[mask],
            weight=weight,
        )

        # fill the met with various systematics considered
        met = np_flat(lltt.MET.pt)
        output["met"][name].fill(
            group=group,
            category=cats,
            # sign=signs,
            btags=btags,
            syst_shift=syst_shift,
            met=met[mask],
            weight=weight,
        )

        met_phi = np_flat(lltt.MET.phi)
        output["met_phi"][name].fill(
            group=group,
            category=cats,
            btags=btags,
            syst_shift=syst_shift,
            met_phi=met_phi[mask],
            weight=weight,
        )

        dR = np_flat(lltt.dR)
        output["dR"][name].fill(
            group=group,
            category=cats,
            btags=btags,
            syst_shift=syst_shift,
            dR=dR[mask],
            weight=weight,
        )

        # fill the Zh->lltt candidate mass spectrum (raw, uncorrected)
        mtt = np_flat((lltt["tt"]["t1"] + lltt["tt"]["t2"]).mass)[mask]
        m4l = np_flat(
            (
                lltt["ll"]["l1"]
                + lltt["ll"]["l2"]
                + lltt["tt"]["t1"]
                + lltt["tt"]["t2"]
            ).mass
        )[mask]

        blind_mask = np.ones(len(m4l), dtype=bool)
        if blind:
            blind_mask[((mtt > 40) & (mtt < 120) & (signs < 0))] = False
        if sum(blind_mask) > 0:
            output["mtt"][name].fill(
                group=group,
                category=cats[blind_mask],
                # sign=signs[blind_mask],
                mass_type="raw",
                btags=btags[blind_mask],
                syst_shift=syst_shift,
                mass=mtt[blind_mask],
                weight=weight[blind_mask],
            )
            for m4l_label in ["m4l", "m4l_reg", "m4l_fine", "m4l_binopt"]:
                output[m4l_label][name].fill(
                    group=group,
                    category=cats[blind_mask],
                    # sign=signs[blind_mask],
                    mass_type="raw",
                    btags=btags[blind_mask],
                    syst_shift=syst_shift,
                    mass=m4l[blind_mask],
                    weight=weight[blind_mask],
                )

            # fill the Zh->lltt candidate mass spectrums (corrected, constrained)
            for mass_label, mass_data in fastmtt_out.items():
                key = mass_label.split("_")[0]  # mtt or m4l
                mass_type = mass_label.split("_")[1]  # corr or cons
                keys = (
                    ["m4l", "m4l_reg", "m4l_fine", "m4l_binopt"]
                    if "m4l" in key
                    else ["mtt"]
                )
                for key in keys:
                    output[key][name].fill(
                        group=group,
                        category=cats[blind_mask],
                        # sign=signs[blind_mask],
                        mass_type=mass_type,
                        btags=btags[blind_mask],
                        syst_shift=syst_shift,
                        mass=mass_data[mask][blind_mask],
                        weight=weight[blind_mask],
                    )
                    if "reducible" in group.lower():
                        output[f"reducible_{key}_{mass_type}"] = col_acc(mass_data)
                    if "data" in group.lower():
                        output[f"data_{key}_{mass_type}"] = col_acc(mass_data)
            if "reducible" in group.lower():
                output["reducible_btag"] = col_acc(btags)
                output["reducible_cat"] = col_acc(cats)
                output["reducible_weight"] = col_acc(weight)
            if "data" in group.lower():
                output["data_btag"] = col_acc(btags)
                output["data_cat"] = col_acc(cats)
                output["data_weight"] = col_acc(weight)

    def apply_lepton_ID_SFs(self, lltt_all, cat):
        lltt, num = ak.flatten(lltt_all), ak.num(lltt_all)
        l1, l2 = lltt["ll"]["l1"], lltt["ll"]["l2"]
        t1, t2 = lltt["tt"]["t1"], lltt["tt"]["t2"]

        # e/mu scale factors
        if cat[:2] == "ee":
            l1_w = lepton_ID_weight(l1, "e", self.eleID_SFs)
            l2_w = lepton_ID_weight(l2, "e", self.eleID_SFs)
        elif cat[:2] == "mm":
            l1_w = lepton_ID_weight(l1, "m", self.muID_SFs)
            l2_w = lepton_ID_weight(l2, "m", self.muID_SFs)

        # also consider hadronic taus
        t1_w, t2_w = np.ones_like(l1_w), np.ones_like(l1_w)
        if cat[2:] == "em":
            t1_w = lepton_ID_weight(t1, "e", self.eleID_SFs)
            t2_w = lepton_ID_weight(t2, "m", self.muID_SFs)
        elif cat[2:] == "et":
            t1_w = lepton_ID_weight(t1, "e", self.eleID_SFs)
        elif cat[2:] == "mt":
            t1_w = lepton_ID_weight(t1, "m", self.muID_SFs)

        # apply ID scale factors
        w = l1_w * l2_w * t1_w * t2_w
        return ak.unflatten(w, num)

    def apply_tau_ID_SFs(self, lltt_all, cat, shift="nom", dm_shift=None):
        lltt, num = ak.flatten(lltt_all), ak.num(lltt_all)
        t1, t2 = lltt["tt"]["t1"], lltt["tt"]["t2"]
        if cat[2:] == "et":
            w = tau_ID_weight(t2, self.tauID_SFs, cat, syst=shift, dm_shift=dm_shift)
        elif cat[2:] == "mt":
            w = tau_ID_weight(t2, self.tauID_SFs, cat, syst=shift, dm_shift=dm_shift)
        elif cat[2:] == "tt":
            w = tau_ID_weight(
                t1, self.tauID_SFs, cat, syst=shift, dm_shift=dm_shift
            ) * tau_ID_weight(t2, self.tauID_SFs, cat, syst=shift, dm_shift=dm_shift)
        else:
            w = np.ones_like(t1.pt)
        return ak.unflatten(w, num)

    def run_fastmtt(self, final_states):
        l1, l2 = final_states.ll.l1, final_states.ll.l2
        t1, t2 = final_states.tt.t1, final_states.tt.t2
        met = final_states.MET
        cats = final_states.cat
        map_it = {"e": 0, "m": 1, "t": 2}
        t1_cats = np.array([map_it[self.categories[cat][2]] for cat in cats])
        t2_cats = np.array([map_it[self.categories[cat][3]] for cat in cats])
        return fastmtt(
            np_flat(l1.pt).astype(np.float32),
            np_flat(l1.eta).astype(np.float32),
            np_flat(l1.phi).astype(np.float32),
            np_flat(l1.mass).astype(np.float32),
            np_flat(l2.pt).astype(np.float32),
            np_flat(l2.eta).astype(np.float32),
            np_flat(l2.phi).astype(np.float32),
            np_flat(l2.mass).astype(np.float32),
            np_flat(t1.pt).astype(np.float32),
            np_flat(t1.eta).astype(np.float32),
            np_flat(t1.phi).astype(np.float32),
            np_flat(t1.mass).astype(np.float32),
            t1_cats.astype(np.int64),
            np_flat(t2.pt).astype(np.float32),
            np_flat(t2.eta).astype(np.float32),
            np_flat(t2.phi).astype(np.float32),
            np_flat(t2.mass).astype(np.float32),
            t2_cats.astype(np.int64),
            np_flat(met.pt * np.cos(met.phi)).astype(np.float32),
            np_flat(met.pt * np.sin(met.phi)).astype(np.float32),
            np_flat(met.covXX).astype(np.float32),
            np_flat(met.covXY).astype(np.float32),
            np_flat(met.covXY).astype(np.float32),
            np_flat(met.covYY).astype(np.float32),
        )

    def postprocess(self, accumulator):
        pass


@nb.njit(
    nb.types.DictType(nb.types.unicode_type, nb.float32[:])(
        nb.float32[::1],
        nb.float32[::1],
        nb.float32[::1],
        nb.float32[::1],
        nb.float32[::1],
        nb.float32[::1],
        nb.float32[::1],
        nb.float32[::1],
        nb.float32[::1],
        nb.float32[::1],
        nb.float32[::1],
        nb.float32[::1],
        nb.int64[::1],
        nb.float32[::1],
        nb.float32[::1],
        nb.float32[::1],
        nb.float32[::1],
        nb.int64[::1],
        nb.float32[::1],
        nb.float32[::1],
        nb.float32[::1],
        nb.float32[::1],
        nb.float32[::1],
        nb.float32[::1],
    )
)
def fastmtt(
    pt_1,
    eta_1,
    phi_1,
    mass_1,
    pt_2,
    eta_2,
    phi_2,
    mass_2,
    pt_3,
    eta_3,
    phi_3,
    mass_3,
    decay_type_3,
    pt_4,
    eta_4,
    phi_4,
    mass_4,
    decay_type_4,
    met_x,
    met_y,
    metcov_xx,
    metcov_xy,
    metcov_yx,
    metcov_yy,
):

    # initialize global parameters
    light_masses = {0: 0.51100e-3, 1: 0.10566}
    m_tau = 1.7768

    # initialize Higgs--> tau + tau decays, tau decay types
    N = len(pt_1)
    m_tt_opt = np.zeros(N, dtype=np.float32)
    m_tt_opt_c = np.zeros(N, dtype=np.float32)
    m_lltt_opt = np.zeros(N, dtype=np.float32)
    m_lltt_opt_c = np.zeros(N, dtype=np.float32)

    for i in range(N):
        if decay_type_3[i] != 2:
            mt1 = light_masses[decay_type_3[i]]
        else:
            mt1 = mass_3[i]
        if decay_type_4[i] != 2:
            mt2 = light_masses[decay_type_4[i]]
        else:
            mt2 = mass_4[i]
        l1 = vec.obj(pt=pt_1[i], eta=eta_1[i], phi=phi_1[i], mass=mass_1[i])
        l2 = vec.obj(pt=pt_2[i], eta=eta_2[i], phi=phi_2[i], mass=mass_2[i])
        t1 = vec.obj(pt=pt_3[i], eta=eta_3[i], phi=phi_3[i], mass=mt1)
        t2 = vec.obj(pt=pt_4[i], eta=eta_4[i], phi=phi_4[i], mass=mt2)

        m_vis = math.sqrt(
            2 * t1.pt * t2.pt * (math.cosh(t1.eta - t2.eta) - math.cos(t1.phi - t2.phi))
        )
        m_vis_1 = mt1
        m_vis_2 = mt2

        if decay_type_3[i] == 2 and m_vis_1 > 1.5:
            m_vis_1 = 0.3
        if decay_type_4[i] == 2 and m_vis_2 > 1.5:
            m_vis_2 = 0.3

        metcovinv_xx, metcovinv_yy = metcov_yy[i], metcov_xx[i]
        metcovinv_xy, metcovinv_yx = -metcov_xy[i], -metcov_yx[i]
        metcovinv_det = metcovinv_xx * metcovinv_yy - metcovinv_yx * metcovinv_xy

        if abs(metcovinv_det) < 1e-10:
            print("Warning! Ill-conditioned MET covariance at event index", i)
            continue

        met_const = 1 / (2 * math.pi * math.sqrt(metcovinv_det))
        min_likelihood, x1_opt, x2_opt = 999, 1, 1  # standard optimization
        min_likelihood_c, x1_opt_c, x2_opt_c = 999, 1, 1  # constrained optimization
        mass_likelihood, met_transfer = 0, 0
        for x1 in np.arange(0, 1, 0.01):
            for x2 in np.arange(0, 1, 0.01):
                x1_min = min(1, math.pow((m_vis_1 / m_tau), 2))
                x2_min = min(1, math.pow((m_vis_2 / m_tau), 2))
                if (x1 < x1_min) or (x2 < x2_min):
                    continue

                t1_x1, t2_x2 = t1 * (1 / x1), t2 * (1 / x2)
                ditau_test = vec.obj(
                    px=t1_x1.px + t2_x2.px,
                    py=t1_x1.py + t2_x2.py,
                    pz=t1_x1.pz + t2_x2.pz,
                    E=t1_x1.E + t2_x2.E,
                )
                nu_test = vec.obj(
                    px=ditau_test.px - t1.px - t2.px,
                    py=ditau_test.py - t1.py - t2.py,
                    pz=ditau_test.pz - t1.pz - t2.pz,
                    E=ditau_test.E - t1.E - t2.E,
                )
                test_mass = ditau_test.mass

                passes_constraint = False
                if (test_mass > 124) and (test_mass < 126):
                    passes_constraint = True

                # calculate mass likelihood integral
                m_shift = test_mass * (1 / 1.15)
                if m_shift < m_vis:
                    continue
                x1_min = min(1.0, math.pow((m_vis_1 / m_tau), 2))
                x2_min = max(
                    math.pow((m_vis_2 / m_tau), 2), math.pow((m_vis / m_shift), 2)
                )
                x2_max = min(1.0, math.pow((m_vis / m_shift), 2) / x1_min)
                if x2_max < x2_min:
                    continue
                J = 2 * math.pow(m_vis, 2) * math.pow(m_shift, -6)
                I_x2 = math.log(x2_max) - math.log(x2_min)
                I_tot = I_x2
                if decay_type_3[i] != 2:
                    I_m_nunu_1 = math.pow((m_vis / m_shift), 2) * (
                        math.pow(x2_max, -1) - math.pow(x2_min, -1)
                    )
                    I_tot += I_m_nunu_1
                if decay_type_4[i] != 2:
                    I_m_nunu_2 = math.pow((m_vis / m_shift), 2) * I_x2 - (
                        x2_max - x2_min
                    )
                    I_tot += I_m_nunu_2
                mass_likelihood = 1e9 * J * I_tot

                # calculate MET transfer function
                residual_x = met_x[i] - nu_test.x
                residual_y = met_y[i] - nu_test.y
                pull2 = residual_x * (
                    metcovinv_xx * residual_x + metcovinv_xy * residual_y
                ) + residual_y * (metcovinv_yx * residual_x + metcovinv_yy * residual_y)
                pull2 /= metcovinv_det
                met_transfer = met_const * math.exp(-0.5 * pull2)

                likelihood = -met_transfer * mass_likelihood
                if likelihood < min_likelihood:
                    min_likelihood = likelihood
                    x1_opt, x2_opt = x1, x2

                if passes_constraint:
                    if likelihood < min_likelihood_c:
                        min_likelihood_c = likelihood
                        x1_opt_c, x2_opt_c = x1, x2

        t1_x1, t2_x2 = t1 * (1 / x1_opt), t2 * (1 / x2_opt)
        p4_ditau_opt = vec.obj(
            px=t1_x1.px + t2_x2.px,
            py=t1_x1.py + t2_x2.py,
            pz=t1_x1.pz + t2_x2.pz,
            E=t1_x1.E + t2_x2.E,
        )
        t1_x1, t2_x2 = t1 * (1 / x1_opt_c), t2 * (1 / x2_opt_c)
        p4_ditau_opt_c = vec.obj(
            px=t1_x1.px + t2_x2.px,
            py=t1_x1.py + t2_x2.py,
            pz=t1_x1.pz + t2_x2.pz,
            E=t1_x1.E + t2_x2.E,
        )

        lltt_opt = vec.obj(
            px=l1.px + l2.px + p4_ditau_opt.px,
            py=l1.py + l2.py + p4_ditau_opt.py,
            pz=l1.pz + l2.pz + p4_ditau_opt.pz,
            E=l1.E + l2.E + p4_ditau_opt.E,
        )
        lltt_opt_c = vec.obj(
            px=l1.px + l2.px + p4_ditau_opt_c.px,
            py=l1.py + l2.py + p4_ditau_opt_c.py,
            pz=l1.pz + l2.pz + p4_ditau_opt_c.pz,
            E=l1.E + l2.E + p4_ditau_opt_c.E,
        )

        m_tt_opt[i] = p4_ditau_opt.mass
        m_tt_opt_c[i] = p4_ditau_opt_c.mass
        m_lltt_opt[i] = lltt_opt.mass
        m_lltt_opt_c[i] = lltt_opt_c.mass

    result_dict = Dict.empty(
        key_type=nb.types.unicode_type,
        value_type=float_array,
    )

    result_dict["mtt_corr"] = m_tt_opt.astype("f")
    result_dict["mtt_cons"] = m_tt_opt_c.astype("f")
    result_dict["m4l_corr"] = m_lltt_opt.astype("f")
    result_dict["m4l_cons"] = m_lltt_opt_c.astype("f")
    return result_dict
