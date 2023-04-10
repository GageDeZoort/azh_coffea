from __future__ import annotations

import logging
import warnings

import awkward as ak
import numpy as np
from coffea import analysis_tools, processor

from azh_analysis.selections.fake_rate_selections import (
    additional_cuts,
    dr_3l,
    get_lepton_count_veto_mask,
    is_prompt_lepton,
    tight_hadronic_taus,
    transverse_mass,
)
from azh_analysis.selections.preselections import (
    build_Z_cand,
    check_trigger_path,
    closest_to_Z_mass,
    dR_ll,
    filter_MET,
    filter_PV,
    get_baseline_electrons,
    get_baseline_muons,
    get_baseline_taus,
    tight_electrons,
    tight_muons,
    trigger_filter,
)
from azh_analysis.utils.corrections import (
    apply_eleES,
    apply_muES,
    apply_tauES,
    apply_unclMET_shifts,
    lepton_ID_weight,
    lepton_trig_weight,
    shift_MET,
    tau_ID_weight_3l,
)
from azh_analysis.utils.histograms import make_fr_hist_stack
from azh_analysis.utils.logging import init_logging

# from coffea.processor import column_accumulator as col_acc


warnings.filterwarnings("ignore")


def flat(a, axis=None):
    return ak.flatten(a, axis=axis)


def np_flat(a, axis=None):
    return ak.to_numpy(ak.flatten(a, axis=axis))


class FakeRateProcessor(processor.ProcessorABC):
    def __init__(
        self,
        source="",
        year="",
        sample_info=None,
        fileset=None,
        sample_dir="../sample_lists/sample_yamls",
        pileup_weights=None,
        lumi_masks=None,
        nevts_dict=None,
        eleID_SFs=None,
        muID_SFs=None,
        tauID_SFs=None,
        muES_SFs=None,
        dyjets_weights=None,
        e_trig_SFs=None,
        m_trig_SFs=None,
        verbose=False,
    ):

        # initialize member variables
        init_logging(verbose=verbose)
        self.info = sample_info
        self.fileset = fileset

        self.eras = {
            "2016preVFP": "Summer16",
            "2016postVFP": "Summer16",
            "2017": "Fall17",
            "2018": "Autumn18",
        }
        self.lumi = {
            "2016preVFP": 35.9 * 1000,
            "2016postVFP": 35.9 * 1000,
            "2017": 41.5 * 1000,
            "2018": 59.7 * 1000,
        }
        self.pileup_weights = pileup_weights
        self.lumi_masks = lumi_masks
        self.nevts_dict = nevts_dict
        self.eleID_SFs = eleID_SFs
        self.muID_SFs = muID_SFs
        self.tauID_SFs = tauID_SFs
        self.muES_SFs = muES_SFs
        self.e_trig_SFs = e_trig_SFs
        self.m_trig_SFs = m_trig_SFs
        self.dyjets_weights = dyjets_weights

    def process(self, events):
        logging.info(f"Processing {events.metadata['dataset']}")
        filename = events.metadata["filename"]

        # organize dataset, year, luminosity
        dataset = events.metadata["dataset"]
        year = dataset.split("_")[-1]
        name = dataset.replace(f"_{year}", "")
        properties = self.info[self.info["name"] == name]
        group = properties["group"][0]
        is_data = "data" in group
        nevts, xsec = properties["nevts"][0], properties["xsec"][0]
        output = make_fr_hist_stack(self.fileset, year)

        # if running on ntuples, need the pre-skim sum_of_weights
        if self.nevts_dict is not None:
            nevts = self.nevts_dict[dataset]
        elif not is_data:
            logging.debug("WARNING: may be using wrong sum_of_weights!")

        # weight by the data-MC luminosity ratio
        sample_weight = self.lumi[year] * xsec / nevts
        if is_data:
            sample_weight = 1

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
        if len(events) == 0:
            return output

        # global weights: sample weight, gen weight, pileup weight
        weights = analysis_tools.Weights(len(events), storeIndividual=True)
        ones = np.ones(len(events), dtype=float)

        # sample weights OR dyjets stitching weights
        if group == "DY":
            njets = ak.to_numpy(events.LHE.Njets)
            weights.add("dyjets_sample_weights", self.dyjets_weights(njets))
        else:  # otherwise weight by luminosity ratio
            weights.add("sample_weight", ones * sample_weight)

        # pileup weights and gen weights
        if (self.pileup_weights is not None) and not is_data:
            weights.add("gen_weight", events.genWeight)
            pu_weights = self.pileup_weights["nom"](events.Pileup.nTrueInt)
            weights.add("pileup_weight", pu_weights)

        # L1 prefiring weights if available
        if not is_data:
            try:
                weights.add(
                    "l1prefire",
                    weight=events.L1PreFiringWeight.Nom,
                    weightUp=events.L1PreFiringWeight.Up,
                    weightDown=events.L1PreFiringWeight.Dn,
                )
                self.has_L1PreFiringWeight = True
            except Exception:
                logging.info(f"No prefiring weights in {dataset}.")
                self.has_L1PreFiringWeight = False

        # grab baseline defined leptons
        baseline_e, e_shifts = apply_eleES(
            get_baseline_electrons(events.Electron),
            "nom",
            "nom",
            is_data=is_data,
        )
        baseline_m, m_shifts = apply_muES(
            get_baseline_muons(events.Muon), self.muES_SFs, "nom", is_data=is_data
        )
        baseline_t, t_shifts = apply_tauES(
            get_baseline_taus(events.Tau),
            self.tauID_SFs,
            "nom",
            "nom",
            "nom",
            is_data=is_data,
        )
        baseline_l = {"e": baseline_e, "m": baseline_m, "t": baseline_t}

        # grab tight electrons and muons, count them
        tight_e = baseline_e[tight_electrons(baseline_e)]
        tight_m = baseline_m[tight_muons(baseline_m)]

        # grab met,
        MET = events.MET
        MET["pt"] = MET.T1_pt
        MET["phi"] = MET.T1_phi
        MET = apply_unclMET_shifts(MET, "nom")
        MET = shift_MET(MET, [e_shifts, m_shifts, t_shifts], is_data=is_data)

        # build ll pairs
        for z_pair in ["ee", "mm"]:
            if (z_pair == "ee") and ("_Electrons" not in filename):
                continue
            if (z_pair == "mm") and ("_Muons" not in filename):
                continue

            # grab ll pair most consistent with the Z mass
            l = tight_e if (z_pair == "ee") else tight_m
            ll = ak.combinations(l, 2, axis=1, fields=["l1", "l2"])
            ll = dR_ll(ll)
            ll = build_Z_cand(ll)
            ll = closest_to_Z_mass(ll)

            # check that one of the leptons matches a trigger object
            mask, tpt1, teta1, tpt2, teta2 = trigger_filter(ll, events.TrigObj, z_pair)
            mask = mask & check_trigger_path(events.HLT, year, z_pair)
            ll = ak.fill_none(ll.mask[mask], [], axis=0)
            if not is_data:
                trig_SFs = self.e_trig_SFs if z_pair == "ee" else self.m_trig_SFs
                wt1 = lepton_trig_weight(tpt1, teta1, trig_SFs, lep=z_pair[0])
                wt2 = lepton_trig_weight(tpt2, teta2, trig_SFs, lep=z_pair[0])
                weights.add("l1_trig_weight", wt1)
                weights.add("l2_trig_weight", wt2)

            # with Z candidates built, consider the jet faking lepton modes
            for mode in ["e", "m", "et", "mt", "tt"]:

                # build all viable (Z->ll)+l pairs
                cat = z_pair + mode[-1]
                lll = ak.cartesian({"ll": ll, "l": baseline_l[mode[-1]]}, axis=1)
                lll["weight"] = weights.weight()
                lll["met"] = MET
                lll["cat"] = cat
                lll = dr_3l(lll, cat)

                # apply lepton count veto
                lll_mask = get_lepton_count_veto_mask(
                    cat,
                    baseline_e,
                    baseline_m,
                    baseline_t,
                )

                # create denominator and numerator regions
                lll_denom = ak.fill_none(lll.mask[lll_mask], [], axis=0)
                lll_denom = additional_cuts(lll_denom, mode)
                if mode == "e":
                    lll_num = lll_denom[tight_electrons(lll_denom["l"])]
                elif mode == "m":
                    lll_num = lll_denom[tight_muons(lll_denom["l"])]
                else:
                    lll_num = lll_denom[tight_hadronic_taus(lll_denom["l"], mode=mode)]
                lll_num = lll_num[~ak.is_none(lll_num, axis=1)]

                # if data, separate denominator and numerator
                if is_data:
                    lll_dict = {
                        ("Denominator", "Data"): lll_denom,
                        ("Numerator", "Data"): lll_num,
                    }
                # if MC, additionally separate prompt and fake
                else:
                    d_prompt_mask = is_prompt_lepton(lll_denom, mode)
                    n_prompt_mask = is_prompt_lepton(lll_num, mode)
                    lll_dict = {
                        ("Denominator", "Fake"): lll_denom[~d_prompt_mask],
                        ("Denominator", "Prompt"): lll_denom[d_prompt_mask],
                        ("Numerator", "Fake"): lll_num[~n_prompt_mask],
                        ("Numerator", "Prompt"): lll_num[n_prompt_mask],
                    }

                # fill hists for each combination of fake/prompt and denom/num
                for label, lll in lll_dict.items():
                    lll = lll[ak.num(lll) == 1]
                    if len(lll) == 0:
                        continue

                    # apply relevant scale factors
                    if not is_data:
                        lll["weight"] = lll["weight"] * self.apply_lepton_ID_SFs(
                            lll,
                            z_pair,
                            mode,
                            is_data=False,
                            numerator=(label[0] == "Numerator"),
                        )
                        # lll = self.apply_ES_shifts(lll, z_pair, mode)

                    self.fill_histos(
                        output,
                        lll,
                        z_pair,
                        mode,
                        numerator=label[0],
                        prompt=label[1],
                        dataset=dataset,
                        name=name,
                        group=group,
                    )

        return output

    def apply_ES_shifts(
        self,
        lll,
        z_pair,
        mode,
        eleES_shift="nom",
        muES_shift="nom",
        tauES_shift="nom",
        efake_shift="nom",
        mfake_shift="nom",
        eleSmear_shift="nom",
    ):

        diffs_list = []
        if len(lll) == 0:
            return lll
        lll, num = ak.flatten(lll), ak.num(lll)
        l1, l2, l = lll.ll.l1, lll.ll.l2, lll.l
        met = lll["met"]
        if z_pair == "ee":
            l1, diffs = apply_eleES(l1, eleES_shift, eleSmear_shift)
            diffs_list.append(diffs)
            l2, diffs = apply_eleES(l2, eleES_shift, eleSmear_shift)
            diffs_list.append(diffs)
        else:
            l1, diffs = apply_muES(l1, muES_shift)
            diffs_list.append(diffs)
            l2, diffs = apply_muES(l2, muES_shift)
            diffs_list.append(diffs)
        if mode == "e":
            l, diffs = apply_eleES(l, eleES_shift, eleSmear_shift)
            diffs_list.append(diffs)
        elif mode == "m":
            l, diffs = apply_muES(l, muES_shift)
            diffs_list.append(diffs)
        else:
            l, diffs = apply_tauES(
                l, self.tauID_SFs, tauES_shift, efake_shift, mfake_shift
            )
            diffs_list.append(diffs)

        # adjust the met
        met_x = met.pt * np.cos(met.phi)
        met_y = met.pt * np.sin(met.phi)
        for diffs in diffs_list:
            met_x = met_x + diffs["x"]
            met_y = met_y + diffs["y"]
            met_p4 = ak.zip(
                {"x": met_x, "y": met_y, "z": 0, "t": 0}, with_name="LorentzVector"
            )
            met["pt"] = met_p4.pt
            met["phi"] = met_p4.phi

        lll["ll"]["l1"] = l1
        lll["ll"]["l2"] = l2
        lll["l"] = l
        lll["met"] = met
        lll = ak.unflatten(lll, num)
        return lll

    def fill_histos(
        self,
        output,
        lll,
        z_pair,
        mode,
        prompt,
        numerator,
        dataset,
        name,
        group,
    ):

        # grab relevant variables
        category = z_pair + mode
        weight = np_flat(lll.weight)
        met = lll.met
        l1, l2, l = lll.ll.l1, lll.ll.l2, lll.l
        pt, eta = np_flat(l.pt), np_flat(l.eta)
        decay_mode = -1 * np.ones_like(weight)
        met = np_flat(lll.met.pt)
        mT = np_flat(transverse_mass(l, lll.met))
        mll = np_flat((l1 + l2).mass)
        # print(category, lll.l.fields)
        if len(mode) > 1:
            decay_mode = np_flat(l.decayMode)
        eta_barrel_bin = r"$|\eta|<1.479$"
        eta_endcap_bin = r"$|\eta|>1.479$"
        barrel_mask = abs(eta) < 1.479
        endcap_mask = abs(eta) > 1.479
        for eta_bin, m in [
            (eta_barrel_bin, barrel_mask),
            (eta_endcap_bin, endcap_mask),
        ]:
            if np.sum(m) == 0:
                continue

            # fill pt
            output["pt"][name].fill(
                group=group,
                category=category,
                prompt=prompt,
                numerator=numerator,
                eta_bin=eta_bin,
                decay_mode=decay_mode[m],
                pt=pt[m],
                weight=weight[m],
            )
            # fill the mass of the dilepton system w/ various systematic shifts
            output["mll"][name].fill(
                group=group,
                category=category,
                prompt=prompt,
                numerator=numerator,
                eta_bin=eta_bin,
                decay_mode=decay_mode[m],
                mll=mll[m],
                weight=weight[m],
            )
            # fill the met with various systematics considered
            output["met"][name].fill(
                group=group,
                category=category,
                prompt=prompt,
                numerator=numerator,
                eta_bin=eta_bin,
                decay_mode=decay_mode[m],
                met=met[m],
                weight=weight[m],
            )
            # fill the transverse mass
            output["mT"][name].fill(
                group=group,
                category=category,
                prompt=prompt,
                numerator=numerator,
                eta_bin=eta_bin,
                decay_mode=decay_mode[m],
                mT=mT[m],
                weight=weight[m],
            )

    def apply_lepton_ID_SFs(self, lll, z_pair, mode, is_data=False, numerator=False):
        if len(lll) == 0:
            return lll
        lll, num = ak.flatten(lll), ak.num(lll)
        l1, l2, l = lll.ll.l1, lll.ll.l2, lll.l

        # e/mu scale factors
        if z_pair == "ee":
            l1_w = lepton_ID_weight(l1, "e", self.eleID_SFs, is_data)
            l2_w = lepton_ID_weight(l2, "e", self.eleID_SFs, is_data)
        elif z_pair == "mm":
            l1_w = lepton_ID_weight(l1, "m", self.muID_SFs, is_data)
            l2_w = lepton_ID_weight(l2, "m", self.muID_SFs, is_data)
        w = l1_w * l2_w

        if numerator:  # weight the fake lepton
            if mode == "e":
                l_w = lepton_ID_weight(l, "e", self.eleID_SFs, is_data)
            elif mode == "m":
                l_w = lepton_ID_weight(l, "m", self.muID_SFs, is_data)
            else:  # the taus have different working points
                l_w = tau_ID_weight_3l(l, self.tauID_SFs, mode)
            w = w * l_w

        return ak.unflatten(w, num)

    def postprocess(self, accumulator):
        pass
