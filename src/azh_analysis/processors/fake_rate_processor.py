from __future__ import annotations

import logging
import warnings

import awkward as ak
import numpy as np
from coffea import analysis_tools, processor

# from coffea.processor import column_accumulator as col_acc
from hist import Hist
from hist.axis import IntCategory, Regular, StrCategory

from azh_analysis.selections.fake_rate_selections import (
    additional_cuts,
    dr_3l,
    is_prompt_lepton,
    lepton_count_veto_3l,
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
    tau_ID_weight,
)
from azh_analysis.utils.pileup import get_pileup_weights

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
        pileup_tables=None,
        lumi_masks=None,
        nevts_dict=None,
        eleID_SFs=None,
        muID_SFs=None,
        tauID_SFs=None,
        dyjets_weights=None,
        e_trig_SFs=None,
        m_trig_SFs=None,
        verbose=False,
    ):

        # initialize member variables
        self.init_logging(verbose=verbose)
        self.info = sample_info

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
        self.pileup_tables = pileup_tables
        self.pileup_bins = np.arange(0, 100, 1)
        self.lumi_masks = lumi_masks
        self.nevts_dict = nevts_dict
        self.eleID_SFs = eleID_SFs
        self.muID_SFs = muID_SFs
        self.tauID_SFs = tauID_SFs
        self.e_trig_SFs = e_trig_SFs
        self.m_trig_SFs = m_trig_SFs
        self.dyjets_weights = dyjets_weights

        # bin variables along axes
        category_axis = StrCategory(
            name="category",
            categories=[],
            growth=True,
        )
        prompt_axis = StrCategory(
            name="prompt",
            categories=[],
            growth=True,
        )
        numerator_axis = StrCategory(
            name="numerator",
            categories=[],
            growth=True,
        )
        pt_axis = StrCategory(
            name="pt_bin",
            categories=[],
            growth=True,
        )
        eta_axis = StrCategory(
            name="eta_bin",
            categories=[],
            growth=True,
        )
        decay_mode_axis = IntCategory(
            [-1, 0, 1, 2, 10, 11, 15],
            name="decay_mode",
        )

        pt = {
            dataset.split(f"_{year}")[0]: Hist(
                category_axis,
                prompt_axis,
                numerator_axis,
                decay_mode_axis,
                pt_axis,
                eta_axis,
                Regular(name="pt", bins=30, start=0, stop=300),
            )
            for dataset in fileset.keys()
        }
        met = {
            dataset.split(f"_{year}")[0]: Hist(
                category_axis,
                prompt_axis,
                numerator_axis,
                decay_mode_axis,
                pt_axis,
                eta_axis,
                Regular(name="met", bins=30, start=0, stop=300),
            )
            for dataset in fileset.keys()
        }

        mll = {
            dataset.split(f"_{year}")[0]: Hist(
                category_axis,
                prompt_axis,
                numerator_axis,
                decay_mode_axis,
                pt_axis,
                eta_axis,
                Regular(name="mll", bins=20, start=60, stop=120),
            )
            for dataset in fileset.keys()
        }

        mT = {
            dataset.split(f"_{year}")[0]: Hist(
                category_axis,
                prompt_axis,
                numerator_axis,
                decay_mode_axis,
                pt_axis,
                eta_axis,
                Regular(name="mT", bins=30, start=0, stop=300),
            )
            for dataset in fileset.keys()
        }

        self.output = processor.dict_accumulator(
            {
                "mll": processor.dict_accumulator(mll),
                "pt": processor.dict_accumulator(pt),
                "met": processor.dict_accumulator(met),
                "mT": processor.dict_accumulator(mT),
            }
        )

    def init_logging(self, verbose=False):
        log_format = "%(asctime)s %(levelname)s %(message)s"
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(level=log_level, format=log_format)
        logging.info("Initializing processor logger.")

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
        filter_MET(events, global_selections, year, UL=True, data=is_data)
        filter_PV(events, global_selections)
        global_mask = global_selections.all(*global_selections.names)
        events = events[global_mask]

        # global weights: sample weight, gen weight, pileup weight
        weights = analysis_tools.Weights(len(events), storeIndividual=True)
        ones = np.ones(len(events), dtype=float)
        if group == "DY":
            njets = ak.to_numpy(events.LHE.Njets)
            weights.add("dyjets_sample_weights", self.dyjets_weights(njets))
        else:  # otherwise weight by luminosity ratio
            weights.add("sample_weight", ones * sample_weight)
        if (self.pileup_tables is not None) and not is_data:
            weights.add("gen_weight", events.genWeight)
            pu_weights = get_pileup_weights(
                events.Pileup.nTrueInt, self.pileup_tables[dataset], self.pileup_bins
            )
            weights.add("pileup_weight", pu_weights)
        if is_data:  # golden json weighleting
            lumi_mask = self.lumi_masks[year]
            lumi_mask = lumi_mask(events.run, events.luminosityBlock)
            weights.add("lumi_mask", lumi_mask)

        # grab baseline defined leptons
        baseline_e = get_baseline_electrons(events.Electron)
        baseline_m = get_baseline_muons(events.Muon)
        baseline_t = get_baseline_taus(events.Tau, loose=True)
        baseline_l = {"e": baseline_e, "m": baseline_m, "t": baseline_t}
        tight_e = baseline_e[tight_electrons(baseline_e)]
        tight_m = baseline_m[tight_muons(baseline_m)]
        e_counts = ak.num(tight_e)
        m_counts = ak.num(tight_m)
        MET = events.MET
        MET["pt"] = MET.T1_pt
        MET["phi"] = MET.T1_phi
        MET = apply_unclMET_shifts(MET, "nom")

        # build ll pairs
        for z_pair in ["ee", "mm"]:
            if (z_pair == "ee") and ("_Electrons" not in filename):
                continue
            if (z_pair == "mm") and ("_Muons" not in filename):
                continue

            l = tight_e if (z_pair == "ee") else tight_m
            ll = ak.combinations(l, 2, axis=1, fields=["l1", "l2"])
            ll = dR_ll(ll)
            ll = build_Z_cand(ll)
            ll = closest_to_Z_mass(ll)
            mask, tpt1, teta1, tpt2, teta2 = trigger_filter(ll, events.TrigObj, z_pair)
            mask = mask & check_trigger_path(events.HLT, year, z_pair)
            # ll = suppress_FSR(ll) # fake rate
            ll = ak.fill_none(ll.mask[mask], [], axis=0)

            trig_SFs = self.e_trig_SFs if z_pair == "ee" else self.m_trig_SFs
            if not is_data:
                weight = np.ones(len(events), dtype=float)
                wt1 = lepton_trig_weight(weight, tpt1, teta1, trig_SFs, lep=z_pair[0])
                wt2 = lepton_trig_weight(weight, tpt2, teta2, trig_SFs, lep=z_pair[0])
                weights.add("l1_trig_weight", wt1)
                weights.add("l2_trig_weight", wt2)

            for mode in ["e", "m", "et", "mt", "tt"]:

                # build all viable (Z->ll)+l pairs
                cat = z_pair + mode[-1]
                lll = ak.cartesian({"ll": ll, "l": baseline_l[mode[-1]]}, axis=1)
                lll["weight"] = weights.weight()
                lll["met"] = MET
                lll["cat"] = cat
                lll = dr_3l(lll, cat)

                # count tight taus, create lepton count veto mask
                tight_t = tight_hadronic_taus(baseline_t, mode=mode)
                t_counts = ak.num(tight_t)
                lll_mask = lepton_count_veto_3l(
                    e_counts,
                    m_counts,
                    t_counts,
                    cat,
                )

                # create denominator and numerator regions
                lll_denom = ak.fill_none(lll.mask[lll_mask], [], axis=0)
                if mode == "e":
                    lll_num = lll_denom[tight_electrons(lll_denom["l"])]
                elif mode == "m":
                    lll_num = lll_denom[tight_muons(lll_denom["l"])]
                else:
                    lll_num = lll_denom[tight_hadronic_taus(lll_denom["l"], mode=mode)]
                lll_num = lll_num[~ak.is_none(lll_num, axis=1)]
                lll_denom = additional_cuts(lll_denom, mode)

                # if data, separate denominator and numerator
                if is_data:
                    lll_dict = {
                        ("Denominator", "Data"): lll_denom,
                        ("Numerator", "Data"): lll_num,
                    }
                # if MC, additionally separate prompt and fake
                else:
                    prompt_mask = is_prompt_lepton(lll_denom, mode)
                    lll_denom_fake = lll_denom[~prompt_mask]
                    lll_denom_prompt = lll_denom[prompt_mask]

                    prompt_mask = is_prompt_lepton(lll_num, mode)
                    lll_num_fake = lll_num[~prompt_mask]
                    lll_num_prompt = lll_num[prompt_mask]

                    # add in lepton ID scale factors
                    lll_dict = {
                        ("Denominator", "Fake"): lll_denom_fake,
                        ("Denominator", "Prompt"): lll_denom_prompt,
                        ("Numerator", "Fake"): lll_num_fake,
                        ("Numerator", "Prompt"): lll_num_prompt,
                    }

                # fill hists in each case
                for label, lll in lll_dict.items():
                    lll = lll[ak.num(lll) == 1]
                    if len(lll) == 0:
                        continue

                    # apply relevant scale factors
                    if not is_data:
                        lll["weight"] = lll["weight"] * self.apply_lepton_ID_SFs(
                            lll, z_pair, mode, is_data=False
                        )
                        lll = self.apply_ES_shifts(lll, z_pair, mode)

                    self.fill_histos(
                        lll,
                        z_pair,
                        mode,
                        numerator=label[0],
                        prompt=label[1],
                        dataset=dataset,
                        name=name,
                        group=group,
                    )

        return self.output

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
        print(l1.mass, l2.mass)
        pt, eta = np_flat(l.pt), np_flat(l.eta)
        decay_mode = -1 * np.ones_like(weight)
        met = np_flat(lll.met.pt)
        mT = np_flat(transverse_mass(l, lll.met))
        mll = np_flat((l1 + l2).mass)
        # print(category, lll.l.fields)
        if len(mode) > 1:
            decay_mode = np_flat(l.decayMode)
        # print(mode, decay_mode)
        for pt_range in [(10, 20), (20, 30), (30, 40), (40, 60), (60, 10**6)]:
            pt_bin = f"${pt_range[0]}<p_T<{pt_range[1]}$ GeV"
            eta_barrel_bin = r"$|\eta|<1.479$"
            eta_endcap_bin = r"$|\eta|>1.479$"
            pt_mask = (pt > pt_range[0]) & (pt <= pt_range[1])
            barrel_mask = (abs(eta) < 1.479) & pt_mask
            endcap_mask = (abs(eta) > 1.479) & pt_mask
            for eta_bin, m in [
                (eta_barrel_bin, barrel_mask),
                (eta_endcap_bin, endcap_mask),
            ]:
                if np.sum(m) == 0:
                    continue

                # fill pt
                self.output["pt"][name].fill(
                    category=category,
                    prompt=prompt,
                    numerator=numerator,
                    pt_bin=pt_bin,
                    eta_bin=eta_bin,
                    pt=pt[m],
                    decay_mode=decay_mode[m],
                    weight=weight[m],
                )
                # fill the mass of the dilepton system w/ various systematic shifts
                self.output["mll"][name].fill(
                    category=category,
                    prompt=prompt,
                    numerator=numerator,
                    pt_bin=pt_bin,
                    eta_bin=eta_bin,
                    decay_mode=decay_mode[m],
                    mll=mll[m],
                    weight=weight[m],
                )
                # fill the met with various systematics considered
                self.output["met"][name].fill(
                    category=category,
                    prompt=prompt,
                    numerator=numerator,
                    pt_bin=pt_bin,
                    eta_bin=eta_bin,
                    decay_mode=decay_mode[m],
                    met=met[m],
                    weight=weight[m],
                )
                # fill the transverse mass
                self.output["mT"][name].fill(
                    category=category,
                    prompt=prompt,
                    numerator=numerator,
                    pt_bin=pt_bin,
                    eta_bin=eta_bin,
                    decay_mode=decay_mode[m],
                    mT=mT[m],
                    weight=weight[m],
                )

    def apply_lepton_ID_SFs(self, lll, z_pair, mode, is_data=False):
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

        # also consider hadronic taus
        if mode == "e":
            l_w = lepton_ID_weight(l, "e", self.eleID_SFs, is_data)
        elif mode == "m":
            l_w = lepton_ID_weight(l, "m", self.muID_SFs, is_data)
        else:
            # print("apply_lepton", z_pair, mode)
            l_w = tau_ID_weight(l, self.tauID_SFs, z_pair + mode)

        # apply ID scale factors
        w = l1_w * l2_w * l_w
        return ak.unflatten(w, num)

    def postprocess(self, accumulator):
        pass
