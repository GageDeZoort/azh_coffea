from __future__ import annotations

import logging
import math
import warnings

import awkward as ak
import numba as nb
import numpy as np
import vector as vec
from coffea import analysis_tools, processor

# from coffea.processor import column_accumulator as col_acc
from hist import Hist
from hist.axis import IntCategory, Regular, StrCategory

from azh_analysis.selections.preselections import (
    build_Z_cand,
    check_trigger_path,
    closest_to_Z_mass,
    dR_ll,
    dR_lltt,
    filter_MET,
    filter_PV,
    get_baseline_bjets,
    get_baseline_electrons,
    get_baseline_jets,
    get_baseline_muons,
    get_baseline_taus,
    get_tight_masks,
    get_tt,
    highest_LT,
    is_prompt,
    lepton_count_veto,
    tight_electrons,
    tight_muons,
    trigger_filter,
)
from azh_analysis.utils.btag import get_btag_effs
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


class AnalysisProcessor(processor.ProcessorABC):
    def __init__(
        self,
        sync=False,
        categories="all",
        collection_vars=None,
        global_vars=None,
        sample_info=None,
        fileset=None,
        sample_dir="../sample_lists/sample_yamls",
        exc1_path="sync/princeton_all.csv",
        exc2_path="sync/desy_all.csv",
        pileup_tables=None,
        lumi_masks=None,
        blind=True,
        nevts_dict=None,
        fake_rates=None,
        eleID_SFs=None,
        muID_SFs=None,
        tauID_SFs=None,
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
    ):

        # initialize member variables
        self.init_logging(verbose=verbose)
        self.sync = sync
        self.info = sample_info
        self.collection_vars = collection_vars
        self.global_vars = global_vars
        self.blind = blind
        if categories == "all":
            self.categories = {
                1: "eeet",
                2: "eemt",
                3: "eett",
                4: "eeem",
                5: "mmet",
                6: "mmmt",
                7: "mmtt",
                8: "mmem",
            }
            self.num_to_cat = {
                **self.categories,
                9: "eeet_fake",
                10: "eemt_fake",
                11: "eett_fake",
                12: "eeem_fake",
                13: "mmet_fake",
                14: "mmmt_fake",
                15: "mmtt_fake",
                16: "mmem_fake",
            }
            self.cat_to_num = {v: k for k, v in self.num_to_cat.items()}
        else:
            self.categories = {i: cat for i, cat in enumerate(categories)}

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
        self.fake_rates = fake_rates
        self.eleID_SFs = eleID_SFs
        self.muID_SFs = muID_SFs
        self.tauID_SFs = tauID_SFs
        self.e_trig_SFs = e_trig_SFs
        self.m_trig_SFs = m_trig_SFs
        self.btag_SFs = btag_SFs
        self.btag_eff_tables = btag_eff_tables
        self.btag_pt_bins = btag_pt_bins
        self.btag_eta_bins = btag_eta_bins
        self.dyjets_weights = dyjets_weights
        self.fastmtt = run_fastmtt
        self.fill_hists = fill_hists

        # cache for the energy scales
        self._eleES_cache = {}
        self._muES_cache = {}
        self._tauES_cache = {}

        # bin variables along axes
        category_axis = StrCategory(
            name="category",
            categories=[],
            growth=True,
        )
        leg_axis = StrCategory(
            name="leg",
            categories=[],
            growth=True,
        )
        btags_axis = IntCategory(
            name="btags",
            categories=[],
            growth=True,
        )
        mass_type_axis = StrCategory(
            name="mass_type",
            categories=[],
            growth=True,
        )
        unclMET_shift_axis = StrCategory(
            name="unclMET_shift",
            categories=[],
            growth=True,
        )
        eleES_shift_axis = StrCategory(
            name="eleES_shift",
            categories=[],
            growth=True,
        )
        muES_shift_axis = StrCategory(
            name="muES_shift",
            categories=[],
            growth=True,
        )
        tauES_shift_axis = StrCategory(
            name="tauES_shift",
            categories=[],
            growth=True,
        )
        efake_shift_axis = StrCategory(
            name="efake_shift",
            categories=[],
            growth=True,
        )
        mfake_shift_axis = StrCategory(
            name="mfake_shift",
            categories=[],
            growth=True,
        )
        eleSmear_shift_axis = StrCategory(
            name="eleSmear_shift",
            categories=[],
            growth=True,
        )

        pt = {
            dataset.split("_")[0]: Hist(
                category_axis,
                leg_axis,
                btags_axis,
                unclMET_shift_axis,
                eleES_shift_axis,
                muES_shift_axis,
                tauES_shift_axis,
                efake_shift_axis,
                mfake_shift_axis,
                eleSmear_shift_axis,
                Regular(name="pt", bins=30, start=0, stop=300),
            )
            for dataset in fileset.keys()
        }
        met = {
            dataset.split("_")[0]: Hist(
                category_axis,
                btags_axis,
                unclMET_shift_axis,
                eleES_shift_axis,
                muES_shift_axis,
                tauES_shift_axis,
                efake_shift_axis,
                mfake_shift_axis,
                eleSmear_shift_axis,
                Regular(name="met", bins=10, start=0, stop=200),
            )
            for dataset in fileset.keys()
        }
        mtt = {
            dataset.split("_")[0]: Hist(
                category_axis,
                mass_type_axis,
                btags_axis,
                unclMET_shift_axis,
                eleES_shift_axis,
                muES_shift_axis,
                tauES_shift_axis,
                efake_shift_axis,
                mfake_shift_axis,
                eleSmear_shift_axis,
                Regular(name="mass", bins=40, start=0, stop=400),
            )
            for dataset in fileset.keys()
        }
        m4l = {
            dataset.split("_")[0]: Hist(
                category_axis,
                mass_type_axis,
                btags_axis,
                unclMET_shift_axis,
                eleES_shift_axis,
                muES_shift_axis,
                tauES_shift_axis,
                efake_shift_axis,
                mfake_shift_axis,
                eleSmear_shift_axis,
                Regular(name="mass", bins=40, start=0, stop=400),
            )
            for dataset in fileset.keys()
        }
        mll = {
            dataset.split("_")[0]: Hist(
                category_axis,
                btags_axis,
                unclMET_shift_axis,
                eleES_shift_axis,
                muES_shift_axis,
                tauES_shift_axis,
                efake_shift_axis,
                mfake_shift_axis,
                eleSmear_shift_axis,
                Regular(name="mll", bins=10, start=60, stop=120),
            )
            for dataset in fileset.keys()
        }

        self.output = processor.dict_accumulator(
            {
                "mtt": processor.dict_accumulator(mtt),
                "m4l": processor.dict_accumulator(m4l),
                "mll": processor.dict_accumulator(mll),
                "pt": processor.dict_accumulator(pt),
                "met": processor.dict_accumulator(met),
            }
        )

    def init_logging(self, verbose=False):
        log_format = "%(asctime)s %(levelname)s %(message)s"
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(level=log_level, format=log_format)
        logging.info("Initializing processor logger.")

    def clear_caches(self):
        self._eleES_cache.clear()
        self._muES_cache.clear()
        self._tauES_cache.clear()

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
        baseline_t = get_baseline_taus(events.Tau)
        baseline_j = get_baseline_jets(events.Jet)
        baseline_b = get_baseline_bjets(baseline_j)
        MET = events.MET
        MET["pt"] = MET.T1_pt
        MET["phi"] = MET.T1_phi

        # seeds the lepton count veto
        e_counts = ak.num(baseline_e[tight_electrons(baseline_e)])
        m_counts = ak.num(baseline_m[tight_muons(baseline_m)])

        # number of b jets used to test bbA vs. ggA
        b_counts = ak.num(baseline_b)

        # build ll pairs
        ll_pairs = {}
        for cat in ["ee", "mm"]:
            if (cat[:2] == "ee") and ("_Electrons" not in filename):
                continue
            if (cat[:2] == "mm") and ("_Muons" not in filename):
                continue

            l = baseline_e if (cat == "ee") else baseline_m
            ll = ak.combinations(l, 2, axis=1, fields=["l1", "l2"])
            ll = dR_ll(ll)
            ll = build_Z_cand(ll)
            ll = closest_to_Z_mass(ll)
            mask, tpt1, teta1, tpt2, teta2 = trigger_filter(ll, events.TrigObj, cat)
            mask = mask & check_trigger_path(events.HLT, year, cat)
            ll = ak.fill_none(ll.mask[mask], [], axis=0)
            ll_pairs[cat] = ll

            trig_SFs = self.e_trig_SFs if cat == "ee" else self.m_trig_SFs
            if not is_data:
                weight = np.ones(len(events), dtype=float)
                wt1 = lepton_trig_weight(weight, tpt1, teta1, trig_SFs, lep=cat[0])
                wt2 = lepton_trig_weight(weight, tpt2, teta2, trig_SFs, lep=cat[0])
                weights.add("l1_trig_weight", wt1)
                weights.add("l2_trig_weight", wt2)

        candidates, cat_tight_masks = {}, {}
        for cat in self.categories.values():
            if (cat[:2] == "ee") and ("_Electrons" not in filename):
                continue
            if (cat[:2] == "mm") and ("_Muons" not in filename):
                continue
            mask = lepton_count_veto(e_counts, m_counts, cat)

            # build 4l final state
            tt = get_tt(baseline_e, baseline_m, baseline_t, cat)
            lltt = ak.cartesian({"ll": ll_pairs[cat[:2]], "tt": tt}, axis=1)
            lltt = dR_lltt(lltt, cat)
            # lltt = build_ditau_cand(lltt, cat, self.cutflow, OS=True)
            lltt = highest_LT(lltt)
            lltt = ak.fill_none(lltt.mask[mask], [], axis=0)

            # determine which legs passed tight selections
            tight_masks = get_tight_masks(lltt, cat)
            l1_tight_mask, l2_tight_mask = tight_masks[0], tight_masks[1]
            t1_tight_mask, t2_tight_mask = tight_masks[2], tight_masks[3]
            tight_mask = l1_tight_mask & l2_tight_mask & t1_tight_mask & t2_tight_mask

            # apply fake weights to estimate reducible background
            if is_data:
                cat_tight_masks[cat + "_fakes"] = tight_masks
                fakes, cands = lltt[~tight_mask], lltt[tight_mask]
                fakes["category"] = self.cat_to_num[cat + "_fakes"]
                cands["category"] = self.cat_to_num[cat]
                candidates[cat + "_fakes"] = fakes
                candidates[cat] = cands
            else:
                prompt_mask = is_prompt(lltt, cat)
                cands = lltt[prompt_mask & tight_mask]
                cands["category"] = self.cat_to_num[cat]
                cands["cat_str"] = cat
                cands["w_lepton_ID"] = self.apply_lepton_ID_SFs(cands, cat)
                candidates[cat] = cands

        # combine the candidates into a single array
        # reject events with multiple candidates from one category
        cands = ak.concatenate(list(candidates.values()), axis=1)
        counts_mask = ak.num(cands) == 1
        sign_mask = ak.flatten(
            (cands[counts_mask].tt.t1.charge * cands[counts_mask].tt.t2.charge) < 0
        )
        lltt = {}
        btags = (b_counts)[counts_mask][sign_mask]
        lltt = {cat: cands[counts_mask][sign_mask] for cat, cands in candidates.items()}
        met = MET[counts_mask][sign_mask]
        jets = baseline_j[counts_mask][sign_mask]
        weight = weights.weight()[counts_mask][sign_mask]

        if not is_data and len(np_flat(met)) > 0:
            # if MC, need to apply the systematic shifts
            shifts = [
                "nom",
                "tauES_down",
                "tauES_up",
                "efake_down",
                "efake_up",
                "mfake_down",
                "mfake_up",
                "eleES_down",
                "eleES_up",
                "muES_down",
                "muES_up",
                "eleSmear_up",
                "eleSmear_down",
                "unclMET_up",
                "unclMET_down",
                "btag_up_correlated",
                "btag_down_correlated",
                "btag_up_uncorrelated",
                "btag_down_uncorrelated",
            ]

            # apply some corrections up front:
            bshift_labels = [
                "central",
                "up_correlated",
                "down_correlated",
                "up_uncorrelated",
                "down_uncorrelated",
            ]
            btag_shifts = {
                bshift: self.apply_btag_corrections(jets, dataset, bshift)
                for bshift in bshift_labels
            }
            met_shifts = {
                "nom": apply_unclMET_shifts(met, "nom"),
                "up": apply_unclMET_shifts(met, "up"),
                "down": apply_unclMET_shifts(met, "down"),
            }

            for shift in shifts:
                # figure out which variable needs to be shifted
                up_or_down = shift.split("_")[-1]
                tauES_shift = up_or_down if ("tauES" in shift) else "nom"
                efake_shift = up_or_down if ("efake" in shift) else "nom"
                mfake_shift = up_or_down if ("mfake" in shift) else "nom"
                eleES_shift = up_or_down if ("eleES" in shift) else "nom"
                muES_shift = up_or_down if ("muES" in shift) else "nom"
                eleSmear_shift = up_or_down if ("eleSmear" in shift) else "nom"
                unclMET_shift = up_or_down if ("unclMET" in shift) else "nom"
                btag_shift = "central"
                if "btag" in shift:
                    btag_shift = shift[5:]

                # apply bjet weights and unclustered energy corrections
                w = weight * btag_shifts[btag_shift]
                met = met_shifts[unclMET_shift]
                for cat, cands in lltt.items():
                    cands["w"] = w * cands["w_lepton_ID"]
                    cands["btags"] = btags
                    cands["met"] = met
                    lltt[cat] = cands

                # apply lepton energy scale corrections
                final_states = self.apply_ES_shifts(
                    lltt,
                    eleES_shift,
                    muES_shift,
                    tauES_shift,
                    efake_shift,
                    mfake_shift,
                    eleSmear_shift,
                )

                # optionally run fastmtt
                if self.fastmtt:
                    fastmtt_out = self.run_fastmtt(final_states)
                else:
                    fastmtt_out = {}

                self.fill_histos(
                    final_states,
                    fastmtt_out,
                    group=group,
                    dataset=dataset,
                    name=name,
                    tauES_shift=tauES_shift,
                    efake_shift=efake_shift,
                    mfake_shift=mfake_shift,
                    muES_shift=muES_shift,
                    eleES_shift=eleES_shift,
                    eleSmear_shift=eleSmear_shift,
                    unclMET_shift=unclMET_shift,
                    btag_shift=btag_shift,
                    blind=(is_data and self.blind),
                )

        self.clear_caches()

        # if 'fake' in cat:
        #    tight_masks = cat_tight_masks[cat]
        #    m0 = tight_masks[0][final_mask][smask]
        #    m1 = tight_masks[1][final_mask][smask]
        #    m2 = tight_masks[2][final_mask][smask]
        #    m3 = tight_masks[3][final_mask][smask]
        #    weight = self.get_fake_weights(lltt, cat.split('_')[0],
        # [m0, m1, m2, m3])

        # if data, do a simple histogram fill
        # if is_data:
        # l1, l2 = ak.flatten(lltt['ll']['l1']), ak.flatten(lltt['ll']['l2'])
        # t1, t2 = ak.flatten(lltt['tt']['t1']), ak.flatten(lltt['tt']['t2'])
        # if run_fastmtt:
        # fastmtt_out = self.run_fastmtt(cat, l1, l2, t1, t2, met)
        #                else: fastmtt_out = {}
        #               group_label = 'reducible' if ('fake' in cat) else group
        #               if self.fill_hists:
        #                   self.fill_histos(lltt, fastmtt_out, met,
        #                                    group=group_label, dataset=dataset,
        #                                    category=cat, bjets=bjet_label, sign=sign,
        #                                    weight=weight, tauES_shift='none',
        #                                    efake_shift='none', mfake_shift='none',
        #                                    muES_shift='none', eleES_shift='none',
        #                                    eleSmear_shift='none', unclMET_shift='none',
        #                                    blind=self.blind)
        #               continue

        return self.output

    def apply_btag_corrections(self, jets, dataset, systematic):
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
        SFs = self.btag_SFs.evaluate(
            systematic, "M", 5, abs(ak.to_numpy(eta)), ak.to_numpy(pt)
        )
        btag_effs = np.array(
            get_btag_effs(
                self.btag_eff_tables,
                self.btag_pt_bins,
                self.btag_eta_bins,
                dataset,
                pt,
                abs(eta),
            )
        )
        w_is_tagged = is_tagged * btag_effs
        w_not_tagged = (1 - btag_effs) * ~is_tagged
        w_MC = w_is_tagged + w_not_tagged
        w_is_tagged = btag_effs * is_tagged * SFs
        w_is_not_tagged = (1 - btag_effs * SFs) * ~is_tagged
        w = (w_is_tagged + w_is_not_tagged) / w_MC
        return ak.prod(ak.unflatten(w, num_j), axis=1)

    def query_eleES_shifts(self, leg, l, cat, eleES_shift, eleSmear_shift):
        key = (l, cat, eleES_shift, eleSmear_shift)
        val = self._eleES_cache.get(key, None)
        if val is None:
            leg, diffs = apply_eleES(leg, eleES_shift, eleSmear_shift)
            self._eleES_cache[key] = (leg, diffs)
            return leg, diffs
        return val[0], val[1]

    def query_muES_shifts(self, leg, l, cat, muES_shift):
        key = (l, cat, muES_shift)
        val = self._muES_cache.get(key, None)
        if val is None:
            leg, diffs = apply_muES(leg, muES_shift)
            self._muES_cache[key] = (leg, diffs)
            return leg, diffs
        return val[0], val[1]

    def query_tauES_shifts(self, leg, l, cat, tauES_shift, efake_shift, mfake_shift):
        key = (l, cat, tauES_shift, efake_shift, mfake_shift)
        val = self._tauES_cache.get(key, None)
        if val is None:
            leg, diffs = apply_tauES(
                leg, self.tauID_SFs, tauES_shift, efake_shift, mfake_shift
            )
            self._tauES_cache[key] = (leg, diffs)
            return leg, diffs
        return val[0], val[1]

    def apply_ES_shifts(
        self,
        lltt,
        eleES_shift=-1,
        muES_shift=-1,
        tauES_shift=-1,
        efake_shift=-1,
        mfake_shift=-1,
        eleSmear_shift=-1,
    ):

        lltt_out = {}
        for cat, lltt_cat in lltt.items():
            lltt_cat, num = ak.flatten(lltt_cat), ak.num(lltt_cat)
            l1, l2 = lltt_cat["ll"]["l1"], lltt_cat["ll"]["l2"]
            t1, t2 = lltt_cat["tt"]["t1"], lltt_cat["tt"]["t2"]
            met = lltt_cat["met"]
            if len(lltt_cat) == 0:
                continue
            diffs_list = []
            if cat[:2] == "ee":
                l1, diffs = self.query_eleES_shifts(
                    l1, "1", cat, eleES_shift, eleSmear_shift
                )
                diffs_list.append(diffs)
                l2, diffs = self.query_eleES_shifts(
                    l2, "2", cat, eleES_shift, eleSmear_shift
                )
                diffs_list.append(diffs)
            if cat[:2] == "mm":
                l1, diffs = self.query_muES_shifts(l1, "1", cat, muES_shift)
                diffs_list.append(diffs)
                l2, diffs = self.query_muES_shifts(l2, "2", cat, muES_shift)
                diffs_list.append(diffs)
            if cat[2] == "e":
                t1, diffs = self.query_eleES_shifts(
                    t1, "3", cat, eleES_shift, eleSmear_shift
                )
                diffs_list.append(diffs)
            if cat[2] == "m":
                t1, diffs = self.query_muES_shifts(t1, "3", cat, muES_shift)
                diffs_list.append(diffs)
            if cat[2] == "t":
                t1, diffs = self.query_tauES_shifts(
                    t1,
                    "3",
                    cat,
                    tauES_shift=tauES_shift,
                    efake_shift=efake_shift,
                    mfake_shift=mfake_shift,
                )
                diffs_list.append(diffs)
            if cat[3] == "m":
                t2, diffs = self.query_muES_shifts(t2, "4", cat, muES_shift)
                diffs_list.append(diffs)
            if cat[3] == "t":
                t2, diffs = self.query_tauES_shifts(
                    t2,
                    "4",
                    cat,
                    tauES_shift=tauES_shift,
                    efake_shift=efake_shift,
                    mfake_shift=mfake_shift,
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

            lltt_cat["ll"]["l1"] = l1
            lltt_cat["ll"]["l2"] = l2
            lltt_cat["tt"]["t1"] = t1
            lltt_cat["tt"]["t2"] = t2
            lltt_cat["met"] = met
            lltt_cat = ak.unflatten(lltt_cat, num)
            lltt_out[cat] = lltt_cat

        return ak.concatenate(list(lltt_out.values()), axis=1)

    def get_fake_weights(self, lltt, cat, tight_masks):
        l1_tight_mask, l2_tight_mask = tight_masks[0], tight_masks[1]
        t1_tight_mask, t2_tight_mask = tight_masks[2], tight_masks[3]
        l1_tight_mask = ak.flatten(l1_tight_mask)
        l2_tight_mask = ak.flatten(l2_tight_mask)
        t1_tight_mask = ak.flatten(t1_tight_mask)
        t2_tight_mask = ak.flatten(t2_tight_mask)
        l1_barrel = ak.flatten(abs(lltt["ll"]["l1"].eta) < 1.479)
        l2_barrel = ak.flatten(abs(lltt["ll"]["l2"].eta) < 1.479)
        t1_barrel = ak.flatten(abs(lltt["tt"]["t1"].eta) < 1.479)
        t2_barrel = ak.flatten(abs(lltt["tt"]["t2"].eta) < 1.479)
        l1l2_fr_barrel = self.fake_rates[cat[:2]]["barrel"]
        l1l2_fr_endcap = self.fake_rates[cat[:2]]["endcap"]

        # fake rate regions
        l1_fake_barrel = l1_barrel & ~l1_tight_mask
        l1_fake_endcap = ~l1_barrel & ~l1_tight_mask
        l2_fake_barrel = l2_barrel & ~l2_tight_mask
        l2_fake_endcap = ~l2_barrel & ~l2_tight_mask
        l1_pt = ak.flatten(lltt["ll"]["l1"].pt)
        l2_pt = ak.flatten(lltt["ll"]["l2"].pt)

        # l1 fake rates: barrel+fake, endcap+fake, or tight
        fr1_barrel = l1l2_fr_barrel(l1_pt)
        fr1_endcap = l1l2_fr_endcap(l1_pt)
        fr1 = (
            (fr1_barrel * l1_fake_barrel)
            + (fr1_endcap * l1_fake_endcap)
            + (np.ones(len(l1_pt)) * l1_tight_mask)
        )

        # l2 fake rates: barrel+fake, endcap+fake, or tight
        fr2_barrel = l1l2_fr_barrel(l2_pt)
        fr2_endcap = l1l2_fr_endcap(l2_pt)
        fr2 = (
            (fr2_barrel * l2_fake_barrel)
            + (fr2_endcap * l2_fake_endcap)
            + (np.ones(len(l2_pt)) * l2_tight_mask)
        )

        # t1 and t2 depend on the type of tau decay being considered
        t1_fake_barrel = t1_barrel & ~t1_tight_mask
        t1_fake_endcap = ~t1_barrel & ~t1_tight_mask
        t2_fake_barrel = t2_barrel & ~t2_tight_mask
        t2_fake_endcap = ~t2_barrel & ~t2_tight_mask
        t1_pt = ak.flatten(lltt["tt"]["t1"].pt)
        t2_pt = ak.flatten(lltt["tt"]["t2"].pt)

        # leptonic decays are easy to handle
        if (cat[2] == "e") or (cat[2] == "m"):
            ll_str = "ee" if cat[2] == "e" else "mm"
            t1_fr_barrel = self.fake_rates[ll_str]["barrel"]
            t1_fr_endcap = self.fake_rates[ll_str]["endcap"]
            fr3_barrel = t1_fr_barrel(t1_pt)
            fr3_endcap = t1_fr_endcap(t1_pt)
            fr3 = (
                (fr3_barrel * t1_fake_barrel)
                + (fr3_endcap * t1_fake_endcap)
                + (np.ones(len(t1_pt)) * t1_tight_mask)
            )

        # hadronic tau decays are not so easy
        elif cat[2] == "t":
            t1_fr_barrel = self.fake_rates["tt"]["barrel"]
            t1_fr_endcap = self.fake_rates["tt"]["endcap"]
            fr3 = np.ones(len(t1_pt)) * t1_tight_mask
            for dm in [0, 1, 10, 11]:
                t1_dm = ak.flatten(lltt["tt"]["t1"].decayMode == dm)
                fr3_barrel = t1_fr_barrel[dm](t1_pt)
                fr3_endcap = t1_fr_endcap[dm](t1_pt)
                t1_fake_barrel_dm = t1_fake_barrel & t1_dm
                t1_fake_endcap_dm = t1_fake_endcap & t1_dm
                fr3 = fr3 + (
                    (fr3_barrel * t1_fake_barrel_dm) + (fr3_endcap * t1_fake_endcap_dm)
                )

        # ditto for the second di-tau leg
        if cat[3] == "m":
            t2_fr_barrel = self.fake_rates["mm"]["barrel"]
            t2_fr_endcap = self.fake_rates["mm"]["endcap"]
            fr4_barrel = t2_fr_barrel(t2_pt)
            fr4_endcap = t2_fr_endcap(t2_pt)
            fr4 = (
                (fr4_barrel * t2_fake_barrel)
                + (fr4_endcap * t2_fake_endcap)
                + (np.ones(len(t2_pt)) * t2_tight_mask)
            )

        elif cat[3] == "t":
            t2_fr_barrel = self.fake_rates[cat[2:]]["barrel"]
            t2_fr_endcap = self.fake_rates[cat[2:]]["endcap"]
            fr4 = np.ones(len(t2_pt)) * t2_tight_mask
            for dm in [0, 1, 10, 11]:
                t2_dm = ak.flatten(lltt["tt"]["t2"].decayMode == dm)
                fr4_barrel = t2_fr_barrel[dm](t2_pt)
                fr4_endcap = t2_fr_endcap[dm](t2_pt)
                t2_fake_barrel_dm = t2_fake_barrel & t2_dm
                t2_fake_endcap_dm = t2_fake_endcap & t2_dm
                fr4 = fr4 + (
                    (fr4_barrel * t2_fake_barrel_dm) + (fr4_endcap * t2_fake_endcap_dm)
                )

        fw1 = ak.nan_to_num(fr1 / (1 - fr1), nan=0, posinf=0, neginf=0)
        fw2 = ak.nan_to_num(fr2 / (1 - fr2), nan=0, posinf=0, neginf=0)
        fw3 = ak.nan_to_num(fr3 / (1 - fr3), nan=0, posinf=0, neginf=0)
        fw4 = ak.nan_to_num(fr4 / (1 - fr4), nan=0, posinf=0, neginf=0)

        apply_w1 = (
            ((~l1_tight_mask & l2_tight_mask & t1_tight_mask & t2_tight_mask) * fw1)
            + ((l1_tight_mask & ~l2_tight_mask & t1_tight_mask & t2_tight_mask) * fw2)
            + ((l1_tight_mask & l2_tight_mask & ~t1_tight_mask & t2_tight_mask) * fw3)
            + ((l1_tight_mask & l2_tight_mask & t1_tight_mask & ~t2_tight_mask) * fw4)
        )
        apply_w2 = (
            (
                (~l1_tight_mask & ~l2_tight_mask & t1_tight_mask & t2_tight_mask)
                * fw1
                * fw2
            )
            + (
                (~l1_tight_mask & l2_tight_mask & ~t1_tight_mask & t2_tight_mask)
                * fw1
                * fw3
            )
            + (
                (~l1_tight_mask & l2_tight_mask & t1_tight_mask & ~t2_tight_mask)
                * fw1
                * fw4
            )
            + (
                (l1_tight_mask & ~l2_tight_mask & ~t1_tight_mask & t2_tight_mask)
                * fw2
                * fw3
            )
            + (
                (l1_tight_mask & ~l2_tight_mask & t1_tight_mask & ~t2_tight_mask)
                * fw2
                * fw4
            )
            + (
                (l1_tight_mask & l2_tight_mask & ~t2_tight_mask & ~t2_tight_mask)
                * fw3
                * fw4
            )
        )
        apply_w3 = (
            (
                (~l1_tight_mask & ~l2_tight_mask & ~t1_tight_mask & t2_tight_mask)
                * fw1
                * fw2
                * fw3
            )
            + (
                (~l1_tight_mask & ~l2_tight_mask & t1_tight_mask & ~t2_tight_mask)
                * fw1
                * fw2
                * fw4
            )
            + (
                (~l1_tight_mask & l2_tight_mask & ~t1_tight_mask & ~t2_tight_mask)
                * fw1
                * fw3
                * fw4
            )
            + (
                (l1_tight_mask & ~l2_tight_mask & ~t1_tight_mask & ~t2_tight_mask)
                * fw2
                * fw3
                * fw4
            )
        )
        apply_w4 = (
            (~l1_tight_mask & ~l2_tight_mask & ~t1_tight_mask & ~t2_tight_mask)
            * fw1
            * fw2
            * fw3
            * fw4
        )
        return apply_w1 - apply_w2 + apply_w3 - apply_w4

    def fill_histos(
        self,
        lltt,
        fastmtt_out,
        dataset,
        name,
        group,
        tauES_shift,
        efake_shift,
        mfake_shift,
        muES_shift,
        eleES_shift,
        eleSmear_shift,
        unclMET_shift,
        btag_shift,
        blind=False,
    ):

        # fill the four-vectors
        label_dict = {
            ("ll", "l1"): "1",
            ("ll", "l2"): "2",
            ("tt", "t1"): "3",
            ("tt", "t2"): "4",
        }
        btags = np_flat(lltt.btags)
        cats = np_flat(lltt.category)
        cats = np.array([self.num_to_cat[c] for c in cats])
        weight = np_flat(lltt.w)

        # fill the lltt leg four-vectors
        for leg, label in label_dict.items():
            p4 = lltt[leg[0]][leg[1]]
            self.output["pt"][name].fill(
                category=cats,
                leg=label,
                btags=btags,
                unclMET_shift=unclMET_shift,
                eleES_shift=eleES_shift,
                muES_shift=muES_shift,
                tauES_shift=tauES_shift,
                efake_shift=efake_shift,
                mfake_shift=mfake_shift,
                eleSmear_shift=eleSmear_shift,
                pt=np_flat(p4.pt),
                weight=weight,
            )

        # fill the mass of the dilepton system w/ various systematic shifts
        mll = np_flat((lltt["ll"]["l1"] + lltt["ll"]["l2"]).mass)
        self.output["mll"][name].fill(
            category=cats,
            btags=btags,
            unclMET_shift=unclMET_shift,
            eleES_shift=eleES_shift,
            muES_shift=muES_shift,
            tauES_shift=tauES_shift,
            efake_shift=efake_shift,
            mfake_shift=mfake_shift,
            eleSmear_shift=eleSmear_shift,
            mll=mll,
            weight=weight,
        )
        # fill the met with various systematics considered
        met = np_flat(lltt.met.pt)
        self.output["met"][name].fill(
            category=cats,
            btags=btags,
            unclMET_shift=unclMET_shift,
            eleES_shift=eleES_shift,
            muES_shift=muES_shift,
            tauES_shift=tauES_shift,
            efake_shift=efake_shift,
            mfake_shift=mfake_shift,
            eleSmear_shift=eleSmear_shift,
            met=met,
            weight=weight,
        )

        # fill the Zh->lltt candidate mass spectrum (raw, uncorrected)
        mtt = np_flat((lltt["tt"]["t1"] + lltt["tt"]["t2"]).mass)
        m4l = np_flat(
            (
                lltt["ll"]["l1"]
                + lltt["ll"]["l2"]
                + lltt["tt"]["t1"]
                + lltt["tt"]["t2"]
            ).mass
        )

        blind_mask = np.zeros(len(m4l), dtype=bool)
        if blind:
            blind_mask = (mtt > 40) & (mtt < 120)
        self.output["m4l"][name].fill(
            category=cats,
            mass_type="raw",
            btags=btags,
            unclMET_shift=unclMET_shift,
            eleES_shift=eleES_shift,
            muES_shift=muES_shift,
            tauES_shift=tauES_shift,
            efake_shift=efake_shift,
            mfake_shift=mfake_shift,
            eleSmear_shift=eleSmear_shift,
            mass=m4l[~blind_mask],
            weight=weight[~blind_mask],
        )

        # fill the Zh->lltt candidate mass spectrums (corrected, constrained)
        for mass_label, mass_data in fastmtt_out.items():
            key = mass_label.split("_")[0]  # mtt or m4l
            mass_type = mass_label.split("_")[1]  # corr or cons
            self.output[key][name].fill(
                category=cats,
                mass_type=mass_type,
                btags=btags,
                unclMET_shift=unclMET_shift,
                eleES_shift=eleES_shift,
                muES_shift=muES_shift,
                tauES_shift=tauES_shift,
                efake_shift=efake_shift,
                mfake_shift=mfake_shift,
                eleSmear_shift=eleSmear_shift,
                mass=mass_data[~blind_mask],
                weight=weight[~blind_mask],
            )

    def apply_lepton_ID_SFs(self, lltt_all, cat, is_data=False):
        lltt, num = ak.flatten(lltt_all), ak.num(lltt_all)
        l1, l2 = lltt["ll"]["l1"], lltt["ll"]["l2"]
        t1, t2 = lltt["tt"]["t1"], lltt["tt"]["t2"]

        # tau_ID_weight(taus, SF_tool, cat, is_data=False, syst='nom', tight=True)

        # e/mu scale factors
        if cat[:2] == "ee":
            l1_w = lepton_ID_weight(l1, "e", self.eleID_SFs, is_data)
            l2_w = lepton_ID_weight(l2, "e", self.eleID_SFs, is_data)
        elif cat[:2] == "mm":
            l1_w = lepton_ID_weight(l1, "m", self.muID_SFs, is_data)
            l2_w = lepton_ID_weight(l2, "m", self.muID_SFs, is_data)

        # also consider hadronic taus
        if cat[2:] == "em":
            t1_w = lepton_ID_weight(t1, "e", self.eleID_SFs, is_data)
            t2_w = lepton_ID_weight(t2, "m", self.muID_SFs, is_data)
        elif cat[2:] == "et":
            t1_w = lepton_ID_weight(t1, "e", self.eleID_SFs, is_data)
            t2_w = tau_ID_weight(t2, self.tauID_SFs, cat)
        elif cat[2:] == "mt":
            t1_w = lepton_ID_weight(t1, "m", self.muID_SFs, is_data)
            t2_w = tau_ID_weight(t2, self.tauID_SFs, cat)
        elif cat[2:] == "tt":
            t1_w = tau_ID_weight(t1, self.tauID_SFs, cat)
            t2_w = tau_ID_weight(t2, self.tauID_SFs, cat)

        # apply ID scale factors
        w = l1_w * l2_w * t1_w * t2_w
        return ak.unflatten(w, num)

    def run_fastmtt(self, final_states):
        l1, l2 = final_states.ll.l1, final_states.ll.l2
        t1, t2 = final_states.tt.t1, final_states.tt.t2
        met = ak.flatten(final_states.met)
        cats = ak.flatten(final_states.category)
        map_it = {"e": 0, "m": 1, "t": 2}
        t1_cats = np.array([map_it[self.num_to_cat[cat][2]] for cat in cats])
        t2_cats = np.array([map_it[self.num_to_cat[cat][3]] for cat in cats])

        return fastmtt(
            np_flat(l1.pt),
            np_flat(l1.eta),
            np_flat(l1.phi),
            np_flat(l1.mass),
            np_flat(l2.pt),
            np_flat(l2.eta),
            np_flat(l2.phi),
            np_flat(l2.mass),
            np_flat(t1.pt),
            np_flat(t1.eta),
            np_flat(t1.phi),
            np_flat(t1.mass),
            t1_cats,
            np_flat(t2.pt),
            np_flat(t2.eta),
            np_flat(t2.phi),
            np_flat(t2.mass),
            t2_cats,
            np_flat(met.pt * np.cos(met.phi)),
            np_flat(met.pt * np.sin(met.phi)),
            np_flat(met.covXX),
            np_flat(met.covXY),
            np_flat(met.covXY),
            np_flat(met.covYY),
            constrain=True,
        )

    def postprocess(self, accumulator):
        pass


@nb.jit(nopython=True, parallel=False)
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
    verbosity=-1,
    delta=1 / 1.15,
    reg_order=6,
    constrain=False,
    constraint_window=np.array([124, 126]),
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
                if (
                    (test_mass > constraint_window[0])
                    and (test_mass < constraint_window[1])
                ) and constrain:

                    passes_constraint = True

                # calculate mass likelihood integral
                m_shift = test_mass * delta
                if m_shift < m_vis:
                    continue
                x1_min = min(1.0, math.pow((m_vis_1 / m_tau), 2))
                x2_min = max(
                    math.pow((m_vis_2 / m_tau), 2), math.pow((m_vis / m_shift), 2)
                )
                x2_max = min(1.0, math.pow((m_vis / m_shift), 2) / x1_min)
                if x2_max < x2_min:
                    continue
                J = 2 * math.pow(m_vis, 2) * math.pow(m_shift, -reg_order)
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

    return {
        "mtt_corr": m_tt_opt,
        "mtt_cons": m_tt_opt_c,
        "m4l_corr": m_lltt_opt,
        "m4l_cons": m_lltt_opt_c,
    }
