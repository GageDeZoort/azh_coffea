from __future__ import annotations

import awkward as ak
import numpy as np
import uproot
import yaml
from coffea import processor
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from coffea.processor import column_accumulator as col_acc

# from processors.analysis_processor import AnalysisProcessor


def open_yaml(f):
    with open(f, "r") as stream:
        try:
            loaded_file = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return loaded_file


def get_sample_info(f):
    return np.genfromtxt(
        f,
        delimiter=",",
        names=True,
        comments="#",
        dtype=np.dtype(
            [
                ("f0", "<U9"),
                ("f1", "<U64"),
                ("f2", "<U32"),
                ("f3", "<U250"),
                ("f4", "<f16"),
                ("f5", "<f8"),
            ]
        ),
    )


def get_fileset(sample_yaml, process="", nfiles=-1):
    fileset = open_yaml(sample_yaml)
    if nfiles > 0:
        fileset = {f: s[:nfiles] for f, s in fileset.items()}
    # filter on a specific process
    fileset = {f: s for f, s in fileset.items() if process in f}
    return fileset


def get_fileset_array(fileset, collection, variable=None):
    values_per_file = []
    for files in fileset.values():
        for f in files:
            try:
                events = NanoEventsFactory.from_root(
                    f, schemaclass=NanoAODSchema
                ).events()
            except BaseException:
                global_redir = "root://cms-xrd-global.cern.ch/"
                f = global_redir + "/store/" + f.split("/store/")[-1]
                events = NanoEventsFactory.from_root(
                    f, schemaclass=NanoAODSchema
                ).events()
            values = events[collection]
            if variable is not None:
                values = values[variable]
            values_per_file.append(values.to_numpy())
    return np.concatenate(values_per_file)


def get_fileset_arrays(
    fileset, collection_vars=None, global_vars=None, analysis=True, sample_info=""
):
    processor_instance = VariableHarvester(
        collection_vars=collection_vars, global_vars=global_vars
    )
    # if analysis:
    #    sample_info = load_sample_info(sample_info)
    #    processor_instance = AnalysisProcessor(sample_info=sample_info,
    #                                           collection_vars=collection_vars,
    #                                           global_vars=global_vars)
    out = processor.run_uproot_job(
        fileset,
        treename="Events",
        processor_instance=processor_instance,
        executor=processor.futures_executor,
        executor_args={"schema": NanoAODSchema, "workers": 20},
    )
    return out


def get_nevts_dict(fileset, year, high_stats=False):
    nevts_dict = {}

    for sample, files in fileset.items():
        sum_of_weights = 0
        for f in files:
            if "1of3_Electrons" not in f:
                continue
            with uproot.open(f) as tree:
                sum_of_weights += tree["hWeights;1"].values()[0]
        nevts_dict[sample] = sum_of_weights

    if "18" in year:
        print("fixing DYJets")
        nevts = (
            nevts_dict["DYJetsToLLM-50_2018"] + nevts_dict["DYJetsToLLM-50_ext1_2018"]
        )
        nevts_dict["DYJetsToLLM-50_2018"] = nevts
        nevts_dict["DYJetsToLLM-50_ext1_2018"] = nevts
        print("fixing WZZ")
        nevts = nevts_dict["WZZ_ext1_2018"] + nevts_dict["WZZ_2018"]
        nevts_dict["WZZ_2018"] = nevts
        nevts_dict["WZZ_ext1_2018"] = nevts
        print("fixing ZZZ")
        nevts = nevts_dict["ZZZ_ext1_2018"] + nevts_dict["ZZZ_2018"]
        nevts_dict["ZZZ_2018"] = nevts
        nevts_dict["ZZZ_ext1_2018"] = nevts
        print("fixing WWW4F")
        nevts = nevts_dict["WWW4F_2018"] + nevts_dict["WWW4F_ext1_2018"]
        nevts_dict["WWW4F_2018"] = nevts
        nevts_dict["WWW4F_ext1_2018"] = nevts

    if "17" in year:
        print("fixing DYjets")
        nevts = (
            nevts_dict["DYJetsToLLM-50_2017"] + nevts_dict["DYJetsToLLM-50_ext1_2017"]
        )
        nevts_dict["DYJetsToLLM-50_2017"] = nevts
        nevts_dict["DYJetsToLLM-50_ext1_2017"] = nevts
        print("fixing WWW4F")
        nevts = nevts_dict["WWW4F_2017"] + nevts_dict["WWW4F_ext1_2017"]
        nevts_dict["WWW4F_2017"] = nevts
        nevts_dict["WWW4F_ext1_2017"] = nevts
        print("fixing ZHToTauTau")
        nevts = (
            nevts_dict["ZHToTauTauM125_2017"] + nevts_dict["ZHToTauTauM125_ext1_2017"]
        )
        nevts_dict["ZHToTauTauM125_2017"] = nevts
        nevts_dict["ZHToTauTauM125_ext1_2017"] = nevts
        print("fixing ZHWW")
        nevts = nevts_dict["HZJHToWW_2017"] + nevts_dict["HZJHToWW_ext1_2017"]
        nevts_dict["HZJHToWW_2017"] = nevts
        nevts_dict["HZJHToWW_ext1_2017"] = nevts

    if "16post" in year:
        print("fixing WWW4F")
        nevts = nevts_dict["WWW4F_2016postVFP"] + nevts_dict["WWW4F_ext1_2016postVFP"]
        nevts_dict["WWW4F_2016postVFP"] = nevts
        nevts_dict["WWW4F_ext1_2016postVFP"] = nevts
        print("fixing ZZZ")
        nevts = nevts_dict["ZZZ_2016postVFP"] + nevts_dict["ZZZ_ext1_2016postVFP"]
        nevts_dict["ZZZ_ext1_2016postVFP"] = nevts
        nevts_dict["ZZZ_2016postVFP"] = nevts
        print("fixing WWZ4F")
        nevts = nevts_dict["WWZ4F_ext1_2016postVFP"] + nevts_dict["WWZ4F_2016postVFP"]
        nevts_dict["WWZ4F_2016postVFP"] = nevts
        nevts_dict["WWZ4F_ext1_2016postVFP"] = nevts
        print("fixing WZZ")
        nevts = nevts_dict["WZZ_2016postVFP"] + nevts_dict["WZZ_ext1_2016postVFP"]
        nevts_dict["WZZ_2016postVFP"] = nevts
        nevts_dict["WZZ_ext1_2016postVFP"] = nevts

    if "16pre" in year:
        print("fixing WWW4F")
        nevts = nevts_dict["WWW4F_2016preVFP"] + nevts_dict["WWW4F_ext1_2016preVFP"]
        nevts_dict["WWW4F_2016preVFP"] = nevts
        nevts_dict["WWW4F_ext1_2016preVFP"] = nevts
        print("fixing ZZZ")
        nevts = nevts_dict["ZZZ_2016preVFP"] + nevts_dict["ZZZ_ext1_2016preVFP"]
        nevts_dict["ZZZ_2016preVFP"] = nevts
        nevts_dict["ZZZ_ext1_2016preVFP"] = nevts
        print("fixing WWZ4F")
        nevts = nevts_dict["WWZ4F_2016preVFP"] + nevts_dict["WWZ_ext1_2016preVFP"]
        nevts_dict["WWZ4F_2016preVFP"] = nevts
        nevts_dict["WWZ_ext1_2016preVFP"] = nevts
        print("fixing WZZ")
        nevts = nevts_dict["WZZ_2016preVFP"] + nevts_dict["WZZ_ext1_2016preVFP"]
        nevts_dict["WZZ_2016preVFP"] = nevts
        nevts_dict["WZZ_ext1_2016preVFP"] = nevts

    return nevts_dict


class VariableHarvester(processor.ProcessorABC):
    def __init__(self, collection_vars=None, global_vars=None):
        self.collection_vars = collection_vars
        self.global_vars = global_vars

        collection_dict = {
            f"{c}_{v}": col_acc(np.array([])) for (c, v) in self.collection_vars
        }
        global_dict = {var: col_acc(np.array([])) for var in global_vars}
        output = {**collection_dict, **global_dict}
        self._accumulator = processor.dict_accumulator(output)

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

        print("...processing", events.metadata["dataset"])

        # organize dataset, year, luminosity
        for (c, v) in self.collection_vars:
            values = events[c][v].to_numpy()
            self.output[f"{c}_{v}"] += col_acc(values)

        for v in self.global_vars:
            values = events[v].to_numpy()
            self.output[v] += col_acc(values)

        return self.output

    def postprocess(self, accumulator):
        return accumulator
