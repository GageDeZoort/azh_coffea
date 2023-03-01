# azh_coffea

## Useful Links:
- AZh Run 2 TWiki: [https://twiki.cern.ch/twiki/bin/viewauth/CMS/AZh-Htautau-Run2](https://twiki.cern.ch/twiki/bin/viewauth/CMS/AZh-Htautau-Run2)
- AZh Run 2 Pre-Approval Talk: [https://indico.cern.ch/event/1200004/contributions/5061637/attachments/2514048/4332974/AZh_Hig-22-004_preapproval_v3.pdf](https://indico.cern.ch/event/1200004/contributions/5061637/attachments/2514048/4332974/AZh_Hig-22-004_preapproval_v3.pdf)
- HIG-22-004 Paper (Draft): [https://gitlab.cern.ch/cms-hcg/cadi/hig-22-004](https://gitlab.cern.ch/cms-hcg/cadi/hig-22-004)
- AN-21-186 Analysis Note (Draft): [https://gitlab.cern.ch/tdr/notes/AN-21-186/-/tree/master](https://gitlab.cern.ch/tdr/notes/AN-21-186/-/tree/master)
- Previous AZh Analysis Note (2016): [AN-2017/276](https://cms.cern.ch/iCMS/jsp/db_notes/noteInfo.jsp?cmsnoteid=CMS%20AN-2021/186)


## Useful Commands:
Several plotting Jupyter notebooks are available. To open a Jupyter notebook on cmslpc, you must first ssh with a -L option specifying the local host: 

```ssh -L localhost:8888:localhost:8888 <USERNAME>@cmslpc-sl7.fnal.gov```

To start the notebook, run `jupyter notebook`, specifying the same port you used in your ssh command:

```jupyter notebook --no-browser --port=8888 --ip 127.0.0.1```


## Quickstart
Most of the repo's useful contents are organized in the `azh_coffea/src` directory. Here's a rundown of this directory's contents:
- `azh_analysis` contains the analysis code itself lives, including Coffea processors, event selection functions, and relevant utilities. It is not yet installable as a package, though it may one day be. 
- `corrections` contains the scale factors, fake rates, and efficiency measurements that are plugged into the analysis. 
- `condor` contains all the necessary scripts to submit analysis jobs to the LPC Condor cluster.
- `notebooks` contains Jupyter notebooks designed to collate intermediate files, test the analysis processors, and produce plots.
- `samples` contains sample lists and scripts to produce their absolte paths. 

Additionally, several scripts designed to run coffea processors are available, the main one being `run_analysis.py`. You can test this script by running:

```python run_analysis.py -s signal_UL -y 2018 --mass 225 --test-mode```

The `source`, or `-s` flag, is defined to be `<data type>_<legacy status>`, e.g. `signal_UL` for ultra-legacy signal code. The `year`, or `-y` flag, denotes the data-taking era, either `2018`, `2017`, `2016postVFP`, or `2016preVFP`. The remaining flags indicate that we're running over 1 AZh signal sample with an A mass of 225 GeV. 

### Running with `condor`
We can scale up and run over the full 225 GeV AZh signal sample by navigating to the `condor` directory. There, we'd run the following command:

```python submit.py -y 2018 -s signal_UL --mass 225 --submit``` 

### Running with `lpcjobqueue`
The Coffea team provides a tool called `lpcjobqueue` (see their repo), which is a Dask-based job queueing plugin for the LPC Condor. To run this code, we need to modify several job submission parameters; therefore, we provide a forked copy of `lpcjobqueue` specific to this code:

[https://github.com/GageDeZoort/lpcjobqueue](https://github.com/GageDeZoort/lpcjobqueue)

To set up job submission via `lpcjobqueue`, you'll need to grab a copy of `bootstrap.sh` and execute it, creating a `shell` script. The `shell` script transfers you to the `/srv` directory and activates the Coffea Singularity shell.

``` 
bash bootstrap.sh
./shell
python run_distributed_analysis.py -y 2018 -s MC_UL --test-mode
```
## Selections
The analysis selections are stored in `selections/preselections.py` and `selections/fake_rate_selections.py`. Additional utilities are available in the `utils/*.py`. 

### Triggers 
Single light lepton triggers are used to identify Z-->ll decays. Trigger selections and filters are applied by functions in ```selections/preselections.py```. The following triggers and filters are used in this analysis:  

| Type | Year | Path | Filter | 
| :--: | :--: | :--: | :----: |
| Single Electron | 2017/18 | Ele35_WPTight_Gsf | HLTEle35WPTightGsfSequence | 
| | 2016 | HLT_Ele25_eta2p1_WPTight_Gsf | hltEle25erWPTightGsfTrackIsoFilter |
| Single Muon | 2017/18 | IsoMu27  | hltL3crIsoL1sMu * Filtered0p07 |
| | 2016 | HLT_IsoMu24 | hltL3crIsoL1sMu * L3trkIsoFiltered0p09  | 
| | 2016 | HLT_IsoTkMu24 | hltL3fL1sMu * L3trkIsoFiltered0p09 | 

The listed trigger filters are the final filters in the respective HLT trigger path. All paths and their respective filters are listed in the [TriggerPaths Git Repo](https://github.com/UHH2/TriggerPaths); given a specific year, you can search for a relevant trigger path and find all of its relevant filters. 

### MET Filters
MET filters are applied to rejecct spurious sources of MET, e.g. cosmic ray contamination. MET filters are applied according to the recommendations in the [MissingETOptionalFiltersRun2 Twiki](https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETOptionalFiltersRun2#2018_2017_data_and_MC_UL). 

### Primary Vertex Filters
The main primary vertex in each event is required to have > 4 degrees of freedom and to satisfy |z| < 24cm and \sqrt{x^2 + y^2} < 2cm. 

### b-Jet Filters
b-jets are required to be baseline jets passing the medium DeepFlavorJet discrimination working points listed in the [BtagRecommendation Twiki](https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation). Relevant b-tag scale factor calculations are detailed in the [BTagSFMethods Twiki](https://twiki.cern.ch/twiki/bin/view/CMS/BTagSFMethods#b_tagging_efficiency_in_MC_sampl).
- [2018 UL](https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL18): `btagDeepFlavB > 0.2783`
- [2017 UL](https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL17): `btagDeepFlavB > 0.3040`
- [2016postVFP UL](https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL16postVFP): `btagDeepFlavB > 0.2489`
- [2016preVFP UL](https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL16preVFP): `btagDeepFlavB > 0.2598`



## Data
### Samples 
Samples are organized in to sample .csv files containing sample names, event counts, etc. and sample .yaml files containing `sample: [file1, file2, ...]` dictionaries. Sample .csv files are listed in `sample_lists/` and sample .yaml files are listed in `sample_lists/sample_yamls/`. 
- 2018: `sample_lists/data_UL_2018.csv`, `sample_lists/sample_yamls/data_UL_2018.yaml`
- 2017: `sample_lists/data_UL_2017.csv`, `sample_lists/sample_yamls/data_UL_2017.yaml`
- 2016postVFP: `sample_lists/data_UL_2016postVFP.csv`, `sample_lists/sample_yamls/data_UL_2016postVFP.yaml`
- 2016preVFP: `sample_lists/data_UL_2016preVFP.csv`, `sample_lists/sample_yamls/data_UL_2016preVFP.yaml`

### Pileup Weights
Following the recommendations in https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJSONFileforData,
- 2018: /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions18/13TeV/PileUp/UltraLegacy/
- 2017: /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp/UltraLegacy/
- 2016: /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions16/13TeV/PileUp/UltraLegacy/

Pileup weights are derived from the ratio of the data pileup distribution (from the corresponding file above) to the relevant MC pileup distribution. These weights are pre-derived and queried at run-time during the analysis. 

### Golden JSONs
Recommended luminosity, golden JSON file information: https://twiki.cern.ch/twiki/bin/view/CMS/TWikiLUM
- 2016APV: 19.52 fb^-1
- 2016: 16.81 fb^-1 
  - `/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions16/13TeV/Legacy_2016/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt`
- 2017: 41.48 fb^-1
  - `/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/Legacy_2017/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt`
- 2018: 59.83 fb^-1
  - `/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions18/13TeV/Legacy_2018/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt`
 
## Simulation
### Samples and Generator Parameters
Sample csv files containing DAS strings, xrootd redirectors, and cross sections are stored the `samples` directory. 

#### Sample Weights
Each sample is weighted by the data-MC luminosity ratio to normalize the expected MC contribution to the observed data contribution. 

#### DY+Jets Stitching 
In order to increase the statistics in the phase space Z+jets events with 1-4 jets, we use dedicated *exclusive* DY+nJets, where n=1,2,3,4,  samples in addition to an *inclusive* sample of DY+Jets with any number of jets. These events have to be carefully weighted to account for the fact that multiple samples contribute relevant events. The process of weighting these events is called [MC stitching](https://arxiv.org/abs/2106.04360). 

