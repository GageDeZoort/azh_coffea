# azh_coffea

## Useful Links:
- AZh Run 2 TWiki: [https://twiki.cern.ch/twiki/bin/viewauth/CMS/AZh-Htautau-Run2](https://twiki.cern.ch/twiki/bin/viewauth/CMS/AZh-Htautau-Run2)
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
python <run analysis>
```
