# azh_coffea

## Useful Links:
- AZh Run 2 TWiki: [https://twiki.cern.ch/twiki/bin/viewauth/CMS/AZh-Htautau-Run2](https://twiki.cern.ch/twiki/bin/viewauth/CMS/AZh-Htautau-Run2)
- HIG-22-004 Paper (Draft): [https://gitlab.cern.ch/cms-hcg/cadi/hig-22-004](https://gitlab.cern.ch/cms-hcg/cadi/hig-22-004)
- AN-21-186 Analysis Note (Draft): [https://gitlab.cern.ch/tdr/notes/AN-21-186/-/tree/master](https://gitlab.cern.ch/tdr/notes/AN-21-186/-/tree/master)
- Previous AZh Analysis Note (2016): [AN-2017/276](https://cms.cern.ch/iCMS/jsp/db_notes/noteInfo.jsp?cmsnoteid=CMS%20AN-2021/186)


## Useful Commands:
Several plotting Jupyter notebooks are availble. To open a Jupyter notebook on cmslpc, you must first ssh with a -L option specifying the local host: 

```ssh -L localhost:8888:localhost:8888 <USERNAME>@cmslpc-sl7.fnal.gov```

To start the notebook, run `jupyter notebook`, specifying the same port you used in your ssh command:

```jupyter notebook --no-browser --port=8888 --ip 127.0.0.1```
