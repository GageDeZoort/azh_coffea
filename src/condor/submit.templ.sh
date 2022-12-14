#!/bin/bash

# make sure this is installed
# python3 -m pip install correctionlib==2.0.0rc6
# pip install --upgrade numpy==1.21.5

# make dir for output
mkdir outfiles

# run code
# pip install --user onnxruntime
python $script --year $year --source $source --sample $sample --start-idx $start_idx --end-idx $end_idx --outfile $outfile --mass $mass

#move output to eos
xrdcp -f *.coffea $eosoutcoffea
xrdcp -f outfiles/* $eosoutpkl
xrdcp -f *.parquet $eosoutparquet
xrdcp -f *.root $eosoutroot

rm *.coffea
rm *.parquet
rm *.root
