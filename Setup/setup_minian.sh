#!/bin/bash
set -e

#
# Create Minian environment
#

BASEDIR=$(dirname "$0")
WS=$( ws_find conda )

echo "==== Creating Minian Environment ===="
source $WS/conda/etc/profile.d/conda.sh
set -x

mamba env create -f \
    $BASEDIR/../Tools/minian/environment.yml
conda activate minian

cd $BASEDIR/../Tools/minian/
pip install -e .

pip install seaborn edlio pywavelets pebble pyfftw

conda clean -y --all
du -sh $WS/conda
