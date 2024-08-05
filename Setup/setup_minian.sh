#!/bin/bash
set -e

#
# Create Minian environment
#

BASEDIR=$(dirname "$0")
source "$BASEDIR/common"
WS=$( ws_find mamba )

echo "==== Creating Minian Environment ===="
source $WS/mamba/etc/profile.d/conda.sh
set -x

mamba env create -f \
    $BASEDIR/../Tools/minian_env.yml
conda activate minian

cd $BASEDIR/../Tools/minian/
pip install -e .

pip install seaborn edlio 'pywavelets==1.4.1' 'pebble==5.0.3' 'pyfftw==0.12.0'

conda clean -y --all
du -sh $WS/
