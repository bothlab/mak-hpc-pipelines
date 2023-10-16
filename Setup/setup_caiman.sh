#!/bin/bash
set -e

#
# Create CaImAn environment
#

BASEDIR=$(dirname "$0")
WS=$( ws_find conda )

echo "==== Creating CaImAn Environment ===="
source $WS/conda/etc/profile.d/conda.sh
set -x

module load devel/cuda/11.6
module load lib/cudnn/8.5.0-cuda-11.6
module load compiler/gnu/12.1

conda env create -f \
    $BASEDIR/../Tools/caiman_setup/caiman-environment.yml
conda activate caiman

cd $BASEDIR/../Tools/caiman_setup/CaImAn/
pip install .

pip install -U protobuf

rm -rf ~/caiman_data
./caimanmanager.py install

conda clean -y --all
du -sh $WS/conda
