#!/bin/bash
set -e

#
# Create DeepLabCut environment
#

BASEDIR=$(dirname "$0")
WS=$( ws_find conda )

echo "==== Creating DeepLabCut Environment ===="
source $WS/conda/etc/profile.d/conda.sh
set -x

module load devel/cuda/11.6
module load lib/cudnn/8.5.0-cuda-11.6
module load compiler/gnu/12.1

conda env create -f \
    $BASEDIR/../Tools/deeplabcut_env.yml
conda activate deeplabcut

conda clean -y --all
du -sh $WS/conda
