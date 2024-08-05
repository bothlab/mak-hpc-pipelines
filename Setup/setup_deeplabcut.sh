#!/bin/bash
set -e

#
# Create DeepLabCut environment
#

BASEDIR=$(dirname "$0")
source "$BASEDIR/common"
WS=$( ws_find mamba )

echo "==== Creating DeepLabCut Environment ===="
source $WS/mamba/etc/profile.d/conda.sh
set -x

if [ -z "$LOCAL_NO_HPC" ]; then
    module load lib/cudnn/9.1.0-cuda-12.2
    module load compiler/gnu/12.1
fi

mamba create -y -n deeplabcut python=3.11
conda activate deeplabcut

if [ "$1" -eq "rocm" ]; then
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
else
    mamba install -y nvidia/label/cuda-12.2.2::cuda
    pip3 install torch torchvision torchaudio
fi

mamba install -y -c conda-forge pytables~=3.8.0

pip3 install git+https://github.com/ximion/DeepLabCut.git@pytorch_dlc#egg=deeplabcut[gui,modelzoo,wandb]

conda clean -y --all
du -sh $WS/
