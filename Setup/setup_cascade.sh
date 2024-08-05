#!/bin/bash
set -e

#
# Create CASCADE environment
#

BASEDIR=$(dirname "$0")
source "$BASEDIR/common"
WS=$( ws_find mamba )

echo "==== Creating CASCADE Environment ===="
source $WS/mamba/etc/profile.d/conda.sh
set -x

mamba env create -f \
    $BASEDIR/../Tools/cascade_env.yml
conda activate cascade

# FIXME: Resolves glibc conflict with the HPC environment, but breaks
# protobuf serialization (which is an okay compromise, currently)
pip install -U --force-reinstall protobuf

cd $BASEDIR/../Tools/Cascade/
pip install -e .

# Cleanup
conda clean -y --all
du -sh $WS/
