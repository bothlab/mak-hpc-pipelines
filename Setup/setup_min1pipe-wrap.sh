#!/bin/bash
set -e

#
# Create MIN1PIPE environment
#

BASEDIR=$(dirname "$0")
WS=$( ws_find conda )

echo "==== Creating MIN1PIPE (Wrap) Environment ===="
source $WS/conda/etc/profile.d/conda.sh
set -x

conda env create -f \
    $BASEDIR/../Tools/min1pipe-wrap_env.yml

conda clean -y --all
du -sh $WS/conda
