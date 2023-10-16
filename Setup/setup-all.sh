#!/bin/bash
set -e
set -x

cd $(dirname "$0")
./create-base-workspace.sh $1

./setup_deeplabcut.sh
#./setup_caiman.sh
#./setup_min1pipe-wrap.sh
./setup_minian.sh
