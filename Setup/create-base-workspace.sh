#!/bin/bash
set -e

#
# Create a new Mambaforge base workspace
#

BASEDIR=$(dirname "$0")
source "$BASEDIR/common"

echo "Using email: $1"
USER_MAIL=$1
if [ -z "$USER_MAIL" ]
then
  echo "No email address specified as first parameter."
  exit 1
fi

# clean the environment
echo "Cleaning up..."
conda deactivate || true
ws_release mamba || true
echo "==== Creating workspace ===="
set -x

ws_allocate -r 7 --mailaddress $USER_MAIL mamba 30
WS=$( ws_find mamba )
cd $WS

wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh" -O mambaforge.sh
bash mambaforge.sh -b -p $WS/mamba
rm mambaforge.sh

set +x
source $WS/mamba/etc/profile.d/conda.sh
set -x

conda activate

mamba install -y pip
mamba install -yc conda-forge \
    toml \
    jinja2 \
    xxhash \
    numpy \
    pint \
    rich
pip install git+https://github.com/bothlab/edlio.git

mamba update -y --all

mamba clean -y --all
du -sh $WS
