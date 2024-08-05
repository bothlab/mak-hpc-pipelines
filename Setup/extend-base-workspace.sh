#!/bin/sh
set -e

#
# Extend time limit on Mamba base workspace
#

echo "Using email: $1"
USER_MAIL=$1
if [ -z "$USER_MAIL" ]
then
  echo "No email address specified as first parameter."
  exit 1
fi

echo "==== Extending workspace time limit ===="
set -x

ws_allocate -x mamba 30 --mailaddress $USER_MAIL
WS=$( ws_find mamba )
cd $WS
set +x
source $WS/mamba/etc/profile.d/conda.sh
set -x
conda activate

conda update -y --all

conda clean -y --all
du -sh $WS/
