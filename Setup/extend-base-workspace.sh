#!/bin/sh
set -e

#
# Extend time limit on conda base workspace
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

ws_allocate -x conda 30 --mailaddress $USER_MAIL
WS=$( ws_find conda )
cd $WS
set +x
source $WS/conda/etc/profile.d/conda.sh
set -x
conda activate

conda update -y --all

conda clean -y --all
du -sh $WS/conda
