#!/bin/bash
set -e

#
# Clone all tools
#

BASEDIR=$(dirname "$0")
cd $BASEDIR
set -x

git clone --depth=1 https://github.com/ximion/minian.git
git clone --depth=1 https://github.com/flatironinstitute/CaImAn.git caiman_setup/CaImAn
