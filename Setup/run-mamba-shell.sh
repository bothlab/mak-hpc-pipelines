#!/bin/bash

# Spawns an interactive shell that has the Mamba environment sourced

BASEDIR="$(dirname "$(realpath "$0")")"

# Create a temporary rc file
TMP_RC=$(mktemp)
echo "source $BASEDIR/common" >> $TMP_RC
echo "WS=\$( ws_find mamba )" >> $TMP_RC
echo "source \$WS/mamba/etc/profile.d/conda.sh" >> $TMP_RC
if [ -f "$HOME/.bashrc" ]; then
    echo "source $HOME/.bashrc" >> $TMP_RC
fi

# Set a trap to delete the temporary RC file when the script exits
trap "rm -f $TMP_RC" EXIT

echo "Running in a new shell, exist to the default environment via CTRL+D, or type 'exit'"
echo ""
exec bash --rcfile $TMP_RC -i
