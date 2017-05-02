#!/bin/env bash

args=("$@")
revelation="ec4187ae-e730-11e6-b9a1-22000b9a448b"
desktop="12a8a7a4-ddb0-11e6-9d11-22000a1e3b52"
relpath() {
    python -c "import os.path; print(os.path.relpath('`pwd`',
        '/u/home/aloukian/simulations'))";
}

for dir in "${args[@]}"; do
    if [ -d "$dir" ]
    then
        pushd $dir > /dev/null
        echo "Entering `pwd`"
        bs-qsub --ttr 10000 -- python ../make-jennifer-plot.py --limits 14250 15250 .
        bs-qsub --ttr 100 -- globus transfer --recursive \
            $revelation:`pwd`/jennifer-figs $desktop:/~/`relpath`/jennifer-figs \
            --label "Transfer jennifer plot $dir"
        popd > /dev/null
    fi
done
