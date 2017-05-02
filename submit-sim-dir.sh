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
        bs-qsub --ttr 10000 -- python make-cfg.py
        bs-qsub --ttr 10000 -- python simulation-meta.py
        bs-qsub --ttr 10000 -- python make-cfg-linear.py
        bs-qsub --ttr 10000 -- python simulation-meta.py
        bs-qsub --ttr 10000 -- python ../make-2dess-plot.py .
        bs-qsub --ttr 100 -- rm -rf /dev/shm/simulations
        bs-qsub --ttr 100 -- globus transfer --recursive \
            $revelation:`pwd`/figures $desktop:/~/`relpath`/figures \
            --label "Transfer $dir"
        popd > /dev/null
    fi
done
