#!/bin/bash

# ---------------------------------------------------------------
# 1. Mesh Generation (parametric blockMesh)
# ---------------------------------------------------------------
echo '--- Step 1: Generate blockMeshDict ---'
cd /case

if [ -n "40060.0" ] && [ -n "495.0" ] && [ -n "530.0" ] && [ -n "8.919244992507192" ] && [ -n "11.20600664271413" ] && [ -n "1.8088853308043262" ]; then
    python contextual_opt/src/cad/generate_blockmesh.py         40060.0 495.0 530.0         8.919244992507192 11.20600664271413 1.8088853308043262
else
    python contextual_opt/src/cad/generate_blockmesh.py
fi

# ---------------------------------------------------------------
# 2. OpenFOAM Execution
# ---------------------------------------------------------------
echo '--- Step 2: OpenFOAM ---'
set +e
source /usr/lib/openfoam/openfoam1912/etc/bashrc
set -e

cd /case/cfd/channelCase

rm -rf constant/polyMesh postProcessing
blockMesh
checkMesh

SIMPLEFOAM_SUCCESS=0

if ! simpleFoam > log.simpleFoam 2>&1; then
    echo "WARN: simpleFoam failed on first attempt, retrying with relaxed settings..."
    cp system/fvSolution.relaxed system/fvSolution
    rm -rf [0-9]* postProcessing constant/polyMesh
    
    if ! simpleFoam > log.simpleFoam.relaxed 2>&1; then
        echo "ERROR: simpleFoam failed even with relaxed settings"
        SIMPLEFOAM_SUCCESS=1
    fi
fi

# ---------------------------------------------------------------
# 3. Flow Rate Extraction
# ---------------------------------------------------------------
echo '--- Step 3: Extract Flow Rate ---'
cd /case

if [  -eq 1 ]; then
    echo "FLOW_RATE:-1.0" > cfd/channelCase/flow_rate.txt
    echo "WARN: CFD simulation failed, returning sentinel value -1.0"
else
    python extract_flow_rate.py > cfd/channelCase/flow_rate.txt
fi
