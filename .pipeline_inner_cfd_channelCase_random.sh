#!/bin/bash

# ---------------------------------------------------------------
# 0. Prepare clean case directory
# ---------------------------------------------------------------
WORKDIR="/case/cfd/channelCase_random"
echo "--- Step 0: Prepare clean case in  ---"
rm -rf "$WORKDIR"
cp -r /case/cfd/channelCase_template "$WORKDIR"

# ---------------------------------------------------------------
# 1. Mesh Generation (parametric blockMesh)
# ---------------------------------------------------------------
echo '--- Step 1: Generate blockMeshDict ---'
cd /case

if [ -n "40029.74633796023" ] && [ -n "508.9194559291867" ] && [ -n "507.6097287632759" ] && [ -n "2.055921344952229" ] && [ -n "17.513780453444888" ] && [ -n "16.098413604958754" ]; then
    python contextual_opt/src/cad/generate_blockmesh.py         40029.74633796023 508.9194559291867 507.6097287632759         2.055921344952229 17.513780453444888 16.098413604958754         --output "cfd/channelCase_random/system/blockMeshDict"
else
    python contextual_opt/src/cad/generate_blockmesh.py         --output "cfd/channelCase_random/system/blockMeshDict"
fi

# ---------------------------------------------------------------
# 2. OpenFOAM Execution
# ---------------------------------------------------------------
echo '--- Step 2: OpenFOAM ---'
set +e
source /usr/lib/openfoam/openfoam1912/etc/bashrc
set -e

cd "$WORKDIR"

blockMesh
checkMesh

echo "CFD Solver is running (check: tail -f cfd/channelCase_random/log.simpleFoam for progress)..."

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

if [ $SIMPLEFOAM_SUCCESS -eq 1 ]; then
    echo "FLOW_RATE:-1.0" > cfd/channelCase_random/flow_rate.txt
    echo "WARN: CFD simulation failed, returning sentinel value -1.0"
else
    python extract_flow_rate.py --case-dir cfd/channelCase_random > cfd/channelCase_random/flow_rate.txt
fi
