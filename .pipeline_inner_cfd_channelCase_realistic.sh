#!/bin/bash

# ---------------------------------------------------------------
# 0. Prepare clean case directory
# ---------------------------------------------------------------
WORKDIR="/case/cfd/channelCase_realistic"
echo "--- Step 0: Prepare clean case in  ---"
rm -rf "$WORKDIR"
cp -r /case/cfd/channelCase_template "$WORKDIR"

# ---------------------------------------------------------------
# 1. Mesh Generation (parametric blockMesh)
# ---------------------------------------------------------------
echo '--- Step 1: Generate blockMeshDict ---'
cd /case

if [ -n "40060.0" ] && [ -n "516.7996847835112" ] && [ -n "511.0289763375539" ] && [ -n "11.454328040354934" ] && [ -n "23.0" ] && [ -n "15.294422515612148" ]; then
    python contextual_opt/src/cad/generate_blockmesh.py         40060.0 516.7996847835112 511.0289763375539         11.454328040354934 23.0 15.294422515612148         --output "cfd/channelCase_realistic/system/blockMeshDict"
else
    python contextual_opt/src/cad/generate_blockmesh.py         --output "cfd/channelCase_realistic/system/blockMeshDict"
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

echo "CFD Solver is running (check: tail -f cfd/channelCase_realistic/log.simpleFoam for progress)..."

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
    echo "FLOW_RATE:-1.0" > cfd/channelCase_realistic/flow_rate.txt
    echo "WARN: CFD simulation failed, returning sentinel value -1.0"
else
    python extract_flow_rate.py --case-dir cfd/channelCase_realistic > cfd/channelCase_realistic/flow_rate.txt
fi
