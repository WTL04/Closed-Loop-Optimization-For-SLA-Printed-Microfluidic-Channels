#!/bin/bash
set -e

# ---------------------------------------------------------------
# 1. CAD Generation
# ---------------------------------------------------------------
echo '--- Step 1: CAD Generation ---'
cd /case

if [ -n "" ] && [ -n "" ] && [ -n "" ]; then
    python contextual_opt/src/single_channel_inlet_outlet_cfd_export.py           
else
    python contextual_opt/src/single_channel_inlet_outlet_cfd_export.py
fi

# ---------------------------------------------------------------
# 2. OpenFOAM Execution
# ---------------------------------------------------------------
echo '--- Step 2: OpenFOAM ---'
set +e
source /usr/lib/openfoam/openfoam1912/etc/bashrc
set -e

cd /case/cfd/channelCase

rm -rf constant/polyMesh
blockMesh
surfaceFeatureExtract
snappyHexMesh -overwrite
checkMesh
simpleFoam

# ---------------------------------------------------------------
# 3. Flow Rate Extraction
# ---------------------------------------------------------------
echo '--- Step 3: Extract Flow Rate ---'
cd /case
python extract_flow_rate.py > cfd/channelCase/flow_rate.txt
