#!/bin/bash
set -e

CASE_DIR="/home/will/Downloads/coding/uni/ml-research/contextual_bayes_opt/cfd/channelCase"
IMAGE="openfoamplus/of_v1912_centos73"

echo "Pulling OpenFOAM Docker image..."
docker pull $IMAGE

echo "Running CFD pipeline in Docker..."
docker run --rm \
  -v "$CASE_DIR:/case" \
  -w /case \
  $IMAGE \
  bash -c "
        source /opt/OpenFOAM/OpenFOAM-v1912/etc/bashrc
        rm -rf constant/polyMesh 1>/dev/null 2>&1 || true
        blockMesh
        surfaceFeatureExtract
        snappyHexMesh -overwrite
        topoSet
        createPatch -overwrite
        simpleFoam
        touch case.foam
    "

echo "Running flow rate extraction..."
cd /home/will/Downloads/coding/uni/ml-research/contextual_bayes_opt
python extract_flow_rate.py >"$CASE_DIR/flow_rate.txt"

echo "CFD run complete."
echo "Flow rate: $(cat "$CASE_DIR/flow_rate.txt")"

