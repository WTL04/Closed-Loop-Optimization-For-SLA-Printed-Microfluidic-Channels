#!/bin/bash
set -e

cd cfd/channelCase

blockMesh
surfaceFeatures
snappyHexMesh -overwrite
topoSet
createPatch -overwrite
simpleFoam
touch case.foam

echo "Running flow rate extraction..."
pvpython /home/will/Downloads/coding/uni/ml-research/contextual_bayes_opt/extract_flow_rate.py >flow_rate.txt

echo "CFD run complete."
echo "Flow rate written to flow_rate.txt"
echo "Open ParaView and load: cfd/channelCase/case.foam"
