#!/bin/bash
set -e

cd cfd/channelCase

blockMesh
surfaceFeatureExtract
snappyHexMesh -overwrite
simpleFoam
touch case.foam

echo "CFD run complete."
echo "Open ParaView and load: cfd/channelCase/case.foam"
