#!/bin/bash
set -e

IMAGE_NAME="unified-cfd-env"
REPO_ROOT="$(pwd)"

# ---------------------------------------------------------------------------
# Parse CBO CAD inputs and delta arguments
# Usage: ./run_cfd.sh <cbo_length_um> <cbo_width_um> <cbo_height_um> <delta_length_um> <delta_width_um> <delta_height_um>
# Expected Physical = CBO_Suggested_CAD + Printer_Delta_Error
# If not provided, export.py falls back to fetching latest from Google Sheets
# ---------------------------------------------------------------------------
CBO_LENGTH_UM="${1:-}"
CBO_WIDTH_UM="${2:-}"
CBO_HEIGHT_UM="${3:-}"
LENGTH_DELTA="${4:-}"
WIDTH_DELTA="${5:-}"
HEIGHT_DELTA="${6:-}"

# Build the Docker image if it does not exist
# NOTE: if you change Dockerfile or req.txt, run: docker rmi unified-cfd-env
#       to force a rebuild on next run
if [[ "$(docker images -q $IMAGE_NAME 2>/dev/null)" == "" ]]; then
  echo "Building Docker image '$IMAGE_NAME'..."
  docker build -t $IMAGE_NAME .
fi

# ---------------------------------------------------------------------------
# Write the inner pipeline script to a temp file in the repo root
# This avoids passing a long inline string to docker run, which breaks
# source commands and makes set -e behave unpredictably
# ---------------------------------------------------------------------------
INNER_SCRIPT="$REPO_ROOT/.pipeline_inner.sh"

cat >"$INNER_SCRIPT" <<INNEREOF
#!/bin/bash

# ---------------------------------------------------------------
# 1. CAD Generation
# ---------------------------------------------------------------
echo '--- Step 1: CAD Generation ---'
cd /case

if [ -n "${CBO_LENGTH_UM}" ] && [ -n "${CBO_WIDTH_UM}" ] && [ -n "${CBO_HEIGHT_UM}" ] && [ -n "${LENGTH_DELTA}" ] && [ -n "${WIDTH_DELTA}" ] && [ -n "${HEIGHT_DELTA}" ]; then
    python contextual_opt/src/cad/single_channel_inlet_outlet_cfd_export.py \
        ${CBO_LENGTH_UM} ${CBO_WIDTH_UM} ${CBO_HEIGHT_UM} \
        ${LENGTH_DELTA} ${WIDTH_DELTA} ${HEIGHT_DELTA}
else
    python contextual_opt/src/cad/single_channel_inlet_outlet_cfd_export.py
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
surfaceFeatureExtract
snappyHexMesh -overwrite
checkMesh
simpleFoam || true

# ---------------------------------------------------------------
# 3. Flow Rate Extraction
# ---------------------------------------------------------------
echo '--- Step 3: Extract Flow Rate ---'
cd /case
python extract_flow_rate.py > cfd/channelCase/flow_rate.txt
INNEREOF

chmod +x "$INNER_SCRIPT"

echo "Starting Unified Pipeline..."
echo "Repo root: $REPO_ROOT"

docker run --rm \
  --entrypoint bash \
  -v "$REPO_ROOT:/case" \
  $IMAGE_NAME \
  /case/.pipeline_inner.sh

# Clean up inner script after run
rm -f "$INNER_SCRIPT"

echo "Pipeline Finished."
echo "Calculated Flow Rate: $(cat cfd/channelCase/flow_rate.txt)"
