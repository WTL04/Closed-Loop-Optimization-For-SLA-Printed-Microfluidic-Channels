#!/bin/bash
set -e

IMAGE_NAME="unified-cfd-env"
REPO_ROOT="$(pwd)"

# ---------------------------------------------------------------------------
# Parse CBO CAD inputs and delta arguments
# Usage: ./run_cfd.sh <cbo_length_um> <cbo_width_um> <cbo_height_um> <delta_length_um> <delta_width_um> <delta_height_um> [case_dir]
# Expected Physical = CBO_Suggested_CAD + Printer_Delta_Error
# If not provided, export.py falls back to fetching latest from Google Sheets
# ---------------------------------------------------------------------------
CBO_LENGTH_UM="${1:-}"
CBO_WIDTH_UM="${2:-}"
CBO_HEIGHT_UM="${3:-}"
LENGTH_DELTA="${4:-}"
WIDTH_DELTA="${5:-}"
HEIGHT_DELTA="${6:-}"
CASE_DIR="${7:-cfd/channelCase}"

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
SAFE_CASE="${CASE_DIR//\//_}"
INNER_SCRIPT="$REPO_ROOT/.pipeline_inner_${SAFE_CASE}.sh"

cat >"$INNER_SCRIPT" <<INNEREOF
#!/bin/bash

# ---------------------------------------------------------------
# 0. Prepare clean case directory
# ---------------------------------------------------------------
WORKDIR="/case/${CASE_DIR}"
echo "--- Step 0: Prepare clean case in ${WORKDIR} ---"
rm -rf "\$WORKDIR"
cp -r /case/cfd/channelCase_template "\$WORKDIR"

# ---------------------------------------------------------------
# 1. Mesh Generation (parametric blockMesh)
# ---------------------------------------------------------------
echo '--- Step 1: Generate blockMeshDict ---'
cd /case

if [ -n "${CBO_LENGTH_UM}" ] && [ -n "${CBO_WIDTH_UM}" ] && [ -n "${CBO_HEIGHT_UM}" ] && [ -n "${LENGTH_DELTA}" ] && [ -n "${WIDTH_DELTA}" ] && [ -n "${HEIGHT_DELTA}" ]; then
    python contextual_opt/src/cad/generate_blockmesh.py \
        ${CBO_LENGTH_UM} ${CBO_WIDTH_UM} ${CBO_HEIGHT_UM} \
        ${LENGTH_DELTA} ${WIDTH_DELTA} ${HEIGHT_DELTA} \
        --output "${CASE_DIR}/system/blockMeshDict"
else
    python contextual_opt/src/cad/generate_blockmesh.py \
        --output "${CASE_DIR}/system/blockMeshDict"
fi

# ---------------------------------------------------------------
# 2. OpenFOAM Execution
# ---------------------------------------------------------------
echo '--- Step 2: OpenFOAM ---'
set +e
source /usr/lib/openfoam/openfoam1912/etc/bashrc
set -e

cd "\$WORKDIR"

blockMesh
checkMesh

echo "CFD Solver is running (check: tail -f ${CASE_DIR}/log.simpleFoam for progress)..."

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

if [ \$SIMPLEFOAM_SUCCESS -eq 1 ]; then
    echo "FLOW_RATE:-1.0" > ${CASE_DIR}/flow_rate.txt
    echo "WARN: CFD simulation failed, returning sentinel value -1.0"
else
    python extract_flow_rate.py --case-dir ${CASE_DIR} > ${CASE_DIR}/flow_rate.txt
fi
INNEREOF

chmod +x "$INNER_SCRIPT"

echo "Starting Unified Pipeline..."
echo "Repo root: $REPO_ROOT"

docker run --rm \
  --entrypoint bash \
  -v "$REPO_ROOT:/case" \
  $IMAGE_NAME \
  "/case/.pipeline_inner_${SAFE_CASE}.sh"

# Clean up inner script after run
rm -f "$INNER_SCRIPT" || true

echo "Pipeline Finished."
echo "Calculated Flow Rate: $(cat ${CASE_DIR}/flow_rate.txt)"
