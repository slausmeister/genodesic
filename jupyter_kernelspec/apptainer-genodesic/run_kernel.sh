#!/usr/bin/env bash
set -euo pipefail

# EDIT THIS: Needs absolute path to the .sif file
SIF="/path/to/genodesic.sif"

# Capture the host’s connection-file path from VS Code
HOST_CONN="$1"

# Prepare a workspace-local Jupyter runtime directory (it’s bind-mounted into the container)
WORKDIR="$PWD"
RUNTIME_DIR="${WORKDIR}/.jupyter/runtime"
mkdir -p "${RUNTIME_DIR}"

# Copy the JSON into the bind-mounted runtime folder so the container can read and write it
NEW_CONN="${RUNTIME_DIR}/$(basename "${HOST_CONN}")"
cp "${HOST_CONN}" "${NEW_CONN}"

# Execute the kernel inside the Singularity image, pointing ipykernel at the copied file
exec singularity exec --nv \
    --bind "${WORKDIR}:${WORKDIR}" \
    "${SIF}" \
    python -m ipykernel_launcher -f "${NEW_CONN}"