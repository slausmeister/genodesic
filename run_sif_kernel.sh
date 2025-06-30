#!/bin/bash
#
# @description : Securely launches the Genodesic Jupyter kernel inside the
#                Singularity container for VS Code.

# Absolute path to the SIF container file.
SIF_FILE="/home/slaus/Genodesic/genodesic.sif"

# --- DO NOT EDIT BELOW THIS LINE ---

# The last argument passed by Jupyter is the path to the connection file.
CONNECTION_FILE="${@: -1}"

# We must bind the directory containing the connection file from the host
# into the container so the kernel can find it.
BIND_DIR=$(dirname "$CONNECTION_FILE")

# Execute the kernel, enabling NVIDIA GPU access (--nv), binding the
# necessary directory, and passing all arguments.
singularity exec \
  --nv \
  --bind "$BIND_DIR" \
  "$SIF_FILE" \
  python -m ipykernel_launcher -f "$@"