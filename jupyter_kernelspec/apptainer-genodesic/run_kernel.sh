#!/bin/bash

CONTAINER_PATH="~/Genodesic/genodesic.sif"

exec apptainer exec --nv \
  --bind "$HOME" \
  "$CONTAINER_PATH" \
  python -m ipykernel_launcher "$@"