#!/bin/bash

# This script downloads and extracts the Schiebinger et al. 2019 dataset
# from the source provided by the Waddington-OT (wot) tutorial.

# Stop the script if any command fails
set -e

# --- Configuration ---
# The URL for the dataset zip file
DATA_URL="https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE122662&format=file"

# The directory where the data will be stored.
# This creates a 'data' folder in the current directory.
OUTPUT_DIR="Data/Schiebinger"
ARCHIVE_FILE="${OUTPUT_DIR}/Schiebinger.tar"

# --- NEW: Flag to track if a download occurred ---
DOWNLOAD_PERFORMED=0

# --- Main Logic ---

echo "--- Schiebinger Dataset Downloader ---"
# Check if the output directory already exists and is not empty
if [ -d "${OUTPUT_DIR}" ] && [ "$(ls -A "${OUTPUT_DIR}")" ]; then
    echo "Directory ${OUTPUT_DIR} already exists and is not empty."
    echo "Continuing: will attempt to extract/unpack any archives found in this directory."
else
    # Create the output directory if it doesn't exist
    echo "Creating output directory: ${OUTPUT_DIR}"
    mkdir -p "${OUTPUT_DIR}"

    # Use wget to download the file. The -c flag allows resuming an interrupted download.
    echo "Downloading dataset from ${DATA_URL}..."
    wget -c -O "${ARCHIVE_FILE}" "${DATA_URL}"
    
    # --- NEW: Set the flag because a download just happened ---
    DOWNLOAD_PERFORMED=1
fi


# Unpack any tar files found in the output directory
for tarfile in "${OUTPUT_DIR}"/*.tar "${OUTPUT_DIR}"/*.tar.gz "${OUTPUT_DIR}"/*.tgz; do
    if [ -f "$tarfile" ]; then
        echo "Unpacking $tarfile..."
        if [[ "$tarfile" == *.tar.gz ]] || [[ "$tarfile" == *.tgz ]]; then
            tar -xzvf "$tarfile" -C "${OUTPUT_DIR}"
        else
            tar -xvf "$tarfile" -C "${OUTPUT_DIR}"
        fi
    fi
done


# --- NEW: Conditional Cleanup Logic ---
# Check the flag. Only run cleanup if the DOWNLOAD_PERFORMED flag is 1.
if [ "$DOWNLOAD_PERFORMED" -eq 1 ]; then
    echo "Cleaning up downloaded archive..."
    rm "${ARCHIVE_FILE}"
else
    echo "Skipping cleanup as no new download was performed."
fi


echo "---"
# Corrected the path in the final output message
echo "Dataset is located in: ${OUTPUT_DIR}"