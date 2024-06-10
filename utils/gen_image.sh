#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <network_url> <seeds>"
    exit 1
fi

# Assign the arguments to variables
NETWORK_URL=$1
SEEDS=$2

# Change directory
echo "Changing directory to /mnt/volume_lon1_01/gen/stylegan3"
cd /mnt/volume_lon1_01/gen/stylegan3 || exit

# Activate conda environment
echo "Activating conda environment 'stylegan3'"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate stylegan3

# Run the Python script
echo "Running the Python script to generate images"
python gen_images.py --outdir='/mnt/volume_lon1_01/Project/movs_classification_2023/generated' --trunc=1 --seeds="$SEEDS" --network="$NETWORK_URL"

# Deactivate conda environment
echo "Deactivating conda environment 'stylegan3'"
conda deactivate

echo "Script execution completed."
