#!/bin/bash

# Name of the Conda environment
ENV_NAME="phonemeFake"
# Path to the environment.yml file
ENV_FILE="environment.yml"

# Function to check if the Conda environment exists
function conda_env_exists() {
    conda env list | grep -q "^${ENV_NAME} "
}

# Check if the Conda environment exists
if conda_env_exists; then
    echo "Conda environment '${ENV_NAME}' already exists."
else
    echo "Conda environment '${ENV_NAME}' does not exist. Creating it..."
    if [[ -f "${ENV_FILE}" ]]; then
        conda env create -f "${ENV_FILE}"
        echo "Environment '${ENV_NAME}' created successfully."
    else
        echo "Error: '${ENV_FILE}' not found. Cannot create environment."
        exit 1
    fi
fi

# Activate the Conda environment
echo "Activating Conda environment '${ENV_NAME}'..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# Check if the activation was successful
if [[ $? -ne 0 ]]; then
    echo "Failed to activate Conda environment '${ENV_NAME}'."
    exit 1
fi

# Run the scripts in order
for script in stage1_transcriptionWhisper.py stage2_infidelingTextGPT.py stage3_dfsynthesis.py; do
    echo "Running ${script}..."
    if [[ -f "${script}" ]]; then
        if [[ "${script}" == "stage1_transcriptionWhisper.py" ]]; then
            python "${script}" --device cuda:7 --datasetDir "/nas/ob_DF_datasets/PhonemeFake/WaveFake"
        elif [[ "${script}" == "stage2_infidelingTextGPT.py" ]]; then
            python "${script}" --root_directory "/nas/ob_DF_datasets/PhonemeFake/WaveFake"
        elif [[ "${script}" == "stage3_dfsynthesis.py" ]]; then
            python "${script}" --device cuda:7  --datasetDir "/nas/ob_DF_datasets/PhonemeFake/WaveFake"
        fi
        if [[ $? -ne 0 ]]; then
            echo "Error: ${script} failed to execute."
            exit 1
        fi
    else
        echo "Error: ${script} not found."
        exit 1
    fi
done

echo "All scripts executed successfully."