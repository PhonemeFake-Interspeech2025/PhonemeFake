#!/bin/bash

# Define dataset directories
ASV_original_wav_path="/nas/ob_DF_datasets/datasets/ASVspoof2021_DF_eval/wav"
ITW_original_wav_path="/nas/ob_DF_datasets/datasets/In_the_Wild_DF_eval/wav"
WaveFake_original_wav_path="/nas/ob_DF_datasets/datasets/WaveFake/wavs"

ASV_PhonemeFake_wav_path="/home/ugrad-su24/ege/PhonemeFake/A_PhonemeFake_ASV21"
ITW_PhonemeFake_wav_path="/home/ugrad-su24/ege/PhonemeFake/A_PhonemeFake_ITW"
WaveFake_PhonemeFake_wav_path="/home/ugrad-su24/ege/PhonemeFake/A_PhonemeFake_WaveFake"

# Define configuration and model paths
config_path="configs/conf-3-linear.yaml"
model_path="pretrained/conf-3-linear.pth"

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <dataset> <version>"
    echo "dataset: ASV21, ITW, WaveFake"
    echo "version: original, phonemeFake"
    exit 1
fi

dataset=$1
version=$2

# Run the evaluation based on the provided arguments
if [ "$dataset" == "ASV21" ]; then
    if [ "$version" == "original" ]; then
        CUDA_VISIBLE_DEVICES=0 bash 03_eval.sh $config_path $ASV_original_wav_path 128 $model_path ./scores/ASV_test.txt
        python evaluate_results.py \
        --score_file ./scores/ASV_presented_scores.txt \
        --metadata_path "/home/ugrad-su24/ege/RawBMamba_PF/A.ASVspoof2021DF_eval/ASV21_metadata.txt"
    elif [ "$version" == "phonemeFake" ]; then
        CUDA_VISIBLE_DEVICES=0 bash 03_eval.sh $config_path $ASV_PhonemeFake_wav_path 128 $model_path ./scores/ASV_PF_test.txt
        python evaluate_results.py \
        --score_file ./scores/ASV_PF_presented_scores.txt \
        --metadata_path "/home/ugrad-su24/ege/RawBMamba_PF/A.ASVspoof2021DF_eval/ASV21_PF_metadata.txt"
    else
        echo "Invalid version: $version"
        exit 1
    fi
elif [ "$dataset" == "ITW" ]; then
    if [ "$version" == "original" ]; then
        CUDA_VISIBLE_DEVICES=1 bash 03_eval.sh $config_path $ITW_original_wav_path 128 $model_path ./scores/ITW_test.txt
        python evaluate_results.py \
        --score_file ./scores/ITW_presented_scores.txt \
        --metadata_path "/home/ugrad-su24/ege/RawBMamba_PF/A.ITW_eval/ITW_metadata.txt"
    elif [ "$version" == "phonemeFake" ]; then
        CUDA_VISIBLE_DEVICES=1 bash 03_eval.sh $config_path $ITW_PhonemeFake_wav_path 128 $model_path ./scores/ITW_PF_test.txt
        python evaluate_results.py \
        --score_file ./scores/ITW_PF_presented_scores.txt \
        --metadata_path "/home/ugrad-su24/ege/RawBMamba_PF/A.ITW_eval/ITW_PF_metadata.txt"
    else
        echo "Invalid version: $version"
        exit 1
    fi
elif [ "$dataset" == "WaveFake" ]; then
    if [ "$version" == "original" ]; then
        CUDA_VISIBLE_DEVICES=2 bash 03_eval.sh $config_path $WaveFake_original_wav_path 128 $model_path ./scores/WaveFake_test.txt
        python evaluate_results.py \
        --score_file ./scores/WaveFake_presented_scores.txt \
        --metadata_path "/home/ugrad-su24/ege/RawBMamba_PF/A.WaveFake_eval/WaveFake_metadata.txt"
    elif [ "$version" == "phonemeFake" ]; then
        CUDA_VISIBLE_DEVICES=2 bash 03_eval.sh $config_path $WaveFake_PhonemeFake_wav_path 128 $model_path ./scores/WaveFake_PF_test.txt
        python evaluate_results.py \
        --score_file ./scores/WaveFake_PF_presented_scores.txt \
        --metadata_path "/home/ugrad-su24/ege/RawBMamba_PF/A.WaveFake_eval/WaveFake_PF_metadata.txt"
    else
        echo "Invalid version: $version"
        exit 1
    fi
else
    echo "Invalid dataset: $dataset"
    exit 1
fi
