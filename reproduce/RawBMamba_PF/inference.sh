#!/bin/bash

# Define dataset directories
ASV_wav_path="/nas/ob_DF_datasets/datasets/ASVspoof2021_DF_eval/wav/"
ASV_PF_wav_path="/home/ugrad-su24/ege/PhonemeFake/A_PhonemeFake_ASV21/"
ITW_wav_path="/nas/ob_DF_datasets/datasets/In_the_Wild_DF_eval/wav/"
ITW_PF_wav_path="/home/ugrad-su24/ege/PhonemeFake/A_PhonemeFake_ITW/"
WaveFake_wav_path="/nas/ob_DF_datasets/datasets/WaveFake/wavs/"
WaveFake_PF_wav_path="/home/ugrad-su24/ege/PhonemeFake/A_PhonemeFake_WaveFake/"

run_evaluation() {
    local test_script=$1
    local score_file=$2
    local model_folder=$3
    local protocol_path=$4
    local wav_path=$5
    local metadata_path=$6

    python $test_script \
        -e $score_file \
        --model_folder $model_folder \
        --protocol_path $protocol_path \
        --wav_path $wav_path

    python evaluate_scores.py \
        --score_file $score_file \
        --metadata_path $metadata_path
}

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <dataset> <version>"
    echo "dataset: ASVspoof2021DF, ITW, WaveFake, all"
    echo "version: original, phonemeFake"
    exit 1
fi

dataset=$1
version=$2

run_all() {
    if [ "$version" == "original" ]; then
        run_evaluation \
            "./A.ASVspoof2021DF_eval/21DF_test.py" \
            "./scores/ASV21_DF_scores.txt" \
            "./models/" \
            "./A.ASVspoof2021DF_eval/ASV21_eval_protocol.txt" \
            "$ASV_wav_path" \
            "/home/ugrad-su24/ege/RawBMamba/A.ASVspoof2021DF_eval/ASV21_metadata.txt"

        run_evaluation \
            "./A.ITW_eval/ITW_test.py" \
            "./scores/ITW_scores.txt" \
            "./models/" \
            "./A.ITW_eval/ITW_eval_protocol.txt" \
            "$ITW_wav_path" \
            "/home/ugrad-su24/ege/RawBMamba/A.ITW_eval/ITW_metadata.txt"

        run_evaluation \
            "./A.WaveFake_eval/WaveFake_test.py" \
            "./scores/WaveFake_scores.txt" \
            "./models/" \
            "./A.WaveFake_eval/WaveFake_eval_protocol.txt" \
            "$WaveFake_wav_path" \
            "/home/ugrad-su24/ege/RawBMamba/A.WaveFake_eval/WaveFake_metadata.txt"
    elif [ "$version" == "phonemeFake" ]; then
        run_evaluation \
            "./A.ASVspoof2021DF_eval/21DF_test.py" \
            "./scores/ASV21_DF_PF_scores.txt" \
            "./models/" \
            "./A.ASVspoof2021DF_eval/ASV21_PF_eval_protocol.txt" \
            "$ASV_PF_wav_path" \
            "/home/ugrad-su24/ege/RawBMamba/A.ASVspoof2021DF_eval/ASV21_PF_metadata.txt"

        run_evaluation \
            "./A.ITW_eval/ITW_test.py" \
            "./scores/ITW_PF_scores.txt" \
            "./models/" \
            "./A.ITW_eval/ITW_PF_eval_protocol.txt" \
            "$ITW_PF_wav_path" \
            "/home/ugrad-su24/ege/RawBMamba/A.ITW_eval/ITW_PF_metadata.txt"

        run_evaluation \
            "./A.WaveFake_eval/WaveFake_test.py" \
            "./scores/WaveFake_PF_scores.txt" \
            "./models/" \
            "./A.WaveFake_eval/WaveFake_PF_eval_protocol.txt" \
            "$WaveFake_PF_wav_path" \
            "/home/ugrad-su24/ege/RawBMamba/A.WaveFake_eval/WaveFake_PF_metadata.txt"
    else
        echo "Invalid version: $version"
        exit 1
    fi
}

if [ "$dataset" == "ASVspoof2021DF" ]; then
    if [ "$version" == "original" ]; then
        run_evaluation \
            "./A.ASVspoof2021DF_eval/21DF_test.py" \
            "./scores/ASV21_DF_scores.txt" \
            "./models/" \
            "./A.ASVspoof2021DF_eval/ASV21_eval_protocol.txt" \
            "$ASV_wav_path" \
            "/home/ugrad-su24/ege/RawBMamba/A.ASVspoof2021DF_eval/ASV21_metadata.txt"
    elif [ "$version" == "phonemeFake" ]; then
        run_evaluation \
            "./A.ASVspoof2021DF_eval/21DF_test.py" \
            "./scores/ASV21_DF_PF_scores.txt" \
            "./models/" \
            "./A.ASVspoof2021DF_eval/ASV21_PF_eval_protocol.txt" \
            "$ASV_PF_wav_path" \
            "/home/ugrad-su24/ege/RawBMamba/A.ASVspoof2021DF_eval/ASV21_PF_metadata.txt"
    else
        echo "Invalid version: $version"
        exit 1
    fi
elif [ "$dataset" == "ITW" ]; then
    if [ "$version" == "original" ]; then
        run_evaluation \
            "./A.ITW_eval/ITW_test.py" \
            "./scores/ITW_scores.txt" \
            "./models/" \
            "./A.ITW_eval/ITW_eval_protocol.txt" \
            "$ITW_wav_path" \
            "/home/ugrad-su24/ege/RawBMamba/A.ITW_eval/ITW_metadata.txt"
    elif [ "$version" == "phonemeFake" ]; then
        run_evaluation \
            "./A.ITW_eval/ITW_test.py" \
            "./scores/ITW_PF_scores.txt" \
            "./models/" \
            "./A.ITW_eval/ITW_PF_eval_protocol.txt" \
            "$ITW_PF_wav_path" \
            "/home/ugrad-su24/ege/RawBMamba/A.ITW_eval/ITW_PF_metadata.txt"
    else
        echo "Invalid version: $version"
        exit 1
    fi
elif [ "$dataset" == "WaveFake" ]; then
    if [ "$version" == "original" ]; then
        run_evaluation \
            "./A.WaveFake_eval/WaveFake_test.py" \
            "./scores/WaveFake_scores.txt" \
            "./models/" \
            "./A.WaveFake_eval/WaveFake_eval_protocol.txt" \
            "$WaveFake_wav_path" \
            "/home/ugrad-su24/ege/RawBMamba/A.WaveFake_eval/WaveFake_metadata.txt"
    elif [ "$version" == "phonemeFake" ]; then
        run_evaluation \
            "./A.WaveFake_eval/WaveFake_test.py" \
            "./scores/WaveFake_PF_scores.txt" \
            "./models/" \
            "./A.WaveFake_eval/WaveFake_PF_eval_protocol.txt" \
            "$WaveFake_PF_wav_path" \
            "/home/ugrad-su24/ege/RawBMamba/A.WaveFake_eval/WaveFake_PF_metadata.txt"
    else
        echo "Invalid version: $version"
        exit 1
    fi
elif [ "$dataset" == "all" ]; then
    run_all
else
    echo "Invalid dataset: $dataset"
    exit 1
fi