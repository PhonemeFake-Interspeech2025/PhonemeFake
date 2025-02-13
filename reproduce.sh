detection_model=$1
dataset=$2
version=$3

if [ "$detection_model" == "mamba" ]; then
    cd reproduce/RawBMamba_PF
    conda activate mamba_fresh
elif [ "$detection_model" == "conf-3" ]; then
    cd reproduce/RawBMamba
    conda activate fairseq
fi

bash inference.sh $dataset $version
conda deactivate
cd ..
cd ..