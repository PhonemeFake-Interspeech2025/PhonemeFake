# SCL-Deepfake-audio-detection

This is official implementation of our work "Balance, Multiple Augmentation, and Re-synthesis: A Triad Training Strategy
for Enhanced Audio Deepfake Detection"


## Preparing

* Setup conda environment and download wav2vec model by running:
```
bash 00_envsetup.sh
bash 01_download_pretrained.sh
```

* Noise data using in this repo is [MUSAN](https://www.openslr.org/17/) and [RIRS_NOISES](https://www.openslr.org/28/)

Please download those dataset by yourself.

### Dataset
We do not redistributed training data as:
* You should download [ASVspoof 2019](https://doi.org/10.7488/ds/2555) dataset and store bona fide samples online in `DATA/asvspoof_2019_supcon/bonafide`. We do not re-distribute that dataset.
* Vocoded sample can be found in Xin Wang et al. work. We use [voc.v4](https://zenodo.org/record/7314976/files/project09-voc.v4.tar) data. After downloading, you should store vocoded samples in `DATA/asvspoof_2019_supcon/vocoded`.

* Please note that training and dev file MUST be converted into `.wav` file.
* Eval set of ASVSpoof 2019 should be copied (or linked) to `DATA/asvspoof_2019_supcon/eval`
For example:
```
ln -s ./ASVspoof/LA/ASVspoof2019_LA_eval/flac/ DATA/asvspoof_2019_supcon/eval/
```

`DATA` folder should look like this:
```
DATA
├── asvspoof_2019_supcon
│   ├── bonafide
│   │   └── leave_bonafide_wav_here
│   ├── eval
│   │   └── leave_eval_wav_here
│   ├── protocol.txt
│   ├── scp
│   │   ├── dev_bonafide.lst
│   │   ├── test.lst
│   │   └── train_bonafide.lst
│   └── vocoded
│       └── leave_vocoded_wav_here
├── asvspoof_2021_DF
│   ├── flac -> /datab/Dataset/ASVspoof/LA/ASVspoof2021_DF_eval/flac
│   ├── protocol.txt
│   └── trial_metadata.txt
└── in_the_wild
    ├── in_the_wild.txt
    ├── protocol.txt
    └── wav -> /datab/Dataset/release_in_the_wild
```

## Configurations
Configuration should be checked and modified before further training or evaluating. Please read configuration files carefully.

- The MUSAN and RIR_NOISES location should be changed

By default, these configurations is set for training.
## Training
```
CUDA_VISIBLE_DEVICES=0 bash 02_train.sh <seed> <config> <data_path> <comment>
```
For example:
```
CUDA_VISIBLE_DEVICES=0 bash 02_train.sh 1234 configs/conf-3-linear.yaml DATA/asvspoof_2019_supcon "conf-3-linear-1234"
```

# Inference Script

The `inference.sh` script is used to run evaluations on different datasets using specified versions of the data. This script automates the process of running the evaluation and computing the scores for the datasets.

## Usage

To run the script, provide the dataset and version as arguments. For example:

```bash
./inference.sh <dataset> <version>
```

## Arguments

- `<dataset>`: The dataset to be evaluated. Possible values are `ASV21`, `ITW`, `WaveFake`.
- `<version>`: The version of the dataset to be evaluated. Possible values are `original` or `phonemeFake`.

## Example Commands

- To run the evaluation on the `ASV21` dataset with the `original` version:
  ```bash
  ./inference.sh ASV21 original
  ```

- To run the evaluation on the `ITW` dataset with the `phonemeFake` version:
  ```bash
  ./inference.sh ITW phonemeFake
  ```

- To run the evaluation on the `WaveFake` dataset with the `original` version:
  ```bash
  ./inference.sh WaveFake original
  ```

## Script Details

The script defines the directories for the datasets and uses a function to run the evaluation and compute the scores. The function takes the following arguments:

- `config_path`: The path to the configuration file.
- `model_path`: The path to the model file.
- `wav_path`: The path to the WAV files.
- `score_file`: The path to the score file.
- `metadata_path`: The path to the metadata file.

The script checks if the correct number of arguments is provided and then runs the evaluation based on the provided dataset and version arguments.

## Dataset Directories

The dataset directories are defined at the beginning of the script:

```bash
ASV_original_wav_path="/nas/ob_DF_datasets/datasets/ASVspoof2021_DF_eval/wav"
ITW_original_wav_path="/nas/ob_DF_datasets/datasets/In_the_Wild_DF_eval/wav"
WaveFake_original_wav_path="/nas/ob_DF_datasets/datasets/WaveFake/wavs"

ASV_PhonemeFake_wav_path="/home/ugrad-su24/ege/PhonemeFake/A_PhonemeFake_ASV21"
ITW_PhonemeFake_wav_path="/home/ugrad-su24/ege/PhonemeFake/A_PhonemeFake_ITW"
WaveFake_PhonemeFake_wav_path="/home/ugrad-su24/ege/PhonemeFake/A_PhonemeFake_WaveFake"
```

These directories are used in the evaluation function to specify the paths to the WAV files for each dataset and version.

## Customized training and evaluating dataset
Please refer to `datautils/eval_only.py` and `datautils/asvspoof_2019.py` for other eval dataset. For augmentation strategies, please refer to `datautils/asvspoof_2019_augall_3.py`.
# Reference
* [SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing)
* [project-NN-Pytorch-scripts](https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts)
* [audio_augmentor](https://github.com/josebeo2016/audio_augmentor)
* [SupContrast](https://github.com/HobbitLong/SupContrast)
