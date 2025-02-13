# RawBMamba
This repository provides the overall framework for training and evaluating audio anti-spoofing systems proposed in [RawBMamba: End-to-End Bidirectional State Space Model for Audio Deepfake Detection](https://arxiv.org/abs/2406.06086). The repository is tailored to be used as a benchmark DeepFake detection model in our paper.

### Mamba Installation
[Mamba](https://github.com/state-spaces/mamba)

### Training

To train RawBMamba:
```
python train.py -o ./save_path/
```
# Inference Script

The `inference.sh` script is used to run evaluations on different datasets using specified versions of the data. This script automates the process of running the evaluation and computing the scores for the datasets.

## Usage

To run the script, provide the dataset and version as arguments. For example:

```bash
./inference.sh <dataset> <version>
```

## Arguments

- `<dataset>`: The dataset to be evaluated. Possible values are `ASVspoof2021DF`, `ITW`, `WaveFake`
- `<version>`: The version of the dataset to be evaluated. Possible values are `original` or `phonemeFake`.

## Example Commands

- To run the evaluation on the `ASVspoof2021DF` dataset with the `original` version:
  ```bash
  ./inference.sh ASVspoof2021DF original
  ```

- To run the evaluation on the `ITW` dataset with the `phonemeFake` version:
  ```bash
  ./inference.sh ITW phonemeFake
  ```

## Script Details

The script defines the directories for the datasets and uses a function `run_evaluation` to run the evaluation and compute the scores. The function takes the following arguments:

- `test_script`: The path to the test script.
- `score_file`: The path to the score file.
- `model_folder`: The path to the model folder.
- `protocol_path`: The path to the protocol file.
- `wav_path`: The path to the WAV files.
- `metadata_path`: The path to the metadata file.

The script checks if the correct number of arguments is provided and then runs the evaluation based on the provided dataset and version arguments.

## Dataset Directories

The dataset directories are defined at the beginning of the script:

```bash
ASV_wav_path="/nas/ob_DF_datasets/ASVspoof2021_DF_eval/wav/"
ASV_PF_wav_path="/home/ugrad-su24/ege/PhonemeFake/A_PhonemeFake_ASV21/"
ITW_wav_path="/nas/ob_DF_datasets/datasets/In_the_Wild_DF_eval/wav/"
ITW_PF_wav_path="/home/ugrad-su24/ege/PhonemeFake/A_PhonemeFake_ITW/"
WaveFake_wav_path="/nas/ob_DF_datasets/datasets/WaveFake/wavs/"
WaveFake_PF_wav_path="/home/ugrad-su24/ege/PhonemeFake/A_PhonemeFake_WaveFake/"
```

These directories are used in the `run_evaluation` function to specify the paths to the WAV files for each dataset and version.

### Pre-trained models

We provide all the pre-trained models under the models folder.

### References

```bibtex
@inproceedings{liu2023leveraging,
  title={Leveraging positional-related local-global dependency for synthetic speech detection},
  author={Liu, Xiaohui and Liu, Meng and Wang, Longbiao and Lee, Kong Aik and Zhang, Hanyi and Dang, Jianwu},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}

@article{mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}

```
