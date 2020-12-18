# On Episodes, Prototypical Networks, and Few-Shot Learning

#### --> Work in progress - code will be released soon <---

This is the codebase for the paper *[On Episodes, Prototypical Networks, and Few-Shot Learning](https://arxiv.org/abs/2012.09831)*, by Steinar Laenen and Luca Bertinetto.
A preliminary version of this work appeared as an oral presentation at (NeurIPS 2020 workshop on meta-learning)[https://meta-learn.github.io/2020/].

## Usage
### With conda

* `conda env create -f environment.pytorch13.yml`
* `source activate few_shot_NCA`
* `pip install -e .`

NOTE: if you are having issues with conda while re-installing the environment, e.g. old packages are used instead of the ones specified by the environment.yml file, try to remove (using conda) pytorch and torchvision and re-installing them with the appropriate cuda-toolkit, e.g. `conda install pytorch torchvision cudatoolkit=10.0 -c pytorch`.

### Data preperation
For miniimagenet, the train/val/test split files need to be downloaded from the [SimpleShot Repo](https://github.com/mileyan/simple_shot/tree/master/split/mini), and should be copied to `<REPO_ROOT>/split/`.

### 2. Download Datasets
For any google drive link, if you want to download it from the command line, you can do that by using:
```angular2
gdown --id FILEID -O FILENAME
```
The FILEID can be found in the link itself, and the FILENAME can be found by clicking the link.

--> When you are done downloading the dataset(s), remove `example` from `./dataset_configs/<dataset>.yaml.example` and edit accordingly. This file won't be committed.

### 2.1 Mini-ImageNet
You can download the dataset from https://drive.google.com/open?id=0B3Irx3uQNoBMQ1FlNXJsZUdYWEE

When setting the `data` argument in `./dataset_configs/miniimagenet.yaml.example`, make sure that it points to the `images` folder:`path/to/miniimagenet/images/`.

### 2.2 Tiered-ImageNet
You can download the dataset from https://drive.google.com/file/d/1g1aIDy2Ar_MViF2gDXFYDBTR-HYecV07/view.
After downloading and unzipping this dataset, you have to run the follow script to generate split files.
```angular2
python src/utils/tieredImagenet.py --data path-to-tiered --split split/tiered/
```

When setting the `data` argument in `./dataset_configs/tieredimagenet.yaml.example`, make sure that it points to the `data` folder:`path/to/tiered-imagenet/data/`.

### 2.4 CIFAR-FS
You can download the dataset from https://drive.google.com/file/d/1pTsCCMDj45kzFYgrnO67BWVbKs48Q3NI/view.
For convenience, it can be downloaded from the command line using 

After downloading and unzipping this dataset, you have to run the follow script to generate split files.
```angular2
python ./scripts/cifarfs_split.py --data path-to-cifar100 --split split/cifar100/
```

When setting the `data` argument in `./dataset_configs/CIFARFS.yaml.example`, make sure that it points to the `images` folder:`path/to/cifar100/images/`.

NOTE: If any of these download links are broken, these datasets can be recreated from the original papers that proposed them: [miniImageNet](https://arxiv.org/abs/1606.04080). [CIFAR-FS](https://arxiv.org/abs/1805.08136), and [tieredImageNet](https://arxiv.org/abs/1803.00676). We put all the images into a single folder, and use .csv files for the train.csv, val.csv, and test.csv splits where there is one row for the image filename and a corresponding row for the label. 

### 3 Loading and Evaluating Models

To train a model, first change directory: `cd ./scripts` and then run:

```angular2
python ./start_training.py
```

 `./src/utils/configs/configuration.py` contains all the arguments that can be specified.


### 4 Run experiments
Experiment scripts are all in `./scripts/bash_scripts/`. Note that some scripts might need to be adjusted to account for which GPUs are used for training.

For the **batch size experiments** run: 
* `./bash_scripts/batch_expm_cifar.sh`
* `./bash_scripts/batch_expm_miniimagenet.sh`

For the **ablation experiments** run: 
* `./bash_scripts/abl_all_cifar.sh` 
* `./bash_scripts/abl_all_miniimagenet.sh` 

To run the **subsampling experiment**:
* `./bash_scripts/sample_NCA_pairs.sh SAMPLE="0.01 0.05 0.1 0.2 0.4 0.8" SEED="0 1 2 3 4"`

To run the experiments to generate the tables at the end of Section 3, run:
* `./bash_scripts/nca_rn12_bestminiimagenet.sh GPU='X,X' SEED="0 1 2 3 4"`
* `./bash_scripts/nca_rn12_cifar.sh GPU='X,X' SEED="0 1 2 3 4"`
* `./bash_scripts/nca_rn12_tiered.sh GPU='X,X' SEED="0 1 2 3 4"`
* `./bash_scripts/protonew_rn12_cifar.sh GPU='X,X' SEED="0 1 2 3 4"`
* `./bash_scripts/protoold_miniimagenet.sh GPU='X,X' SEED="0 1 2 3 4"`
* `./bash_scripts/protoold_tiered.sh GPU='X,X' SEED="0 1 2 3 4"`
* `./bash_scripts/protold_cifar.sh GPU='X,X' SEED="0 1 2 3 4"`

#### How to get accuracy + conf interval over 5 seeds
Results for all 10.000 episodes get saved in `./results/numpy_results`. We have an example script in `./scripts/extract_mean_conf.py`, where the correct regex for the experiment needs to be written to extract the mean + conf interval for the test and validation set.
