# Image Classification on ImageNet with Decentralized Training

## Introduction

The folder contains the experiment code for training ResNet-50 on ImageNet-1K. It includes both AllReduce training and decentralized training.

## Reproduce Experiments

### Python Environment

1. Create a virtual environment using either `virtualenv` or `conda` (**>=3.11**), and activate the environment.

2. Install dependencies
    ```
    # Install latest version of PyTorch (check https://pytorch.org/get-started/locally/ for your platform)
    pip install torch torchvision torchaudio
    # Install dali with cuda-11.0 as it comes with cuda dependencies
    pip install --extra-index-url https://pypi.nvidia.com --upgrade nvidia-dali-cuda110
    # Install other dependencies
    pip install wandb seaborn loguru scipy tqdm tomli-w pydantic
    # Install decentralized training wrapper
    pip install decent-dp
    ```

3. One has to login to wandb for uploading the metrics before runing the experiments. Check this [link](https://docs.wandb.ai/quickstart/) for details.
    ```
    wandb login
    ```

The versions of dependencies under test are listed in [`requirements.txt`](./requirements.txt) (Python version 3.12.8) for reference.

> [!NOTE]
> The codebase of the wrapper for decentralized training (`decent-dp`) is not directly available in this repository, and it's installed as a dependency and the source code is available at [Decent-DP](https://github.com/WangZesen/Decent-DP).

### Prepare Data

Since it needs to sign an agreement for downloading the dataset, only an instruction is provided here.

Download the ImageNet (ILSVRC 2012) dataset from [here](https://www.image-net.org/).

Put the data under `./data/Imagenet` and arrage the files like
```
data/Imagenet/
├── dev
│   └── ILSVRC2012_devkit_t12
├── meta.bin
├── train
│   ├── n01440764
│   ├── n01443537
│   ├── n01484850
│   ├── n01491361
│   ├── n01494475
│   ├── ...
├── val
│   ├── n01440764
│   ├── n01443537
│   ├── n01484850
│   ├── n01491361
│   ├── n01494475
│   ├── ...
```

> [!NOTE]
> Note that the images in validation dataset should also be arranged by classes like in `train`.

> [!WARNING]
> **Avoid the IO bottleneck to achieve best possible efficiency**
> An efficient data loader is one of the key factors to a overall efficient training systems. To avoid the bottleneck cause by disk IO, the data MUST be placed in local SSD otherwise the speedup brought by other optimization techniques (like decentralized training or AMP training) will be hindered. In our experiments on shared clusters, the data is stored in a network drive. `preload` in the training configuration controls whether the data will be copied to local SSDs of compute nodes before the training. To make the preload work, shards of the data needs to be created. See next section for details.

#### (Optional) Create Shards for Fast Preloading

To be added.

### Train

The experiments are conducted on a data center using Slurm as the scheduler. To run the training with four A40 GPUs, 

```
sbatch -A <PROJECT_ACCOUNT> scripts/train/4xA40.sh $(which torchrun) config/data/imagenet.toml config/train/resnet50.toml
```
where `<PROJECT_ACCOUNT>` is the slurm project account.

One can extract the command in [`script/train/4xA40.sh`](./script/train/4xA40.sh) to run seperately if the system is not based on slurm.

The evaulation on the validation set is done along with the training.

