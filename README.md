# From Promise to Practice: Realizing High-performance Decentralized Training

[![arXiv](https://img.shields.io/badge/arXiv-2401.11998-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2410.11998) 
[![OpenReview](https://img.shields.io/badge/OpenReview-Paper-blue)](https://openreview.net/forum?id=lo3nlFHOft)



Welcome to the official repository for **"From Promise to Practice: Realizing High-Performance Decentralized Training."** This repository contains the code and experimental setups for the decentralized training of deep neural networks, including **ResNet, Transformers, and GPT-2**.

## 🚀 Overview

Decentralized training has shown promise in enabling scalability beyond traditional **All-Reduce** approaches. This repository provides implementations and benchmarks of decentralized optimizers, including our **Accumulated Decentralized Adam (AccumAdam)** algorithm, which allows for overlapping computation and communication and great generalization performance as AllReduce training.

Our experiments span across:

- **Image classification** (ResNet-50 on ImageNet-1K)
- **Neural Machine Translation** (Transformer on WMT14 En-De, En-Fr)
- **Language Model Pretraining** (GPT-2 on OpenWebText)

## 📂 Repository Structure

```
Decentralized-Training-Exp/
├── cnn/           # Image classification experiments (ResNet-50 on ImageNet-1K)
├── transformer/   # Neural machine translation experiments (Transformer on WMT14)
├── gpt2/          # Language modeling experiments (GPT-2 on OpenWebText)
├── .gitignore     # Git ignore rules
├── LICENSE        # License
└── README.md      # This file
```

## 🚀 Running Experiments

### **ResNet-50 Training on ImageNet-1K**

Please refer to [`cnn/README.md`](./cnn/README.md) for details.

### **Transformer Training on WMT14**

Please refer to [`transformer/README.md`](./transformer/README.md) for details.

### **GPT-2 Pretraining on OpenWebText**

Please refer to [`gpt2/README.md`](./gpt2/README.md) for details.

## 🥝 PyTorch Extension

The PyTorch extension that facilitates the decentralized training is wrapped as a Python package: **Decent-DP** ([Github](https://github.com/WangZesen/Decent-DP), [PyPI](https://pypi.org/project/decent-dp/)).

## 📊 Benchmark Results

Our experiments demonstrate that decentralized training can achieve **better scalability** and **comparable generalization** to All-Reduce training while reducing communication overhead.


## 📝 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{wang2024decentralized,
  title={From Promise to Practice: Realizing High-Performance Decentralized Training},
  author={Wang, Zesen and Zhang, Jiaojiao and Wu, Xuyang and Johansson, Mikael},
  journal={arXiv preprint arXiv:2410.11998},
  year={2024}
}
```


## 🤝 Contributing

We welcome contributions! Feel free to open an issue or submit a pull request to improve the repository.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

