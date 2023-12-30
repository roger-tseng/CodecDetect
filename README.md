### Getting started
`requirements.txt` must be installed for execution. We state our experiment environment for those who prefer to simulate as similar as possible. 
- Installing dependencies
```
pip install -r requirements.txt
```
- Our environment (for GPU training)
  - Python 3.8.18
  - GCC 11.2.0
  - GPU: 1 NVIDIA Tesla V100
    - About 32GB is required to train AASIST using a batch size of 32
  - gpu-driver: 470.161.03

### Running
```
bash run.sh
```

### Acknowledgements
This repository is built on top of several open source projects. 
- [ASVspoof 2021 baseline repo](https://github.com/asvspoof-challenge/2021/tree/main/LA/Baseline-RawNet2)
-  https://github.com/eurecom-asp/RawGAT-ST-antispoofing
-  https://github.com/clovaai/aasist
