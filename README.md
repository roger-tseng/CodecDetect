## Train Fake Speech Detectors on CodecFake

<p align="center">  
    <a href="https://arxiv.org/abs/2406.07237">Paper</a>,
    <a href="https://huggingface.co/datasets/rogertseng/CodecFake">Dataset</a>,
    <a href="https://codecfake.github.io/">Project Page</a>
</p>
<p align="center">  
    <i>Interspeech 2024</i>
</p>

**TL;DR**: We show that better detection of deepfake speech from codec-based TTS systems can be achieved by training models on speech re-synthesized with neural audio codecs.
We also release the CodecFake dataset for this purpose.

### Data Download

See [[here]](https://github.com/roger-tseng/CodecFake). 
If using ZIP files is preferred, please use [this commit (3abd4aa)](https://github.com/roger-tseng/CodecDetect/commit/3abd4aa50e4c0c1a696430ab8a942c99e7278721).

### Environment
`requirements.txt` must be installed for execution. We state our experiment environment for those who prefer to simulate as similar as possible. 
```
pip install -r requirements.txt
```
- Our environment (for GPU training)
  - Python 3.8.18
  - GCC 11.2.0
  - GPU: 1 NVIDIA Tesla V100 32GB
  - gpu-driver: 470.161.03

### Running
1. Training

About 32GB GPU RAM is required to train AASIST using a batch size of 32. <br/>
Available codecs are listed in the script. 
```bash
bash train.sh <codec_name>
```
2. Evaluation

First, add paths to trained checkpoints to the script.
Then adjust subsets to evaluate on.
```bash
bash eval_all.sh
```

### Acknowledgements
This repository is built on top of several open source projects. 
- [ASVspoof 2021 baseline repo](https://github.com/asvspoof-challenge/2021/tree/main/LA/Baseline-RawNet2)
-  https://github.com/eurecom-asp/RawGAT-ST-antispoofing
-  https://github.com/clovaai/aasist
