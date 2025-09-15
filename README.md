<div align="center">

<h1>
  <span style="font-weight:900; letter-spacing:2px;">
    <span style="color:#76b900; font-weight:900;">ZeroMSF</span>:
    <span style="color:#76b900;">Zero</span>-shot 
    <span style="color:#76b900;">M</span>onocular 
    <span style="color:#76b900;">S</span>cene 
    <span style="color:#76b900;">F</span>low Estimation in the Wild
  </span>
</h1>

<p style="font-size:1.1em; color:#ff6b6b; font-weight:bold; margin:10px 0;">
  🏆 CVPR 2025 Oral & Best Paper Award Candidate
</p>

<p>
  <a href="https://arxiv.org/pdf/2501.10357">
    <img src="https://img.shields.io/badge/-Paper-b31b1b?style=flat-square&logo=arxiv&logoColor=white">
  </a>
  &nbsp;  &nbsp;
  <a href="https://research.nvidia.com/labs/lpr/zero_msf/">
    <img src="https://img.shields.io/badge/Project%20Page-76b900?style=flat-square&logo=nvidia&logoColor=white">
  </a>
</p>

<p align="center">
  <a href="https://lynl7130.github.io/"><b>Yiqing Liang<sup>1,2</sup></b></a> &nbsp;  &nbsp;
  <a href="https://abadki.github.io/"><b>Abhishek Badki<sup>1,*</sup></b></a> &nbsp;  &nbsp;
  <a href="https://suhangpro.github.io"><b>Hang Su<sup>1,*</sup></b></a> &nbsp;  &nbsp;
  <a href="https://jamestompkin.com/"><b>James Tompkin<sup>2</sup></b></a> &nbsp;  &nbsp;
  <a href="https://oraziogallo.github.io"><b>Orazio Gallo<sup>1</sup></b></a>
</p>
<p style="font-size:0.95em; color:#888; margin-bottom:-0.5em;">* indicates equal contribution</p>

<br>

<p style="font-size:0.95em; color:#888; margin-bottom:-0.5em;"><sup>1</sup> <img src="media/nvidialogo.png" height="20" alt="NVIDIA" style="vertical-align: middle;"/> &nbsp; <sup>2</sup> <img src="media/brownlogo.svg" height="20" alt="Brown University" style="vertical-align: middle;"/></p>

<br>
<br>

</div>

ZeroMSF is a model for monocular scene flow that jointly estimates geometry and motion in a zero-shot feedforward fashion. It leverages a joint geometry-motion estimation architecture and a scale-adaptive optimization strategy. The model exhibits strong generalization abilities, benefiting from a diverse training set compiled from six data sources.


## Installation

Testing environment: 
* Hardware: Single NVIDIA RTX 3090 GPU 
* System: Ubuntu 24.04, CUDA 12.6


Pull code with all submodules: 
```bash
git clone --recurse-submodules git@github.com:NVlabs/zero-msf.git
```

Install with conda: 

```bash
# conda environment
conda create -y --name zero_msf python=3.10
conda activate zero_msf

# install pytorch
# here showing pytorch 2.6 and cuda 12.6
# see https://pytorch.org/get-started/locally/ for other versions
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.cuda.is_available())" # make sure can see CUDA by printing True

# install most dependencies
pip install -r requirements.txt

# (optional) install curope
conda install -y -c conda-forge libstdcxx-ng
cd zmsf/mast3r/dust3r/croco/models/curope
python setup.py build_ext --inplace 
cd -
```


## Demo

First, download model checkpoint from [Google Drive](https://drive.google.com/drive/folders/1hnTlB8WYgF5jWlvX1mXgXG1Y4f-9Q2E7) and put it under `checkpoints/`. 

Input sample should be provided as a folder containing two images. If more than two images are found within the input folder, only the first two are used. Some samples can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1pom7XwwgLnJo10yqtzzE57JS2M5pa20f). 

* Run demo: 
```bash
model_config=zmsf_shift+dynamicreplica+vkitti+kubric+pointodyssey+spring_adap_241106

python demo.py \
     --model zmsf/configs/${model_config}.yaml \
     --data PATH_INPUT_FOLDER \
     --output PATH_OUTPUT_FOLDER
```

Estimated point clouds and scene flow can be found under `PATH_OUTPUT_FOLDER`. 

* Visualize results with viser: 
```bash
python visualize_viser.py \
     --steps 50 \
     --data PATH_OUTPUT_FOLDER
```


## Citation
```bibtex
@InProceedings{liang2025zeroshot,
    author    = {Liang, Yiqing and Badki, Abhishek and Su, Hang and Tompkin, James and Gallo, Orazio},
    title     = {Zero-Shot Monocular Scene Flow Estimation in the Wild},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {21031-21044}
}
```
