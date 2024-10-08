ED-Copilot
=========================
This repository contains the source code and datasets for [ED-Copilot: Reduce Emergency Department Wait Time with Language Model Diagnostic Assistance](https://arxiv.org/pdf/2402.13448) accepted to ICML 2024.
## Requirements and Setup
MIMIC-IV-ED can be downloaded from https://physionet.org/content/mimic-iv-ed/2.2

MIMIC-IV can be downloaded from https://physionet.org/content/mimiciv/2.2/

After downloading MIMIC dataset, then put them into corresponding data folder\
Create a new Anaconda environment and install setup:
~~~
conda create --name ed_copilot python=3.11.5
conda activate ed_copilot
pip install -r requirements.txt
~~~
### 1. Benchmark Data Generation
~~~
python3 benchmark/extract_master_dataset.py --mimic4_path "../data" --output_path "../data"
~~~
### 2. Data Processing and Splitting
~~~
cd data
python3 extract_lab_results.py
python3 merge_lab_results.py
python3 split.py
~~~
### 3. Supervised Finetuning (For multiple GPUs)
~~~
torchrun --standalone --nproc-per-node=gpu main_sft.py
~~~
### 4. Reinforcement Learning
~~~
python3 main.py
~~~
### 5. ED-Copilot Test
The trained SFT and RL checkpoint for critical outcome can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1Jl0c2euAFwUy7O6GblWss38pRycBHj5t) and put under `/outputs/critical/`
~~~
python3 main.py --mode test
~~~
## Acknowledge
Some parts of our code are adapted from the [MIMIC-IV-ED benchmark](https://github.com/nliulab/mimic4ed-benchmark) repository. 
## Citations
Please cite the following paper if you find the benchmark and code helpful for your research.
```
@misc{sun2024edcopilot,
      title={ED-Copilot: Reduce Emergency Department Wait Time with Language Model Diagnostic Assistance}, 
      author={Liwen Sun and Abhineet Agarwal and Aaron Kornblith and Bin Yu and Chenyan Xiong},
      year={2024},
      eprint={2402.13448},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
