ED-Copilot
=========================
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
### 4. ED-Copilot Test
~~~
python3 main.py --mode test
~~~
## Acknowledge
Some parts of our code are adapted from the [MIMIC-IV-ED benchmark](https://github.com/nliulab/mimic4ed-benchmark) repository. 

