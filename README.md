# Deep Reinforcement Learning for Online Optimal Execution Strategies

Micheli, Alessandro & Monod, Mélodie (2024). Deep Reinforcement Learning for Online Optimal Execution Strategies. arXiv. https://doi.org/10.48550/ARXIV.2410.13493

## Warranty
Imperial makes no representation or warranty about the accuracy or completeness of the data nor that the results will not constitute in infringement of third-party rights. Imperial accepts no liability or responsibility for any use which may be made of any results, for the results, nor for any reliance which may be placed on any such work or results.

## Cite

```bibtex
@misc{micheli2024,
      title={Deep Reinforcement Learning for Online Optimal Execution Strategies}, 
      author={Alessandro Micheli and Mélodie Monod},
      year={2024},
      eprint={2410.13493},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.13493},
      doi={https://doi.org/10.48550/arXiv.2410.13493}
}
```

## System Requirements
* macOS or UNIX
* This release has been checked on Red Hat Enterprise Linux release 8.5 (Ootpa) and macOS Sonoma 14.1.2

## Installation
A `yml` file is provided and can be used to build a conda virtual environment containing all dependencies. Create the environment using:
```bash
cd deep_rl_liquidation
conda env create -f deep_rl_liquidation.yml
```

## Usage 

</details>

<details>
<summary> Reproduce results of experiment "Convergence to Optimal Execution Strategy" </summary>

### 1. Setup
First, specify the following directories at the top of the `submit_jobs_experiment_1.sh` file:

* Repository Directory (`INDIR`): The directory where the repository is located.
* Output Directory (`OUTDIR`): The directory where the results will be stored.

```bash
INDIR="/Users/melodiemonod/git/deep_rl_liquidation"
OUTDIR="/Users/melodiemonod/projects/2024/deep_rl_liquidation"
```

Second, open a terminal and navigate to the repository directory, then execute the `submit_jobs_experiment_1.sh` script:

```bash
cd deep_rl_liquidation
bash submit_jobs_experiment_1.sh
```

### 2. Running Experiments
The script will generate folders in the output directory, each containing a bash script for an experiment.

Go to the output directory, locate the experiment folder and navigate into it. Run the experiment by executing the bash script within that folder. For example:
```bash
cd $OUTDIR
cd experiment_1-exponential_decay_kernel
bash experiment_1-exponential_decay_kernel.sh
```

Repeat these steps for each experiment folder created in `$OUTDIR`.

### 3. Plot results

Open the Jupyter notebook `plots/plot_experiment_1.ipynb`, update the results path to match your output directory, and execute each code block sequentially.

Open the Jupyter notebook `plots/plot_auxiliary_qfunction_experiment.ipynb`, update the results path to match your output directory, and execute each code block sequentially.


</details>

<details>
<summary> Reproduce results of experiment "Online Learning in a Dynamic Environment" </summary>

To reproduce these results, you must have first run the "Convergence to Optimal Execution Strategy" experiment for the exponential decay kernel (previous section) as the algorithm leverages pre-trained weights and memory.

### 1. Setup
First, specify the following directories at the top of the `submit_jobs_experiment_2.sh` file:

* Repository Directory (`INDIR`): The directory where the repository is located.
* Output Directory (`OUTDIR`): The directory where the results will be stored.

```bash
INDIR="/Users/melodiemonod/git/deep_rl_liquidation"
OUTDIR="/Users/melodiemonod/projects/2024/deep_rl_liquidation"
```

Second, open a terminal and navigate to the repository directory, then execute the `submit_jobs_experiment_2.sh` script:

```bash
cd deep_rl_liquidation
bash submit_jobs_experiment_2.sh
```

### 2. Running Experiments
The script will generate folders in the output directory, each containing a bash script for an experiment.

Go to the output directory, locate the experiment folder and navigate into it. Run the experiment by executing the bash script within that folder. For example:
```bash
cd $OUTDIR
cd experiment_2-exponential_decay_kernel_decrease
bash experiment_2-exponential_decay_kernel_decrease.sh
```

Repeat these steps for each experiment folder created in `$OUTDIR`.

### 3. Plot results

Open the Jupyter notebook `plots/plot_experiment_2.ipynb`, update the results path to match your output directory, and execute each code block sequentially.




