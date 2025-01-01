## Pixel Classification of Remote Sensing Data - Assessing Impervious and Pervious Surfaces for Rainfall Analysis

### Abstract



**Author:**         Julian Kraft  
**Tutor:**          Dr. Johann Junghardt  
**Institution:**    Zurich University of Applied Sciences (ZHAW)
**Program:**        BSc Natural Resource Sciences  
**Project:**        Term Paper 2  
**Date:**           2025-01-23

**Data:** 

**Paper:** [link](LaTeX/main.pdf)
**Visualizations:** [link](code/visualizations.ipynb)

### Repository Content

This repository provides all the relevant code, data and training logs as well as the LaTeX source code and
and all visualizations used in the term paper.

### Repository Structure

- `LaTeX/`: LaTeX source code of the term paper
- `code/`: Python code for the CNN model and the evaluation


### Usage

To set up the environment run the following command:

```bash
conda env create -f environment.yml
```

After switching to the `./code` directory and activate the environment and run the following command:

```bash
conda activate sa2
pip install -e .
```

To run an experiment run from the `./code` directory:

```bash
python run_model.py --device='gpu' -o --dev_run --batch_size=50
```

Available arguments are:




