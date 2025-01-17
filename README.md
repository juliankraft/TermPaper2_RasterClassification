## Pixel Classification of Remote Sensing Data - Assessing Impervious and Pervious Surfaces

### Keywords
classification, deep learning, machinelearning, remote sensing, land cover classification, perviousness, surface sealing

### Abstract

Understanding the distribution of impervious and pervious surfaces is critical 
for effective urban planning, environmental management, and rainfall impact analysis. 
This study explores the use of convolutional neural networks (CNN) for 
pixel-based classification of aerial remote sensing data to assess surface sealing. 
Leveraging high-resolution SwissImage RS data, the analysis employs a simplified 
ResNet-18 architecture adapted for four-channel inputs, including RGB and 
near-infrared bands. A comprehensive workflow was developed, encompassing 
data preprocessing, augmentation, and hyperparameter tuning. The best-performing 
model achieved a classification accuracy of 0.927 for simplified surface perviousness, 
demonstrating the potential of deep learning to improve upon traditional 
geoprocessing methods. While challenges such as mixed pixels and class imbalances remain, 
this research highlights promising avenues for future advancements 
in remote sensing through the integration of advanced neural architectures and self-supervised learning.

**Author:**         Julian Kraft<br>  
**Tutor:**          Dr. Johann Junghardt<br>  
**Institution:**    Zurich University of Applied Sciences (ZHAW)<br>
**Program:**        BSc Natural Resource Sciences<br>  
**Project:**        Term Paper 2<br>  
**Date:**           2025-01-23<br>

**Paper:** [link](./LaTeX/main.pdf)<br>
**Visualizations:** [link](./code/analysis/visualizations.ipynb)

### Repository Content

This repository provides all the relevant code, data and training logs as well as the LaTeX source code and
and all visualizations used in the term paper.

### Repository Structure

- `LaTeX/`: LaTeX source code of the term paper
- `code/`: Python code for the CNN model and the evaluation
- `hpc_submit/`: Scripts to submit the training jobs to the HPC

### Environment

The environment used to run this model and the evaluation was created using Anaconda. The config file is available as `environment.yml`.

Additionally, the code developed in this project can be installed as a developer package. To do so, run:

```bash
pip install -e ./code
```

### License

This repository is licensed under the **CC0 1.0 Universal (Public Domain Dedication)**. 

To the extent possible under law, the authors of this repository have waived all copyright and related or neighboring rights to this work. 

For more details, see the [LICENSE](./LICENSE) file or visit the [Creative Commons Legal Code](https://creativecommons.org/publicdomain/zero/1.0/legalcode).


