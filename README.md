# Sartorius Cell Instance Segmentation

Neurological disorders, including neurodegenerative diseases such as Alzheimer's and brain tumors, are a leading cause of death and disability across the globe. However, it is hard to quantify how well these deadly disorders respond to treatment. One accepted method is to review neuronal cells via light microscopy, which is both accessible and non-invasive. Unfortunately, segmenting individual neuronal cells in microscopic images can be challenging and time-intensive. Accurate instance segmentation of these cells—with the help of computer vision—could lead to new and effective drug discoveries to treat the millions of people with these disorders

Current solutions have limited accuracy for neuronal cells in particular. In internal studies to develop cell instance segmentation models, the neuroblastoma cell line SH-SY5Y consistently exhibits the lowest precision scores out of eight different cancer cell types tested. This could be because neuronal cells have a very unique, irregular and concave morphology associated with them, making them challenging to segment with commonly used mask heads. 

The aim for this competition is to detect and delineate distinct objects of interest, the neuronal cell types. 

[Link to competition](https://www.kaggle.com/c/sartorius-cell-instance-segmentation)



## Metric
mean average precision at different intersection over union (IoU) thresholds. Essentially the better the model is able to detect the segmentation area over the groundtruth the better the precision. 

## Contents
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)

## Installation
### Requirements
- CUDA
- detectron2
- pytorch==1.10
```
pip install -e .
```

## Getting started
1. Data setup
```
kaggle competitions download -c sartorius-cell-instance-segmentation
```
2. Unzip data into `data` folder in root project directory
3. From root project directory train the model
```
python experiments/mask_rcnn/train.py
```

## Project Structure
```

sartorius-cell-instance-segmentation
├── cell_segmentation       # cell_segmentation library.
|   ├── data                # data loading, transforms etc.
|   ├── models              # model templates, specifically utilising and sourcing well known libraries.
|   └── utils               # useful scripts for visualisation and any useful transforms.
├── data                    # data directory where sartorius data is extracted to.
├── experiments             # model experiments.
├── notebooks               # Example, usage and exploration notebooks.
└── tests                   # Unit tests.

```

## TODO Checklist
- [x] Structure dataset
- [ ] Data exploration
- [x] Baseline results on one epoch
- [x] Custom configuration
- [x] Training
- [x] Evaluation
- [x] Submission output
- [ ] Resolve submission issue
- [ ] Package up python packages for submission
- [ ] Feature Engineering
- [ ] Augmentation
- [ ] Cross Validation
- [ ] GCP training
