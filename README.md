# Text-Enhanced Graph Attention Hashing for Cross-Modal Retrieval


### 1. Introduction

This is the source code of paper "Text-Enhanced Graph Attention Hashing for Cross-Modal Retrieval".


We have uploaded the complete source code and the generated hash codes to the repository for testing. The test.zip archive contains the test scripts we provided for evaluation purposes..

### 2. Requirements

- python 3.11
- pytorch 2.1.0
- ...

Device:
NVIDIA RTX-3090Ti GPU with 128 GB RAM


### 3. Get dataset

You should generate the following `*.mat` file for each dataset. The structure of directory `./data` should be:
```
    dataset
    ├── coco
    │   ├── caption.mat 
    │   ├── index.mat
    │   └── label.mat 
    ├── flickr25k
    │   ├── caption.mat
    │   ├── index.mat
    │   └── label.mat
    └── nuswide
        ├── caption.mat
        ├── index.mat 
        └── label.mat
```

Please preprocessing dataset to appropriate input format.

### 4. Train

After preparing the python environment, and dataset, we can train the TEGAH model.

### 5. Test

unzip test.zip

python test.py

## Acknowledegements
[GCDH](https://github.com/Zjut-MultimediaPlus/GCDH)
