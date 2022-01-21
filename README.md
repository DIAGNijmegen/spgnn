# Structure and Position-Aware Graph Neural Network for Airway Labeling

## News

- The source code (v1.0) is now available.

## Background

This repository is for [Structure and Position-Aware Graph Neural Network for Airway Labeling](https://arxiv.org/abs/2201.04532), by [Weiyi Xie](https://xieweiyi.github.io/), from Diagnostic Image Analysis Group, Radboud University Medical Center.

### Citation
If you find this useful in your research, please consider citing:

	@inproceedings{Xie22,
	    author={Xie, Weiyi and Jacobs, Colin and Charbonnier, Jean-Paul and van Ginneken, Bram},
	    title={Structure and position-aware graph neural network for airway labeling},
	    booktitle={arXiv preprint arXiv:2201.04532},   
	    year={2022},   
	}

### Table Of Contents
- [Introduction](#introduction)
- [Usage](#usage)
- [Main Results](#main-results)

## Introduction

![Screenshot](https://github.com/DIAGNijmegen/spgnn/blob/main/airway_screenshot.png)

We present a novel graph-based approach for labeling the anatomical branches of a given airway tree segmentation. The proposed method formulates airway labeling as a branch classification problem in the airway tree graph, where branch features are extracted using convolutional neural networks (CNN) and enriched using graph neural networks. Our graph neural network is structure-aware by having each node aggregate information from its local neighbors and position-aware by encoding node positions in the graph. The algorithm is also publicly available as an <a href="https://grand-challenge.org/algorithms/airway-anatomical-labeling/">algorithm</a> served on the grand-challenge website.

## Usage
 - Please check `/docker_base/install_files/requirements.in` for the required packages/versions to install.
 - To build a docker image for this algorithm, `cd` into `/docker_base/`, and run `docker build --tag=spgnn .`, please also check the `Dockerfile` regarding which Cuda, Python, and Cudnn versions are installed.
 - Regarding DGL library, in the `Dockerfile`, we build it from the latest sourcecode. We suggest you at least install 0.6.x. 0.4.x cannot be used because of bugs related to the implementation of graph attention networks.
 - Our method is a two-stage method. Therefore, you need to first train the CNN network
 - Before training, you can pre-build and store airway graphs to files. To do so, you run the function `generate_tree_data` in `prepare_data.py`. This function will generate trees and store them to `/derived/conv` under your dataset root path. Once trees are built, you can start training your CNN models. Once CNNs are trained, you can run the function `generate_conv_embeddings` in `prepare_data.py` to store CNN features to files. This allows you to train GNN networks. 
 - For training, run train.py.
 - For testing, run test.py.
 - For visualization using t-SNE (Fig.5 in the paper), run plot_embeddings.py.
 - For each experiment, you use a specfic setting file located in `/exp_settings` as the input argument pass to your training or testing script using `--smp=`. In the setting file, you define training hyper-parameters, network architectures, and paths to your data.

# Main Results
Tab 1. Branch Classification Accuracy (ACC(%)) and Topological Distance (TD) of the CNN, GATS, and the proposed SPGNN methods (in mean ± standard deviation). The overall branch classification accuracy is measured over all target labels on average. Multiply accumulate operations (MACs) and the number of parameters are shown as measures of computational complexity. Testing time consumption indicates the run-time efficiency. The overall topological distance is the average of TD
on all target labels. Boldface denotes the best result. 

|Method     |ACC (%)   |TD|MACS |#Params|Testing time(second)|
|:---------:|:---:|:-----:|:----:|:----:|:----:|
|CNN |83.83±7.37  |2.41±0.67|**6.42G**|**67.49M**|**14.25±9.65**|
|GATS  |89.84±5.44|2.02±0.61|6.62G|69.52M|16.12±8.69|
|**SPGNN**|**91.18±4.97**|**1.80±0.50**|6.67G|70.09M|16.98±9.79|

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Some implementations are inspired by
[LSPE](https://github.com/vijaydwivedi75/gnn-lspe)
