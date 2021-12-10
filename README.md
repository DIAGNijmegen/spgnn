# Structure and position aware graph neural network for airway labeling

This is an implementation of our airway labeling algorithm (paper under review).
The algorithm takes an airway segmentation map as the input, where an airway tree and branch segments were extracted and presented in different colors.


## Installation

 - Please check /docker_base/install_files/requirements.in for the required packages.
 - check our dockerFile regarding how to setup docker environment, including the information regarding cuda and cudnn versions when we setup experimental environment.
 - Regarding DGL library, we suggest you install 0.6.x. 0.4.x cannot be used because of bugs related to the implementation of graph attention networks.


## Usage

Before training, you can pre-build and store airway graphs to files so training can be faster. To do so, you run the function generate_tree_data in prepare_data.py. This function will generate trees and store them to /derived/conv from your archive root. Once trees are ready, you can start training your CNN models. Once CNNs are trained, you can run the function generate_conv_embeddings in prepare_data.py to store CNN features to files. This allows you to train GNN more efficiently.  

- For training, run train.py
- For testing, run test.py
- For visualization of t-SNE, run plot_embeddings.py
- For each experiment, you use a specfic setting file located in /exp_settings. In the setting file, you define training hyperparameters, network architectures, and data locations.

## License
[MIT](https://choosealicense.com/licenses/mit/)
