# Convolutional_Deep_Belief_Network
This code contains how to create a convolutional DBN from stacked convolutional RBM, configure it and train it layerwise. 


## Description of the project
A convolutional deep belief network (CDBN) is a deep network which consists in a stack of convolutional restricted boltzmann machine (CRBM). 
Because the gradient of the network is intractable, a greedy layer-wise training procedure is used. 
More details can be found [here](https://www.cs.princeton.edu/~rajeshr/papers/icml09-ConvolutionalDeepBeliefNetworks.pdf) and [here](https://papers.nips.cc/paper/3048-greedy-layer-wise-training-of-deep-networks.pdf) and [here](https://www.cs.toronto.edu/~hinton/science.pdf).
This project contains 4 files, `CDBN.py`, `CRBM.py`, `DATA_HANDLER.py` (in scr folder) and `cdbn_emnist.ipynb`.
Below is a description of each file, what it does and how to use it.


### Model overview
<img src="https://i.stack.imgur.com/J7FZG.jpg" width="600" height="500">


## How to use
1. `CRBM.py`
This file is the building block of the whole network since it contains the class that is necessary for ONE crbm to function properly. 
Many parameters are included such as dimension of input and hidden, parameter to inialitize hidden unit, whether they are gaussian or not, whether to use probabilistic max pooling, whether to use sparsity, etc.
For one crbm, one can compute its energy, infer the probability forward or backward, draw samples forward or backward, do contrastive divergence. This is the most complicated part of the project since the contrastive divergence does not rely on computation of gradient but rather make use of Gibbs sampling. 

2. `CDBN.py`
This file is the class that represent the whole network and can be composed of several crbm that stacked together. First the network is created empty and then layers can be added successively. A final softmax layer can also be added. After locking the network, the whole network can be trained by training each layer successively.

3. `DATA_HANDLER.py`
This file contains a class that is used to prepare data for a correct loading in the model.

4. `cdbn_emnist.ipynb`
This file is a jupyter notebook containing a full working example of CDBN used on the EMNIST dataset. 
  - The notebook performs training and prediction on the EMNIST dataset. Hyperparameters have been tuned to give a decent accuracy score;
  - It goes further by exploring the internal representation of the model, to see how the weights have been tuned for the classification task. For each layer of the CRBN the filters are visualized and interpreted. Activations on input images are visualized to see how filters highligts different features of the digits/letters;
  - It is then showed how the model effectively builds a hierarchical representation that can be extracted with the hierarchical clustering algorithm and visualized with a dendogram;
  - Then the CDBN is tested against a convolutional neural network to assess and compare robustness to noise injection and semi-supervised learning.

## Improvements and modifications
The [original project](https://github.com/arthurmeyer/Convolutional_Deep_Belief_Network) from was built on `tensorflow 0.12` and `python 3`. I brought it up to date with `python 3.7` and `tensorflow 2.7.0`. I birefly summarize the most relevant modifications I made:
-   replacing the legacy tensorflow functions with the newer ones, adapting the modified behaviour to the task they were originally intended for;
-   fixing a problem of NaN's values that would propagate during training: this was caused by the exponential activation function used in the *probability max pooling* layer which would lead to overflows. I solved this by replacing the exponential function with a similar, but less steep and more "bounded" *softplus* function; 
-   allowing the model to perform *semi-supervised* learning by accepting a dataset in which only a fraction of the data is provided with labels. This is explored in the notebook;
-   implementing auto calculation of the input dimension for the deeper layers from the output dimension of the previous ones.

### Updated requirements
- Python 3.7
- Tensorflow 2.7.0 (makes use of compatibility features to support tensorflow 1 code)
- Numpy


