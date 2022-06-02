# How To Train Your Occamy

The `net_mnist.c` is the first network that implements a full training routine, which can be simulated in _banshee_ . 
As the name states we are working on the MNIST dataset using the following network configuration:

1. Simple Linear Layer with SoftMax activation
2. SGD Backward pass

The weights and biases are initialized using the data obtained from a PyTorch Golden Model.