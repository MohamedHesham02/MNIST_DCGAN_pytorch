# MNIST_GAN_Pytorch

### In this Machine Learning approach, we're focusing on generation of images based on Generative Adverserial Network (GAN) which is specialized in producing fake images that network is trained on.

### GAN approach at all have found success in medical imaging tasks, including medical image enhancement, segmentation, classification, reconstruction, and synthesis. 

### We're using Mnist basic datasets for taking a demo of GAN implementation.

### There are 4 trials of various methods of GAN to evaluate these methods after 8 epochs 

### The 4 Methods are:

 1- Fully Connected Network 
 2- Fully Connected Network with Batch Normalization 
 3- Deep Convolutional Generative Adversarial Networks (DCGAN)
 4- Deep Convolutional Generative Adversarial Networks with Spectral Normalization 

### 1 - Fully Connected Network 
 To train the model from scratch use this command line:

```
python Fully_Connected/Fully_connected.py
```

 To test the model and see the results of network with inputting noise images use this command line:

```
python Fully_Connected/FC_test.py
```

### 2- Fully Connected Network with Batch Normalization
 To train the model from scratch use this command line:

```
python FC_Batch_normalized/FC_wth_batchnorm.py
```
 To test the model and see the results of network with inputting noise images use this command line:

```
python FC_Batch_normalized/FC_BN_test.py
```

### 3- Deep Convolutional Generative Adversarial Networks (DCGAN)
 To train the model from scratch use this command line:

```
python DCGAN/DCGAN.py
```
 To test the model and see the results of network with inputting noise images use this command line:

```
python DCGAN/DCGAN_test.py
```

### 4- Deep Convolutional Generative Adversarial Networks with Spectral Normalization 
 To train the model from scratch use this command line:

```
python DCGAN_spect/DCGAN_spect.py
```
 To test the model and see the results of network with inputting noise images use this command line:

```
python DCGAN_spect/DCGAN_SP_test.py
```
