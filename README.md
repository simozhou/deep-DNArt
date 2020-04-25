# deep-DNArt
repository of the code produced during the Copenhagen Bioinformatics Hackaton

## Project description
Both art and DNA constist of a dual essence: if on one side they are unique, personal, they stand as herald of our very own personality, on the other they retain traits that are common to all human being; the need to communicate, the emotions we feel, the pathologies we carry and the peculiarities of our traits are not far apart elements. Are what makes us humans, both unique and unrecognizable, a *serial print that changes all the times*. 

And as unique-serial prints as we are, who is the artist that best could tell us apart? Art and DNA come together to test the ability of machines to descern uniqueness and essence of human being. 

### Karyospectrum: vintage biology is the new black
Albeit being a strongly dynamic process, the creation of these art pieces was driven by one main focus: embed idiosyncracy on picture. So we decided to put up a machine GAN who would learn how to draw beautiful paintings from noise, and then feed to the *generator* SNPs data so to see the art that would be created from them. During the whole process we were seeing the outputted pictures as blocks of ~16-20 squares, all ordered. This "piling" of pictures reminded us of the karyotype, the process by which photographs of chromosomes are taken in order to determine the chromosome complement of an individual, including the number of chromosomes and any abnormalities [1]. This idea of "sorting chromosomes" inspired us to *update* karyotyping with 22 (+ the sex chromosome which we excluded for ease of simplicity) machine generated pictures from the chromosomes SNPs. These pictures are wildly abstract, similar to a collaboration between Pollock and Warhol (or just slightly messy if we're being honest)!! In their uniqueness, they entail the same textures and colors. Here we've found back the aims that we set at the beginning of the journey, and thus decided the the art piece was complete.

### Techinques and structure
A Deep Convolutional Adversarial Network was built loosely based on the code of [2], which gave us incredible insights on how to structure the network and on hyperparameter tweaking. Here reported the structure of the two networks:

```
DISCRIMINATOR
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_5 (Conv2D)            (None, 128, 128, 32)      2432      
_________________________________________________________________
leaky_re_lu_5 (LeakyReLU)    (None, 128, 128, 32)      0         
_________________________________________________________________
dropout_6 (Dropout)          (None, 128, 128, 32)      0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 64, 64, 64)        51264     
_________________________________________________________________
leaky_re_lu_6 (LeakyReLU)    (None, 64, 64, 64)        0         
_________________________________________________________________
dropout_7 (Dropout)          (None, 64, 64, 64)        0         
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 32, 32, 128)       204928    
_________________________________________________________________
leaky_re_lu_7 (LeakyReLU)    (None, 32, 32, 128)       0         
_________________________________________________________________
dropout_8 (Dropout)          (None, 32, 32, 128)       0         
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 32, 32, 256)       819456    
_________________________________________________________________
leaky_re_lu_8 (LeakyReLU)    (None, 32, 32, 256)       0         
_________________________________________________________________
dropout_9 (Dropout)          (None, 32, 32, 256)       0         
_________________________________________________________________
flatten_2 (Flatten)          (None, 262144)            0         
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 262145    
_________________________________________________________________
activation_6 (Activation)    (None, 1)                 0         
=================================================================


GENERATOR
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_4 (Dense)              (None, 98304)             39419904  
_________________________________________________________________
batch_normalization_5 (Batch (None, 98304)             393216    
_________________________________________________________________
activation_7 (Activation)    (None, 98304)             0         
_________________________________________________________________
reshape_2 (Reshape)          (None, 64, 64, 24)        0         
_________________________________________________________________
dropout_10 (Dropout)         (None, 64, 64, 24)        0         
_________________________________________________________________
up_sampling2d_3 (UpSampling2 (None, 128, 128, 24)      0         
_________________________________________________________________
conv2d_transpose_5 (Conv2DTr (None, 128, 128, 12)      7212      
_________________________________________________________________
batch_normalization_6 (Batch (None, 128, 128, 12)      48        
_________________________________________________________________
activation_8 (Activation)    (None, 128, 128, 12)      0         
_________________________________________________________________
up_sampling2d_4 (UpSampling2 (None, 256, 256, 12)      0         
_________________________________________________________________
conv2d_transpose_6 (Conv2DTr (None, 256, 256, 6)       1806      
_________________________________________________________________
batch_normalization_7 (Batch (None, 256, 256, 6)       24        
_________________________________________________________________
activation_9 (Activation)    (None, 256, 256, 6)       0         
_________________________________________________________________
conv2d_transpose_7 (Conv2DTr (None, 256, 256, 3)       453       
_________________________________________________________________
batch_normalization_8 (Batch (None, 256, 256, 3)       12        
_________________________________________________________________
activation_10 (Activation)   (None, 256, 256, 3)       0         
_________________________________________________________________
conv2d_transpose_8 (Conv2DTr (None, 256, 256, 3)       228       
=================================================================

```

Keras [3] on Tensorflow GPU was the library of choice.
The network was trained on various subsets of data, but the one who shown best results was the `landscape` dataset, consisting of 3608 images of natural and urban landascapes.

The final model was trained for a total of 1100 epochs, and the objective for stopping training was mostly based on Discriminator loss and accuracy measures and *prettyness* of the pictures produced.

### Future perspectives
Of course this is just the beginning of what might be a beautiful art form! WWith larger datasets and a little more time & patience in fine-tuning the parameters we're sure we could make something even prettier than this!



[1] https://en.wikipedia.org/wiki/Karyotype
[2] https://towardsdatascience.com/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0
[3] https://keras.io/
