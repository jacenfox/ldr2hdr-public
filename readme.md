# ICCV 2017
-------

This code implements the algorithm introduced in:

* Jinsong Zhang and Jean-François Lalonde, Learning High Dynamic Range from Outdoor Panoramas, International Conference on Computer Vision (ICCV), 2017.

This code takes as input a single LDR omnidirectional panorama, and converts it to HDR automatically.

For more details, please see our project webpage: http://vision.gel.ulaval.ca/~jflalonde/publications/projects/learningHDR/

*Important*: if you use this code, please cite the paper above!


## Getting started
Execute the `ldr2hdr.py` to generate HDR image.

Download the data from our [project webpage](http://vision.gel.ulaval.ca/~jflalonde/publications/projects/learningHDR/#data).
The provided data contains all the LDR and HDR images used in the training and test. 
You may also need to download the [SUN360 dataset](http://vision.princeton.edu/projects/2012/SUN360/) to train the domain adaptation model. Example images can be found in the `./examples` folder which have been aligned by centering the sun.

The network is defined in `ldr2hdr_net.py`. Two pre-trained models are also provided in the `model_*` folder.

## Requirements
Our model is trained with [TensorFlow](https://www.tensorflow.org/); and [OpenEXR](http://www.openexr.com/) is used to write HDR images.
