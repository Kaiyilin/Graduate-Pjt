# Graduate-Pjt
The summary to the work I have done in graduate school

## Project 1: Classification and visualisation of chemotherapy induced cognitive impairment in Volumetric Convolutional Neural Network     



## Project 2: One step image registration for Positron Emission Tomography (PET) image using GAN

In this project, I used general advasarial network (GAN) to simplify the PET and MRI image coregistration
</br>
All the image data used in this project are from [ADNI dataset](http://adni.loni.usc.edu), in this study I used 100 Raw PET 3D Neuro Dynamic images and their coresponding Co-Registered Processed PET images.

### Phase 1: Check the image shape, range and distribution

### Phase 2: Data_Cleaning, Data_preprocessing

From Phase 1, I know the data is complicated.</br>
To simplify the data, I selected the data with identical distribution and the image size less than (128, 128, 128) </br>
Then standardised Raw PET 3D Neuro Dynamic images.</br>

### Phase 3: Construct a pix2pix model using Tensorflow

The original paper of pix2pix can be seen at [here](https://arxiv.org/abs/1611.07004)</br>
In my project, I modified the generator into a self-constructed volumetric [U-Net](https://arxiv.org/abs/1505.04597)</br>
The discriminator is another self-constructed volumetric Convolutional Neural Network (CNN)

Fig. Generator
![](Images/loss.png)

Fig. Discriminator
![](Images/loss.png)

### Phase 4: Evaluate the model

Fig. Results
![](Images/loss.png)
