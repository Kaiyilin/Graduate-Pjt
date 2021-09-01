# Project 2: One step image registration for Positron Emission Tomography (PET) image using GAN

In this project, I used general advasarial network (GAN) to simplify the PET and MRI image coregistration
</br>
All the image data used in this project are from [ADNI dataset](http://adni.loni.usc.edu), in this study I used 100 Raw PET 3D Neuro Dynamic images and their coresponding Co-Registered Processed PET images.

### Phase 1: Check the data

Fig 1. The data distribution of raw images </br>
<img src="./Data_distribution/Raw_standardised_resample/kdeplot_all.png" alt="./Data_distribution/Raw_standardised_resample/kdeplot_all.png" width="600"/>


I know that the range for each image are different.

### Phase 2: Data_Cleaning, Data_preprocessing

From Phase 1, I know the data is complicated.</br>
To simplify the data, I selected the data with identical distribution and the image size less than (128, 128, 128) </br>
Then standardised Raw PET 3D Neuro Dynamic images based on the preproceesing steps that listed on ADNI website.</br>

Here's the results of the pre-processed data

Fig 2. The data distribution of preprocessed raw data and target data </br>
<img src="./Data_distribution/Raw_standardised_resample/kdeplot_007_S_1206_20120220.nii.png" alt="./Data_distribution/Raw_standardised_resample/kdeplot_007_S_1206_20120220.nii.png" width="600"/>

<img src="./Data_distribution/comparison/kdeplot_002_S_4225_20131015.nii.png" alt="./Data_distribution/comparison/kdeplot_002_S_4225_20131015.nii.png" width="600"/>


### Phase 3: Construct a pix2pix model using Tensorflow

The original paper of pix2pix can be seen at [here](https://arxiv.org/abs/1611.07004)</br>
In my project, I modified the generator into a self-constructed volumetric [U-Net](https://arxiv.org/abs/1505.04597)</br>
The discriminator is another self-constructed volumetric Convolutional Neural Network (CNN)

The overall architectures of generators and discriminators can be seen at [here](./Data_distribution/Report/)



Fig 3. The learning curve of pix2pix model </br>

<img src="./Data_distribution/itpjt/Figure_1.png" alt="./Data_distribution/itpjt/Figure_1.png" width="600"/>


### Phase 4: Evaluate the model
Finally, here's the comparison between the predicted image and the target normalised image

Fig. Results </br>
<img src="./Evaluation/testResult1/TestResult1.gif" width="600"/>
