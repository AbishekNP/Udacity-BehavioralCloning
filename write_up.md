## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* To teach a simulated version of car to replicate your style of driving.
* A Convolutional Neural Network is trained using Keras to capture the driving pattern.
* Train and Validate that model.
* Your car must drive 1 succesful lap around the simulated track.


[//]: # (Image References)

[image1]: ./images/rgb.png "RGB image"
[image2]: ./images/nvidia.png "Architecture"
[image3]: ./images/cropped.png "Cropped Image"
[image4]: ./images/flipped.png "Augmented Image"
[image5]: ./images/loop.png "Process image"



# [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

# Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

# Writeup / README


# Included files:

* model.py
* video.mp4
* model.h5
* write_up.md 


# 1. Brief view of the training data:

![alt text][image1]

## Model Architecture and Training strategy:

### 1) I've implemented model proposed by Nvidia in their "End to End Deep Learning for Self Driving Cars paper" :

The model architecture is as follows:
![alt text][image2] 

The methodology followed in this task:
![alt text][image5]

I've initially cropped the top 70px and bottom 25px from the training image to
reduce the noisy impact of the external features on the model. This includes features like sky, bushes etc....
Then, I've normalized the image to a standard scale.

The network consists of:

[Conv_2D_1] ==> Filters=24 ; kernel_size=(5,5) ; Stride=2 ; Activation='relu';
[BatchNormalization] layer

[Conv_2D_2] ==> Filters=36 ; kernel_size=(5,5) ; Stride=2 ; Activation='relu';
[BatchNormalization] layer

[Conv_2D1_3] ==> Filters=48 ; kernel_size=(5,5) ; Stride=2 ; Activation='relu';
[BatchNormalization] layer

[Conv_2D_4] ==> Filters=64 ; kernel_size=(3,3) ; Stride=3 ; Activation='relu';
[BatchNormalization] layer

[Conv_2D_5] ==> Filters=64 ; kernel_size=(3,3) ; Stride=3 ; Activation='relu';
[BatchNormalization] layer

[FLATTEN] layer

[Dense_1] ==> units=1164 ; Activation='relu'

[Dense_2] ==> units=100 ; Activation='relu'
[Dropout] ==> drop_rate=0.2

[Dense_3] ==> units=50 ; Activation='relu'

[Dense_4] ==> units=10 ; Activation='relu'

[Dense_5] ==> units=1 


### 2. Attempts to reduce overfitting in the model:

* Dropout layers were added at the end of 2nd Dense layer.
* BatchNormalization was performed at regular intervals so the model can focus only on what really matters.
* The model was fed with good training data, including few recovery scenes.
* Image augmentation was done on the fly to get more input data for the model to generalize on.


### 3.Model Parameter tuning:

* I've used the Adam optimizer for this model. 'Adam' is an Adaptive Optimizer, so not much manual work was required in that 
context.
* I also tried the SGD + Nesterov Momentum instead of Adam, but decided to stick with the forme after performance comparison.
* I also gave a try for the ELU activation function but stuck with ReLu due to the same above mentioned reasons.

### 3:Appropriate training data:

* This probably played the biggest factor to get a good run.
* Suggested methods to collect efficient steering angle data was using a joystick, but I had to stick to keyboard input
due to hardware limitations. This made my job a bit harder than usual.
* I collected data from 3 different cameras, including the 'Center', 'Left', 'Right' , just so that the model could generalize well on the input data.
* Input data was cropped to reduce noise from the outliers.

![alt text][image4]

### 4:Solution Design Approach:

* I initally passed in a cropped and normalized training images to the 1st Convolutional Layer consisting of 24 filters.
* This was followed by 4 more CNN's[Keeping in mind the dimensions of the input images].
* Later I decided to add few BatchNormalization layers at regular intervals just to make sure that my model focuses only on the important features.
* One of my major observation from the training was that validation_loss is not a sole factor to decide the performance of a model. I've seen models with a bit 1.x validation_loss outperform models with a vaidation_loss in ranges of 0.0x .
I've used image augmentation to increase the training data to the model to both reduce overfitting as well as to generalize the model.
* Even after many refinements my model was having a hard time dealing with sharp edges due to the keyboard entered input. But after tons of parameter trials and newwly refined input data, I was able to get way better results.
* At the end of the process, my car was able to succesfully complete a lap around the 1st track.

### 5:Training and Validation data:

* I've split 20% of my input data for validation purposes. This was done using: Keras train_test_split function.
* My final dataset had about 17k images from the 3 cameras, of which 20% was considerd for validation. Moreover I also generated augmented images on the fly for better model generalization.
* All my input input images were cropped accordingly to reduce noise and were normalized by the scale :(X / 255.0) - 0.5 , where X is each of the input image. The reason to use 225.0 to scale the imags is due to the reason that the pixel value of each image lies between 0 and 255. 
* The input images read from cv2.imread() is taken in the BGR format whereas the drive.py file reads images in the RGB format.




# Discussion
## 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* The main matter of concern to me was the input data. Limited to keyboard generated data was a bit hard with maintaining the accuracy. 

* I would also prefer 'Transfer Learning' to be used here. Some of the best alternates includes: ResNet-50, Inception-V3, VGG-19 etc..... Thes architectures are pretrained on the largest available image dataset called the 'ImageNet', which consistes of over a million images. Just freezing a few of their end layers and adding our custom layers could make our life easier. Ways to implement them are given in great detail in the Keras documentation.

* I had to run the simulator locally due to some lagging issues with my online one. This made it really hard to try out different possibilites as I was also not able access my GPU.

* Though my model successfully finishes the lap, it seemed to be making sudden turns at sharp turnings. Better training data could reasonably fix this issue and smoothen the steering angle predictions. 




   
