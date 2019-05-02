# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/final_model.png "Model Visualization"
[training]: ./output_images/training_data.png "Training Image"
[loss30]: ./output_images/loss_30.png "Loss30"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My Approach is using nVidia Autonomous Car Group model.

#### 2. Attempts to reduce overfitting in the model

The model do not contains dropout layers in order to reduce overfitting because the number of epoch is very small (5 epochs)

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 93).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.  I used two laps data of the first track. My training data contains images of center, left, and right cameras. I also added flipped images to my training data.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a lap data and a powerful network: nVidia Autonomous Car Group. On the first track, the car went straight to the lake. I considered lack of training data.

Next, I used two laps data and nVidia model.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (makeModel method in model.py) is shown in the following image.


![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving.   After the collection process, I had 2647 number of data points.

I also used images of left and right cameras. Then, I added flipped images to my training data. 

![alt text][training]

By above process, the number of my training data is 15,882. (2647 * 6)


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

The following picture shows the training.

![alt text][loss30]

When focusing on the validation loss, I can not see the good effect after about 5 epochs.
So, the number of epochs in my code is 5.
