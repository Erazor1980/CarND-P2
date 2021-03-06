#**Traffic Sign Recognition** 

##Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/dataDistribution.PNG "Data Distribution"
[image2]: ./images/tf_1.png
[image3]: ./images/tf_2.png
[image4]: ./images/tf_3.png
[image5]: ./images/tf_4.png
[image6]: ./images/tf_5.png
[image7]: ./images/softmax.PNG

---

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/Erazor1980/CarND-P2/blob/master/Traffic_Sign_Classifier_LC.ipynb)

###Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3 (changed afterwards to 32x32x1 -> grayscale)
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed

![alt text][image1]

###Design and Test a Model Architecture

All images were converted to grayscale and normalized, so that the pixel values lie between -1 and 1. The mean pixel value is around 0, what should lead to better training performance.


My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   							| 
| Convolution 4x4     	| 1x1 stride, same padding 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride				|
| Convolution 4x4	    | 1x1 stride, same padding      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride				|
| Fully connected		| Input = 400, output = 120        									|
| RELU					|												|
| Fully connected		| Input = 120, output = 84        									|
| RELU					|												|
| Fully connected		| Input = 84, output = 43        									|
 

	
To train the model, I used the AdamOptimizer from TensorFlow. The batch size was 64, the number of epochs 30. For the learning rate I chose 0.001.

The architecture was a slightly modified version of the LeNet, which already had good results on the MNIST data set. To reach at least 93% accuracy some layers have been modified. Some thoughts on the reasoning behind choosing this architecture:
* Convolutions are great in learning complex objects: first layer can recognize basic lines/curves, next could detect shapes/blobs and finally large, more complex objects can be recognized
* Pooling layer are helpful in generalizing feature representations, which can reduce overfitting
* Rectified Linear Units (RELUs) are used because they are pretty simple and have simple derivatives, besides this it showed its effectivness in the LeNet already
* Cross entropy loss function fits very good to our problem: only 1 of 43 classes is correct, so we can simply compare the result vector to the 1-hot encoded label vector

My final model results were:
* validation set accuracy of 94.1%
* test set accuracy of 91.5%

###Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image2] ![alt text][image3] ![alt text][image4] 
![alt text][image5] ![alt text][image6]

Possible difficulties for the classifier:

* The first image (50 km/h) might be difficult to classify because it is a bit rotated around 2 axes.
* The stop sign could cause some problems because of its background and some noise in the image.
* The no passing sign is a bit blurry and another sign starts directly above it.
* The priority road sign looks good, but the different backgrounds (sky/ground) could possible cause problems
* The last traffic sign has a good quality (one background, centred, etc.)

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 50 km/h      		| 50 km/h   									| 
| Stop     		| Stop 										|
| No passing		| No passing											|
| Priority Road	      	| Priority Road					 				|
| Right-of-way at the next intersection			| Right-of-way at the next intersection      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 91.5%.

For the first image, the model is relatively sure that this is a 50 km/h sign (probability of 0.26, the second best vote has 0.06 for 100 km/h ). The top five on the logits are

| Probability         	|     Prediction	        		| 
|:---------------------:|:---------------------------------------------:| 
| .26         		| 50 km/h   					| 
| .06     		| 100 km/h					|
| .05			| Roundabout mandatory				|
| .01	      		| Wild animals crossing			 	|
| .01			| Go straight or left				|


For the second image, the model is sure that this is a stop sign (0.46, second best vote has 0.15). The top five on the logits are

| Probability         	|     Prediction	        		| 
|:---------------------:|:---------------------------------------------:| 
| .46         		| Stop       					| 
| .15     		| Turn right ahead				|
| .07			| Keep right					|
| .03	      		| Yield					 	|
| .004			| No entry      					|

For the third image, the model predicted no passing with 0.21 probability. The top five on the logits are

| Probability         	|     Prediction	        		| 
|:---------------------:|:---------------------------------------------:| 
| .21         		| No passing 					| 
| .11     		| No entry 					|
| .05			| Go straight or left				|
| .05	      		| Slippery Road				 	|
| .02			| End of no passing				|

For the fourth and fifth images, the model is pretty sure what they are:
0.45 for priority road (second best only 0.03!) and 0.57 for Right-of-way at the next intersection (second best 0.36 for the pedestrians sign).

Using soft max on the logit predictions leads to 1.0 probability almost on each traffic sign... The soft max results are shown in the image below:
![alt text][image7]
