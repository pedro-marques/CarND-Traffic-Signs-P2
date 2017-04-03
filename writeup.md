#**Traffic Sign Recognition**

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./images/bar-chart.png "Visualization"
[image4]: ./32x32_signs/children_crossing_32_32.png "Traffic Sign 1"
[image5]: ./32x32_signs/end_of_speed_limit_30km2_32_32.png "Traffic Sign 2"
[image6]: ./32x32_signs/no_entry2_32_32.png "Traffic Sign 3"
[image7]: ./32x32_signs/pedestrians_32_32.png "Traffic Sign 4"
[image8]: ./32x32_signs/road_work_32_32.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/pedro-marques/CarND-Traffic-Signs-P2/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used python's len function and numpy's unique function to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed across the different classes

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fifth code cell of the IPython notebook.

I have decided against converting the images to grayscale because I believe that color is an important factor in traffic sign classification, and by converting the image to grayscale I would ignore the color feature and would only consider color intensity.

So I only used normalization and brightness augmentation as the preprocessing techniques, I normalized the image data because the neural network works well with small numbers, some of the images from the training set were darker than others so using the brightness augmentation function seemed like a good idea to improve accuracy and feature learning ...

![alt text][image2]

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the seventh code cell of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by using the train_test_split function from the sklearn.model_selection. ...

My final training set had 107576 number of images. My validation set and test set had 4410 and 12630 number of images.

The sixth code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because the data set was very unbalanced in terms of number of examples per class, some classes had as much as 2000 examples and others only 180. To add more data to the the data set, I set an upper limit of images to all the classes and for each image of a given class I would create n number of copies until the number for that class reached the upper limit.

I have experimented with vxy10's image transformation and brightness augmentation for the data augmentation, but the best validation, test accuracy and also better probabilities using the images downloaded from the internet were obtained when I augmented the data without performing any modification to the image and only applied the normalization using the formula: Xnorm =( X/127.5)–1 - X being the input.

Here is the link to vxy10's -> https://github.com/vxy10/ImageAugmentation


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the eighth cell of the ipython notebook.

My final model consisted of the following layers:

| Layer         		    |     Description	        					            |
|:---------------------:|:---------------------------------------------:|
| Input         		    | 32x32x3 RGB image   							            |
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	  |
| RELU					        |	Activation function										        |
| Max pooling	      	  | 2x2 stride, valid padding, outputs 14x14x6 	  |
| Convolution 3x3	      | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU                  | Activation function                           |
| Max pooling	      	  | 2x2 stride, valid padding, outputs 5x5x16 	  |
| Flatten               | Input 5x5x16, Outputs 400 (1D)                |
| Fully connected		    | Input 400, Outputs 120                        |
| RELU                  | Activation function                           |
| Fully Connected       | Input 120, Outputs 84                         |
| RELU                  | Activation function                           |
| Fully Connected       | Input 84, Outputs 43                          |



####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eighth cell of the ipython notebook.

To train the model, I used the AdamOptimizer, a batch size of 128, 10 epochs, a learing rate of 0.001

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth and tenth cells of the Ipython notebook.

My final model results were:
* validation set accuracy of 0.951
* test set accuracy of 0.938

If a well known architecture was chosen:
* What architecture was chosen?
I have used the LeNet-5 architecture
* Why did you believe it would be relevant to the traffic sign application?
I have seen it previously being used to classify images.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
My fear of the approach I have taken is that the training set and the validation set might be too similar, but the test accuracy was actually quite high using this method, even higher than when using the image transformation technique so I have decided to stick with it. I had a a validation accuracy of 99% percent and test accuracy of 92% with this method, there appears be overfitting occurring here, but when testing on the images downloaded from the web it predicted with better accuracy than when I used the transformation. I have done tons of tests with this possibilities, using the default validation data provided, splitting the data, using only the augmented brightness, this proved the best option for me.

UPDATE: Decided to leave out the splitting and use the given validation set, the results were satisfactory for me and I don't feel like the training and validation set are similar.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because the same sign on the train data is of different color, and the figures are on the opposite direction

The second one might also be difficult because the sign does not exist on the train data the only one closer to it would be End of speed limit (80km/h) class id 6

The third one I believe will be a little easier than the previous two because it is pretty much similar to the ones on the train data set

The fourth one is also hard because again there is no data similar to it on the train set

The fifth one should be dead on ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the twelfth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			            |     Prediction	        					            |
|:---------------------:|:---------------------------------------------:|
| Children Crossing     | Road work   									                |
| End of speed limit    | End of speed limit (80km/h)                   |
  (30km/h) Zone
| Pedestrians				    | Turn left ahead										            |
| No entry	      		  | No entry					 				                    |
| Road work			        | Road work     							                  |


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This compares unfavorably to the accuracy on the test set of 93% but I have given some complicated images, because some of them don't exist on the training data set, but the ones that were similar got predicted correctly, I was hoping the children crossing sign, although different from the ones from train data would still be predicted accurately...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 12th cell and for displaying the predictions is on the 15th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a road work sign (probability of 0.9) - bad signal, although the image may have a similar look to the road work sign, ultimately it is a children crossing sign, so the model was wrong, the correct sign is not in the probabilities. The top five soft max probabilities were

| Probability         	|     Prediction	        					            |
|:---------------------:|:---------------------------------------------:|
| .95         			    | Road work   									                |
| .04     				      | Bicycles crossing 										        |
| .00					          | Road narrows on the right											|
| .00	      			      | Slippery road					 				                |
| .00				            | Dangerous curve to the right      						|


For the second image, the model has no idea of which sign this could possibly be, it is a end of speed limit 30 km/h zone sign, which does not exist on the test, validation and training data, the sign with highest probability selected by the model was the keep right sign but with only 0.11, I was hoping it would go and predict the end of speed limit 80 km/h on the top five but sadly no. The top five soft max probabilities were

| Probability         	|     Prediction	        					            |
|:---------------------:|:---------------------------------------------:|
| .34         			    | End of speed limit (80km/h                    |
| .28     				      | End of all speed and passing                  |
| .10					          | Speed limit (20km/h)											    |
| .08	      			      | Roundabout mandatory				 				          |
| .05				            | Speed limit (60km/h)     							        |

The third image is a no entry sign but with a red X on it, frankly I have never seen the sign like that, but the model has predicted with fair accuracy that it was a no entry sign (probability of 0.9). The top five soft max probabilities were

| Probability         	|     Prediction	        					            |
|:---------------------:|:---------------------------------------------:|
| .97         			    | No entry 									                    |
| .01     				      | Stop 										                      |
| .00					          | Beware of ice/snow											      |
| .00	      			      | Priority road					 				                |
| .00				            | No passing for vehicles over 3.5 metric tons  |


The fourth image is a pedestrian sign but unlike any other the model has ever seen, so it has some difficulties on predicting which sign it really is, it predicts with fairly high probability the turn left ahead sign (0.7) and quite frankly wrong, but given the color of the sign and the figures position I can understand. The top five soft max probabilities were

| Probability         	|     Prediction	        					            |
|:---------------------:|:---------------------------------------------:|
| .70         			    | Turn left ahead 									            |
| .23     				      | Keep right 										                |  
| .04					          | Ahead only											              |
| .00	      			      | Turn right ahead					 				            |
| .00				            | Go straight or left     							        |


The fifth and final image is a road work sign and the model predicted with absolute certainty that it was a road work sign (probability 1.0), well done. The top five soft max probabilities were

| Probability         	|     Prediction	        					            |
|:---------------------:|:---------------------------------------------:|
| 1.0         			    | Road work 									                  |
| .00     				      | Bicycles crossing 										        |  
| .00					          | Traffic signals											          |
| .00	      			      | Dangerous curve to the left					 				  |
| .00				            | Road narrows on the right    							    |
