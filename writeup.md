
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/veh_notveh.JPG
[image2]: ./output_images/HOG.JPG
[image3]: ./output_images/spatial.JPG
[image4]: ./output_images/color_hist.JPG
[image5]: ./output_images/HOG2.JPG
[image6]: ./output_images/heatmaps_and_output.JPG
[image7]: ./output_images/search_area.JPG
[video1]: ./output_images/project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The relevant code can be ound in cell "In11" of the notebook VehicleDetectiontracking.ipynb. 

I started by reading in all the `vehicle` and `non-vehicle` images.  Here are two examples from each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to observe what the `skimage.hog()` output looks like.

Here are examples of application of HOG to the gray scale images of arbitrarily selected car and noncar images; with HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

Please notice that for the training of classifier, YUV colorspace and all three channels were used in the project.

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and observed the detection success in the test_video. As the number of orientations used are decreased, the features for car and non-car objects start 
to be similar, and unuseful for classification. A high resolution in orientations is desirable. In the end I decided to stick with the following:

name             | value
------------- | --
Orientations | 9
pix_per_cell | 8
cells_per_block | 2

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using spatial features, color features and hog features. For spatial features, I used a spatial cell size of 32x32.
Here are examples of (not normalized) spatial feature vectors from examples of each class:

![alt text][image3]

An example of the color histogram features is given below. In this example, and later for training of the classifier, YUV colorspace and histogram bin number of 32 was used.

![alt text][image4]

And another example of the hog feature:
![alt text][image5]

As a classifier, I used a linear SVM. I used class `CalibratedClassifierCV` from the scikit-learn library. This classifier not only returns the predicted labels, but also the prediction
probabilities for the labels. I used this feature in order to eliminate some false positives.
I normalized and scaled the feature vectors using a scaler. This prevents the classifier to be biased to specific features.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

For the feature search, I limited my search area to y_start=400, y_end=656. I used the whole x-range for the search. See below image for the search area:

![alt text][image7]

I used four different window sizes for feature search: (64,64), (96,96), (128,128) and (256,256).

For the window sliding, I made use of find_cars function supplied in lecture. I only made some modifications regarding the color spaces. This method was useful as it creates 
the hog image for whole search ares, and scales it according to provided window scales. The window sizes I chose correspond to scales 1, 1.5, 2 and 4 respectively. 

My initial approach was to use smallest window size (64,64) in the topmost part of the search area, i.e between y=[400,528]. However, this resulted in some false negatives
in my pipeline. So I decided to use small windows for the whole search area as well. 

As for overlapping of the search windows, I chose a cell step of 1, in the find_cars function.


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector.
I used heatmaps and predicition probability values from the classifier to overcome false positives  Here are some example images for heatmaps and resulting detection windows:

![alt text][image6]
---

### Video Implementation

####1. Provide a link to your final video output.  
Here's a [link to my video result](./output_images/project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in **each third frame** of the video. The reason I added this downsampling is for limited performance of the 
pipeline and creating new video from images. When all of the frames are used, the processing takes 4.5 hours, even on AWS instance.
This downsampling is not very illogical, as the video is 25fps, which means implemented pipeline update period is around 0.125seconds.

For detection of the vehicles, I used a detection probability threshold. I only used detections with probability>0.9 as positives.
From the positive detections I created a heatmap. The heatmap is thresholded to filter out further false positives. 
Finally in order to smooth the pipeline output, I used a deque to average 5 consecutive heatmaps.
I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the averaged heatmap.  
I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Implemented solution is beneficial to understand how a linear classifier is used to detect objects in the camera stream. However it is certainly not suitable
for realtime implementation. Processing of only one frame took in the order of seconds. My next aim is to survey and implement a CNN approach, as applied for
TrafficSignDetection project, and observe the time performance. 

While the pipeline was performing well to detect vehicles, one can observe that 00:43 when the white car is leaving the frame and not seen completely,
the pipeline fails to classify it as a car. This might be due to the reason that partial images were not present in the training dataset. However, a 
tracking algorithm might be introduced into the pipeline in order to use vehicle motion model and predict the position of the vehicles in order to overcome
such kind of issues. Such a solution may also help for lost objects when the lighting conditions change and the classifier fails to detect.
