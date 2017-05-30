**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/input-frame.png
[image2]: ./output_images/hot-windows.png
[image3]: ./output_images/heat-map.png
[image4]: ./output_images/final-detection.png
[video1]: output_images/project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

In `features_extractor.py` I defined a class called `VehicleFeatureExtractor`. This class has a number of methods
that extract various feature from an image.  HOG features in particular are extracted in the `get_hog_features`
 method around line 49.  The code snippet is below.
 
 ```python

    def get_hog_features(self, image,
                         vis=False, feature_vec=True):
        # Call with two outputs if vis==True
        if vis:
            features, hog_image = hog(image, orientations=self.orient,
                                      pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                                      cells_per_block=(self.cell_per_block, self.cell_per_block),
                                      transform_sqrt=True,
                                      visualise=vis, feature_vector=feature_vec)
            return features, hog_image
        # Otherwise call with one output
        else:
            features = hog(image, orientations=self.orient,
                           pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                           cells_per_block=(self.cell_per_block, self.cell_per_block),
                           transform_sqrt=True,
                           visualise=vis, feature_vector=feature_vec)
            return features

```

  Around line 116 the `extract_features()` method is defined that takes a list of files and extract features for 
  every file.
```python
    def extract_features(self, imgs):
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for file in imgs:
            # Read in each one by one
            image = mpimg.imread(file)
            image_features = self.single_img_features(image)
            features.append(image_features)
        # Return list of feature vectors
        return features

``` 

In on line 75 of `train_classifier.py` this method is called on the training set.
 ```python
    X = feature_extractor.extract_features(image_paths)
```

####2. Explain how you settled on your final choice of HOG parameters.

I settled on my choice of HOG and other feature parameters experimentally.  An intuitive approach may have been to look 
at plots of HOG and other  features and then choose parameters based on my perceptions at the time.  I decided however,
I would implement the pipeline first and then tune parameters based on the behavior of the system at run time.  This 
way, my assumptions would be validated by actual running code. During development, I did however plot images for 
debugging purposes and to convince myself the system was doing what I thought it should be doing.  
In the end I chose the following set of parameters.


| Parameter          | Value          | 
|:-----------------:|:--------------:| 
| color space       | YCrCb          |
| hog channel       | Y (Luminance)  |
| pix per cell      | 8              | 
| cell per block    | 2              |
| spatial bin size  | (32,32)        |
| # histogram bins  | 32             |

I knew the choice of color-space would be critical. Because of prior experience, I had an intuition that color 
representation needed to include some type of "lightness" value like in HSL or perhaps something else like the value
component of HLV.  I experimented with all three of channels of RGB, HSV, LUV, and YCrCb.  In the end YCrCb seemed to 
perform best.  Additionally it seemed the Cr and Cb components were of little value, 
so I dropped them.  These channels don't tend to yield much in terms of shape information. Additionally the color information lost due 
to the missing Cr and Cb channels seems to be better represented by a color histogram and spatial binning.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I chose to use a support vector machine (SVM) with a linear kernel.  SKLearn has a few implementations
for this type of classifier.  I started out using the `svm.SVC` classifier.  As it turns out however, this classifier is 
relatively slow when training larger data sets.  After a day of training, I looked for other options and found `svm.LinearSVC()`.
This classifier trained in a reasonable amount of time.  In my code, the `CarClassifier` class in `classifier.py` is 
a thin wrapper around calls to `svm.SVC`.  This class has methods for fitting the classifier and saving the state of the trained
classifier.  The entire training pipeline is implemented in `train_classifier.py`  In this file all of the training parameters 
are specified.  The training data is loaded and fit using the SVM and the resultant model and it's parameters are 
saved to disk.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code implementing the sliding window search is found in a class called `ObjectDetector` around line 59 in `detector.py`.  
The `ObjectDetector` class accepts a classifier and a feature extractor in its constructor.  This class can be used to 
detect an arbitrary object depending on the classifier and feature extractor passed to it. The `search_windows()` method 
in the code snippet below accepts a list of windows as input and performs the classification on each 
window.

```python
    def search_windows(self, img, windows):

        # 1) Create an empty list to receive positive detection windows
        on_windows = []
        # 2) Iterate over all windows in the list
        for window in windows:
            # 3) Extract the test window from original image
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            # 4) Extract features for that window using single_img_features()
            features = self.feature_extractor.single_img_features(test_img)
            # 5) Scale extracted features to be fed to classifier
            test_features = self.classifier.scaler.transform(np.array(features).reshape(1, -1))
            # 6) Predict using your classifier
            prediction = self.classifier.predict(test_features)
            # 7) If positive (prediction == 1) then save the window
            if prediction == 1:
                on_windows.append(window)
        # 8) Return windows for positive detections
        return on_windows

```

`slide_window()` around line 19 of the same file generates the list of windows in the method above.

The entire pipeline is comes together in `detect_vehicles.py`

The relavent code is below:

```python
def main():

    ...
    
    # setup processing pipeline
    classifier = CarClassifier('car_classifier_model.pkl')
    feature_extractor = VehicleFeatureExtractor.load('feature_extractor_settings.pkl')

    processor = Processor(heatmap_threshold=2)

    # Add a region to search for vehicles
    processor.add_detector(ObjectDetector(classifier, feature_extractor,
                                          xy_window=(100, 100),
                                          xy_overlap=(0.75, 0.75),
                                          y_start_stop=(360, 535)))

    # Add another region to search for vehicles at a different scale
    processor.add_detector(ObjectDetector(classifier, feature_extractor,
                                          xy_window=(60, 60),
                                          xy_overlap=(0.65, 0.65),
                                          x_start_stop=(450,1055),
                                          y_start_stop=(370, 485)))

    # Add yet another region to search for vehicles at a different scale
    processor.add_detector(ObjectDetector(classifier, feature_extractor,
                                          xy_window=(105, 105),
                                          xy_overlap=(0.7, 0.6),
                                          x_start_stop=(192,1280),
                                          y_start_stop=(350, 555)))

    # combine lane detection and vehicle detection.
    def process_frame(image):
        ...
        output_image = processor.process_frame(image, output_image)
        return output_image

    # Process clips
    clip = VideoFileClip('.\\project_video.mp4')
    output_clip = clip.fl_image(process_frame)

    output_clip.write_videofile('.\\project_video_output.mp4', audio=False)
```

The choices for window size, scale, and overlap were all determined experimentally.  In the end, I decided to create 
three overlapping regions within th frame to search at different scales.  Each call to `add_detector()` adds
a different region to search for objects.
  
####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Below are some example images:

![alt text][image1]
![hot windows][image2]
![image3]
![image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](output_images/project_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The `ObjectDetector.get_detections()` method returns a list of windows where the classifier classified the window to 
contain a vehicle.  Since at time the classifier would produce false positives from time to time, I used two similar 
techniques to filter these false positives out.
1. Create a heat map from the detection list and then set a heat threshold.  The assumption here is that overlapping 
windows will create a higher number of detections in a given area making that area more "hot".  False positives 
are less likely and may be filtered out by setting an appropriate threshold.  This is done once per frame
2. Create a second heat map that is used to track detections from frame to frame.  Here more heat is added 
when a detection is made.  However in this case, a little heat is subtracted on every frame.  Whenever the 
heat level reaches a certain threshold, this detection is considered valid and the `scipy.ndimage.measurements.label()`
function is used to label contiguous areas that meet the threshold value.  These areas are then drawn onto the
output frame.

The code for the steps outlined above car found in the `process_frame()` at line 45 of `processor.py`

Technique 1 implemented in the code snippet below
```python
    def process_frame(self, image, output_image=None, return_heat_map=False, return_hot_windows=False, process_as_video=True):
        norm_image = cv2.normalize(image, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        heat_map = np.zeros_like(image[:, :, 0]).astype(np.float)

        for detector in self.detector_list:
            hot_windows = detector.get_detections(norm_image)
            heat_map = self._add_heat(heat_map, hot_windows)

        heat_map = self._apply_threshold(heat_map)
        labels = label(heat_map)

        ...
```

Technique 2 is implemented in the snippet below.  *Note - Technique 2 is only performed when analysing a series of
frames such as in a video*
```python
    def process_frame(self, image, output_image=None, return_heat_map=False, return_hot_windows=False, process_as_video=True):

        ...
        
        labels = label(heat_map)

        if process_as_video:
            # if video examine multiple frames to help filter
            # false positives.
            if self.display_heat_map is None:
                self.display_heat_map = np.zeros_like(image[:, :, 0]).astype(np.float)

            #keep track of previous detections using a heatmap
            # remove heat if we don't have a current detection at a given location.
            self.display_heat_map[self.display_heat_map > 0] -= 1
            self.display_heat_map[(labels[0] > 0) & (self.display_heat_map <= 20)] += 2

            # mask out areas that don't have enough heat (history of detections)
            mask = np.zeros_like(self.display_heat_map)
            mask[self.display_heat_map > 8] = 1
            draw_labels = label(mask)
            
            ...
```

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One of the biggest issues with my implementation is that it is slow.  Ideally this pipline should run in real time. 
One technique that could have been used to help make it more efficient is to compute the HOG features once per frame and 
subsample HOG output for each of the windows.  Additionally I did look at scanning multiple window groups in seperate 
threads.  The hope was to take advantage of multiple cores.  My estimate is that this alone could have reduced the execution
time by nearly a third.  Nevertheless, being new to python, I found a reduction in performance.  I 
suspect this was due to the global interpreter lock.  

Another issue arises from the fact that multiple frames are needed to verify a detection in the current implementation.
This works fine if the scene changes slowly or if the frame rate is sufficiently high to have repetitions across frames.  On
the other hand, if the frame rate is reduced (perhaps as an optimisation) or if an object is changing position quickly 
from frame to frame, the filter which reduces false positives may also filter out legitimate detections as well.  Also 
the classifier depends on a decent image with little to no motion blur.  This works well in brightly lit scens with fast shutter
speeds.  Dimly lit scenes with longer shutter speeds may be a problem.
