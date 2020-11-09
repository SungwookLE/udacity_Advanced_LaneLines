## Writeup_this_Project
* AUTHOR/DATE: SungwookLE, '20.11/08
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/calibration1.jpg "(Input Image) distorted"
[image2]: ./output_images/undist/undistort1.png "(Output Image) undistorted"
[image3]: ./test_images/test1.jpg "(Input Image) Road Transformed"
[image4]: ./output_images/binary/binary_comb1.jpg "(Output Image) Binary Example"
[image5]: ./output_images/warp/warp1.jpg "(Output Image) Warp Example"
[image6]: ./output_images/find_lane_pix/sliding_window/sliding_find_lane1.jpg "(Output Image) Sliding Window Find Lane Line"
[image7]: ./output_images/find_lane_pix/prior_search/prior_find_lane1.jpg "(Output Image) Prior Search Find Lane Line"
[image8]: ./output_images/inverse_warp/inversewarp1.jpg "(Output Image) Visualized Inverse Warp with Curvature"

[video1]: ./project_video.mp4 "(Input Video) Video"
[video2]: ./output_videos/OUT_project_video.mp4 "(Output Video) Final Results"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first and second code cell of the IPython notebook located in "./advanced_lanelines.ipynb".

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection using `cv2.findChessboardCorners()`

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]
(Input Image)
![alt text][image2]
(Output Image)

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Distortion-corrected image is the first step while processing image. The code for this step is contained in the first and second cell of the IPython notebook located in "./advanced_lanelines.ipynb".

![alt text][image1]
(Input Image)
![alt text][image2]
(Output Image)


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

To create a threshold binary image, I write the code in the third cell of the IPython notedbook loacated in "./advanced_lanelines.ipynb".
It is composed of two parts. one is using 'hls-color-space', another one is using 'gradient-method'.

'hls-color-space': convert the HLS color space using `cv2.cvtColor(, cv2.COLOR_RGB2HLS)`, and divide into each H,L,S channel. And finding the pixels which are in threshold boundaries
'gradiend-method': convert the GRAY color space using `cv2.cvtColor(, cv2.COLOR_RGB2GRAY)`, and get the sobel gradient using `sobelx = cv2.Sobel(gray, cv2.CV_64F, 1,0, ksize=sobel_kernel)`.
you can know that sobelx is x directional gradient in (1,0). y-direction gradient is calculated using (0,1). and then, under 255 scaling is done for thresholding.
one morething, here is directional thresholding using `direction = np.arctan2(abs_sobely, abs_sobelx)`.
All scaled gradient image pixels are searched wheter they are in threshold boundaries or not like this `binary_mag[ (scaled_sobelxy >= threshold[0]) & (scaled_sobelxy <= threshold[1]) ] = 1`

Combination method is customized by myself like this `comb_bin[ ((sample_color_thd ==1)) | (( sample_gradient_thd_x==1) ) | ( (sample_gradient_thd_mag==1) & (sample_gradient_thd_dir==1) & ( sample_gradient_thd_y==1) ) ]=1`. Why i used like that, is color_space is shown clear performance but, gradient method worked with some noise. Following images are my reulst.


![alt text][image3]
(Input Image)
![alt text][image4]
(Output Image)


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I write the code in the fourth cell of the IPython notebook located in "./advanced_lanelines.ipynb".
My perspective function called `perspective_img()`. The explanation of this function is following.
(1) undistorted_image is input to `perspective_img()`
(2) Source pts are input to `perspective_img()`: Source Pts are well choosed for bird-eye view results
(3) Warp the Image using `cv2.getPerspectiveTransform(src_pts, dst_pts)` and `cv2.warpPerspective`

using the opencv member function, the perspective image is well implemented

I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [ [560,465], [765,465],  [1080, 650], [265, 650] ])
offset = 50
    dst = np.float32 ( [[ offset, offset] ,[img.shape[1]-offset, offset] , [img.shape[1]-offset, img.shape[0]-offset] , [offset , img.shape[0]-offset] ])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image3]
(Input Image)
![alt text][image5]
(Output Image)


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I write the code in the fifth and sixth cell of the IPython notebook located in "./advanced_lanelines.ipynb".
fifth cell is for finding the lane line pixels using sliding window method. and sixth cell is for finding the lane pixels using prior search method.

---
I explain the sliding window method first,
the function that i made is `fit_polynomial() and find_lane_pixels()`. Function inputs are undistored and binary and warped image.
and Function returs the (out_img, left_fitx, right_fitx, ploty).

that function is finding the satisfying pixel in each sliding windows.
sliding windows are vertical n-divided window for each image 
(1) get histogram value using `np.sum( binary, axis= 0 )`
(2) get the initial point using `np.argmax()`
(3) preparing the nonzero pixel index using `np.array(binary.nonzero())`
(4) each vertical divided window, 'for-loop' works to find proper pixels (that's why it is called sliding window).
`good_left_inds = ( (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]` 
`good_right_inds= ( (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]`

(5) Using proper pixels, the polynomial coefficients are calcuated `np.polyfit(y,x,2)`
(6) Calculating the pts using coefficients

![alt text][image6]
(Sliding Window)

---
I explain the prior search method,
the function that i made is `search_around_ploy() and fit_poly()`. Function inputs are undistorted and binary and warped image and prior roi.
and function returns the ( out_img, left_fitx, right_fitx, ploty).

that function is finding the satisfying in prior ROI(Region of Interest, in 2nd spline bound)
(1) finding pixels which are in prior boundaries
`left_lane_inds = (nonzerox > (init_tune[0][0] * (nonzeroy)**2 + init_tune[0][1] * nonzeroy + init_tune[0][2] - margin ) ) & (nonzerox < (init_tune[0][0] * (nonzeroy)**2 + init_tune[0][1] * nonzeroy + init_tune[0][2] + margin ))`
`right_lane_inds = (nonzerox > (init_tune[1][0] * (nonzeroy)**2 + init_tune[1][1] * nonzeroy + init_tune[1][2] - margin ) ) & (nonzerox < (init_tune[1][0] * (nonzeroy)**2 + init_tune[1][1] * nonzeroy + init_tune[1][2] + margin ) )`
(2) finding the polynomial coefficients in `fit_poly()`

![alt text][image7]
(Prior Search)

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I write the code in the seventh cell of the IPython notebook in "./advanced_lanelies.ipynb".
the function is `measure_curvature()`. Inputs are left_fitx(pts), right_fitx(pts), ploty(pts), ratio = (xm_per_pix, ym_per_pix).
Outputs are left_curverad(m), right_curverad(m), mean_curverad(m), left_of_center(m)

Using the method of 'Calculation of R Curve'
'R curve = (1+(2Ay+B)**2)**(3/2)/abs(2*A)'

​the position of the vehicle with respect to center(from left) is calculated as follow: (center_pix - left_x_pix) * xm_per_pix

you can check the results on output video as text on top.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I write the code in the eighth cell of the IPython notebook in "./advanced_lanelines.ipynb".
First, left_fit points and right_fit points are gathered to stack array as 'pts'
Second, using `cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))`, fiiling the green color in area
Third, using `cv2.getPerspectiveTransform(), cv2.warpPerspective()` the inverse warp is implemented


![alt text][image3]
(Input Image)
![alt text][image8]
(Output Image)

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

You can check the code pipeline in "./project_main.py".
And the Final video results are as follow

Here's a [link to my video result](./output_videos/OUT_project_video.mp4)
(Output Video) Final Results
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

the rectangle ROI region is so sensitive. I think there are needed some alternative tuning parameter according to vertical movement and the speed of the vehicle and the radius of curvature

