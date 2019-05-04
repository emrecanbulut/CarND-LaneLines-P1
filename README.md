# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./test_images_output/grayscaled.png "Grayscale"
[image2]: ./test_images_output/canny_edged.png "Canny Edges"
[image3]: ./test_images_output/masked_edges.png "Masked Edges"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. 
First, I converted the images to grayscale and later applied Gaussian smoothing which resulted in:

![grayscale_image][image1]

Then, I applied the `canny` function that detected the edges:

--![canny_edges_image][image2]

Afterwards, I masked the image focusing only on the lines that I cared about.

--![masked_edges_image][image3]

Afterwards, I applied `hough_lines` transform which returned the lines(x,y pairs of end points of the lines) in the image.
Then, in the `draw_lines()` function, I classified the lines into `right_lines` and `left_lines` using the lines' slopes by the simple logic below,

 * If the slope is negative, then it should be the left line of the lane.
 * If the slope is positive, then it should be the right line of the lane.

Then, I also kept track of how average slope was changing for each side, right and left. 
While calculating the average slope, I thought a line of 5 pixel long must not affect the average slope as much as a line of 50 pixel long. Therefore, I weighed their effect on the average slope by their length (in `find_new_avg_slope()`). Also, if the slope of next line differed relatively too much from the average, then I ignored that line and not considered as part of the lane.
My tolerance value for relative error is 5%.

After filtering out the lines that I considered as 'noise', to draw a single line, for each side:
* I first found the endpoint with the lowest y-coordinate. That is, I found the highest point which is on a line detected on the image. Let's call it `P1`.
* Secondly, I found the found the endpoint with the highest y-coordinate. That is, I found the lowest point which is on a line detected on the image. Let's call it `P2`.
* Then, using the `find_midway()` I found the middle point of `P1`and `P2`. Call that point `P3`.
* Knowing the slope, and a point on the line, `P3`, I calculated the intersection points between that line and `y=A` where A was the y-coordinate of the end points of the line I wanted to draw.
* Finally, I drew a line between those intersection points.

### 2. Identify potential shortcomings with your current pipeline

One potential shortcoming would be what would happen when the first line among detected lines was an inaccurate line. Since the my filtering algorithm was based on the average slope, if the first line was inaccurate, average slope would be inaccurate. Thus, yielding a totally different result.

Another shortcoming could be the shaking effect of the lines on videos. I think my lines were overfitting and if even a relatively small side of a line was not detected, the detected lane line would lean towards either the right or the left of the lane line.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to check several lines to initialize the average slope. Then decide from there to eliminate the noise. E.g. if the slope of 3 lines is 0.6 in average, it should make around 0.6 when you check any 2 of these 3 lines. Based on this, I could have initialized the average slope with a better condifidence level.
