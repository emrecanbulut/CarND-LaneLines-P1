#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import cv2
#%matplotlib inline
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
#from IPython.display import HTML

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    left_lines=[]
    right_lines = []
    avg_left_slope=0.
    avg_right_slope=0.
    lowest_point_left = ()
    bottom_point_left = ()
    lowest_point_right = ()
    bottom_point_right = ()
    relative_error = 0.
    tolerance = 0.05

    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y2-y1)/(x2-x1)
            #if slope is negative, it is a left line
            if slope < 0:
                if avg_left_slope == 0:
                    avg_left_slope = slope
                else:
                    relative_error = abs((slope - avg_left_slope)/avg_left_slope)
                    if relative_error > tolerance:
                        continue
                    else:
                        avg_left_slope = find_new_avg_slope(avg_left_slope, slope, len(left_lines))
                left_lines.append(line)
            #if slope is positive, it is a right line
            elif slope > 0 :
                if avg_right_slope == 0:
                    avg_right_slope = slope
                else:
                    relative_error = abs((slope - avg_right_slope)/avg_right_slope)
                    if relative_error > tolerance:
                        continue
                    else:
                        avg_right_slope = find_new_avg_slope(avg_right_slope, slope, len(right_lines))
                right_lines.append(line)
    #img.shape[0] = height (y)
    #img.shape[1] = width (x)
    lowest_point_right = find_lowest_y_point(right_lines, max(img.shape[0], img.shape[1])) #find_intersection_point( a_right_point, avg_right_slope, 350)
    bottom_point_right = find_intersection_point(lowest_point_right,avg_right_slope, img.shape[0]-1)

    lowest_point_left = find_lowest_y_point(left_lines, max(img.shape[0], img.shape[1])) #find_intersection_point( a_left_point, avg_left_slope, 350)
    bottom_point_left = find_intersection_point(lowest_point_left,avg_left_slope, img.shape[0]-1)

    cv2.line(img, lowest_point_right, bottom_point_right , color, thickness)
    cv2.line(img, lowest_point_left, bottom_point_left, color, thickness)

'''
def draw_hough_lines(img, lines, color=[255,255, 0], thickness=5):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1,y1), (x2,y2) , color, thickness)
'''

#Finds the bottom point on the image that intersects with the specified line
def find_intersection_point(point_on_line, slope, intersection_y):
    x0, y0 = point_on_line
    x1=int(round((intersection_y - y0)/slope))+x0
    return (x1, intersection_y)

#Finds the point which has the lowest y value in a list of lines
def find_lowest_y_point(lines, max_pixel):
    lowest_point=(max_pixel, max_pixel)
    for line in lines:
        for x1,y1,x2,y2 in line:
            if y1<lowest_point[1]:
                lowest_point = (x1, y1)
            if y2<lowest_point[1]:
                lowest_point = (x2, y2)
    return lowest_point

def find_new_avg_slope(old_slope, slope_to_add, line_size):
    new_slope = ((old_slope*line_size)+slope_to_add)/(line_size+1)
    return new_slope

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=1., β=0.9, γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    if len(initial_img.shape) != 3:
        initial_img = cv2.cvtColor(initial_img,cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(initial_img, α, img, β, γ)

# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.

# Read in and grayscale the image
def read(filename):
    image = mpimg.imread('test_images/'+filename)
    return image
    
def make_canny(image):
    gray = grayscale(image)
    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 7
    blur_gray = gaussian_blur(gray, kernel_size)

    # Define our parameters for Canny and apply
    low_threshold = 80
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)

    return edges
def mask_image(edges_img):
    # This time we are defining a four sided polygon to mask
    imshape = edges_img.shape
    #img.shape[0] = height (y)     #img.shape[1] = width (x)
    vertices = np.array([[(0,imshape[0]),(imshape[1]*0.46, imshape[0]*0.62), (imshape[1]*0.54, imshape[0]*0.62), (imshape[1],imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges_img, vertices)
    return masked_edges
def hough_transform(original_img, masked_edges, filename, showEnabled=True):
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 10     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 15 # minimum number of pixels making up a line
    max_line_gap = 30    # maximum gap in pixels between connectable line segments

    
    line_img = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

    lines_edges = weighted_img(line_img, original_img)
    if showEnabled==True:
        plt.imshow(lines_edges)
        plt.title(filename + ' - Lane lines marked')
        
        if not os.path.exists('test_images_output'):
            os.makedirs('test_images_output')

        plt.savefig('test_images_output/'+filename)
        return None
    else:
        return lines_edges


def work_on_images():
    for file in os.listdir("test_images/"):
        init_img = read(file)
        edges = make_canny(init_img)
        masked_edges = mask_image(edges)
        #combined_img = weighted_img(init_img, masked_edges)
        hough_transform(init_img, masked_edges, file)

def work_on_video(filename):
    white_output = 'test_videos_output/'+ filename
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
    clip1 = VideoFileClip("test_videos/"+filename) #solidWhiteRight.mp4")
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    edges = make_canny(image)
    masked_edges = mask_image(edges)
    #uncomment 2 lines below for testing
    #combined_img = weighted_img(image, masked_edges)
    #result = hough_transform(combined_img, masked_edges, '', False)
    result = hough_transform(image , masked_edges, '', False)
    return result

if __name__ == '__main__':
    #work_on_images()
    work_on_video("solidWhiteRight.mp4")
    work_on_video("solidYellowLeft.mp4")
