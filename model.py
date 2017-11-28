"""
Advanced Lane Finding Project

The goals / steps of this project are the following:

Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
Apply a distortion correction to raw images.
Use color transforms, gradients, etc., to create a thresholded binary image.
Apply a perspective transform to rectify binary image ("birds-eye view").
Detect lane pixels and fit to find the lane boundary.
Determine the curvature of the lane and vehicle position with respect to center.
Warp the detected lane boundaries back onto the original image.
Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
"""
import os
import cv2
import traceback
import numpy as np
import matplotlib.pyplot as plt
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

IMG_SIZE = (1280, 720)

# Define a class to receive the characteristics of each line detection
class Line():
    """
    Class to keep track of things like where your last several detections of the lane lines were and what
    the curvature was, so you can properly treat new detections
    """
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        # x values for detected line pixels
        self.allx = []
        # y values for detected line pixels
        self.ally = []

def visualize(img_data, img_data_2):
    # # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    f.tight_layout()
    # ax1.imshow(cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB))
    ax1.imshow(img_data)
    ax1.set_title('Original Image', fontsize=12)
    try:
        ax2.imshow(cv2.cvtColor(img_data_2, cv2.COLOR_BGR2RGB))
        print(traceback.format_exc())
    except:
        ax2.imshow(img_data_2)
    ax2.set_title('Pipeline Result', fontsize=12)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

def calibrate_camera(path):
    """
    Chessboard size 9x6
    :param path:
    :return:
    """
    nx = 9  # the number of inside corners in x
    ny = 6  # the number of inside corners in y

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Collect object points and image points from all the images
    for image_file_name in os.listdir(path):
        # Read image using opencv
        img_data = cv2.imread(os.path.join(path, image_file_name))  # BGR
        # Convert to gray
        gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    # Camera calibration, given object points, image points, and the shape of the grayscale image
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, IMG_SIZE, None, None)

    return mtx, dist


def generate_binary(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    """
    Find lane line by using HLS color space and sobel gradients
    :param img:
    :param s_thresh:
    :param sx_thresh:
    :return: binary image
    """
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.uint8(np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255)

    return color_binary

def perspective_transform():

    # Four source coordinates
    # src = np.float32(
    #     [[760, 500],  # top right
    #     [1000, 650],  # bottom right
    #     [305, 650],  # bottom left
    #     [530, 500]]  # top left
    # )
    #
    # # Four desired coordinates
    # dst = np.float32(
    #     [[990, 500],
    #      [1000, 650],
    #      [305, 650],
    #      [330, 500]]
    # )

    # w, h = 1280, 720
    # x, y = 0.5 * w, 0.8 * h
    # src = np.float32([[200. / 1280 * w, 720. / 720 * h],
    #                   [453. / 1280 * w, 547. / 720 * h],
    #                   [835. / 1280 * w, 547. / 720 * h],
    #                   [1100. / 1280 * w, 720. / 720 * h]])
    #
    # dst = np.float32([[(w - x) / 2., h],
    #                   [(w - x) / 2., 0.82 * h],
    #                   [(w + x) / 2., 0.82 * h],
    #                   [(w + x) / 2., h]])

    src = np.float32([[(200, 720), (570, 470), (720, 470), (1130, 720)]])
    dst = np.float32([[(350, 720), (350, 0), (980, 0), (980, 720)]])

    # Compute the perspective transform, M, given source and destination points:
    M = cv2.getPerspectiveTransform(src, dst)

    # # Compute the inverse perspective transform:
    Minv = cv2.getPerspectiveTransform(dst, src)

    return M, Minv

def find_lane_histogram(binary_warped):
    """
    We did not find a lane line in the previous frame, lets search using histogram
    :param binary_warped:
    :return: Polynomial coefficients
    """
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    # histogram = np.sum(binary_warped[binary_warped.shape[0] / 2:, :], axis=0)
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                      (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                      (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if (not any(righty)) or (not any(lefty)):
        print('ERROR')
        return [], [], []

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit

def find_lane_using_previous(binary_warped):
    """
    Lane line detected in previous frame, start searching using previous data
    :param binary_warped:
    :return: Polynomial coefficients
    """
    left_fit = line_left_data.current_fit
    right_fit = line_right_data.current_fit

    # Assume you now have a new warped binary image from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                         left_fit[1] * nonzeroy + left_fit[
                                                                             2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                           right_fit[1] * nonzeroy + right_fit[
                                                                               2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if (not any(righty)) or (not any(lefty)):
        return [], []

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit

def calc_lane(binary_warped, left_fit, right_fit):
    """

    :param binary_warped:
    :param left_fit:
    :param right_fit:
    :return:
    """
    #
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # ------------------------------------------------------
    # Now we have polynomial fits and we can calculate the radius of curvature as follows:

    # Define y-value where we want radius of curvature. I'll choose the maximum y-value, corresponding to the
    # bottom of the image
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])
    # print(left_curverad, right_curverad)
    # Example values: 1926.74 1908.48

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    # Now our radius of curvature is in meters
    print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m

    # --------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------
    problem = False
    max_diff = max(abs(abs(left_fitx) - abs(right_fitx))) / 195.0
    print('max_diff: {0} meters. max is 3.7 meters.'.format(max_diff))

    if max_diff > 4.3:
        problem = True

    if (not any(left_fit)) or (not any(right_fit)):
        problem = True

    # Validate curvature
    if line_left_data.radius_of_curvature and line_right_data.radius_of_curvature:
        print(abs(line_left_data.radius_of_curvature - left_curverad))
        print(abs(line_right_data.radius_of_curvature - right_curverad))

    return problem, left_fitx, right_fitx, ploty, left_curverad, right_curverad

def find_lane(binary_warped):
    """

    :param binary_warped:
    :return:
    """
    # Compute the Polynomial coefficients
    if not line_left_data.detected:
        # We did not find a lane line in the previous frame, lets search using histogram
        left_fit, right_fit = find_lane_histogram(binary_warped)
        print('using histogram')
    else:
        # Lane line detected in previous frame, start searching using previous data
        left_fit, right_fit = find_lane_using_previous(binary_warped)
        print('using previous')

    problem, left_fitx, right_fitx, ploty, left_curverad, right_curverad = calc_lane(binary_warped, left_fit, right_fit)

    # try histogram
    if problem and line_left_data.detected:
        left_fit, right_fit = find_lane_histogram(binary_warped)
        problem, left_fitx, right_fitx, ploty, left_curverad, right_curverad = calc_lane(binary_warped, left_fit, right_fit)

        # use last average
        if problem:
            line_left_data.detected = False
            line_right_data.detected = False
            return line_left_data.bestx, line_right_data.bestx, ploty

    # -----------------------------------------------------------------
    # Save curve/line/lane data
    # -----------------------------------------------------------------
    # Detection - was the line detected in the last iteration?
    line_left_data.detected = True
    line_right_data.detected = True
    # x values for detected line pixels
    line_left_data.allx.append(left_fitx)
    line_right_data.allx.append(right_fitx)
    # y values for detected line pixels
    # line_left_data.ally = None
    # line_right_data.ally = None
    # Current fit - polynomial coefficients for the most recent fit
    line_left_data.current_fit = left_fit
    line_right_data.current_fit = right_fit
    # Curve - radius of curvature of the line in some units
    line_left_data.radius_of_curvature = left_curverad
    line_right_data.radius_of_curvature = right_curverad
    # x values of the last n fits of the line
    line_left_data.recent_xfitted = left_fitx
    line_right_data.recent_xfitted = right_fitx
    # average x values of the fitted line over the last 3 iterations
    line_left_data.bestx = np.average(line_left_data.allx[-3:], axis=0)
    line_right_data.bestx = np.average(line_right_data.allx[-3:], axis=0)
    # polynomial coefficients averaged over the last n iterations
    line_left_data.best_fit = None
    line_right_data.best_fit = None

    return left_fitx, right_fitx, ploty

def draw_lane_lines(img_data):
    """

    :param img_data:
    :return:
    """
    # Using the camera matrix and distortion coeff, undistort the image
    undist = cv2.undistort(src=img_data, cameraMatrix=mtx, distCoeffs=dist, newCameraMatrix=None, dst=mtx)

    binary = generate_binary(img_data)

    # Warp an image using the perspective transform, M:
    binary_warped = cv2.warpPerspective(src=binary, M=M, dsize=IMG_SIZE, flags=cv2.INTER_LINEAR)

    # Split the binary warped image to its RGB channels
    r, g, b = cv2.split(binary_warped)

    # Find lane lines
    left_fitx, right_fitx, ploty = find_lane(g)

    # -----------------------------------------------------------------
    # OVERLAY
    # -----------------------------------------------------------------
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(b).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, IMG_SIZE)
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    return result

def calibrate_camera_and_pers_transform():
    global mtx, dist, M, Minv

    # Calibrate the camera (matrix)
    mtx, dist = calibrate_camera(r'camera_cal/')

    # Perspective Transform
    M, Minv = perspective_transform()

def run_on_test_images():
    """

    :return:
    """
    calibrate_camera_and_pers_transform()

    path = r'test_images/'

    for image_file_name in os.listdir(path):
        # Distortion correction (coefficients)

        # Color & Gradient threshold to create a binary image
        # Perspective transform
        # Detect lane lines
        # Determine the lane curvature

        # Read image using opencv
        img_data = cv2.imread(os.path.join(path, image_file_name))  # BGR

        result = draw_lane_lines(img_data)

        # Plot the result
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        f.tight_layout()
        ax1.imshow(cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image', fontsize=12)
        ax2.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        ax2.set_title('Pipeline Result', fontsize=12)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

        pass


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    img_final = draw_lane_lines(image)
    return img_final

def run_on_video():
    calibrate_camera_and_pers_transform()

    global line_left_data, line_right_data
    line_left_data = Line()
    line_right_data = Line()

    white_output = 'project_video_output.mp4'
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    # clip1 = VideoFileClip("project_video.mp4").subclip(41, 44)
    clip1 = VideoFileClip("project_video.mp4").subclip(21, 23)
    # clip1 = VideoFileClip("project_video.mp4")

    white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)

if __name__ == '__main__':
    # run_on_test_images()
    run_on_video()