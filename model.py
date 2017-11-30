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
import pandas as pd
import traceback
import numpy as np
import matplotlib.pyplot as plt
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger('model')

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

class Calc_Lane_Results():

    def __init__(self):
        self.problem = False
        self.issue = None
        self.left_fitx = None
        self.right_fitx = None
        self.ploty = None
        self.left_curverad = None
        self.right_curverad = None
        self.avg_curverad = None
        self.center_offset_meters = None
        self.lane_width = None

class Blah():
    def __init__(self):
        self.stats_df = pd.DataFrame()
        self.num_frame = 0
        self.line_left_data = Line()
        self.line_right_data = Line()

    def process_image(self, image):
        # NOTE: The output you return should be a color image (3 channel) for processing video below
        # TODO: put your pipeline here,
        # you should return the final output (image where lines are drawn on lanes)
        img_final = self.draw_lane_lines(image)
        return img_final

    def draw_lane_lines(self, img_data):
        """

        :param img_data:
        :return:
        """
        # Using the camera matrix and distortion coeff, undistort the image
        undist = cv2.undistort(src=img_data, cameraMatrix=mtx, distCoeffs=dist, newCameraMatrix=None, dst=mtx)

        binary = generate_binary(undist)

        # Warp an image using the perspective transform, M:
        binary_warped = cv2.warpPerspective(src=binary, M=M, dsize=IMG_SIZE, flags=cv2.INTER_LINEAR)

        # Split the binary warped image to its RGB channels
        r, g, b = cv2.split(binary_warped)

        # Find lane lines
        left_fitx, right_fitx, ploty, avg_curverad, current_stats_df = self.find_lane(r, g, b)

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

        current_stats_df['num_frame'] = self.num_frame
        self.stats_df = self.stats_df.append(current_stats_df)

        # Add text overlay
        cv2.putText(img=result, text='Avg curverad: {0:,.2f}'.format(avg_curverad), org=(10, 20),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255, 255, 255))
        cv2.putText(img=result, text='Frame: {0}'.format(str(self.num_frame)), org=(10, 50),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255, 255, 255))
        try:
            cv2.putText(img=result, text='Offset from center: {0:,.2f} meters'.format(current_stats_df['center_offset_meters'][0]),
                        org=(10, 80), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255, 255, 255))
            cv2.putText(img=result, text='Lane width: {0:,.2f} meters'.format(current_stats_df['lane_width'][0]),
                        org=(10, 110), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255, 255, 255))
        except:
            pass

        self.num_frame += 1

        # -----------------------------------------------------
        # Overlay
        # -----------------------------------------------------
        # visualize_lane_lines(g, self.line_left_data.current_fit, self.line_right_data.current_fit)
        # binary_warped_with_lines = cv2.imread('temp.png')
        #
        # binary_warped_thumbnail = cv2.resize(binary_warped, (0, 0), fx=0.3, fy=0.3)
        # binary_warped_with_lines_thumbnail = cv2.resize(binary_warped_with_lines, (0, 0), fx=0.3, fy=0.3)
        #
        # x_offset = 800
        # y_offset = 25
        # result[y_offset:y_offset + binary_warped_thumbnail.shape[0], x_offset:x_offset + binary_warped_thumbnail.shape[1]] = binary_warped_thumbnail
        # y_offset = 200
        # result[y_offset:y_offset + binary_warped_with_lines_thumbnail.shape[0], x_offset:x_offset + binary_warped_with_lines_thumbnail.shape[1]] = binary_warped_with_lines_thumbnail
        # plt.imshow(result)
        # plt.show()

        return result

    def find_lane(self, r, g, b):
        """
        Given the RGB channels of the binary warped image, find the lane lines using various methods. This method
        is called ONCE per lane line. calc_line() is called multiple times to find the most optimal line.
        :param r: binary warped red
        :param g: binary warped green
        :param b: binary warped blue
        :return: fitted left and right lines and stats
        """
        logger.debug('\n-------------------------------------------------------------')

        # Compute the Polynomial coefficients
        if not self.line_left_data.detected:
            # We did not find a lane line in the previous frame, lets search using histogram
            left_fit, right_fit = find_lane_histogram(g)  # use green channel
            method = 'histogram g'
        else:
            # Lane line detected in previous frame, start searching using previous data
            left_fit, right_fit = self.find_lane_using_previous(g)  # use green channel
            method = 'previous g'

        lane_results = self.calc_lane(g, left_fit, right_fit)

        # try histogram
        # if lane_results.problem and self.line_left_data.detected:
        if lane_results.problem:
            left_fit, right_fit = find_lane_histogram(g)
            method = 'histogram retry g'
            lane_results = self.calc_lane(g, left_fit, right_fit)

            # try blue channel
            if lane_results.problem:
                left_fit, right_fit = find_lane_histogram(b)
                method = 'histogram retry b'
                lane_results = self.calc_lane(b, left_fit, right_fit)

                # use last average. check if line detected in last frame -- cant use average if no lines.
                if lane_results.problem and self.line_left_data.detected:
                    method = 'use last average'
                    logger.debug('Using method: {0}'.format(method))
                    self.line_left_data.detected = False
                    self.line_right_data.detected = False

                    avg_curverad = (self.line_left_data.radius_of_curvature + self.line_right_data.radius_of_curvature) / 2

                    lane_results.left_fitx = self.line_left_data.bestx
                    lane_results.right_fitx = self.line_right_data.bestx
                    lane_results.avg_curverad = avg_curverad

                    left_fit = self.line_left_data.current_fit
                    right_fit = self.line_right_data.current_fit

        # -----------------------------------------------------------------
        # Save curve/line/lane data
        # -----------------------------------------------------------------
        # Detection - was the line detected in the last iteration?
        self.line_left_data.detected = True
        self.line_right_data.detected = True
        # x values for detected line pixels
        self.line_left_data.allx.append(lane_results.left_fitx)
        self.line_right_data.allx.append(lane_results.right_fitx)
        # y values for detected line pixels
        # line_left_data.ally = None
        # line_right_data.ally = None
        # Current fit - polynomial coefficients for the most recent fit
        self.line_left_data.current_fit = left_fit
        self.line_right_data.current_fit = right_fit
        # Curve - radius of curvature of the line in some units
        self.line_left_data.radius_of_curvature = lane_results.left_curverad
        self. line_right_data.radius_of_curvature = lane_results.right_curverad
        # average x values of the fitted line over the last 3 iterations
        self.line_left_data.bestx = np.average(self.line_left_data.allx[-5:], axis=0)
        self.line_right_data.bestx = np.average(self.line_right_data.allx[-5:], axis=0)
        # x values of the last n fits of the line
        if any(left_fit):
            self.line_left_data.recent_xfitted.append(left_fit)
        if any(right_fit):
            self.line_right_data.recent_xfitted.append(right_fit)
        # polynomial coefficients averaged over the last 3 iterations
        self.line_left_data.best_fit = np.average(self.line_left_data.recent_xfitted[-5:], axis=0)
        self.line_right_data.best_fit = np.average(self.line_right_data.recent_xfitted[-5:], axis=0)

        # Save stats for analysis
        if lane_results.left_curverad and lane_results.right_curverad:
            current_stats_df = pd.DataFrame(data={'left_curverad': lane_results.left_curverad,
                                              'right_curverad': lane_results.right_curverad,
                                              'avg_curverad': lane_results.avg_curverad,
                                              'method': method,
                                              'center_offset_meters': lane_results.center_offset_meters,
                                              'lane_width': lane_results.lane_width,
                                              'issue': lane_results.issue},
                                        index=[0])
        else:
            current_stats_df = pd.DataFrame()

        logger.debug('-------------------------------------------------------------')

        return lane_results.left_fitx, lane_results.right_fitx, lane_results.ploty, \
               lane_results.avg_curverad, current_stats_df

    def calc_lane(self, binary_warped, left_fit, right_fit):
        """
        Given a binary warped image and the polynomial coefficients for the left and right lane, calculate the
        polynomial fits and then check if it makes sense. If it doesn't make sense, then this method will be
        called again with other parameters.
        :param binary_warped: binary warped image
        :param left_fit: left lane polynomial coefficients
        :param right_fit: right lane polynomial coefficients
        :return: Calc_Lane_Results class which contains all the results like fit, curve, lane width, etc
        """
        problem = False
        issue = None

        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

        if any(left_fit) and any(right_fit):
            # Given polynomial coefficients, calculate the polynomial fits for left and right lane
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

            # ------------------------------------------------------
            # Now we have polynomial fits and we can calculate the radius of curvature as follows:

            # Define y-value where we want radius of curvature. I'll choose the maximum y-value, corresponding to the
            # bottom of the image
            y_eval = np.max(ploty)

            # Define conversions in x and y from pixels space to meters
            ym_per_pix = 30 / 720  # meters per pixel in y dimension
            xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

            # Fit new polynomials to x,y in world space
            left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
            right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
            # Calculate the new radii of curvature ..and our radius of curvature is in meters
            left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / \
                            np.absolute(2 * left_fit_cr[0])
            right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / \
                             np.absolute(2 * right_fit_cr[0])
            avg_curverad = (left_curverad + right_curverad) / 2

            # Calc max lane width
            max_lane_width = max(abs(abs(left_fitx) - abs(right_fitx))) / 195.0

            # Calc Center Offset
            camera_position = 1280 / 2
            lane_center = (right_fitx[719] + left_fitx[719]) / 2
            center_offset_pixels = abs(camera_position - lane_center)
            center_offset_meters = center_offset_pixels * xm_per_pix

            # Calc lane width
            lane_width = abs(right_fitx[719] - left_fitx[719]) * xm_per_pix
        else:
            # left_fit and right_fit is blank
            problem = True
            left_fitx = self.line_left_data.current_fit
            right_fitx = self.line_right_data.current_fit
            left_curverad = self.line_left_data.radius_of_curvature
            right_curverad = self.line_right_data.radius_of_curvature
            avg_curverad = (left_curverad + right_curverad) / 2
            center_offset_meters, lane_width = 0, 0

        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------
        if (not any(left_fit)) or (not any(right_fit)):
            issue = 'histogram could not find lane'
            logger.debug(issue)
            problem = True
        elif (max_lane_width > 4.0) or (max_lane_width < 2.5):
            issue = 'BAD max_lane_width error: {0}'.format(max_lane_width)
            logger.debug(issue)
            problem = True
        elif avg_curverad < 400.0:
            issue = 'BAD avg curve: {0}'.format(avg_curverad)
            logger.debug(issue)
            problem = True
        elif center_offset_meters > 0.41:
            issue = 'BAD center_offset_meters: {0}'.format(center_offset_meters)
            logger.debug(issue)
            problem = True
        elif lane_width > 3.5:
            issue = 'BAD lane_width: {0}'.format(lane_width)
            logger.debug(issue)
            problem = True

        # Save results
        lane_results = Calc_Lane_Results()
        lane_results.problem = problem
        lane_results.issue = issue
        lane_results.left_fitx = left_fitx
        lane_results.right_fitx = right_fitx
        lane_results.ploty = ploty
        lane_results.left_curverad = left_curverad
        lane_results.right_curverad = right_curverad
        lane_results.avg_curverad = avg_curverad
        lane_results.center_offset_meters = center_offset_meters
        lane_results.lane_width = lane_width
        return lane_results

    def find_lane_using_previous(self, binary_warped):
        """
        Lane line detected in previous frame, start searching using previous data
        :param binary_warped:
        :return: Polynomial coefficients
        """
        left_fit = self.line_left_data.current_fit
        right_fit = self.line_right_data.current_fit

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

def visualize_lane_lines(binary_warped, left_fit, right_fit):
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

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                    ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                     ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    # plt.imshow(result)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(result)
    ax.plot(left_fitx, ploty, color='yellow')
    ax.plot(right_fitx, ploty, color='yellow')
    fig.savefig('temp.png')

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
        logger.debug('ERROR')
        return [], []

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit

def calibrate_camera_and_pers_transform():
    global mtx, dist, M, Minv

    # Calibrate the camera (matrix)
    mtx, dist = calibrate_camera(r'camera_cal/')

    # Perspective Transform
    M, Minv = perspective_transform()

def run_on_test_images():
    """
    Distortion correction (coefficients)
    Color & Gradient threshold to create a binary image
    Perspective transform
    Detect lane lines
    Determine the lane curvature
    :return:
    """
    calibrate_camera_and_pers_transform()

    path = r'test_images/'

    for image_file_name in os.listdir(path):
        # Read image using opencv
        img_data = cv2.imread(os.path.join(path, image_file_name))  # BGR

        blah = Blah()
        result = blah.draw_lane_lines(img_data)

        # Plot the result
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        f.tight_layout()
        ax1.imshow(cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image', fontsize=12)
        ax2.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        ax2.set_title('Pipeline Result', fontsize=12)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        # plt.show()

        # Save image
        cv2.imwrite(os.path.join(r'output_images/', image_file_name), result)  # BGR

def run_on_video():
    calibrate_camera_and_pers_transform()

    blah = Blah()

    white_output = 'project_video_output.mp4'
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    # clip1 = VideoFileClip("project_video.mp4").subclip(0, 25)
    # clip1 = VideoFileClip("project_video.mp4").subclip(21, 23)
    clip1 = VideoFileClip("project_video.mp4")

    white_clip = clip1.fl_image(blah.process_image)  # NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)

    print(blah.stats_df.to_string())
    blah.stats_df.to_csv(r'test.csv')

if __name__ == '__main__':
    run_on_test_images()
    # run_on_video()