import numpy as np
import cv2


def abs_sobel_thresh(gray_img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    if (orient == 'x'):
        sobel = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, sobel_kernel)

    # Apply threshold
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return grad_binary


def mag_thresh(gray_image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobelxy = np.sqrt(sobelx ** 2 + sobely ** 2)

    scaled_sobel = np.uint8(255 * abs_sobelxy / np.max(abs_sobelxy))

    # Apply threshold
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

    return mag_binary


def dir_threshold(gray_image, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Calculate gradient direction
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    absdir = np.arctan2(abs_sobely, abs_sobelx)
    dir_binary = np.zeros_like(absdir)
    dir_binary[(absdir >= thresh[0]) & (absdir <= thresh[1])] = 1

    # Apply threshold
    return dir_binary


def hls_select(image, thresh=(0, 255)):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    s_channel = hls[:, :, 2]

    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1

    return binary_output


def get_top_down_mask(camera_calibrator, image):
    dst = camera_calibrator.undistort(image)
    # gray_dst = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)

    # # Choose a Sobel kernel size
    # ksize = 9  # Choose a larger odd number to smooth gradient measurements
    #
    # # Apply each of the thresholding functions
    # gradx = abs_sobel_thresh(gray_dst, orient='x', sobel_kernel=ksize, thresh=(30, 100))
    # grady = abs_sobel_thresh(gray_dst, orient='y', sobel_kernel=ksize, thresh=(30, 100))
    # mag_binary = mag_thresh(gray_dst, sobel_kernel=ksize, mag_thresh=(30, 100))
    # dir_binary = dir_threshold(gray_dst, sobel_kernel=ksize, thresh=(0.7, 1.3))
    #

    # After all of the experimentation, it turns out for this particular project, hls performed the best.
    h_bin = hls_select(dst, thresh=(215, 255))

    #
    # combined = np.zeros_like(dir_binary)
    # combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    # warped_image = camera_calibrator.warp(combined)
    warped_image = camera_calibrator.warp(h_bin)

    return warped_image


def fit_lanes(binary_warped, num_windows=9, margin=100, min_pixel_threshold=50,
              world_conversion_factor=(30 / 720, 3.7 / 700)):
    """
    Fits lane line curves using a sliding window search to locate lane lines, and
    then fits the curves for left and right lane lines using a polynomial fit.

    :param binary_warped: top down looking image mask depicting lane lines to be curve
           fit.
    :param num_windows: The number of windows used to determine the fit.
    :param margin: 
    :param min_pixel_threshold: Minimum number of pixels required to qualify a window
           as containing lane lines.
    :return: 2 vectors of polynomial coefficients. One for the left and right lane lines.  
             left_fit, right_fit
    """
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2):, :], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / num_windows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(num_windows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
        nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
        nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > min_pixel_threshold:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > min_pixel_threshold:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    return poly_fit_points(leftx, lefty, rightx, righty, world_conversion_factor)


def fit_lanes_next_frame(binary_warped, left_fit, right_fit, margin=100, world_conversion_factor=(30 / 720, 3.7 / 700)):
    """
    Fits the curve of the current frame using the fit from the last frame
    as a starting point.

    :param binary_warped: top down looking image mask depicting lane lines to be curve
           fit.
    :param left_fit: 2nd order polynomial coefficients to use as a starting point for 
                     locating the new left lane line. 
    :param right_fit: 2n order polynomial coefficients to use as a starting point for
                      locating the new right lane line.
    :param margin:

    :param world_conversion_factor:

    :return:  2 vectors of polynomial coefficients. One for the left and right lane lines.  
              left_fit, right_fit
    """
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
        nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = (
        (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
            nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    return poly_fit_points(leftx, lefty, rightx, righty, world_conversion_factor)


def poly_fit_points(leftx, lefty, rightx, righty, world_conversion_factor=(30 / 720, 3.7 / 700)):
    left_fit = None
    if len(leftx) > 0 and len(lefty) > 0:
        left_fit = np.polyfit(lefty, leftx, 2)

    right_fit = None
    if len(rightx) > 0 and len(righty) > 0:
        right_fit = np.polyfit(righty, rightx, 2)

    world_left_fit = None
    if len(leftx) > 0 and len(lefty) > 0:
        world_left_fit = np.polyfit(lefty * world_conversion_factor[1], leftx * world_conversion_factor[0], 2)

    world_right_fit = None
    if len(rightx) > 0 and len(righty) > 0:
        world_right_fit = np.polyfit(righty * world_conversion_factor[1], rightx * world_conversion_factor[0], 2)

    # use our fit to determine the lane center
    return left_fit, right_fit, world_left_fit, world_right_fit


def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
    max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
    return output


def find_window_centroids(image, window_width, window_height, margin):
    window_centroids = []  # Store the (left,right) window centroid positions per level
    window = np.ones(window_width)  # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template

    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3 * image.shape[0] / 4):, :int(image.shape[1] / 2)], axis=0)
    l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
    r_sum = np.sum(image[int(3 * image.shape[0] / 4):, int(image.shape[1] / 2):], axis=0)
    r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(image.shape[1] / 2)

    # Add what we found for the first layer
    window_centroids.append((l_center, r_center))

    # Go through each layer looking for max pixel locations
    for level in range(1, (int)(image.shape[0] / window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(
            image[int(image.shape[0] - (level + 1) * window_height):int(image.shape[0] - level * window_height), :],
            axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width / 2
        l_min_index = int(max(l_center + offset - margin, 0))
        l_max_index = int(min(l_center + offset + margin, image.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center + offset - margin, 0))
        r_max_index = int(min(r_center + offset + margin, image.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
        # Add what we found for that layer
        window_centroids.append((l_center, r_center))

    return window_centroids


def solve_poly(coefficients, input):
    y = 0
    i = len(coefficients)
    for idx in range(i, 0, -1):
        y += coefficients[i - idx] * input ** (idx - 1)

    return y


def rolling_average(new_sample, iteration, prev_avg=None):
    """
    Calculates the rolling average using Welford’s method
    :param new_sample: The new sample
    :param iteration:  current iteration or the current number of samples
    :param prev_avg: Previous average
    :return: 
    """
    # Add 1 because this function is 1 based not 0.
    iteration += 1
    if iteration == 1:
        avg = new_sample
    else:
        avg = prev_avg + (new_sample - prev_avg) / iteration
    return avg


def exp_approx_rolling_average(new_sample, iteration, prev_avg=None):
    """
    Calculates the exponentially weighted moving average
    :param new_sample: The new sample
    :param iteration:  current iteration or the current number of samples
    :param prev_avg: Previous average
    :return: 
    """

    iteration += 1
    if iteration == 1:
        avg = new_sample
    else:
        avg = prev_avg - prev_avg / iteration
        avg += new_sample / iteration;

    return avg;
