# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

import camera
import lane_functions as process

import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import imageio

imageio.plugins.ffmpeg.download()


class LaneDetector():
    def __init__(self, camera_calibrator):
        self.camera_calibrator = camera_calibrator

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
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

        self.window_width = 50
        self.window_height = 80
        self.margin = 100

        self.world_conversion_factor = (3.7 / 700, 30 / 720)

        self.max_smoothing_samples = 7
        self.num_smoothing_samples = 0

    def process_frame(self, image):

        top_down_image_mask = process.get_top_down_mask(self.camera_calibrator, image)

        if not self.detected:
            left_fit, right_fit, world_left_fit, world_right_fit = \
                process.fit_lanes(top_down_image_mask, world_conversion_factor=self.world_conversion_factor)
        else:
            left_fit, right_fit, world_left_fit, world_right_fit = \
                process.fit_lanes_next_frame(top_down_image_mask,
                                             self.current_fit[0],
                                             self.current_fit[1],
                                             world_conversion_factor=self.world_conversion_factor)

        self.detected = self._do_sanity_check(left_fit, right_fit);

        if self.detected:
            self.current_fit = np.array([left_fit, right_fit])

            self.best_fit = process.rolling_average(self.current_fit, self.num_smoothing_samples, self.best_fit)
            if self.num_smoothing_samples < self.max_smoothing_samples:
                self.num_smoothing_samples += 1

            # Calculate the vehicle offset in pixels then convert to world coordinates.
            self.line_base_pos = \
                self.calc_lane_offset(left_fit, right_fit, image_shape=image.shape) * self.world_conversion_factor[0]

        elif self.best_fit is not None:
            self.current_fit = self.best_fit

        l_radius, r_radius = self.calc_radius(world_left_fit, world_right_fit)
        self.radius_of_curvature = (l_radius + r_radius) / 2

        # buffer = LaneDetector.draw_lane(calib, image, left_fit, right_fit)
        if self.best_fit is not None:
            buffer = LaneDetector.draw_lane(self.camera_calibrator, image, self.best_fit[0], self.best_fit[1],
                                            top_down_image_mask)
            self.draw_info(buffer)
        else:
            return image

        return buffer

    def detect_lanes(self, input_video_clip_file, output_video_clip):
        clip = VideoFileClip(input_video_clip_file)
        output_clip = clip.fl_image(self.process_frame)

        output_clip.write_videofile(output_video_clip, audio=False)

    def _do_sanity_check(self, left_fit, right_fit):
        if left_fit is None or right_fit is None:
            return False

        # a basic sanity check is to make sure our left and right polynomials don't
        # differ too much.
        val = np.abs(left_fit - right_fit)
        val = val[0:2]
        val = val[np.where(val > 1)]
        return val.size == 0

    @staticmethod
    def draw_lane(camera_calibarator, image, left_fit, right_fit, detected_lanes=None):
        """

        :param camera_calibarator: Instance of Calibrator class used for distorting
               and undistoriting frame images.
        :param image: unwarped frame image. 
        :param left_fit: coefficients for curve fit of left lane in warped coordinates  
        :param right_fit: coefficients for curv fit of right lane in warped coordinates
        :param detected_lanes:
        :return: Image with lane lines drawn on it.  
        """

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(image).astype(np.uint8)

        # Generate x and y values for plotting
        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Merge lane mask
        if detected_lanes is not None:
            detected_lanes = camera_calibarator.unwarp(detected_lanes)
            image[detected_lanes == 1] = (255, 0, 0)

        # Draw the lane onto the warped blank image
        cv2.fillPoly(warp_zero, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = camera_calibarator.unwarp(warp_zero)

        # Combine the result with the original image
        result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

        return result

    def draw_info(self, image):
        font = cv2.FONT_HERSHEY_SIMPLEX

        if self.radius_of_curvature <= 1500:
            radius_string = 'Turn Radius: {:.0f}(m)'.format(self.radius_of_curvature)
        else:
            radius_string = 'No Curve'
        cv2.putText(image, radius_string, (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        center_offset_string = 'Center Offset: {:.2f}(m)'.format(self.line_base_pos)
        cv2.putText(image, center_offset_string, (10, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return

    def calc_radius(self, left_fit, right_fit):
        # y_eval = np.max(ploty)
        y_eval = 700 * self.world_conversion_factor[1]
        left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
        right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])
        return left_curverad, right_curverad

    def calc_lane_offset(self, left_fit, right_fit, image_shape):
        y = image_shape[0]

        left_side = process.solve_poly(left_fit, y)
        right_side = process.solve_poly(right_fit, y)

        lane_center = (right_side - left_side) / 2
        frame_center = y / 2
        center_offset = frame_center - lane_center

        return center_offset



