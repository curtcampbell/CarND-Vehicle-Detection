import numpy as np
import cv2
from threadpool import *
from scipy.ndimage.measurements import label
from threading import Thread, Lock


class Processor:

    def __init__(self, heatmap_threshold=0):
        self.detector_list = []
        self.heatmap_threshold = heatmap_threshold
        self.thread_pool = ThreadPool(5)
        self.display_heat_map = None
        # self.lock = Lock()

    def add_detector(self, detector):
        self.detector_list.append(detector)

    @staticmethod
    def _add_heat(heat_map, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heat_map[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        return heat_map

    def _apply_threshold(self, heat_map):
        # Zero out pixels below the threshold
        heat_map[heat_map <= self.heatmap_threshold] = 0
        # Return thresholded map
        return heat_map

    # def _task(self, image, heat_map, detector):
    #     hot_windows = detector.get_detections(image)
    #     # only one writes to the heat map at a time.
    #     self.lock.acquire()
    #     try:
    #         heat_map = Processor._add_heat(heat_map, hot_windows)
    #     finally:
    #         self.lock.release()

    def process_frame(self, image, output_image=None, return_heat_map=False, return_hot_windows=False, process_as_video=True):
        norm_image = cv2.normalize(image, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        heat_map = np.zeros_like(image[:, :, 0]).astype(np.float)

        for detector in self.detector_list:
            hot_windows = detector.get_detections(norm_image)
            heat_map = self._add_heat(heat_map, hot_windows)

        heat_map = self._apply_threshold(heat_map)
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
        else:
            draw_labels = labels

        # render to our output buffer if we have one.
        if output_image is not None:
            image = output_image

        labeled_image = self.draw_labeled_bboxes(image, draw_labels, color=(0, 255, 0), thick=2)

        ret_val = [labeled_image]

        if return_heat_map:
            ret_val.append(heat_map)

        if return_hot_windows:
            hot_window_image = self.draw_boxes(image, hot_windows)
            ret_val.append(hot_window_image)

        return tuple(ret_val)

    def draw_labeled_bboxes(self, img, labels, color=(0, 0, 255),thick=6):
        # Iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], color, thick)
        # Return the image
        return img

    def draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=6):
        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy
