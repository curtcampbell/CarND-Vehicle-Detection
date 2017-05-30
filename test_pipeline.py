from camera import *
from lane_detector import *

from processor import *
from classifier import *
from detector import *
from features_extractor import VehicleFeatureExtractor

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip


def plot(image, title=None, cmap=None):
    plt.figure()
    if title is not None:
        plt.title(title)
    plt.imshow(image, cmap=cmap)



# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img


def main():

    calib = CameraCalibrator()
    calib.load_calibaration("cam_calib.p")
    lane_detector = LaneDetector(calib)

    # setup processing pipeline
    classifier = CarClassifier('car_classifier_model.pkl')
    feature_extractor = VehicleFeatureExtractor.load('feature_extractor_settings.pkl')

    processor = Processor(heatmap_threshold=2)

    # processor.add_detector(ObjectDetector(classifier,
    #                                       feature_extractor,
    #                                       xy_window=(158, 96),
    #                                       xy_overlap=(0.75,0.75),
    #                                       y_start_stop=[350, 720]))

    # processor.add_detector(ObjectDetector(classifier, feature_extractor,
    #                                       xy_window=(50, 40),
    #                                       xy_overlap=(0.70, 0.70),
    #                                       x_start_stop=(550, 1020),
    #                                       y_start_stop=(375, 500)))

    # image 1 and 2
    processor.add_detector(ObjectDetector(classifier, feature_extractor,
                                          xy_window=(100, 100),
                                          xy_overlap=(0.75, 0.75),
                                          y_start_stop=(360, 535)))

    # image 3
    processor.add_detector(ObjectDetector(classifier, feature_extractor,
                                          xy_window=(60, 60),
                                          xy_overlap=(0.65, 0.65),
                                          x_start_stop=(450,1055),
                                          y_start_stop=(370, 485)))

    # image 4 5
    processor.add_detector(ObjectDetector(classifier, feature_extractor,
                                          xy_window=(105, 105),
                                          xy_overlap=(0.7, 0.6),
                                          x_start_stop=(192,1280),
                                          y_start_stop=(350, 555)))

    def process_frame(image):
        output_image = lane_detector.process_frame(image)
        output_image = processor.process_frame(image, output_image)
        return output_image

    # Process image
    # test_file_name = '.\\Capture\\capture-8.png'

    test_file_name = '.\\test_images\\test1.jpg'
    in_image = mpimg.imread(test_file_name)
    norm_image = cv2.normalize(in_image, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    out_image, heat_map, hot_windows = processor.process_frame(norm_image, return_heat_map=True, return_hot_windows=True, process_as_video=False)
    plot(in_image, "Input frame")
    plot(out_image, "Final detection")
    plot(heat_map, "Heat map", cmap='hot')
    plot(hot_windows, "Hot windows")
    plt.show()

    # # Process clips
    # # clip = VideoFileClip('.\\test_video.mp4')
    # clip = VideoFileClip('.\\project_video.mp4')
    # # output_clip = clip.subclip(0, 4).fl_image(processor.process_frame)
    # # output_clip = clip.fl_image(processor.process_frame)
    # output_clip = clip.fl_image(process_frame)
    #
    # output_clip.write_videofile('.\\project_video_output.mp4', audio=False)


if __name__ == "__main__":
    main()