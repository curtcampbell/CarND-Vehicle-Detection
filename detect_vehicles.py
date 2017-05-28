from camera import *
from lane_detector import *

from processor import *
from classifier import *
from detector import *
from features_extractor import VehicleFeatureExtractor

from moviepy.editor import VideoFileClip


def main():

    # Lane detection code
    calib = CameraCalibrator()
    calib.load_calibaration("cam_calib.p")
    lane_detector = LaneDetector(calib)

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
        output_image = lane_detector.process_frame(image)
        output_image = processor.process_frame(image, output_image)
        return output_image

    # Process clips
    clip = VideoFileClip('.\\project_video.mp4')
    output_clip = clip.fl_image(process_frame)

    output_clip.write_videofile('.\\project_video_output.mp4', audio=False)


if __name__ == "__main__":
    main()