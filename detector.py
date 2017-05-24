import numpy as np
import cv2


class ObjectDetector:

    def __init__(self, classifier, feature_extractor,
                 x_start_stop=[None, None],
                 y_start_stop=[None, None],
                 xy_window=(64, 64),
                 xy_overlap=(0.5, 0.5),
                 heatmap_threshold=0):
        self.classifier = classifier
        self.feature_extractor = feature_extractor
        self.x_start_stop = x_start_stop
        self.y_start_stop = y_start_stop
        self.xy_window = xy_window
        self.xy_overlap = xy_overlap

    def slide_window(self, image):
        # If x and/or y start/stop positions not defined, set to image size
        if self.x_start_stop[0] is None:
            self.x_start_stop[0] = 0
        if self.x_start_stop[1] is None:
            self.x_start_stop[1] = image.shape[1]
        if self.y_start_stop[0] is None:
            self.y_start_stop[0] = 0
        if self.y_start_stop[1] is None:
            self.y_start_stop[1] = image.shape[0]
        # Compute the span of the region to be searched
        xspan = self.x_start_stop[1] - self.x_start_stop[0]
        yspan = self.y_start_stop[1] - self.y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(self.xy_window[0] * (1 - self.xy_overlap[0]))
        ny_pix_per_step = np.int(self.xy_window[1] * (1 - self.xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_buffer = np.int(self.xy_window[0] * (self.xy_overlap[0]))
        ny_buffer = np.int(self.xy_window[1] * (self.xy_overlap[1]))
        nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
        ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs * nx_pix_per_step + self.x_start_stop[0]
                endx = startx + self.xy_window[0]
                starty = ys * ny_pix_per_step + self.y_start_stop[0]
                endy = starty + self.xy_window[1]

                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows
        return window_list

    def search_windows(self, img, windows):

        # 1) Create an empty list to receive positive detection windows
        on_windows = []
        # 2) Iterate over all windows in the list
        for window in windows:
            # 3) Extract the test window from original image
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            # 4) Extract features for that window using single_img_features()
            features = self.feature_extractor.single_img_features(test_img)
            # 5) Scale extracted features to be fed to classifier
            test_features = self.classifier.scaler.transform(np.array(features).reshape(1, -1))
            # 6) Predict using your classifier
            prediction = self.classifier.predict(test_features)
            # 7) If positive (prediction == 1) then save the window
            if prediction == 1:
                on_windows.append(window)
        # 8) Return windows for positive detections
        return on_windows

    def get_detections(self, image):
        window_to_search = self.slide_window(image)
        detection_windows = self.search_windows(image, window_to_search)
        return detection_windows

