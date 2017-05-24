import numpy as np
import pickle
import cv2
import glob
import matplotlib.image as mpimg
from skimage.feature import hog


class VehicleFeatureExtractor:
    """
    A class used to extract features that can be used to
    detect vehicles in a frame.
    """

    def __init__(self,
                 color_space='RGB', spatial_size=(32, 32),
                 hist_bins=32, orient=9,
                 pix_per_cell=8, cell_per_block=2, hog_channel=0,
                 spatial_feat=True, hist_feat=True, hog_feat=True):

        self.color_space = color_space
        self.spatial_size = spatial_size
        self.hist_feat = hist_feat
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hog_channel = hog_channel
        self.spatial_feat = spatial_feat
        self.hist_bins = hist_bins
        self.orient = orient
        self.hog_feat = hog_feat

        return

    def color_hist(self, image, bins_range=(0., 1.)):
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(image[:, :, 0], bins=self.hist_bins, range=bins_range)
        channel2_hist = np.histogram(image[:, :, 1], bins=self.hist_bins, range=bins_range)
        channel3_hist = np.histogram(image[:, :, 2], bins=self.hist_bins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    def bin_spatial(self, img):
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(img, self.spatial_size).ravel()
        return features.astype(np.float64)

    def get_hog_features(self, image,
                         vis=False, feature_vec=True):
        # Call with two outputs if vis==True
        if vis:
            features, hog_image = hog(image, orientations=self.orient,
                                      pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                                      cells_per_block=(self.cell_per_block, self.cell_per_block),
                                      transform_sqrt=True,
                                      visualise=vis, feature_vector=feature_vec)
            return features, hog_image
        # Otherwise call with one output
        else:
            features = hog(image, orientations=self.orient,
                           pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                           cells_per_block=(self.cell_per_block, self.cell_per_block),
                           transform_sqrt=True,
                           visualise=vis, feature_vector=feature_vec)
            return features

    def single_img_features(self, image):
        # 1) Define an empty list to receive features
        img_features = []
        # 2) Apply color conversion if other than 'RGB'
        if self.color_space != 'RGB':
            if self.color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif self.color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif self.color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif self.color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif self.color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            else:
                #default to rgb
                feature_image = np.copy(image)
        else:
            feature_image = np.copy(image)
        # 3) Compute spatial features if flag is set
        if self.spatial_feat:
            spatial_features = self.bin_spatial(feature_image)
            # 4) Append features to list
            img_features.append(spatial_features)

        # 5) Compute histogram features if flag is set
        if self.hist_feat:
            hist_features = self.color_hist(feature_image)
            # 6) Append features to list
            img_features.append(hist_features)

        # 7) Compute HOG features if flag is set
        if self.hog_feat:
            if self.hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(self.get_hog_features(feature_image[:, :, channel],
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = self.get_hog_features(feature_image[:, :, self.hog_channel], vis=False, feature_vec=True)
            # 8) Append features to list
            img_features.append(hog_features)

        # 9) Return concatenated array of features
        return np.concatenate(img_features)

    def extract_features(self, imgs):
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for file in imgs:
            # Read in each one by one
            image = mpimg.imread(file)
            image_features = self.single_img_features(image)
            features.append(image_features)
        # Return list of feature vectors
        return features

    def save(self, file_name='feature_extractor_settings.pkl'):

        output = open(file_name, 'wb')
        pickle.dump(self, output)

        output.close()

    @staticmethod
    def load(file_name):
        # we open the file for reading
        file = open(file_name, 'rb')

        obj = pickle.load(file)
        file.close()

        return obj

