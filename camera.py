import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob, pickle

from skimage.transform._geometric import warp


class CameraCalibrator:

    mtx = None
    dist = None
    rvecs = None
    tvecs = None
    perspective_mtx = None
    inverse_perspective_mtx = None

    def __init__(self):
        self.img_points = []
        self.obj_points = []
        self.perspective_mtx = self.get_perspective_transform()
        self.inverse_perspective_mtx = self.get_perspective_transform(is_inverse=True)

    def save_calibration(self, file_name):
        pickle.dump(self, open(file_name, "wb"))

    def load_calibaration(self, file_name):
        calib = pickle.load( open( file_name, "rb" ) )
        self.mtx = calib.mtx
        self.dist = calib.dist
        self.rvecs = calib.rvecs
        self.tvecs = calib.tvecs

    def calibrate(self, image_name_pattern, nx, ny):
        objp = np.zeros((nx * ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        images = glob.glob(image_name_pattern)

        for image in images:

            img = mpimg.imread(image)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            if ret == True:
                self.img_points.append(corners)
                self.obj_points.append(objp)

        ret, self.mtx, self.dist, self.rvecs, self.tvecs =\
            cv2.calibrateCamera(self.obj_points, self.img_points, gray.shape[::-1], None, None)

    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

    def get_perspective_transform(self, is_inverse=False):

        src_pts = np.float32([[262, 684], [586, 457], [695, 457], [1019, 684]])
        dst_pts = np.float32([[262, 684], [262, 0], [1048, 0], [1048, 684]])
        if(is_inverse == False):
            return cv2.getPerspectiveTransform(src_pts, dst_pts)
        else:
            return  cv2.getPerspectiveTransform(dst_pts, src_pts)

    def warp(self, image):
        img_size = (image.shape[1], image.shape[0])
        return cv2.warpPerspective(image, self.perspective_mtx, img_size)

    def unwarp(self, image):
        img_size = (image.shape[1], image.shape[0])
        return cv2.warpPerspective(image, self.inverse_perspective_mtx, img_size)



