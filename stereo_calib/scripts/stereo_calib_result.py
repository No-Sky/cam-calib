import sys
import os
import cv2
import numpy as np
import glob

class stereoCameral(object):
    def __init__(self):
        # 左相机内参数
        self.cam_matrix_left = np.asarray([[1.01371452e+03, 0.00000000e+00, 9.46057581e+02],
                                           [0.00000000e+00, 1.01374887e+03,
                                               5.18433343e+02],
                                           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        # 右相机内参数
        self.cam_matrix_right = np.asarray([[1.00961974e+03, 0.00000000e+00, 9.67193243e+02],
                                            [0.00000000e+00, 1.01001406e+03,
                                                4.93110992e+02],
                                            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

        # 左右相机畸变系数:[k1, k2, p1, p2, k3]
        self.distortion_l = np.asarray([[-3.69051939e-01,  1.61718172e-01, -7.83087722e-05,  2.58040137e-04,
                                         -3.75137954e-02]])
        self.distortion_r = np.asarray([[-3.67265340e-01,  1.55719616e-01, -1.88885026e-04,  4.45146376e-04,
                                         -3.39251782e-02]])
        # 旋转矩阵

        self.R = np.asarray([[9.99947132e-01, -9.92989790e-03,  2.67023488e-03],
                             [9.93049946e-03,  9.99950669e-01, -2.12118829e-04],
                             [-2.66799684e-03,  2.38624381e-04,  9.99996412e-01]])
        # 平移矩阵
        self.T = np.asarray([[-9.16059047e+01],
                             [-7.31339482e-03],
                             [1.68750703e+00]])

        self.baseline = self.T[0]

def undistortion(image, camera_matrix, dist_coeff):
    # 消除畸变
    undistortion_image = cv2.undistort(image, camera_matrix, dist_coeff)

    return undistortion_image

# 获取畸变校正和立体校正的映射变换矩阵、重投影矩阵
# @param：config是一个类，存储着双目标定的参数:config = stereoconfig.stereoCamera()


def getRectifyTransform(height, width, config):
    # 读取内参和外参
    left_K = config.cam_matrix_left
    right_K = config.cam_matrix_right
    left_distortion = config.distortion_l
    right_distortion = config.distortion_r
    R = config.R
    T = config.T

    # 计算校正变换
    height = int(height)
    width = int(width)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        left_K, left_distortion, right_K, right_distortion, (width, height), R, T, alpha=-1)

    map1x, map1y = cv2.initUndistortRectifyMap(
        left_K, left_distortion, R1, P1, (width, height), cv2.CV_16SC2)
    map2x, map2y = cv2.initUndistortRectifyMap(
        right_K, right_distortion, R2, P2, (width, height), cv2.CV_16SC2)
    # print(width, height)

    return map1x, map1y, map2x, map2y, Q


# 畸变校正和立体校正
def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
    rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_LINEAR)
    rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_LINEAR)

    return rectifyed_img1, rectifyed_img2


def read_images(cal_path):
    filepath = glob.glob(cal_path + '/*.jpg')
    filepath.sort()
    return filepath


def main():
    imgPath = "/media/sky/files/slam_study/calibration/data/23_06_22_17_36_46"
    imagesL = read_images(os.path.join(imgPath, "left"))
    imagesR = read_images(os.path.join(imgPath, "right"))
    imgLSavePath = os.path.join(imgPath, "rectify/left")
    imgRSavePath = os.path.join(imgPath, "rectify/right")
    if os.path.exists(imgLSavePath) is False:
        os.makedirs(imgLSavePath)
    if os.path.exists(imgRSavePath) is False:
        os.makedirs(imgRSavePath)

    height, width = 1080, 1920
    config = stereoCameral()    # 读取相机内参和外参

    for i in range(len(imagesL)):
        imgL = cv2.imread(imagesL[i])
        imgR = cv2.imread(imagesR[i])
        
        # cv2.imshow("undistortion before", imgL)
        # 去畸变
        # imgL = undistortion(imgL, config.cam_matrix_left, config.distortion_l)
        # imgR = undistortion(imgR, config.cam_matrix_right, config.distortion_r)
        
        # cv2.imshow("undistortion after", imgL)
        
        # 去畸变和几何极线对齐
        map1x, map1y, map2x, map2y, Q = getRectifyTransform(
            height, width, config)
        iml_rectified, imr_rectified = rectifyImage(
            imgL, imgR, map1x, map1y, map2x, map2y)
        
        # cv2.imshow("rectify after", iml_rectified)
        # cv2.waitKey(0)
        

        # 保存图片
        cv2.imwrite(os.path.join(imgLSavePath, "%04d.jpg" % i), iml_rectified)
        cv2.imwrite(os.path.join(imgRSavePath, "%04d.jpg" % i), imr_rectified)
        
    print("rectify done!")
    
if __name__ == '__main__':
    main()
        
        
