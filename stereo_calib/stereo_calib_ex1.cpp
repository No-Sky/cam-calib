#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#include <dirent.h>
#include <sys/types.h>

using namespace std;
using namespace cv;

void loadImagePath(std::string imgDirPath, std::vector<std::string> &vimgPath) {

  DIR *pDir;
  struct dirent *ptr;
  if (!(pDir = opendir(imgDirPath.c_str()))) {
    std::cout << "Folder doesn't Exist!" << std::endl;
    return;
  }

  while ((ptr = readdir(pDir)) != 0) {
    if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
      vimgPath.push_back(imgDirPath + "/" + ptr->d_name);
    }
  }
  sort(vimgPath.begin(), vimgPath.end());

  closedir(pDir);
}

void getImgsPoints(vector<Mat> imgs, vector<vector<Point2f>> &imgsPoints,
                   Size board_size) {
  for (int i = 0; i < imgs.size(); i++) {
    Mat img1 = imgs[i];
    Mat gray1;
    cvtColor(img1, gray1, COLOR_BGR2GRAY);
    vector<Point2f> img1_points;
    findChessboardCorners(gray1, board_size, img1_points); //计算方格标定板角点
    // find4QuadCornerSubpix(gray1, img1_points,
    //                       Size(5, 5)); //细化方格标定板角点坐标
    cv::cornerSubPix(
        gray1, img1_points, cv::Size(11, 11), cv::Size(-1, 1),
        cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30,
                         0.01)); // 亚像素角点坐标
    imgsPoints.push_back(img1_points);
  }
}

void saveIntrinsicParams(cv::Mat &M1, cv::Mat &D1, cv::Mat &M2, cv::Mat &D2) {
  // save intrinsic parameters
  FileStorage fs("intrinsics.yml", FileStorage::WRITE);
  if (fs.isOpened()) {
    fs << "M1" << M1 << "D1" << D1 << "M2" << M2 << "D2" << D2;
    fs.release();
  } else
    cout << "Error: can not save the intrinsic parameters\n";
}

void saveExtrinsicParams(cv::Mat &R, cv::Mat &T, cv::Mat &R1, cv::Mat &R2,
                         cv::Mat &P1, cv::Mat &P2, cv::Mat &Q) {
  FileStorage fs("extrinsics.yml", FileStorage::WRITE);
  if (fs.isOpened()) {
    fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2"
       << P2 << "Q" << Q;
    fs.release();
  } else
    cout << "Error: can not save the extrinsic parameters\n";
}

int main(int args, char** argv) {
  //读取所有图像
  vector<Mat> imgLs;
  vector<Mat> imgRs;
  string imgLName;
  string imgRName;
  vector<string> imgLPath;
  loadImagePath("/media/sky/files/slam_study/calibration/data/23_06_22_17_36_46/left",
                imgLPath);
  vector<string> imgRPath;
  loadImagePath("/media/sky/files/slam_study/calibration/data/23_06_22_17_36_46/right",
                imgRPath);

  for (int i = 0; i < imgLPath.size(); i++) {
    Mat imgL = imread(imgLPath[i]);
    Mat imgR = imread(imgRPath[i]);
    imgLs.push_back(imgL);
    imgRs.push_back(imgR);
  }

  Size board_size = Size(11, 8); //方格标定板内角点数目（行，列）
  vector<vector<Point2f>> imgLsPoints;
  vector<vector<Point2f>> imgRsPoints;
  getImgsPoints(imgLs, imgLsPoints, board_size);
  getImgsPoints(imgRs, imgRsPoints, board_size);

  //生成棋盘格每个内角点的空间三维坐标
  Size squareSize = Size(50, 50); //棋盘格每个方格的真实尺寸
  vector<vector<Point3f>> objectPoints;
  for (int i = 0; i < imgLsPoints.size(); i++) {
    vector<Point3f> tempPointSet;
    for (int j = 0; j < board_size.height; j++) {
      for (int k = 0; k < board_size.width; k++) {
        Point3f realPoint;
        // 假设标定板为世界坐标系的z平面，即z=0
        realPoint.x = j * squareSize.width;
        realPoint.y = k * squareSize.height;
        realPoint.z = 0;
        tempPointSet.push_back(realPoint);
      }
    }
    objectPoints.push_back(tempPointSet);
  }

  //图像尺寸
  Size imageSize;
  imageSize.width = imgLs[0].cols;
  imageSize.height = imgLs[0].rows;

  Mat Matrix1, dist1, Matrix2, dist2, rvecs, tvecs;
  calibrateCamera(objectPoints, imgLsPoints, imageSize, Matrix1, dist1, rvecs,
                  tvecs, 0);
  calibrateCamera(objectPoints, imgRsPoints, imageSize, Matrix2, dist2, rvecs,
                  tvecs, 0);
  std::cout << "M1: \n"
            << Matrix1 << "\n"
            << "D1: \n"
            << dist1 << "\n"
            << "M2: \n"
            << Matrix2 << "\n"
            << "D2: \n"
            << dist2 << "\n"
            << std::endl;
  saveIntrinsicParams(Matrix1, dist1, Matrix2, dist2);

  //双目相近进行标定
  Mat R, T, E, F;
  double rms = stereoCalibrate(objectPoints, imgLsPoints, imgRsPoints, Matrix1,
                               dist1, Matrix2, dist2, imageSize, R, T, E, F,
                               CALIB_USE_INTRINSIC_GUESS);
  std::cout << "Stereo Calibration done with RMS error = " << rms << std::endl;

  //计算校正变换矩阵
  Mat R1, R2, P1, P2, Q;
  stereoRectify(Matrix1, dist1, Matrix2, dist2, imageSize, R, T, R1, R2, P1, P2,
                Q, 0);

  std::cout << "R: \n"
            << R << "\n"
            << "T: \n"
            << T << "\n"
            << "R1: \n"
            << R1 << "\n"
            << "R2: \n"
            << R2 << "\n"
            << "P1: \n"
            << P1 << "\n"
            << "P2: \n"
            << P2 << "\n"
            << "Q: \n"
            << Q << "\n"
            << std::endl;
  saveExtrinsicParams(R, T, R1, R2, P1, P2, Q);

  //计算校正映射矩阵
  Mat map11, map12, map21, map22;
  initUndistortRectifyMap(Matrix1, dist1, R1, P1, imageSize, CV_16SC2, map11,
                          map12);
  initUndistortRectifyMap(Matrix2, dist2, R2, P2, imageSize, CV_16SC2, map21,
                          map22);

  for (int i = 0; i < imgLs.size(); i++) {
    //进行校正映射
    Mat img1r, img2r;
    remap(imgLs[i], img1r, map11, map12, INTER_LINEAR);
    remap(imgRs[i], img2r, map21, map22, INTER_LINEAR);

    //拼接图像
    Mat result;
    hconcat(img1r, img2r, result);

    //绘制直线，用于比较同一个内角点y轴是否一致
    line(result, Point(0, imgLsPoints[i][0].y),
         Point(result.cols, imgLsPoints[i][0].y), Scalar(0, 0, 255), 2);
    // for (int j = 0; j < result.rows; j += 16)
    //     line(result, Point(0, imgLsPoints[i][j].y),
    //      Point(result.cols, imgLsPoints[i][j].y), Scalar(0, 0, 255), 2);

    namedWindow("校正后结果", WINDOW_NORMAL);
    imshow("校正后结果", result);
    waitKey(0);
  }
  return 0;
}
