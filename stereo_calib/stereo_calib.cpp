//双目相机标定 stereo_calib.cpp
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <ctype.h>
#include <iostream>
#include <iterator>
#include <ostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include <dirent.h>
#include <sys/types.h>

template <typename T>
std::ostream &operator<<(std::ostream &out, const std::vector<T> &ve) {
  out << "[";
  char delim[3] = {'\0', ' ', '\0'};
  for (auto &item : ve) {
    out << delim << item;
    delim[0] = ',';
  }
  out << "]\n";
  return out;
}

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

//摄像头的分辨率
const int imageWidth = 1920;
const int imageHeight = 1080;
//横向的角点数目
const int boardWidth = 11;
//纵向的角点数目
const int boardHeight = 8;
//总的角点数目
const int boardCorner = boardWidth * boardHeight;
//相机标定时需要采用的图像帧数
const int frameNumber = 24;
//标定板黑白格子的大小 单位是mm
const int squareSize = 50;
//标定板的总内角点
const cv::Size boardSize = cv::Size(boardWidth, boardHeight);
cv::Size imageSize = cv::Size(imageWidth, imageHeight);

cv::Mat R, T, E, F;
// R旋转矢量 T平移矢量 E本征矩阵 F基础矩阵
std::vector<cv::Mat> rvecs; // R
std::vector<cv::Mat> tvecs; // T
//左边摄像机所有照片角点的坐标集合
std::vector<std::vector<cv::Point2f>> imagePointL;
//右边摄像机所有照片角点的坐标集合
std::vector<std::vector<cv::Point2f>> imagePointR;
//各图像的角点的实际的物理坐标集合
std::vector<std::vector<cv::Point3f>> objRealPoint;
//左边摄像机某一照片角点坐标集合
std::vector<cv::Point2f> cornerL;
//右边摄像机某一照片角点坐标集合
std::vector<cv::Point2f> cornerR;

cv::Mat rgbImageL, grayImageL;
cv::Mat rgbImageR, grayImageR;

cv::Mat intrinsic;
cv::Mat distortion_coeff;
//校正旋转矩阵R，投影矩阵P，重投影矩阵Q
cv::Mat Rl, Rr, Pl, Pr, Q;

//映射表
cv::Mat mapLx, mapLy, mapRx, mapRy;
cv::Rect validROIL, validROIR;
//图像校正之后，会对图像进行裁剪，其中，validROI裁剪之后的区域
/*事先标定好的左相机的内参矩阵
fx 0 cx
0 fy cy
0  0  1
*/
cv::Mat cameraMatrixL =
    (cv::Mat_<double>(3, 3) << 013.882047219104, 0, 946.3430491448021, 0,
     1013.905239617196, 518.5861232462403, 0, 0, 1);
//获得的畸变参数
cv::Mat distCoeffL =
    (cv::Mat_<double>(5, 1) << -0.3691712171064356, 0.1618375698369081,
     -9.406354268972872e-05, 0.000241917824769911, -0.03757177488367499);
/*事先标定好的右相机的内参矩阵
fx 0 cx
0 fy cy
0  0  1
*/
cv::Mat cameraMatrixR =
    (cv::Mat_<double>(3, 3) << 21009.822936743211, 0, 967.3826649740179, 0,
     1010.200648624123, 493.2886734664576, 0, 0, 1);
cv::Mat distCoeffR =
    (cv::Mat_<double>(5, 1) << -0.3673211899693686, 0.1557541373281207,
     -0.0002188379716904138, 0.0004518684107884466, -0.03393382018853325);

/*计算标定板上模块的实际物理坐标*/
void calRealPoint(std::vector<std::vector<cv::Point3f>> &obj, int boardWidth,
                  int boardHeight, int imgNumber, int squareSize) {
  std::vector<cv::Point3f> imgpoint;
  for (int rowIndex = 0; rowIndex < boardHeight; rowIndex++) {
    for (int colIndex = 0; colIndex < boardWidth; colIndex++) {
      imgpoint.push_back(
          cv::Point3f(rowIndex * squareSize, colIndex * squareSize, 0));
    }
  }
  for (int imgIndex = 0; imgIndex < imgNumber; imgIndex++) {
    obj.push_back(imgpoint);
  }
}

void outputCameraParam(void) {
  /*保存数据*/
  /*输出数据*/
  cv::FileStorage fs("../result/intrisics.yml", cv::FileStorage::WRITE);
  if (fs.isOpened()) {
    fs << "cameraMatrixL" << cameraMatrixL << "cameraDistcoeffL" << distCoeffL
       << "cameraMatrixR" << cameraMatrixR << "cameraDistcoeffR" << distCoeffR;
    fs.release();
    std::cout << "cameraMatrixL=:" << cameraMatrixL << std::endl
              << "cameraDistcoeffL=:" << distCoeffL << std::endl
              << "cameraMatrixR=:" << cameraMatrixR << std::endl
              << "cameraDistcoeffR=:" << distCoeffR << std::endl;
  } else {
    std::cout << "Error: can not save the intrinsics!!!!" << std::endl;
  }

  fs.open("../result/extrinsics.yml", cv::FileStorage::WRITE);
  if (fs.isOpened()) {
    fs << "R" << R << "T" << T << "Rl" << Rl << "Rr" << Rr << "Pl" << Pl << "Pr"
       << Pr << "Q" << Q;
    std::cout << "R=" << R << std::endl
              << "T=" << T << std::endl
              << "Rl=" << Rl << std::endl
              << "Rr" << Rr << std::endl
              << "Pl" << Pl << std::endl
              << "Pr" << Pr << std::endl
              << "Q" << Q << std::endl;
    fs.release();
  } else {
    std::cout << "Error: can not save the extrinsic parameters\n";
  }
}

int main(int argc, char **argv) {
  cv::Mat img;
  int goodFrameCount = 0;

  std::vector<std::string> imgLPath;
  loadImagePath("/media/sky/files/calibration/data/23_06_22_17_36_46/left",
                imgLPath);
  std::vector<std::string> imgRPath;
  loadImagePath("/media/sky/files/calibration/data/23_06_22_17_36_46/right",
                imgRPath);

  while (goodFrameCount < frameNumber) {
    /* 读取左边的图像 */
    rgbImageL = cv::imread(imgLPath[goodFrameCount]);
    cv::imshow("chessboardL", rgbImageL);
    cv::cvtColor(rgbImageL, grayImageL, cv::COLOR_RGB2GRAY);
    /* 读取左边的图像 */
    rgbImageR = cv::imread(imgRPath[goodFrameCount]);
    cv::imshow("chessboardR", rgbImageR);
    cv::cvtColor(rgbImageR, grayImageR, cv::COLOR_RGB2GRAY);

    bool isFindL, isFindR;
    isFindL = cv::findChessboardCorners(rgbImageL, boardSize, cornerL,
                                        cv::CALIB_CB_ADAPTIVE_THRESH |
                                            cv::CALIB_CB_NORMALIZE_IMAGE);
    isFindR = cv::findChessboardCorners(rgbImageR, boardSize, cornerR,
                                        cv::CALIB_CB_ADAPTIVE_THRESH |
                                            cv::CALIB_CB_NORMALIZE_IMAGE);

    if (isFindL == true && isFindR == true) {
      cv::cornerSubPix(
          grayImageL, cornerL, cv::Size(11, 11), cv::Size(-1, 1),
          cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30,
                           0.01));
      cv::drawChessboardCorners(rgbImageL, boardSize, cornerL, isFindL);
      cv::imshow("chessboardL", rgbImageL);
      imagePointL.push_back(cornerL);

      cv::cornerSubPix(
          grayImageR, cornerR, cv::Size(11, 11), cv::Size(-1, -1),
          cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30,
                           0.01));
      cv::drawChessboardCorners(rgbImageR, boardSize, cornerR, isFindR);
      cv::imshow("chessboardR", rgbImageR);
      imagePointR.push_back(cornerR);

      goodFrameCount++;
      std::cout << "the image" << goodFrameCount << " is good" << std::endl;
    } else {
      std::cout << "the image is bad please try again" << std::endl;
    }
    if (cv::waitKey(10) == 'q') {
      break;
    }
  }

  //计算实际的校正点的三维坐标，根据实际标定格子的大小来设置
  calRealPoint(objRealPoint, boardWidth, boardHeight, frameNumber, squareSize);
  std::cout << "cal real successful" << std::endl;

  //   cv::Mat cameraMatrix[2], distCoeffs[2];
  //   cameraMatrix[0] =
  //       initCameraMatrix2D(objRealPoint, imagePointR, imageSize, 0);
  //   cameraMatrix[1] =
  //       initCameraMatrix2D(objRealPoint, imagePointL, imageSize, 0);

  //标定摄像头
  double rms = cv::stereoCalibrate(
      objRealPoint, imagePointL, imagePointR, cameraMatrixL, distCoeffL,
      cameraMatrixR, distCoeffR, cv::Size(imageWidth, imageHeight), R, T, E, F,
      cv::CALIB_USE_INTRINSIC_GUESS,
      cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100,
                       1e-5));
  std::cout << "Stereo Calibration done with RMS error = " << rms << std::endl;

  cv::stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR,
                    imageSize, R, T, Rl, Rr, Pl, Pr, Q,
                    cv::CALIB_ZERO_DISPARITY, -1, imageSize, &validROIL,
                    &validROIR);

  bool isVerticalStereo = fabs(Pr.at<double>(1, 3)) > fabs(Pr.at<double>(0, 3));

  //摄像机校正映射
  cv::initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pl, imageSize,
                              CV_32FC1, mapLx, mapLy);
  cv::initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize,
                              CV_32FC1, mapRx, mapRy);

  cv::Mat rectifyImageL, rectifyImageR;
  cv::cvtColor(grayImageL, rectifyImageL, cv::COLOR_GRAY2BGR);
  cv::cvtColor(grayImageR, rectifyImageR, cv::COLOR_GRAY2BGR);

  cv::imshow("Recitify Before", rectifyImageL);
  std::cout << "按Q1退出..." << std::endl;
  //经过remap之后，左右相机的图像已经共面并且行对准了
  cv::Mat rectifyImageL2, rectifyImageR2;
  cv::remap(rectifyImageL, rectifyImageL2, mapLx, mapLy, cv::INTER_LINEAR);
  cv::remap(rectifyImageR, rectifyImageR2, mapRx, mapRy, cv::INTER_LINEAR);

  cv::imshow("rectifyImageL", rectifyImageL2);
  cv::imshow("rectifyImageR", rectifyImageR2);

  outputCameraParam();

  //显示校正结果
  cv::Mat canvas;
  double sf;
  int w, h;
  if (!isVerticalStereo) {
    sf = 600. / MAX(imageSize.width, imageSize.height);
    w = cvRound(imageSize.width * sf);
    h = cvRound(imageSize.height * sf);
    canvas.create(h, w * 2, CV_8UC3);
  } else {
    sf = 300. / MAX(imageSize.width, imageSize.height);
    w = cvRound(imageSize.width * sf);
    h = cvRound(imageSize.height * sf);
    canvas.create(h * 2, w, CV_8UC3);
  }

  //左图像画到画布上
  cv::Mat canvasPart = canvas(cv::Rect(0, 0, w, h));
  cv::resize(rectifyImageL2, canvasPart, canvasPart.size(), 0, 0,
             cv::INTER_AREA);
  cv::Rect vroiL(cvRound(validROIL.x * sf), cvRound(validROIL.y * sf),
                 cvRound(validROIL.width * sf), cvRound(validROIL.height * sf));
  cv::rectangle(canvasPart, vroiL, cv::Scalar(0, 0, 255), 3, 8);

  std::cout << "Painted ImageL" << std::endl;

  //右图像画到画布上
  canvasPart = canvas(cv::Rect(w, 0, w, h));
  cv::resize(rectifyImageR2, canvasPart, canvasPart.size(), 0, 0,
             cv::INTER_LINEAR);
  cv::Rect vroiR(cvRound(validROIR.x * sf), cvRound(validROIR.y * sf),
                 cvRound(validROIR.width * sf), cvRound(validROIR.height * sf));
  cv::rectangle(canvasPart, vroiR, cv::Scalar(0, 255, 0), 3, 8);

  std::cout << "Painted ImageR" << std::endl;

  //画上对应的线条
  for (int j = 0; j < canvas.rows; j += 16)
    cv::line(canvas, cv::Point(0, j), cv::Point(canvas.cols, j),
             cv::Scalar(0, 255, 0), 1, 8);

  cv::imshow("rectified", canvas);

  std::cout << "wait key" << std::endl;
  cv::waitKey(0);

  return 0;
}