#include <algorithm>
#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <stdio.h>
#include <time.h>
#include <vector>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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

int main(int args, char **argv) {
  std::vector<std::string> imgPath;
  loadImagePath("/media/sky/files/calibration/data/23_06_22_17_36_46/left",
                imgPath);
  std::ofstream fout("../result/left_calib_result.txt"); /* 保存标定结果的文件 */

  // 读取每一幅图像，从中提取出角点，然后对角点进行亚像素精确化
  int image_count = 0;                    /* 图像数量 */
  cv::Size image_size;                    /* 图像的尺寸 */
  cv::Size board_size = cv::Size(11, 8); /* 标定板上每行、列的角点数 */
  std::vector<cv::Point2f> image_points_buf; /* 缓存每幅图像上检测到的角点 */
  std::vector<std::vector<cv::Point2f>>
      image_points_seq; /* 保存检测到的所有角点 */
  // std::string filename;      // 图片名
  std::vector<std::string> filenames;

  for (auto it = 0; it < imgPath.size(); it++) {
    ++image_count;
    cv::Mat imageInput = cv::imread(imgPath[it]);
    filenames.emplace_back(imgPath[it]);
    // 读入第一张图片时获取图片大小
    if (image_count == 1) {
      image_size.width = imageInput.cols;
      image_size.height = imageInput.rows;
    }

    // std::cout << "idx: " << it << "\n"
    //           << "width: " << imageInput.cols << " " << "height: " << imageInput.rows << std::endl; 

    // 提取角点
    if (!cv::findChessboardCorners(imageInput, board_size, image_points_buf)) {
      std::cout << "**" << imgPath[it] << "** cannot find chessboard corners!"
                << std::endl;
      exit(1);
    } else {
      cv::Mat view_gray;
      // 转灰度图
      cv::cvtColor(imageInput, view_gray, cv::COLOR_RGB2GRAY);

      /* 亚像素精确化 */
      // image_points_buf 初始的角点坐标向量，同时作为亚像素坐标位置的输出
      // Size(5,5) 搜索窗口大小
      // （-1，-1）表示没有死区
      // TermCriteria 角点的迭代过程的终止条件,
      // 可以为迭代次数和角点精度两者的组合
      //参数：
      // type –	终止条件类型:
      // maxCount –	计算的迭代数或者最大元素数
      // epsilon – 当达到要求的精确度或参数的变化范围时，迭代算法停止
      // type可选：
      // TermCriteria::COUNT //达到最大迭代次数 =TermCriteria::MAX_ITER
      // TermCriteria::EPS	//达到精度
      // TermCriteria::COUNT + TermCriteria::EPS //以上两种同时作为判定条件
      cv::cornerSubPix(
          view_gray, image_points_buf, cv::Size(5, 5), cv::Size(-1, -1),
          cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30,
                           0.1));

      image_points_seq.emplace_back(image_points_buf); // 保存亚像素角点

      // 在图像上显示角点位置
      // 用于在图像中标记角点
      cv::drawChessboardCorners(view_gray, board_size, image_points_buf, false);

      cv::imshow("Camera Calibration", view_gray);

      cv::waitKey(500); // 暂停0.5s
    }
  }

  int cornersNum = board_size.width * board_size.height; // 每张图片上总的角点数

  // ========================= 以下是相机标定 ==========================

  /* 棋盘三维信息 */
  cv::Size square_size =
      cv::Size(50, 50); /* 实际测量得到的标定板上每个棋盘格的大小 */
  std::vector<std::vector<cv::Point3f>>
      object_points; /* 保存标定板上角点的三维坐标 */

  /* 内外参数 */
  cv::Mat cameraMatrix =
      cv::Mat(3, 3, CV_32FC1, cv::Scalar::all(0)); /* 相机内参矩阵 */
  std::vector<int> point_counts; // 每幅图像中角点的数量
  cv::Mat distCoeffs =
      cv::Mat(1, 5, CV_32FC1,
              cv::Scalar::all(0)); /* 相机的5个畸变系数：k1,k2,p1,p2,k3 */
  std::vector<cv::Mat> tvecsMat; /* 每幅图像的旋转向量 */
  std::vector<cv::Mat> rvecsMat; /* 每幅图像的平移向量 */

  /* 初始化标定板上角点的三维坐标 */
  int i, j, t;
  for (t = 0; t < image_count; t++) {
    std::vector<cv::Point3f> tempPointSet;
    for (i = 0; i < board_size.height; i++) {
      for (j = 0; j < board_size.width; j++) {
        cv::Point3f realPoint;

        /* 假设标定板放在世界坐标系中z=0的平面上 */
        realPoint.x = i * square_size.width;
        realPoint.y = j * square_size.height;
        realPoint.z = 0;
        tempPointSet.push_back(realPoint);
      }
    }
    object_points.push_back(tempPointSet);
  }

  /* 初始化每幅图像中的角点数量，假定每幅图像中都可以看到完整的标定板 */
  for (i = 0; i < image_count; i++) {
    point_counts.push_back(board_size.width * board_size.height);
  }

  /* 开始标定 */
  // object_points 世界坐标系中的角点的三维坐标
  // image_points_seq 每一个内角点对应的图像坐标点
  // image_size 图像的像素尺寸大小
  // cameraMatrix 输出，内参矩阵
  // distCoeffs 输出，畸变系数
  // rvecsMat 输出，旋转向量
  // tvecsMat 输出，位移向量
  // 0 标定时所采用的算法
  // 执行标定操作 * RMS error    |  RMSE
  double rms = calibrateCamera(object_points, image_points_seq, image_size, cameraMatrix,
                  distCoeffs, rvecsMat, tvecsMat, 0);
  fout << "RMS error report by calibrateCamera: " << rms << std::endl << std::endl;

  //------------------------标定完成------------------------------------

  // -------------------对标定结果进行评价------------------------------

  double total_err = 0.0; /* 所有图像的平均误差的总和 */
  double err = 0.0;       /* 每幅图像的平均误差 */
  std::vector<cv::Point2f> image_points2; /* 保存重新计算得到的投影点 */
  fout << "每幅图像的标定误差：\n";

  for (i = 0; i < image_count; i++) {
    std::vector<cv::Point3f> tempPointSet = object_points[i];

    /* 通过得到的摄像机内外参数，对空间的三维点进行重新投影计算，得到新的投影点
     */
    projectPoints(tempPointSet, rvecsMat[i], tvecsMat[i], cameraMatrix,
                  distCoeffs, image_points2);

    /* 计算新的投影点和旧的投影点之间的误差*/
    std::vector<cv::Point2f> tempImagePoint = image_points_seq[i];
    cv::Mat tempImagePointMat = cv::Mat(1, tempImagePoint.size(), CV_32FC2);
    cv::Mat image_points2Mat = cv::Mat(1, image_points2.size(), CV_32FC2);

    for (int j = 0; j < tempImagePoint.size(); j++) {
      image_points2Mat.at<cv::Vec2f>(0, j) =
          cv::Vec2f(image_points2[j].x, image_points2[j].y);
      tempImagePointMat.at<cv::Vec2f>(0, j) =
          cv::Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);
    }
    err = norm(image_points2Mat, tempImagePointMat, cv::NORM_L2);
    total_err += err /= point_counts[i];
    fout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << std::endl;
  }
  fout << "总体平均误差：" << total_err / image_count << "像素" << std::endl
       << std::endl;

  //-------------------------评价完成---------------------------------------------

  //-----------------------保存定标结果-------------------------------------------
  cv::Mat rotation_matrix =
      cv::Mat(3, 3, CV_32FC1, cv::Scalar::all(0)); /* 保存每幅图像的旋转矩阵 */
  fout << "相机内参数矩阵：" << std::endl;
  fout << cameraMatrix << std::endl << std::endl;
  fout << "畸变系数：\n";
  fout << distCoeffs << std::endl << std::endl << std::endl;
  for (int i = 0; i < image_count; i++) {
    fout << "第" << i + 1 << "幅图像的旋转向量：" << std::endl;
    fout << tvecsMat[i] << std::endl;

    /* 将旋转向量转换为相对应的旋转矩阵 */
    Rodrigues(tvecsMat[i], rotation_matrix);
    fout << "第" << i + 1 << "幅图像的旋转矩阵：" << std::endl;
    fout << rotation_matrix << std::endl;
    fout << "第" << i + 1 << "幅图像的平移向量：" << std::endl;
    fout << rvecsMat[i] << std::endl << std::endl;
  }
  fout << std::endl;

  //--------------------标定结果保存结束-------------------------------

  //----------------------显示定标结果--------------------------------

  cv::Mat mapx = cv::Mat(image_size, CV_32FC1);
  cv::Mat mapy = cv::Mat(image_size, CV_32FC1);
  cv::Mat R = cv::Mat::eye(3, 3, CV_32F);
  std::string imageFileName;
  std::stringstream StrStm;
  for (int i = 0; i != image_count; i++) {
    initUndistortRectifyMap(cameraMatrix, distCoeffs, R, cameraMatrix,
                            image_size, CV_32FC1, mapx, mapy);
    cv::Mat imageSource = cv::imread(filenames[i]);
    cv::Mat newimage = imageSource.clone();
    remap(imageSource, newimage, mapx, mapy, cv::INTER_LINEAR);
    StrStm.clear();
    imageFileName.clear();
    StrStm << i + 1;
    StrStm >> imageFileName;
    imageFileName += "_d.jpg";
    imwrite(imageFileName, newimage);
  }

  fout.close();

  return 0;
}