#pragma once

#include<iostream>
#include<opencv2/opencv.hpp>
#include<Eigen/Dense>
#include<opencv2/opencv.hpp>
#include<ros/console.h>
#include "parameters.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/CameraFactory.h"
using namespace std;
using namespace camodocal;
class FeatureTracker
{
public:
    FeatureTracker();
    //跟踪图片
    map<int, vector< Eigen::Matrix<double, 7, 1>>>  trackImage(double t, cv::Mat &Imgleft, cv::Mat &Imgright); 
    void readIntrinsicParameter(const vector<string> &calib_file);
    cv::Mat imTrack;
    
private:

    bool inBorder(cv::Point2f pt);
    void setMask(vector<cv::Point2f> &cur_pts);
    vector<cv::Point2f> undistortedPts(const vector<cv::Point2f> &pts, camodocal::CameraPtr cam);   //去畸变并投影得到归一化坐标
    vector<cv::Point2f> ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts, map<int,cv::Point2f> &id_pts);

    void drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight, 
                               vector<int> &curLeftIds,
                               vector<cv::Point2f> &curLeftPts, 
                               vector<cv::Point2f> &curRightPts,
                               map<int, cv::Point2f> &prevLeftPtsMap);

    cv::Mat  prev_img;   
    double cur_time, prev_time;  //用来计算速度
    cv::Mat mask;
    vector<cv::Point2f>  prev_pts, prev_right_pts  ,n_pts;    //用来光流跟踪
    map<int, cv::Point2f>  prev_id_pts, prev_right_id_pts;   //用来计算特征点速度
    vector<int> ids, right_ids;      //特征点的id
    vector<int> track_cnt;   //特征点的跟踪次数
    int n_id;
    vector<camodocal::CameraPtr> m_camera;   //相机模型，主要用liftProjective函数

    map<int, cv::Point2f> prevLeftPtsMap;  //存上次左相机的特征点，用来显示相对位移
};