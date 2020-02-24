#pragma once

#include<iostream>
#include<opencv2/opencv.hpp>
#include<Eigen/Dense>
#include<opencv2/opencv.hpp>
#include<ros/ros.h>
#include "camodocal/camera_models/PinholeCamera.h"
using namespace std;

class FeatureTracker
{
public:
    FeatureTracker();
    ~FeatureTracker();
    //跟踪图片
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> trackImage(double t, cv::Mat &Imgleft, cv::Mat &Imgright); 
    
private:

    bool inBorder(cv::Point2f pt);
    void setMask(vector<cv::Point2f> &cur_pts);
    vector<cv::Point2f> undistortedPts(const vector<cv::Point2f> &pts, camodocal::CameraPtr cam);   //去畸变并投影得到归一化坐标
    vector<cv::Point2f> ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts, map<int,cv::Point2f> &id_pts);
    cv::Mat  prev_img;   
    double cur_time, prev_time;  //用来计算速度
    int row, col ;
    cv::Mat mask;
    vector<cv::Point2f>  prev_pts, prev_right_pts  ,n_pts;    //用来光流跟踪
    map<int, cv::Point2f>  prev_id_pts, prev_right_id_pts;   //用来计算特征点速度
    vector<int> ids, right_ids;      //特征点的id
    vector<int> track_cnt;   //特征点的跟踪次数
    int n_id;
    int MIN_DIST;  //特征点之间的最小距离
    int MAX_CNT;  //一帧图像中特征点数目的上阈值
    vector<camodocal::CameraPtr> m_camera;   //相机模型，主要用liftProjective函数
};