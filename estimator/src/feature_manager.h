#pragma once

#include<vector>
#include<map>
#include<eigen3/Eigen/Dense>
#include<list>
#include<opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include<ros/console.h>
#include<ros/assert.h>

#include"parameters.h"


using namespace std;
using namespace Eigen;


class FeaturePerFrame
{
public:
    Vector3d point, pointRight;
    Vector2d uv, uvRight;
    Vector2d velocity, velocityRight;
    bool is_stereo;
    FeaturePerFrame(const Eigen::Matrix<double, 7, 1> &_point)
    {
        point.x() = _point(0);
        point.y() = _point(1);
        point.z() = _point(2);
        uv.x() = _point(3);
        uv.y() = _point(4);
        velocity.x() = _point(5); 
        velocity.y() = _point(6); 
        is_stereo = false;
    }
    void rightObservation(const Eigen::Matrix<double, 7, 1> &_point)
    {
        pointRight.x() = _point(0);
        pointRight.y() = _point(1);
        pointRight.z() = _point(2);
        uvRight.x() = _point(3);
        uvRight.y() = _point(4);
        velocityRight.x() = _point(5); 
        velocityRight.y() = _point(6); 
        is_stereo = true;
    }
};

class FeaturePerId
{
  public:
    const int feature_id;
    int start_frame;
    vector<FeaturePerFrame> feature_per_frame;
    int used_num;
    double estimated_depth;   //估计的深度是开始帧的左相机坐标系下测量的
    int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;

    FeaturePerId(int _feature_id, int _start_frame): feature_id(_feature_id), start_frame(_start_frame),used_num(0), estimated_depth(-1.0), solve_flag(0)
    {
    }
};

class FeatureManager
{
public:
    bool addFeatureCheckParallax(int frame_count,const map<int, vector< Eigen::Matrix<double,7,1>>> &image);

    void initFramePoseByPnP(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[]);

    void triangulate(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[]);
    
    bool FeatureManager::solvePoseByPnP(Eigen::Matrix3d &R, Eigen::Vector3d &P, vector<cv::Point2f> &pts2D, vector<cv::Point3f> &pts3D);

    void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                        Eigen::Vector2d &point0, Eigen::Vector2d &point1, Eigen::Vector3d &point_3d);
                
private:
    double compensatedParallax2();
    const Matrix3d *Rs;
    Matrix3d ric[2];
    list<FeaturePerId> feature;    
};