#pragma once

#include<opencv2/opencv.hpp>
#include<eigen3/Eigen/Dense>
#include<map>
#include<queue>
#include<mutex>
#include"featuretracker.h"
using namespace std;
using namespace Eigen;


class Estimator
{
public:
 	Estimator();

    ~Estimator();
	
	/**
	 @description: 设置一些必要的参数
	*/
	void setParameter();

	/**
	 @description: 对双目图像进行处理，得到特征点存到featureBuf中
	 @param: time[in]  左图像的时间戳
	@param: imgleft[in] 左图像opencv格式
	@param: imgright[in] 右图像opencv格式
	*/
	void inputImage(double time, cv::Mat imgleft, cv::Mat imgright);

	/**
	@description: 对IMU数据进行处理
	@param: time[in] imu数据的时间戳
	@return: acc[in] 加速度计的数据
	@return: gyr[in] 陀螺仪的数据
	*/
	void inputIMU(double time, Vector3d acc, Vector3d gyr);


	void clearStatus();


	FeatureTracker featureTracker;

private:
	int inputImageCnt;
	mutex mBuf;  //特征点Buf的互斥量
	queue<pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>>> featureBuf;


};