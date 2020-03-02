/*******************************************************
 * Copyright (C) 2020, Unmanned Driving  Group, Harbin Institute of Technology University
 * @Decription: This file is part of VLOAM.
 * @Date: 2020-2-20
 * @Author: YZH (734313835@qq.com)
 *                 
 *******************************************************/

#include <iostream>
#include<thread>
#include<mutex>
#include<queue>
#include <ros/ros.h>
#include<sensor_msgs/Imu.h>
#include<sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include"estimator.h"
#include "parameters.h"
#include "pub_topic.h"

using namespace std;
using namespace Eigen;

Estimator estimator;
mutex m_buf;     //保护数据存入
//queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::ImageConstPtr> imgleft_buf;
queue<sensor_msgs::ImageConstPtr> imgright_buf;


void imu_callback(const sensor_msgs::ImuConstPtr & imu_msg)
{
	double t = imu_msg->header.stamp.toSec();
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Vector3d acc(dx, dy, dz);
    Vector3d gyr(rx, ry, rz);
    estimator.inputIMU(t, acc, gyr);
	return;
}

void ImageLeft_callback(const sensor_msgs::ImageConstPtr & image_msg)
{
	m_buf.lock();
    imgleft_buf.push(image_msg);
    m_buf.unlock();
	return;
}


void ImageRight_callback(const sensor_msgs::ImageConstPtr & image_msg)
{
	m_buf.lock();
    imgright_buf.push(image_msg);
    m_buf.unlock();
	return;
}

//将ros的图像消息转成opencv格式，注意格式的转换，
cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    cv::Mat img = ptr->image.clone();
    return img;
}

//图像消息话题数据同步，并传给estimator进行处理，注意互斥锁保证数据不被污染
void sync_process()
{
	while(1){
		cv::Mat imgleft, imgright;
		std_msgs::Header header;
		double time=0;
		m_buf.lock();
		if (!imgleft_buf.empty() && !imgright_buf.empty())
		{
			double time0 = imgleft_buf.front()->header.stamp.toSec();
			double time1 = imgleft_buf.front()->header.stamp.toSec();
			if(time0 < time1 - 0.003)
				imgleft_buf.pop();
			else if(time0 > time1 + 0.003)
				imgright_buf.pop();
			else{
				time = imgleft_buf.front()->header.stamp.toSec();
				header = imgleft_buf.front()->header;
				imgleft = getImageFromMsg(imgleft_buf.front());
				imgleft_buf.pop();
				imgright = getImageFromMsg(imgright_buf.front());
				imgright_buf.pop();
			}
		}
		m_buf.unlock();
		if(!imgleft.empty())
			estimator.inputImage(time, imgleft, imgright);

		std::chrono::milliseconds dura(2);
		std::this_thread::sleep_for(dura);
	}
	
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "estimator_node");
	ros::NodeHandle n("~"); 
	//设置rosconsole调试级别
	ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug);
	ROS_INFO("Estimator_node start!");
	 if(argc != 2){
        printf("please intput: rosrun estimator estimator_exc [config file] \n");
        return 1;
    }

	string config_file = argv[1];
	readParameters(config_file);
	registerPub(n);
    estimator.setParameter();

	//订阅双目相机的图像消息和imu数据
	ROS_WARN("waiting for image and imu...");
	ros::Subscriber sub_imu=n.subscribe("/kitti/oxts/imu" , 2000 , imu_callback  , ros::TransportHints().tcpNoDelay());
	ros::Subscriber sub_image_left = n.subscribe(IMAGE0_TOPIC, 100 , ImageLeft_callback);
	ros::Subscriber sub_image_right = n.subscribe(IMAGE1_TOPIC , 100 , ImageRight_callback);

	thread sync_thread{ sync_process};

	ros::spin(); 
    return 0;
}
