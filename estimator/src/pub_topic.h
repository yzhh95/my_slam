#pragma once

#include<ros/ros.h>
#include <sensor_msgs/Image.h>

extern ros::Publisher  pub_image_track;


void registerPub(ros::NodeHandle &n);
