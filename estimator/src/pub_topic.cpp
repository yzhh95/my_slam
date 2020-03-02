#include "pub_topic.h"
ros::Publisher  pub_image_track;

void registerPub(ros::NodeHandle &n)
{
    pub_image_track = n.advertise<sensor_msgs::Image>("image_track", 1000);
}