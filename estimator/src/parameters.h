#pragma once

#include <ros/ros.h>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>
#include <map>

using namespace std;

const double FOCAL_LENGTH = 460.0;
const int WINDOW_SIZE = 10;
const double INIT_DEPTH = -1.0;

extern double MIN_PARALLAX;
extern std::vector<Eigen::Matrix3d> RIC;
extern std::vector<Eigen::Vector3d> TIC;
extern double SOLVER_TIME;
extern int NUM_ITERATIONS;
extern std::string IMU_TOPIC;
extern int ROW, COL;
extern std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
extern std::vector<std::string> CAM_NAMES;
extern int MAX_CNT;
extern int MIN_DIST;
extern double F_THRESHOLD;

void readParameters(std::string config_file);
