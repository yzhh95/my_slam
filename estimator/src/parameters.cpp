#include "parameters.h"

double MIN_PARALLAX;   //视差阈值
std::vector<Eigen::Matrix3d> RIC;   
std::vector<Eigen::Vector3d> TIC;
int ROW, COL;
int MAX_CNT;    //每帧的特征点数最大值
int MIN_DIST;    //特征点之间的最小距离
std::string IMU_TOPIC;   
std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
std::vector<std::string> CAM_NAMES;    //相机的模型文件
int SHOW_TRACK;  //观察跟踪特征点 
int NUM_ITERATIONS;  
double F_THRESHOLD;
double SOLVER_TIME;   

void readParameters(std::string config_file)
{
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
        std::cerr << "ERROR: Wrong path to settings" << std::endl;

    fsSettings["image0_topic"] >> IMAGE0_TOPIC;
    fsSettings["image1_topic"] >> IMAGE1_TOPIC;
    MAX_CNT = fsSettings["max_cnt"];
    MIN_DIST = fsSettings["min_dist"];
    F_THRESHOLD = fsSettings["F_threshold"];
    SOLVER_TIME = fsSettings["max_solver_time"];
    NUM_ITERATIONS = fsSettings["max_num_iterations"];
    MIN_PARALLAX = fsSettings["keyframe_parallax"];
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;
    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    SHOW_TRACK=fsSettings["show_track"];


    cv::Mat cv_T;
    Eigen::Matrix4d T;
    fsSettings["body_T_cam0"] >> cv_T;
    cv::cv2eigen(cv_T, T);
    RIC.push_back(T.block<3, 3>(0, 0));
    TIC.push_back(T.block<3, 1>(0, 3));
    fsSettings["body_T_cam1"] >> cv_T;
    cv::cv2eigen(cv_T, T);
    RIC.push_back(T.block<3, 3>(0, 0));
    TIC.push_back(T.block<3, 1>(0, 3));

    int pn = config_file.find_last_of('/');
    std::string configPath = config_file.substr(0, pn);
    std::string cam0Calib,  cam1Calib;
    fsSettings["cam0_calib"] >> cam0Calib;
    std::string cam0Path = configPath + "/" + cam0Calib;
    CAM_NAMES.push_back(cam0Path);
    fsSettings["cam1_calib"] >> cam1Calib;
    std::string cam1Path = configPath + "/" + cam1Calib; 
    CAM_NAMES.push_back(cam1Path);
    
    fsSettings.release();
}
