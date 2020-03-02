#include "featuretracker.h"

double distance(cv::Point2f pt1, cv::Point2f pt2)
{
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

bool  FeatureTracker::inBorder(cv::Point2f pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x <COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

template<typename T>
void reduceVector(vector<T> &v, vector<uchar> status)
{
    int j = 0;
    for (size_t i = 0; i < v.size(); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void FeatureTracker::setMask(vector<cv::Point2f> &cur_pts)
{
    mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;
    for (size_t i = 0; i < cur_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(cur_pts[i], ids[i])));
    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });
    cur_pts.clear();
    ids.clear();
    track_cnt.clear();
    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255)
        {
            cur_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

FeatureTracker::FeatureTracker()
{
    n_id=0;
}

vector<cv::Point2f> FeatureTracker::undistortedPts(const vector<cv::Point2f> &pts, camodocal::CameraPtr cam)
{
    vector<cv::Point2f> un_pts;
    for (unsigned int i = 0; i < pts.size(); i++)
    {
        Eigen::Vector2d a(pts[i].x, pts[i].y);
        Eigen::Vector3d b;
        cam->liftProjective(a, b);
        un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
    }
    return un_pts;
}

vector<cv::Point2f> FeatureTracker::ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts, map<int,cv::Point2f> &id_pts)
{
    vector<cv::Point2f> pts_velocity;
    map<int, cv::Point2f> cur_id_pts;
    for (size_t i = 0; i < ids.size(); i++)
        cur_id_pts.insert(make_pair(ids[i], pts[i]));
    if (!id_pts.empty())
    {
        double dt = cur_time - prev_time;
        for (size_t i = 0; i < cur_id_pts.size(); i++)
        {
            map<int, cv::Point2f>::iterator it;
            it = id_pts.find(ids[i]);
            if( it!= id_pts.end()) 
            {
                double v_x = (pts[i].x - it->second.x) / dt;
                double v_y = (pts[i].y - it->second.y) / dt;
                pts_velocity.push_back(cv::Point2f(v_x, v_y));
            }
            else
                pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    else
        for (unsigned int i = 0; i < cur_id_pts.size(); i++)
            pts_velocity.push_back(cv::Point2f(0, 0));
    id_pts=cur_id_pts;
    return pts_velocity;
}

map<int, vector<Eigen::Matrix<double, 7, 1>>> FeatureTracker::trackImage(double t, cv::Mat &Imgleft, cv::Mat &Imgright)
{
    cur_time = t;
    vector<cv::Point2f> cur_pts, cur_un_pts, pts_velocity, cur_right_pts, cur_un_right_pts, right_pts_velocity;

    if (prev_pts.size() > 0)
    {
        vector<uchar> status;
        vector<float> err;
        cv::calcOpticalFlowPyrLK(prev_img, Imgleft, prev_pts, cur_pts, status, err, cv::Size(21, 21), 3);   //上一帧跟当前帧的特征点数量是一样的
        //返检
        vector<uchar> reverse_status;
        vector<cv::Point2f> reverse_pts = prev_pts;
        cv::calcOpticalFlowPyrLK(Imgleft, prev_img, cur_pts, reverse_pts, reverse_status, err, cv::Size(21, 21), 1,
                                 cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
        for (size_t i = 0; i < status.size(); i++)
        {
            if (status[i] && reverse_status[i] && inBorder(cur_pts[i]) && distance(prev_pts[i], reverse_pts[i]) <= 0.5)
                status[i] = 1;
            else
                status[i] = 0;
        }
        reduceVector<cv::Point2f>(cur_pts, status);
        reduceVector<int>(ids, status);
        reduceVector<int>(track_cnt, status);
    }

    for (auto &n : track_cnt)
        n++;

    setMask(cur_pts); //设置掩模防止特征点过密
    int n_max_cnt = MAX_CNT - static_cast<int>(cur_pts.size());
    if (n_max_cnt > 0)
    {
        n_pts.clear();
        cv::goodFeaturesToTrack(Imgleft, n_pts, n_max_cnt, 0.01, MIN_DIST, mask);
        for (auto &p : n_pts)
        {
            cur_pts.push_back(p);
            ids.push_back(n_id++);
            track_cnt.push_back(1);
        }
    }

    cur_un_pts=undistortedPts(cur_pts, m_camera[0]);
    pts_velocity= ptsVelocity(ids, cur_un_pts, prev_id_pts);
    if(!Imgright.empty())
    {
        if(!cur_pts.empty())
        {
            vector<cv::Point2f> reverseLeftPts;
            vector<uchar> status, statusRightLeft;
            vector<float> err;
            cv::calcOpticalFlowPyrLK(Imgleft, Imgright,cur_pts, cur_right_pts, status, err, cv::Size(21,21), 3);
            cv::calcOpticalFlowPyrLK(Imgright,Imgleft, cur_right_pts, reverseLeftPts, statusRightLeft, err, cv::Size(21,21), 3);
 
             for(size_t i = 0; i < status.size(); i++)
             {
                if(status[i] && statusRightLeft[i] && inBorder(cur_right_pts[i]) && distance(cur_pts[i],reverseLeftPts[i])<=0.5)
                    status[i] = 1;
                else
                    status[i]  = 0;
             }
            right_ids =ids;
            reduceVector<int>(right_ids, status);
            reduceVector<cv::Point2f>(cur_right_pts, status);
            cur_un_right_pts=undistortedPts(cur_right_pts, m_camera[1]);
            right_pts_velocity= ptsVelocity(right_ids, cur_un_right_pts, prev_right_id_pts);
        }
    }

    if(SHOW_TRACK)
        drawTrack(Imgleft, Imgright, ids, cur_pts, cur_right_pts, prevLeftPtsMap);

    prevLeftPtsMap.clear();
    for(size_t i = 0; i < cur_pts.size(); i++)
        prevLeftPtsMap[ids[i]] = cur_pts[i];

    prev_img=Imgleft;
    prev_pts=cur_pts;
    prev_time=cur_time;
    map<int, vector<Eigen::Matrix<double, 7,1>>> featureFrame;
    for(size_t i=0; i < ids.size(); i++)
    {
        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity<<cur_un_pts[i].x , cur_un_pts[i].y , 1.0 , cur_pts[i].x , cur_pts[i].y, pts_velocity[i].x , pts_velocity[i].y ;
        int feature_id= ids[i];
        featureFrame[feature_id].emplace_back( xyz_uv_velocity);
    }
    for(size_t i=0; i <right_ids.size(); i++)
    {
        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity<<cur_un_right_pts[i].x , cur_un_right_pts[i].y , 1 , cur_right_pts[i].x , cur_right_pts[i].y, right_pts_velocity[i].x , right_pts_velocity[i].y ;
        int feature_id= right_ids[i];
        featureFrame[feature_id].emplace_back(xyz_uv_velocity);
    }
    return featureFrame;
}

void FeatureTracker::readIntrinsicParameter(const vector<string> &calib_file)
{
    for(size_t i=0;i < calib_file.size();i++)
    {
        ROS_INFO("reading parameter of camera %s",calib_file[i].c_str());
        camodocal::CameraPtr camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file[i]);
        m_camera.push_back(camera);
    }
}

void FeatureTracker::drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight, 
                               vector<int> &curLeftIds,
                               vector<cv::Point2f> &curLeftPts, 
                               vector<cv::Point2f> &curRightPts,
                               map<int, cv::Point2f> &prevLeftPtsMap)
{
    int cols = imLeft.cols;
    cv::hconcat(imLeft, imRight, imTrack);
    cv::cvtColor(imTrack, imTrack, CV_GRAY2RGB);

    for (size_t j = 0; j < curLeftPts.size(); j++)
    {
        double len = std::min(1.0, 1.0 * track_cnt[j] / 20);
        cv::circle(imTrack, curLeftPts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    }
    for (size_t i = 0; i < curRightPts.size(); i++)
    {
        cv::Point2f rightPt = curRightPts[i];
        rightPt.x += cols;
        cv::circle(imTrack, rightPt, 2, cv::Scalar(0, 255, 0), 2);
    }

    map<int, cv::Point2f>::iterator mapIt;
    for (size_t i = 0; i < curLeftIds.size(); i++)
    {
        int id = curLeftIds[i];
        mapIt = prevLeftPtsMap.find(id);
        if(mapIt != prevLeftPtsMap.end())
            cv::arrowedLine(imTrack, curLeftPts[i], mapIt->second, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
    }
}