#include"estimator.h"

Estimator::Estimator()
{
	processThread = std::thread(&Estimator::processMeasurements, this);
}

Estimator::~Estimator()
{
	    processThread.join();
        printf("join thread \n");
}

void Estimator::processMeasurements()

{
	while(1)
	{
		printf("process measurments\n");
		if(!featureBuf.empty())
		{
			pair<double, map<int, vector<Eigen::Matrix<double, 7, 1>>>> feature;
			mBuf.lock();
            feature = featureBuf.front();
			featureBuf.pop();
			mBuf.unlock();
			processImage(feature.second, feature.first);
			
			std_msgs::Header header;
            header.frame_id = "world";
            header.stamp = ros::Time(feature.first);
            //pubOdometry(*this, header);
			//pubCameraPose(*this, header);
			//pubTF(*this, header);
		}
		if(!Flag_thread)
			break;
		chrono::milliseconds dura(2);
        this_thread::sleep_for(dura);
	}
}

void Estimator::clearState(){
	 inputImageCnt = 0;
}

void Estimator::setParameter(){
	return;
}

void Estimator::inputImage(double time, cv::Mat imgleft, cv::Mat imgright){
	inputImageCnt++;
	map<int, vector<Eigen::Matrix<double, 7, 1>>> featureFrame;
	featureFrame=featureTracker.trackImage(time,imgleft,imgright);
	if(inputImageCnt%2==0){
		mBuf.lock();
		featureBuf.push(make_pair(time,featureFrame));
		mBuf.unlock();
	}
		return;
}

void Estimator::inputIMU(double time, Vector3d acc, Vector3d gyr){
	return;
}

void Estimator::processImage(const map<int, vector< Eigen::Matrix<double, 7, 1>>> &image, const double t)
{
    ROS_DEBUG("Adding feature points %lu", image.size());
	if(f_manager.addFeatureCheckParallax(frame_count,image))
		marginalization_flag=0;
	else
		marginalization_flag=1;
	if(solver_flag==0)
	{
		f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
		f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
		optimization();
		if(frame_count==WINDOW_SIZE)
		{
			solver_flag=1;
			slideWindow();
                ROS_INFO("Initialization finish!");
		}
		else
		{
		    frame_count++;
            int prev_frame = frame_count - 1;
			//有必要吗？？
            Ps[frame_count] = Ps[prev_frame];
            Rs[frame_count] = Rs[prev_frame];
		}
	}
	else
	{
		f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
		f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
		optimization();
		set<int> removeIndex;
        outliersRejection(removeIndex);
        f_manager.removeOutlier(removeIndex);
		 if (failureDetection())
		 {
           	ROS_WARN("failure detection!");
            clearState();
            setParameter();
            ROS_WARN("system reboot!");
            return;
		 }
		slideWindow();
	}
}

bool Estimator::failureDetection()
{

}


void Estimator::slideWindow()
{
    if (marginalization_flag == 0)
    {
        double t_0 = Headers[0];
		Matrix3d back_R0 = Rs[0];
        Vector3d back_P0 = Ps[0];
        if (frame_count == WINDOW_SIZE)
        {
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Headers[i] = Headers[i + 1];
                Rs[i].swap(Rs[i + 1]);
                Ps[i].swap(Ps[i + 1]);
            }
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];

			Matrix3d R0, R1;
			Vector3d P0, P1;
			R0 = back_R0 * ric[0];
			R1 = Rs[0] * ric[0];
			P0 = back_P0 + back_R0 * tic[0];
			P1 = Ps[0] + Rs[0] * tic[0];
			f_manager.removeBackShiftDepth(R0, P0, R1, P1);


        }
    }
    else
    {
        if (frame_count == WINDOW_SIZE)
        {
            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];
           	f_manager.removeFront(frame_count);
        }
    }
}


void Estimator::outliersRejection(set<int> &removeIndex)
{
    for (auto &it_per_id : f_manager.feature)
    {
        double err = 0;
        int errCnt = 0;
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;
        double depth = it_per_id.estimated_depth;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i != imu_j)
            {
                Vector3d pts_j = it_per_frame.point;             
                double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], 
                                                    Rs[imu_j], Ps[imu_j], ric[0], tic[0],
                                                    depth, pts_i, pts_j);
                err += tmp_error;
                errCnt++;
            }
            if(it_per_frame.is_stereo)
            {
                
                Vector3d pts_j_right = it_per_frame.pointRight;
				double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], 
													Rs[imu_j], Ps[imu_j], ric[1], tic[1],
													depth, pts_i, pts_j_right);
				err += tmp_error;
				errCnt++;
            }
        }
        double ave_err = err / errCnt;
        if(ave_err * FOCAL_LENGTH > 3)
            removeIndex.insert(it_per_id.feature_id);
    }
}

double Estimator::reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                                 Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj, 
                                 double depth, Vector3d &uvi, Vector3d &uvj)
{
    Vector3d pts_w = Ri * (rici * (depth * uvi) + tici) + Pi;
    Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
    Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();
    double rx = residual.x();
    double ry = residual.y();
    return sqrt(rx * rx + ry * ry);
}
