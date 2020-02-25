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
			pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>> feature;
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

void Estimator::clearStatus(){
	 inputImageCnt = 0;
}

void Estimator::setParameter(){
	return;
}

void Estimator::inputImage(double time, cv::Mat imgleft, cv::Mat imgright){
	inputImageCnt++;
	map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
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
		f_manager.initFramePoseByPnp();
		f_manager.triangulate();
		optimization();
		if(frame_count==WINDOW_SIZE)
		{
			udpateLastestStates();
			solver_flag=1;
			slideWindow();
                ROS_INFO("Initialization finish!");
		}
		else
		{
		    frame_count++;
            int prev_frame = frame_count - 1;
            Ps[frame_count] = Ps[prev_frame];
            Vs[frame_count] = Vs[prev_frame];
            Rs[frame_count] = Rs[prev_frame];
		}
	}
	else
	{
		initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
		triangulate(frame_count, Ps, Rs, tic, ric);
		optimization();
		//去除外点
		 if (failureDetection())
		 	cout<<"fail"<<endl;
		slideWindow();
		//移除失败点
		updateLatestStates();

	}
}
