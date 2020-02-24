#include"estimator.h"

Estimator::Estimator(){

}

Estimator::~Estimator(){

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