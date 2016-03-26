/*
@author Derue François-Xavier
francois.xavier.derue<at>gmail.com
*/
#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>

/*
-----------------------------------------------------------------------------
This class provides an easy interface to use the different matcher of openCV.
Different options must be provided in #define macros
-----------------------------------------------------------------------------
ex : 

MatchEngine matchEngine(distanceMeasure,crossCheck); 
distanceMeasure : NORM_L1, NORM_L2, HAMMING....
crossCheck : bidirectional match
if GPU_MATCHER
matchEngine.match<cuda::GpuMat>(d_kp_gpu_query, d_kp_gpu_train,knn?); // if data already on gpu, faster this way
else
matchEngine.match<Mat>(d_kp_cpu_query,d_kp_cpu_train,knn?); // if data on cpu


bool knn : indicates if used knnMatcher. If so, use David Lowe ratio to select final match.
*/


#define GPU_MATCHER 0 // matching on GPU : only BF matcher
#define CROSS_CHECK 0 // flag for one-to-one match query <-> train
#if !CROSS_CHECK
#define KNN 1
#define LOWE_RATIO .75f
#else
#define KNN 0
#endif
#define FLANN 0 //use FLANN instead of BF (cpu only)

using namespace std;
using namespace cv;
class MatchEngine
{
public:
	vector<DMatch> v_DMatch;
private:
	bool m_crossCheck;
#if GPU_MATCHER
	Ptr<cuda::DescriptorMatcher> matcher;
	cuda::GpuMat matchArrayGpu;
#else
	DescriptorMatcher* matcher;
#endif

public:
	MatchEngine(int normType = NORM_L2, bool crossCheck = false)
	{
#if GPU_MATCHER
		matcher = cuda::DescriptorMatcher::createBFMatcher(NORM_L2);
#elif FLANN
		matcher = new FlannBasedMatcher;
#else
		m_crossCheck = crossCheck;
		matcher = new BFMatcher(normType, crossCheck);
#endif
	}
	template<typename T>void match(T queryDesc, T trainDesc, bool knn = true);
	template<>void match<cuda::GpuMat>(cuda::GpuMat queryDesc, cuda::GpuMat trainDesc, bool knn);

};

template<typename T>
void MatchEngine::match(T queryDesc, T trainDesc, bool knn)
{
	v_DMatch.resize(0);
#if GPU_MATCHER
	cuda::GpuMat queryDesc_gpu, trainDesc_gpu;
	queryDesc_gpu.upload(queryDesc);
	trainDesc_gpu.upload(trainDesc);
#if KNN
	vector<vector<DMatch>>v_v_DMatch;
	matcher->knnMatchAsync(queryDesc_gpu, trainDesc_gpu, v_v_DMatch, 2);
	float thr_dl = LOWE_RATIO;
	for (int i = 0; i < v_v_DMatch.size(); i++){
		float ratio = v_v_DMatch[i][0].distance / v_v_DMatch[i][1].distance;
		if (ratio < thr_dl)v_DMatch.push_back(v_v_DMatch[i][0]);
	}
#else
	matcher->matchAsync(queryDesc_gpu, trainDesc_gpu, matchArrayGpu);
#endif
#else
	if (knn && !m_crossCheck){
		vector<vector<DMatch>>v_v_DMatch;
		matcher->knnMatch(queryDesc, trainDesc, v_v_DMatch, 2);
		//lowe ratio test
		float thr_dl = LOWE_RATIO;
		for (int i = 0; i < v_v_DMatch.size(); i++){
			float ratio = v_v_DMatch[i][0].distance / v_v_DMatch[i][1].distance;
			if (ratio < thr_dl)v_DMatch.push_back(v_v_DMatch[i][0]);
		}
	}
	else{
		matcher->match(queryDesc, trainDesc, v_DMatch);
	}
#endif
}

#if GPU_MATCHER
template<> //specialization if descriptor already on gpu
void MatchEngine::match<cuda::GpuMat>(cuda::GpuMat queryDesc_gpu, cuda::GpuMat trainDesc_gpu, bool knn)
{
	v_DMatch.resize(0);
#if KNN
	vector<vector<DMatch>>v_v_DMatch;
	matcher->knnMatchAsync(queryDesc_gpu, trainDesc_gpu, matchArrayGpu, 2);
	matcher->knnMatchConvert(matchArrayGpu, v_v_DMatch);
	float thr_dl = LOWE_RATIO;
	for (int i = 0; i < v_v_DMatch.size(); i++){
		float ratio = v_v_DMatch[i][0].distance / v_v_DMatch[i][1].distance;
		if (ratio < thr_dl)v_DMatch.push_back(v_v_DMatch[i][0]);
	}
#else //crossCheck
	matcher->matchAsync(queryDesc_gpu, trainDesc_gpu, matchArrayGpu);
	matcher->matchConvert(matchArrayGpu, v_DMatch);
#endif

}
#endif