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
-------------------------------------------------------
KpEngine allows to extract and describe different
types of features provided by openCV 3 with
a simple framework. Different options must be specified
in #define macro.
ex : 
KpEngine kpEngine;
// if same extractor and descriptor
kpEngine.extrAndDescrKp(im);

// if different extractor and descriptor
kpEngine.extractKp(im); 
kpEngine.describeKp(im);

// if need to download descriptors and keypoints from gpu if GPU used 
kpEngine1.getDescFromGpu();
kpEngine.getKpFromGpu();
-------------------------------------------------------
Features that can be used
- D : Descriptor
- E : Extractor
xfeatures2d::SIFT : D+E
xfeatures2d::SURF : D+E
xfeatures2d::DAISY : D
xfeatures2d::FREAK : D
xfeatures2d::LATCH : D
BRISK : D+E
ORB : D+E
KAZE : D+E
AKAZE : D+E
------ GPU Implementation -------
cuda::SURF_CUDA : D+E // does not work with small image
cuda::ORB : D+E but D alone is not working !
cuda::FastFeatureDetector : E
*/

#define KP_EXTRACTOR xfeatures2d::SIFT//xfeatures2d::SURF//xfeatures2d::SIFT//cuda::SURF_CUDA//cuda::SURF_CUDA//xfeatures2d::SIFT
#define KP_DESCRIPTOR xfeatures2d::SIFT//xfeatures2d::SURF//xfeatures2d::SIFT//cuda::SURF_CUDA//cuda::SURF_CUDA//xfeatures2d::SURF//xfeatures2d::SIFT
#define KP_GPU 0 // to activate if use Gpu Feature
#define SURF_GPU_EXTRACTOR 0 // flag needed only for SURF_GPU 
#define SURF_GPU_DESCRIPTOR 0 // flag needed only for SURF_GPU 

using namespace std;
using namespace cv;

class KpEngine
{
public:
#if KP_GPU
	cuda::GpuMat im_gpu;
	cuda::GpuMat d_kp_gpu;
	cuda::GpuMat v_kp_gpu;
#endif
#if SURF_GPU_EXTRACTOR
	KP_EXTRACTOR kpExtractor;
#else
	Ptr<KP_EXTRACTOR> kpExtractor;
#endif
#if SURF_GPU_DESCRIPTOR
	KP_DESCRIPTOR kpDescriptor;
#else
	Ptr<KP_DESCRIPTOR> kpDescriptor;
#endif
	Mat d_kp;
	vector<KeyPoint> v_kp;

public:

	KpEngine(){
#if !SURF_GPU_EXTRACTOR
		kpExtractor = KP_EXTRACTOR::create();
#endif
#if !SURF_GPU_DESCRIPTOR
		kpDescriptor = KP_DESCRIPTOR::create();
#endif
	}
	~KpEngine(){}

	void extrAndDescrKp(Mat& im);
	void extractKp(Mat& im);
	void describeKp(Mat& im);
	void getKpFromGpu();
	void getDescFromGpu();
};

