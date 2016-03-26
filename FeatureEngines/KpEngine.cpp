#include "KpEngine.h"

using namespace std;
using namespace cv;

void KpEngine::extractKp(Mat& im)
{
#if KP_GPU
	Mat imGray;
	cvtColor(im, imGray, CV_BGR2GRAY);
	im_gpu.upload(imGray);
	try{
#if SURF_GPU_EXTRACTOR
		kpExtractor(im_gpu, cuda::GpuMat(), v_kp_gpu);
#else
		kpExtractor->detectAsync(im_gpu, v_kp_gpu);
#endif
	}
	catch (cv::Exception e){
		cerr << "this feature can not extract" << endl;
	}
#else
	try{
		kpExtractor->detect(im, v_kp);
	}
	catch (cv::Exception e){
		cerr << "this feature can not extract" << endl;
	}
#endif
}

void KpEngine::describeKp(Mat& im)
{
#if KP_GPU
	try{
#if SURF_GPU_DESCRIPTOR
		kpDescriptor(im_gpu, cuda::GpuMat(), v_kp_gpu, d_kp_gpu, true);
#else
		kpDescriptor->computeAsync(im_gpu, v_kp_gpu, d_kp_gpu);
#endif
	}
	catch (cv::Exception e){
		cerr << "!!!! this feature can not describe" << endl;
	}
#else
	try{
		kpDescriptor->compute(im, v_kp, d_kp);
	}
	catch (cv::Exception e){
		cerr << "this feature can not describe" << endl;
	}
#endif
}

void KpEngine::extrAndDescrKp(Mat& im)
{
#if KP_GPU
	try{
		CV_Assert(im.channels() == 3);
		Mat imGray;
		cvtColor(im, imGray, CV_BGR2GRAY);
		im_gpu.upload(imGray);
#if (SURF_GPU_DESCRIPTOR & SURF_GPU_EXTRACTOR)
		kpExtractor(im_gpu, cuda::GpuMat(), v_kp_gpu, d_kp_gpu);
#else
		kpExtractor->detectAndComputeAsync(im_gpu, cuda::GpuMat(), v_kp_gpu, d_kp_gpu);
#endif
	}
	catch (cv::Exception e){
		extractKp(im);
		describeKp(im);
	}
#else

	try{
		kpExtractor->detectAndCompute(im, Mat(), v_kp, d_kp);
	}
	catch (cv::Exception e){
		extractKp(im);
		describeKp(im);
	}
#endif
}
void KpEngine::getKpFromGpu()
{
#if SURF_GPU_EXTRACTOR
	kpExtractor.downloadKeypoints(v_kp_gpu, v_kp);
#elif KP_GPU
	kpExtractor->convert(v_kp_gpu, v_kp);
#endif
}
void KpEngine::getDescFromGpu()
{
#if KP_GPU
	//d_kp_gpu.download(d_kp);
	d_kp = Mat(d_kp_gpu);
#endif
}