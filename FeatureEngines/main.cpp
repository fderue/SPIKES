#include <opencv2/opencv.hpp>
#include "KpEngine.h"
#include "MatchEngine.h"

using namespace std;
using namespace cv;

int main(int argc, const char** argv)
{
	Mat im1 = imread("D:/Videos/CVPR_benchmark/tiger1/img/0001.jpg");
	Mat im2 = imread("D:/Videos/CVPR_benchmark/tiger1/img/0002.jpg");
	CV_Assert(im1.data != nullptr);
	CV_Assert(im2.data != nullptr);


	KpEngine kpEngine1,kpEngine2;
	//kpEngine.extrAndDescrKp(im);
	kpEngine1.extractKp(im1);
	kpEngine1.describeKp(im1);
	kpEngine1.getDescFromGpu();


	kpEngine2.extractKp(im2);
	kpEngine2.describeKp(im2);
	kpEngine2.getDescFromGpu();

	MatchEngine matchEngine(NORM_L2,true);
#if GPU_MATCHER
	matchEngine.match<cuda::GpuMat>(kpEngine1.d_kp_gpu, kpEngine2.d_kp_gpu);
#else
	matchEngine.match<Mat>(kpEngine1.d_kp, kpEngine2.d_kp,false);
#endif

	Mat out;
	kpEngine1.getKpFromGpu();
	kpEngine2.getKpFromGpu();
	drawMatches(im1, kpEngine1.v_kp, im2, kpEngine2.v_kp, matchEngine.v_DMatch, out);
	imshow("out", out);
	waitKey();
	cout << "done" << endl;

	return 1;
}