/*
@author Derue François-Xavier
francois.xavier.derue<at>gmail.com

List of Superpixel method :
SEEDS
SLIC (cpu)
SLI_CUDA (gpu)
*/


#pragma once
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include "Slic/Slic.h"
#include "SLIC_CUDA/SLIC_cuda.h"
#include "Segmentor.h"
#include "ParamSpikeS_T.h"


using namespace cv;
using namespace std;
using namespace cv::ximgproc;


class SeedsEngine :public Segmentor
{
private:

	Ptr<SuperpixelSEEDS> seeds;
	int num_iterations;
	int prior;
	bool double_step;
	int num_levels;
	int num_histogram_bins;
public:

	SeedsEngine(int Nspx = 100);
	~SeedsEngine(){}

	void Init(Mat& imBGR, int diamSpx_or_Nspx = 16, float wc = 35, Segmentor::InitType initType = SIZE);
	void Segment(Mat& frame);
};


class SlicEngine :public Segmentor,private Slic
{
private:
	unique_ptr<Slic> slic;
	int m_nSpx_or_size;
	Slic::InitType m_typeInit;
	float m_wc;

public:

	SlicEngine(int nSpx_or_size = 16, float wc=35, Slic::InitType typeInit=SLIC_SIZE);
	~SlicEngine(){}

	void Init(Mat& imBGR, int diamSpx_or_Nspx = 16, float wc = 35, Segmentor::InitType initType = SIZE);
	void Segment(Mat& frame);
};

class SlicEngine_CUDA : public Segmentor
{
private:
	unique_ptr<SLIC_cuda> slic_cuda;
public:
	SlicEngine_CUDA(int diamSpx_or_Nspx = 16, float wc=35,SLIC_cuda::InitType initType=SLIC_cuda::SLIC_SIZE);
	~SlicEngine_CUDA(){}

	virtual void Init(Mat& imBGR);
	void Init(Mat& imBGR, int diamSpx_or_Nspx = 16, float wc = 35, Segmentor::InitType initType = SIZE);
	void Segment(Mat& frame);
};