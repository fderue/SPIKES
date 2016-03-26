#include "SpxEngine.h"

//=============== SEEDS ===============
SeedsEngine::SeedsEngine(int Nspx_e)
{
	num_iterations = 5;
	prior = 5;
	double_step = false;
	cm_nSpx = Nspx_e; // expected number of superpixel
	num_levels = 4;
	num_histogram_bins = 5;
}

void SeedsEngine::Init(Mat& imBGR, int diamSpx_or_Nspx, float wc , Segmentor::InitType initType )
{
	m_widthFrame = imBGR.cols;
	m_heightFrame = imBGR.rows;
	cm_labels = Mat(m_heightFrame, m_widthFrame, CV_32S, Scalar(-1));

	if (initType == Segmentor::SIZE) cm_nSpx = m_widthFrame*m_heightFrame / (diamSpx_or_Nspx*diamSpx_or_Nspx);
	else cm_nSpx = diamSpx_or_Nspx;

	seeds = createSuperpixelSEEDS(m_widthFrame, m_heightFrame, imBGR.channels(), cm_nSpx,
		num_levels, prior, num_histogram_bins, double_step);


	int true_n = cm_nSpx;
	while (seeds->getNumberOfSuperpixels()<true_n)
	{
		cm_nSpx += 50;
		seeds = createSuperpixelSEEDS(m_widthFrame, m_heightFrame, imBGR.channels(), cm_nSpx, num_levels, prior, num_histogram_bins, double_step);
	}
	cm_nSpx = seeds->getNumberOfSuperpixels();

}

void SeedsEngine::Segment(Mat& frame)
{
	Mat convertedFrame;
	cvtColor(frame, convertedFrame, CV_BGR2Lab); // choice of the color for gathering px into spx
	seeds->iterate(convertedFrame, num_iterations); //segmentation
	cm_nSpx = seeds->getNumberOfSuperpixels(); // true number of spx;
	seeds->getLabels(cm_labels);
}


//=============== SLIC ===============
SlicEngine::SlicEngine(int nSpx_or_size, float wc, Slic::InitType typeInit)
{
	m_nSpx_or_size = nSpx_or_size;
	m_typeInit = typeInit;
	m_wc = wc;
	slic = make_unique<Slic>();
}

void SlicEngine::Init(Mat& imBGR, int diamSpx_or_Nspx , float wc , Segmentor::InitType initType)
{
	m_widthFrame = imBGR.cols;
	m_heightFrame = imBGR.rows;

	Slic::InitType it;
	if (initType == SIZE)it = Slic::SLIC_SIZE;
	else it = Slic::SLIC_NSPX;
	slic->initialize(imBGR, diamSpx_or_Nspx, wc, it);
}
void SlicEngine::Segment(Mat& imBGR)
{
	slic->generateSpx(imBGR);
	SlicEngine::cm_nSpx = slic->getNspx();
	cm_labels = slic->getLabels();
}

//=============== SLIC CUDA ===============
SlicEngine_CUDA::SlicEngine_CUDA(int diamSpx_or_Nspx, float wc, SLIC_cuda::InitType initType)
{
	slic_cuda = make_unique<SLIC_cuda>(diamSpx_or_Nspx, wc, initType);
}


void SlicEngine_CUDA::Init(Mat& imBGR)
{
	slic_cuda->Initialize(imBGR);
}

void SlicEngine_CUDA::Init(Mat& imBGR, int diamSpx_or_Nspx, float wc, Segmentor::InitType initType)
{
	SLIC_cuda::InitType it;
	if (initType == SIZE)it = SLIC_cuda::SLIC_SIZE;
	else it = SLIC_cuda::SLIC_NSPX;
	slic_cuda->setParam(diamSpx_or_Nspx,wc,it);
	slic_cuda->Initialize(imBGR);
}
void SlicEngine_CUDA::Segment(Mat& imBGR)
{
	slic_cuda->Segment(imBGR);
	cm_labels = slic_cuda->getLabels();
	cm_labels.convertTo(cm_labels, CV_32S);
	cm_nSpx = slic_cuda->getNspx();
}

