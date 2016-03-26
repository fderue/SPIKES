#include "Superpixel.h"

using namespace std;
using namespace cv;


void Superpixel::computeMean(){
	xy.x = 0; xy.y = 0;
	color[0] = 0;
	color[1] = 0;
	color[2] = 0;
	for (Pixel& px : v_pixels){
		xy += px.xy;
		color += px.color;
	}
	if (v_pixels.size() != 0){
		//xy /= (float)v_pixels.size();
		xy.x /= v_pixels.size();
		xy.y /= v_pixels.size();
		color /= (float)v_pixels.size();
	}
}

void Superpixel::computeHisto(int nBin1d)
{
	CV_Assert(!v_pixels.empty());
	Mat pxMat(1,v_pixels.size(), CV_32FC3);
	Vec3f* pxMat_ptr = (Vec3f*)pxMat.data;
	for (int i = 0; i < v_pixels.size(); i++){
		pxMat_ptr[i] = v_pixels[i].color;
	}


	CV_Assert(pxMat.isContinuous());
	switch (v_pixels[0].colorSpace){
	case Pixel::BGR:
		funUtils::hist3D(pxMat, histo, nBin1d, funUtils::BGR);
		break;
	case Pixel::HSV:
		funUtils::hist3D(pxMat, histo, nBin1d, funUtils::HSV);
		break;
	case Pixel::Lab:
		funUtils::hist3D(pxMat, histo, nBin1d, funUtils::Lab);
		break;
	}	
}

void Superpixel::alight(Mat& out, Vec3b color){
	if (out.channels() == 3){
		for (int i = 0; i < v_pixels.size(); i++){
			out.at<Vec3b>(v_pixels[i].xy.y, v_pixels[i].xy.x) = color;
		}
	}
	else{
		for (int i = 0; i < v_pixels.size(); i++){
			out.at<uchar>(v_pixels[i].xy.y, v_pixels[i].xy.x) = color[0];
		}
	}
}

Mat Superpixel::getFeatMat()
{
	Mat feat;
	switch (this->featType)
	{
	case MEAN_COLOR:
		return (Mat_<float>(1, 3) << color[0], color[1], color[2]);
		break;
	case HISTO3D:
		return Mat(1, histo.size[0] * histo.size[1] * histo.size[2], CV_32F, (float*)histo.data);
		break;
	default:
		break;
	}
}

