/*
@author Derue François-Xavier
francois.xavier.derue<at>gmail.com

Useful image processing functions to be used with openCV

*/
#pragma once
#include <opencv2/opencv.hpp>
#include <fstream>

using namespace std;
namespace funUtils{

	enum HistColor{
		BGR,
		HSV,
		Lab
	};

	void getGrabCutSeg(cv::Mat& inIm, cv::Mat& mask_fgnd, cv::Rect ROI);
	cv::Mat makeMask(cv::Rect ROIin, int wFrame, int hFrame, float scale = 2, bool fullFrame = false);
	void adaptROI(cv::Rect& ROI, int wFrame, int hFrame);
	void hist3D(cv::Mat& image, cv::Mat& hist, int Nbin,HistColor histColorSpace);
	cv::Rect giveAdaptRect(const cv::Mat& frame, const cv::Rect& ROIo);
	cv::Rect giveScaleRect(const cv::Rect& ROI, float scale = 2);
	cv::Rect giveRegion(cv::Point center, cv::Size s, int factor);
	void printHist3D(cv::Mat& histo3d);
	vector<string> get_all_files_names_within_folder(string folder);
	cv::Rect getGndT(string f);
	void genSubWindByPoint(cv::Rect base, float scale, cv::Point delta, vector<cv::Point>& ULP, vector<cv::Point>& BRP, cv::Mat& frame);

	template<typename T> cv::Mat computeIntImage(cv::Mat& image){
		CV_Assert(image.isContinuous());
		cv::Mat intImage = image.clone();

		for (int j = 1; j < image.cols; j++){
			T* intImage_ptr_i = intImage.ptr<T>(0);
			T* image_ptr = image.ptr<T>(0);
			intImage_ptr_i[j] = image_ptr[j] + intImage_ptr_i[j - 1];
		}
		for (int i = 1; i< image.rows; i++){
			intImage.at<T>(i, 0) = image.at<T>(i, 0) + intImage.at<T>(i - 1, 0);
		}

		for (int i = 1; i < image.rows; i++){
			T* intImage_ptr_i = intImage.ptr<T>(i);
			T* intImage_ptr_i_1 = intImage.ptr<T>(i - 1);
			T* image_ptr = image.ptr<T>(i);

			for (int j = 1; j < image.cols; j++){
				intImage_ptr_i[j] = image_ptr[j] + intImage_ptr_i[j - 1] + intImage_ptr_i_1[j] - intImage_ptr_i_1[j - 1];
			}
		}
		copyMakeBorder(intImage, intImage, 1, 0, 1, 0, BORDER_CONSTANT, Scalar(0));
		return intImage;
	}

	/*compute the sum over a rectangle limited by [ul to br]
	-------------------------------------------------------
	-----------------ul-------------------ur---------------
	-----------------|--------------------|----------------
	-----------------|--------------------|----------------
	-----------------|--------------------|----------------
	-----------------|--------------------|----------------
	-----------------bl-------------------br---------------
	-------------------------------------------------------*/
	template<typename T> T getIntegrale(cv::Mat& intImage, cv::Point ul, cv::Point br){

		if (!(br.x >= 0 && br.y >= 0 && br.x < intImage.cols - 1 && br.y < intImage.rows - 1 && ul.x >= 0 && ul.y >= 0 && ul.x < intImage.cols - 1 && ul.y < intImage.rows - 1 && ul.x <= br.x && ul.y <= br.y)){
			cout << "br " << br << endl;
			cout << "ul " << ul << endl;
		}
		CV_Assert(br.x >= 0 && br.y >= 0 && br.x<intImage.cols - 1 && br.y< intImage.rows - 1 && ul.x >= 0 && ul.y >= 0 && ul.x<intImage.cols - 1 && ul.y<intImage.rows - 1 && ul.x <= br.x && ul.y <= br.y);

		cv::Point ul_1 = ul;
		br.x += 1; br.y += 1;
		cv::Point ur(br.x, ul_1.y);
		cv::Point bl(ul_1.x, br.y);

		return intImage.at<T>(br.y, br.x) + intImage.at<T>(ul_1.y, ul_1.x) - intImage.at<T>(ur.y, ur.x) - intImage.at<T>(bl.y, bl.x);
	}
}


