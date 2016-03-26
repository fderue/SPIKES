/*
@author Derue François-Xavier
francois.xavier.derue<at>gmail.com

Interface for Superpixel Segmentation method

*/


#pragma once
#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

class Segmentor
{
protected:
	int cm_nSpx;
	Mat cm_labels;
	int m_widthFrame, m_heightFrame;

public:
	static enum InitType{
		SIZE,
		NSPX
	};
public:
	Segmentor(){}
	virtual ~Segmentor(){}
	virtual void Init(Mat& imBGR, int diamSpx_or_Nspx = 16, float wc = 35, Segmentor::InitType initType = SIZE) = 0;
	virtual void Segment(Mat&) = 0;
	Mat getLabels(){ return cm_labels; }
	int getNSpx(){ return cm_nSpx; }
	void DisplayContours(Mat& image, Scalar colour){
		const int dx8[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
		const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
		/* Initialize the contour vector and the matrix detailing whether a pixel
		* is already taken to be a contour. */
		vector<Point> contours;
		vector<vector<bool> > istaken;
		for (int i = 0; i < image.rows; i++) {
			vector<bool> nb;
			for (int j = 0; j < image.cols; j++) {
				nb.push_back(false);
			}
			istaken.push_back(nb);
		}
		/* Go through all the pixels. */
		for (int i = 0; i<image.rows; i++) {
			for (int j = 0; j < image.cols; j++) {

				int nr_p = 0;

				/* Compare the pixel to its 8 neighbours. */
				for (int k = 0; k < 8; k++) {
					int x = j + dx8[k], y = i + dy8[k];

					if (x >= 0 && x < image.cols && y >= 0 && y < image.rows) {
						if (istaken[y][x] == false && cm_labels.at<int>(i, j) != cm_labels.at<int>(y, x)) {
							nr_p += 1;
						}
					}
				}
				/* Add the pixel to the contour list if desired. */
				if (nr_p >= 2) {
					contours.push_back(Point(j, i));
					istaken[i][j] = true;
				}

			}
		}
		/* Draw the contour pixels. */
		for (int i = 0; i < (int)contours.size(); i++) {
			image.at<Vec3b>(contours[i].y, contours[i].x) = Vec3b((uchar)colour[0], (uchar)colour[1], (uchar)colour[2]);
		}
	}
};

