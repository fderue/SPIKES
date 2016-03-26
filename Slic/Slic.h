/*
SLIC : CPU version
@author Derue François-Xavier
francois.xavier.derue<at>gmail.com

This class implement the superpixel segmentation "SLIC Superpixels",
Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi, Pascal Fua, and Sabine Susstrunk,
EPFL Technical Report no. 149300, June 2010.
Copyright (c) 2010 Radhakrishna Achanta [EPFL]. All rights reserved.

*/

#pragma once

#define MAXIT 5 //#iterations for clustering

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

typedef vector< vector<int> > vec2di;

struct center;
class Slic
{
private:
	int m_nSpx;
    float m_wc;
    int m_width;
    int m_height;
    int m_diamSpx;

    Mat m_labels;
    vector<vector<float> > m_allDist;
    vector<center> m_allCenters;

	void enforceConnectivity();// (optional)
	void findCenters(Mat& frame);
	void updateCenters(Mat& frame);

public:

    Slic(){}
    ~Slic(){}

	static enum InitType{
		SLIC_SIZE,
		SLIC_NSPX
	};

	/*SLIC_SIZE -> initialize by specifying the spx size
	SLIC_NSPX -> initialize by specifying the number of spx*/
    void initialize(Mat& frame,int nspx,float wc,Slic::InitType type);
	void resetVariables();
    void generateSpx(Mat& frame);

	Mat getLabels();
	int getNspx();
	int getSspx();
    void display_contours(Mat& image,Scalar colour=Scalar(255,0,0));
};

struct center
{
    Point xy;
    float Lab[3];
    center():xy(Point(0,0)){
        Lab[0] = 0.f;
        Lab[1] = 0.f;
        Lab[1] = 0.f;
    }
};



