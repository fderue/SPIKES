/*
@author Derue François-Xavier
francois.xavier.derue<at>gmail.com

Spikes_T implements the SPiKeS-based tracker proposed in 
"SPiKeS: Superpixel-Keypoints Structure for Robust Visual Tracking"

Ex :
SpikeS tracker;
tracker.initialize(frame0,ROI0);
for(each frame){
	tracker.track(frame);
	displayState(frame); //display result
	imshow("result",frame);
	tracker.update();
}
*/


#pragma once

#include <opencv2\opencv.hpp>
#include <memory>
#include <list>
#include "FunUtils/funUtils.h"
#include "SpxEngine.h"
#include "FeatureEngines/KpEngine.h"
#include "FeatureEngines/MatchEngine.h"
#include "SpikeS.h"
#include <queue>
#include <math.h>


using namespace std;
using namespace cv;

class SpikeS_T
{
private:
	
	//Images
	Mat Frame0, Frame_t;
	//Engines
	unique_ptr<Segmentor> spxEngine;
	KpEngine kpEngine;
	unique_ptr<MatchEngine> matchEngine;

	//Pre-allocated buffer
	SpikeS* buffSpikeS_Model;
	SpikeS* buffSpikeS_Frame;
	KeyPoint2* buffKp2_Fgnd;
	KeyPoint2* buffKp2_Bgnd;
	KeyPoint2* buffKp2_Frame;
	
	//Pools
	list<SpikeS*> l_spikesModel, l_spikesFrame;
	list<KeyPoint2*> l_kp2ModelF, l_kp2ModelB;
	list<KeyPoint2*> l_kp2Frame;

	//Matching
	Mat scoreMat, matchMat;
	int nMatchKp_for, nMatchKp_back;
	int nMatchSpikeS;
	int nMatchSpikeSBBEst; //matching spikes in the estimated BoundingBox 

	//State
	int timeT;
	Rect StateInit;
	Rect m_State, StateEst;
	Point m_position;
	Point positionEst;

	Point dPos, dPosEst;
	Point dPosTotale = 0;
	bool noOcc;
	bool posEstOk;
	//Memory management
	queue<SpikeS*> q_spikesModel;
	queue<KeyPoint2*> q_kp2ModelF, q_kp2ModelB;
	int m_maxSpikesModel;

	//For optimization
	vector<float> v_rankedW_SpikeS, v_rankedW_Kp2F, v_rankedW_Kp2B;
	Mat maskBB, maskAroundBB, maskSegment;

	//Extra
	float m_diam0;



public:
	SpikeS_T();
	~SpikeS_T();

	/*Initialization :
	1) Target localized with ROI0
	2) Coarse Target Seg with grabCut at pxLevel (optional)
	3) Select Spx which overlaps segmentation (optional)
	4) Create Spx Fgnd/Bgnd model 
	5) Create Kp Fgnd/Bgnd model
	6) Create SpiKeS model
	*/
	void initialize(Mat& frame0, Rect ROI0);

	/*Tracking :
	1) Extract SpiKeS from current frame
	2) Match SpiKeS (model <-> current frame)
	3) Estimate State (location)
	*/
	void track(Mat& frame);

	/*Update
	1) Check occlusion
	2) Update the SpiKes model if no occlusion
	*/
	void update();

	Rect getState(){ return m_State; }

	//----- Display Routines ------
	void displaySegmentation(Mat& im);
	void displayModelSpx(Mat& im);
	void displayModelKpFB(Mat& im);
	void displayModelSpikeS(Mat& im);
	void displayFrameSpikeS(Mat& im);
	void displayMatchKp(Mat& frame1, Mat& frame2, Mat& output);
	void displayMatchSpikeS(Mat& frame1, Mat& frame2, Mat& output);
	void displayMatchSpikesMask(Mat& im8UC1);
	void displayFgnd(Mat& im);
	void displayState(Mat& im);
	void displayVote(Mat& im);

	//----- Subroutines -----------
private:
	void computeSpikesMatchMat();
	void constraintMatch();
	void setFBSpikes(const Mat& fgndMask, const Mat& bgndMask, const Mat& grabSegMask, const list<SpikeS*>& l_sf_in, list<SpikeS*>& l_sf_out);
	void extractSpikeS(Mat& frame);
	void matchSpikeS();
	void matchKp2();
	void resetMatches();
	void estimatePos();
	bool checkPosUpdate();
	void updatePos();
	void checkOcclusion(Rect boundingBox);
	void updateModel();
	void resetTracker();
};