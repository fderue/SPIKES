/*
Derue François-Xavier
francois.xavier.derue@gmail.com

This class implements the new feature SPiKeS :
a superpixel customized with keypoints

*/


#pragma once
#include <opencv2\opencv.hpp>
#include "Superpixel.h"
#include "ParamSpikeS_T.h"
#include <memory>
#include <list>
#define _USE_MATH_DEFINES
#include <math.h>


struct KeyPoint2
{
	KeyPoint kp;
	KeyPoint2* p_matched;
	Mat mat_kpDescr;
	bool isFor;
	float w;
	KeyPoint2(){}
	KeyPoint2(KeyPoint& kp, Mat& mat_kpDescr,bool isFor = false, KeyPoint2* p_matched=nullptr) : kp(kp), mat_kpDescr(mat_kpDescr),isFor(isFor), p_matched(p_matched){}
	KeyPoint2(const KeyPoint2& kp2) :kp(kp2.kp), p_matched(kp2.p_matched), mat_kpDescr(kp2.mat_kpDescr.clone()), isFor(kp2.isFor), w(kp2.w){}
	KeyPoint2& operator=(KeyPoint2& kp2){
		kp = kp2.kp;
		p_matched = kp2.p_matched;
		mat_kpDescr = kp2.mat_kpDescr.clone();
		isFor = kp2.isFor;
		w = kp2.w;
		return *this;
	}



	void set(KeyPoint& kp, Mat& mat_kpDescr,float w=1,bool isFor = false, KeyPoint2* p_matched = nullptr){
		this->kp = kp;
		this->mat_kpDescr = mat_kpDescr;
		this->p_matched = p_matched;
		this->w = w;
		this->isFor = isFor;
	}
	void updateFeat(float alpha){
		CV_Assert(p_matched != nullptr);
		kp.pt = (1 - alpha)*kp.pt + alpha*(*p_matched).kp.pt;
		kp.angle = (1 - alpha)*kp.angle + alpha*(*p_matched).kp.angle;
		mat_kpDescr = (1 - alpha)*mat_kpDescr + alpha*(*p_matched).mat_kpDescr;
	}
};

struct BranchKp2
{
	KeyPoint2* p_kp2;
	Point relpos;
	Point relpos_inv;
	float theta_inv;

	BranchKp2(){}
	BranchKp2(KeyPoint2* p_kp2, Point& relpos) :p_kp2(p_kp2), relpos(relpos)
	{
		float theta = p_kp2->kp.angle / 180.f*M_PI;
		Point inv = Point((relpos.x*cos(theta) - relpos.y*sin(theta)), relpos.x*sin(theta) + relpos.y*cos(theta));// rotation invariance
		//relpos_inv = relpos; // no transformation invariance (otherwise add cos+sin...)
		relpos_inv = inv;
		theta_inv = atan2((float)relpos_inv.y, (float)relpos_inv.x);
	}
};

class SpikeS:public Superpixel
{
public:
	float m_searchR;
	vector<BranchKp2> v_branch;
	SpikeS* p_matched;
	Point voteVector;
	float w;
	float phi;
public:
	SpikeS() :Superpixel(){ m_searchR = 10; p_matched = nullptr; }
	SpikeS(Point xy, Vec3f color, int radius = 10, ColorSpace colorSpace = BGR, FeatType featType = MEAN_COLOR, State neut = NEUT) :Superpixel(xy, color, colorSpace, featType, neut){ m_searchR = radius; p_matched = nullptr; }
	~SpikeS(){}


	void createBranchesKp2(const list<KeyPoint2*>& l_kp2);
	void drawMe(Mat& im);
	void initVote(Point position, float w0, float phi);
	void updateFeat(float alpha);
	void updateHist(float alpha);

};