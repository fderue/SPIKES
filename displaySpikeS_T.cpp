#include "SpikeS_T.h"
using namespace std;
using namespace cv;

void SpikeS_T::displayModelSpx(Mat& im)
{
	displaySegmentation(im);
	for (auto& it = l_spikesModel.begin(); it != l_spikesModel.end(); it++)
		if((*it)->state==Superpixel::FGND)(*it)->alight(im, Vec3b(0, 255, 0));
}

void SpikeS_T::displayModelKpFB(Mat& im)
{
	for (auto& it = l_kp2ModelB.begin(); it != l_kp2ModelB.end(); it++)
		circle(im, (*it)->kp.pt, 1, Scalar(255, 0, 0), 1, CV_AA);
	for (auto& it = l_kp2ModelF.begin(); it != l_kp2ModelF.end(); it++)
		circle(im, (*it)->kp.pt, 1, Scalar(0, 255, 255), 1, CV_AA);

}

void SpikeS_T::displayModelSpikeS(Mat& im)
{
	displaySegmentation(im);
	for (auto& it = l_spikesModel.begin(); it != l_spikesModel.end(); it++){
		(*it)->alight(im, (*it)->color);
		(*it)->drawMe(im);
	}
}

void SpikeS_T::displaySegmentation(Mat& im)
{
	spxEngine->DisplayContours(im, Scalar(0, 0, 0));
}

void SpikeS_T::displayFrameSpikeS(Mat& im)
{
	displaySegmentation(im);
	for (auto& it = l_spikesFrame.begin(); it != l_spikesFrame.end(); it++)
		(*it)->drawMe(im);
}

void SpikeS_T::displayMatchKp(Mat& frame1, Mat& frame2, Mat& output)
{
	// creation de l'output image
	int outWidth = frame1.cols + frame2.cols;
	int outHeight = frame1.rows;
	output = Mat(outHeight, outWidth, CV_8UC3, Scalar(0, 0, 0));

	auto xstep1 = frame1.step.p[1];
	auto ystep1 = frame1.step.p[0];
	auto xstep2 = frame2.step.p[1];
	auto ystep2 = frame2.step.p[0];
	auto xstep_out = output.step.p[1];
	auto ystep_out = output.step.p[0];

	// image gauche
	for (int j = 0; j < frame1.rows; j++)
	{
		for (int i = 0; i < frame1.cols; i++)
		{
			for (int ch = 0; ch < 3; ch++)
			{
				*(output.data + j*ystep_out + i*xstep_out + ch) = *(frame1.data + ystep1*j + xstep1*i + ch);
			}
		}
	}
	// image droite
	for (int j = 0; j < frame2.rows; j++)
	{
		for (int i = 0; i < frame2.cols; i++)
		{
			for (int ch = 0; ch < 3; ch++)
			{
				*(output.data + j*ystep_out + (ystep1 + i*xstep_out) + ch) = *(frame2.data + ystep2*j + xstep2*i + ch);
			}
		}
	}
	//=== draw line of match of foreground =====

	for (auto& kp2 : l_kp2ModelF)
	{
		if (kp2->p_matched != nullptr)
		{
			Point first = kp2->kp.pt-Point2f(dPosTotale.x,dPosTotale.y);
			Point p = kp2->p_matched->kp.pt;

			Scalar  colorM = Scalar(rand() % 255, rand() % 255, rand() % 255);
			circle(output, first, 1, colorM, 2, CV_AA);
			circle(output, Point(frame1.cols + p.x, p.y), 1, colorM, 2, CV_AA);
			line(output, first, Point(frame1.cols + p.x, p.y), Scalar(255, 0, 0), 1, CV_AA);
		}
	}
	//=== draw line of match of background =====
	for (auto&kp2 : l_kp2ModelB)
	{
		if (kp2->p_matched != nullptr)
		{
			Point first = kp2->kp.pt;
			Point p = kp2->p_matched->kp.pt;

			Scalar  colorM;
			int radius;
			if (maskBB.at<uchar>((int)kp2->p_matched->kp.pt.y, (int)kp2->p_matched->kp.pt.x))
			{
				colorM = Scalar(0, 255, 0);
				radius = 3;
				circle(output, first, radius, colorM, 1, CV_AA);
				circle(output, Point(frame1.cols + p.x, p.y), radius, colorM, 1, CV_AA);
				line(output, first, Point(frame1.cols + p.x, p.y), Scalar(0, 0, 255), 1, CV_AA);
			}
			//else
			/*{
				colorM = Scalar(rand() % 255, rand() % 255, rand() % 255);
				radius = 1;
				circle(output, first, radius, colorM, 1, CV_AA);
				circle(output, Point(frame1.cols + p.x, p.y), radius, colorM, 1, CV_AA);
				line(output, first, Point(frame1.cols + p.x, p.y), Scalar(0, 0, 255), 1, CV_AA);
			}*/
		}
	}
}

void SpikeS_T::displayMatchSpikeS(Mat& frame1, Mat& frame2, Mat& output)
{
	// creation de l'output image
	int outWidth = frame1.cols + frame2.cols;
	int outHeight = frame1.rows;
	output = Mat(outHeight, outWidth, CV_8UC3, Scalar(0, 0, 0));

	auto xstep1 = frame1.step.p[1];
	auto ystep1 = frame1.step.p[0];
	auto xstep2 = frame2.step.p[1];
	auto ystep2 = frame2.step.p[0];
	auto xstep_out = output.step.p[1];
	auto ystep_out = output.step.p[0];

	// image gauche
	for (int j = 0; j<frame1.rows; j++)
	{
		for (int i = 0; i<frame1.cols; i++)
		{
			for (int ch = 0; ch<3; ch++)
			{
				*(output.data + j*ystep_out + i*xstep_out + ch) = *(frame1.data + ystep1*j + xstep1*i + ch);
			}
		}
	}
	// image droite
	for (int j = 0; j<frame2.rows; j++)
	{
		for (int i = 0; i<frame2.cols; i++)
		{
			for (int ch = 0; ch<3; ch++)
			{
				*(output.data + j*ystep_out + (ystep1 + i*xstep_out) + ch) = *(frame2.data + ystep2*j + xstep2*i + ch);
			}
		}
	}

	//=== draw line of matc =====

	for (auto& spkp : l_spikesModel)
	{
		if (spkp->p_matched != nullptr && spkp->xy.x != 0)
		{
			//Point first = spkp.mySpx->xy;
			Point first = spkp->xy;
			Point p = spkp->p_matched->xy;
			line(output, first, Point(frame1.cols + p.x, p.y), Scalar(0, 0, 255), 1, CV_AA);
			//line(output, first, Point(frame1.cols + p.x, p.y), Scalar(rand() % 255, rand() % 255, rand() % 255), 1, CV_AA);
		}
	}
}

void SpikeS_T::displayState(Mat& im)
{
	//rectangle(im, StateEst, Scalar(0, 0, 255), 1, CV_AA);
	rectangle(im, m_State, Scalar(0, 255, 0),2,CV_AA);
}

void SpikeS_T::displayVote(Mat& im)
{
	displaySegmentation(im);
	for (auto& it : l_spikesModel){
		if (it->p_matched != nullptr){
			arrowedLine(im, it->p_matched->xy, it->p_matched->xy + it->voteVector, Scalar(255, 0, 0), 1, CV_AA);
		}
	}
}

void SpikeS_T::displayMatchSpikesMask(Mat& im8UC1)
{
	for (auto& spikes : l_spikesFrame){
		if (spikes->p_matched != nullptr ||spikes->state == Superpixel::FGND){
			spikes->alight(im8UC1, Vec3b(255, 0, 0));
		}
	}
}

void SpikeS_T::displayFgnd(Mat& image)
{
	image = Mat(Frame_t.size(), CV_8UC3, Scalar(0));
	for (auto& spikes : l_spikesModel){
		if (spikes->p_matched!=nullptr){
			for (auto px : spikes->p_matched->v_pixels){
				image.at<Vec3b>(px.xy.y, px.xy.x) = Frame_t.at<Vec3b>(px.xy.y, px.xy.x);
			}
		}
	}
}