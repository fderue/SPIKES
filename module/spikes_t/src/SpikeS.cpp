#include "spikes_t/SpikeS.h"


void SpikeS::createBranchesKp2(const list<KeyPoint2*>& l_kp2)
{
	v_branch.resize(0);
	float radius2 = pow(m_searchR, 2);

	for (auto& it : l_kp2)
	{
		float d2 = pow((it->kp.pt.x - xy.x), 2) + pow((it->kp.pt.y - xy.y), 2);
		if (d2<radius2)
		{
			Point relpos = Point((int)it->kp.pt.x, (int)it->kp.pt.y) - xy;
			BranchKp2 branch(it, relpos);
			v_branch.push_back(branch);
		}
	}
}

void SpikeS::initVote(Point position, float w0, float phi0)
{
	w = w0;
	phi = phi0;
	voteVector = position - xy;
}

void::SpikeS::drawMe(Mat& im)
{
	for (int i = 0; i < v_branch.size(); i++){
		line(im, xy, v_branch[i].p_kp2->kp.pt, Scalar(0, 255, 0), 1,CV_AA);
		circle(im, v_branch[i].p_kp2->kp.pt, 1,Scalar(0,0,255),1,CV_AA);
	}
}

void SpikeS::updateHist(float alpha)
{
	float* hist_ptr = histo.ptr<float>();
	int a, b;
	int col_row = histo.size[0] * histo.size[1];
	for (int k = 0; k < histo.size[2]; k++)
	{
		a = k*col_row;
		for (int i = 0; i < histo.size[0]; i++)
		{
			b = i*histo.size[1];
			for (int j = 0; j < histo.size[1]; j++)
			{
				int idx = a + b + j;
				hist_ptr[idx] = (1 - alpha)*hist_ptr[idx] + alpha*p_matched->histo.ptr<float>()[idx];
			}
		}
	}
}
void SpikeS::updateFeat(float alpha)
{
#if FEAT_TYPE==Superpixel::MEAN_COLOR
	CV_Assert(p_matched != nullptr);
	color = (1 - alpha)*color + alpha*p_matched->color;
#else
	updateHist(alpha);
#endif
}