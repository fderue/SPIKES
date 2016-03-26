#include "SpikeS_T.h"


//------------------- Match SpikeS ----------------------

int countOnes(Mat& col_j, vector<int>& idx_ones)
{
	int counter = 0;
	//int* col_j_ptr = col_j.ptr<int>();
	for (int i = 0; i<col_j.rows; i++)
	{
		if (col_j.at<int>(i, 0) == 1)
		{
			counter++;
			idx_ones.push_back(i);
		}
	}
	return counter;
}

int findMaxIdxInMat1D(Mat v)
{
	double maxv, minv;
	Point maxIdx, minIdx;

	cv::minMaxLoc(v, &minv, &maxv, &minIdx, &maxIdx);
	if (v.rows == 1) return maxIdx.x;
	else return maxIdx.y;
}



float inline fun_sm(float d_theta, float e, float radius)
{
	float ne = 2 * radius;
	return (float)exp(-pow(e / ne, 2));
	//return (float)exp(-(pow(d_theta / M_PI, 2) + pow(e / ne, 2))); // min = exp(-2) = 0.135 et max = exp(0) = 1;
}
float inline fun_sTot(float dc, float s1, float normFactorColor)
{	
	return exp(-dc/normFactorColor) + s1; // min = exp(-thr_c)->thr_c = 0.5 : min = exp(-0.5) = 0.605
}


float computeScore(const SpikeS &spikes1,const SpikeS& spikes2, Superpixel::FeatType featType) // spkp1 should be from the l_spikesModel !
{
	float thr_c, dc, normFactor;
	if (featType == Superpixel::HISTO3D)
	{
		thr_c = THR_SPIKES_COLOR_HIST; // color threshold 
		normFactor = 1.f;
		dc = (float)compareHist(spikes1.histo, spikes2.histo, HIST_COMP); // histogramme Comparison
	}
	else if (featType == Superpixel::MEAN_COLOR)
	{
		thr_c = THR_SPIKES_COLOR_MC; // color threshold 
		normFactor = 441.f;
		dc = norm(spikes1.color - spikes2.color); // color L2 distance
	}
	if (dc < thr_c)
	{
		//==== sm = weighted sum of good kp_match =====
		double sm = 0;
		for (int i = 0; i < spikes1.v_branch.size(); i++)
		{
			if (spikes1.v_branch[i].p_kp2->p_matched != nullptr)
			{
				for (int j = 0; j < spikes2.v_branch.size(); j++)
				{
					if (spikes1.v_branch[i].p_kp2->p_matched == spikes2.v_branch[j].p_kp2)
					{
						double dif = abs(spikes1.v_branch[i].theta_inv - spikes2.v_branch[j].theta_inv);
						double d_theta = min(dif, 2 * M_PI - dif); //in radian
						float e = (float)norm(spikes1.v_branch[i].relpos_inv - spikes2.v_branch[j].relpos_inv);
						sm += fun_sm((float)d_theta, e, spikes2.m_searchR);
						break;
					}
				}
			}
		}
		float sTot = fun_sTot(dc, (float)sm, normFactor);
		return sTot;
	}
	else
	{
		return 0;
	}
}


void SpikeS_T::computeSpikesMatchMat()
{
	//===== compute Matrix of the score and checkbox match i_j =========
	scoreMat = Mat((int)l_spikesModel.size(), (int)l_spikesFrame.size(), CV_32F, Scalar(0));
	matchMat = Mat(scoreMat.size(), CV_32S, Scalar(0));
	int i, j;
	list<SpikeS*>::iterator it_spikes_m, it_spikes_f;
	for (i = 0, it_spikes_m = l_spikesModel.begin(); i < l_spikesModel.size(); i++, it_spikes_m++){
		int best_j = -1;
		float score_max = 0;
		float* scoreMat_r = scoreMat.ptr<float>(i);
		for (j = 0, it_spikes_f = l_spikesFrame.begin(); j<l_spikesFrame.size(); j++, it_spikes_f++){
			float score = computeScore(*(*it_spikes_m), *(*it_spikes_f), FEAT_TYPE); // compute score (observation model)
			scoreMat_r[j] = score;
			if (score>score_max){
				score_max = score;
				best_j = j;
			}
		}
		if (best_j != -1){
			matchMat.at<int>(i, best_j) = 1;
		}
	}

	//======= ensure one to one match =========
	int global_counter = 1;
	int ite = 0;
	Mat col_j_score, col_j;

	while (global_counter != 0 && ite <= MAX_IT_CMM){
		global_counter = 0;
		for (int j = 0; j<matchMat.cols; j++){
			vector<int> idx_ones;
			col_j = matchMat.col(j);
			int n_ones = countOnes(col_j, idx_ones);
			if (n_ones>1){
				global_counter++;
				col_j_score = scoreMat.col(j);
				int best_i_in_col_j = findMaxIdxInMat1D(col_j_score);
				for (int b = 0; b < idx_ones.size(); b++){
					int other_i = idx_ones[b];
					if (other_i != best_i_in_col_j){
						matchMat.at<int>(other_i, j) = 0;//deselect
						scoreMat.at<float>(other_i, j) = 0; //delete this maximum
					}
				}
			}
		}
		ite++;
	}
}

void SpikeS_T::constraintMatch()
{
	nMatchSpikeS = 0;
	// score constraint
	float thr_min = exp(-THR_SPIKES_COLOR_HIST);
	float thr_score = (nMatchKp_for > 0) ? thr_min + 1 : thr_min;
	list<SpikeS*>::iterator it_spikes_m, it_spikes_f;
	int i, j;
	for (i = 0, it_spikes_m = l_spikesModel.begin(); i < scoreMat.rows; i++, it_spikes_m++){
		float* scoreMat_ptr = scoreMat.ptr<float>(i);
		int* matchMat_ptr = matchMat.ptr<int>(i);
		for (j = 0, it_spikes_f = l_spikesFrame.begin(); j<scoreMat.cols; j++, it_spikes_f++){
			if (matchMat_ptr[j] == 1 && scoreMat_ptr[j]>thr_score){
				(*it_spikes_m)->p_matched = (*it_spikes_f);
				(*it_spikes_f)->p_matched = (*it_spikes_m);
				(*it_spikes_f)->voteVector = (*it_spikes_m)->voteVector;
				nMatchSpikeS++;
				break;
			}
		}
	}

	// motion constraint
	float thr_motion = norm(dPos) + THR_MOTION_FAC * l_spikesModel.back()->diamSpx;{
		for (auto& spikes : l_spikesModel){
			if (spikes->p_matched != nullptr && norm(spikes->p_matched->xy - spikes->xy) > thr_motion){
				nMatchSpikeS--;
				spikes->p_matched->p_matched = nullptr;
				spikes->p_matched = nullptr;
			}
		}
	}
	
}