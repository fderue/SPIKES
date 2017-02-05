#include "spikes_t/SpikeS_T.h"

void SpikeS_T::checkOcclusion(Rect boudingBox)
{
	Rect adapt_m_State = funUtils::giveAdaptRect(Frame0, boudingBox);
	//Bouding Box mask
	maskBB = Scalar(0); 
	maskBB(adapt_m_State) = 1;

	int nKpMatchBB = 0;
	for (auto& kp2 : l_kp2ModelB)
	{
		if (kp2->p_matched != nullptr && maskBB.at<uchar>(kp2->p_matched->kp.pt.y, kp2->p_matched->kp.pt.x))
		{
			nKpMatchBB++;
		}
	}
#if DEBUG_MODE
	cout << "nKpMatch in BB : " << nKpMatchBB << endl;
#endif
	noOcc = nKpMatchBB<THR_OCC;
}



template<typename T>
void updateModel_T(list<T*>& l_T,float alpha_f, float beta_w,const Mat& maskArea, vector<float>& v_rankedW)
{
	CV_Assert(maskArea.type() == CV_8U);
	v_rankedW.resize(l_T.size());
	int i;
	list<T*>::iterator it;
	for (i=0,it = l_T.begin(); it != l_T.end(); it++,i++){
		if ((*it)->p_matched != nullptr && maskArea.at<uchar>((*it)->p_matched->xy.y, (*it)->p_matched->xy.x)){
			(*it)->updateFeat(alpha_f);
			(*it)->w = (1 - beta_w)*(*it)->w + beta_w;
		}
		else{
			(*it)->w = (1 - beta_w)*(*it)->w;
		}
		v_rankedW[i] = (*it)->w;
	}
}

template<>
void updateModel_T<KeyPoint2>(list<KeyPoint2*>& l_T, float alpha_f, float beta_w, const Mat& maskArea, vector<float>& v_rankedW)
{
	v_rankedW.resize(l_T.size());
	int i;
	list<KeyPoint2*>::iterator it;
	for (i = 0, it = l_T.begin(); it != l_T.end(); it++, i++){
		if ((*it)->p_matched != nullptr && (maskArea.empty()||maskArea.at<uchar>((*it)->p_matched->kp.pt.y, (*it)->p_matched->kp.pt.x))){
			(*it)->updateFeat(alpha_f);
			(*it)->w = (1 - beta_w)*(*it)->w + beta_w;
		}
		else{
			(*it)->w = (1 - beta_w)*(*it)->w;
		}
		v_rankedW[i] = (*it)->w;
	}
}


template<typename T>
void deleteModel_T(list<T*>& l_T, int maxNumber, vector<float>& v_rankedW, queue<T*>& q_T, const Mat& maskArea)
{
	int idx_b;
	float weakest_w = 0;
	if ((idx_b = (int)v_rankedW.size() - maxNumber) > 0) {// if too much Kps, set a threshold
		std::sort(v_rankedW.begin(), v_rankedW.end());
		weakest_w = v_rankedW[idx_b];
	}
	int c = 0;
	list<T*>::iterator it;
	for ( it = l_T.begin(); it != l_T.end(); it++){
		if ((*it)->w < weakest_w) // erase the weakest kps
		{
			c++;
			q_T.push(*it); // add address of erased T to reuse memory
			it = l_T.erase(it); //remove from pointer from list
			it--;
		}
	}
	it = l_T.end();
	/*while (l_T.size() > maxNumber){
		c++;
		it--;
		q_T.push(*it);
		it = l_T.erase(it);
		it--;
	}*/
#if DEBUG_MODE
	cout << "Deleted : " << c << endl;
#endif
}

template<>
void deleteModel_T<KeyPoint2>(list<KeyPoint2*>& l_T, int maxNumber, vector<float>& v_rankedW, queue<KeyPoint2*>& q_T, const Mat& maskArea)
{
	int idx_b;
	float weakest_w = 0;
	if ((idx_b = (int)v_rankedW.size() - maxNumber) > 0) {// if too much Kps, set a threshold
		std::sort(v_rankedW.begin(), v_rankedW.end());
		weakest_w = v_rankedW[idx_b];
	}
	int c = 0;
	list<KeyPoint2*>::iterator it;
	for (it = l_T.begin(); it != l_T.end(); it++){
		if ((*it)->w < weakest_w || (l_T.size()>1 &&!maskArea.empty() && ((*it)->p_matched!=nullptr && !maskArea.at<uchar>((*it)->p_matched->kp.pt.y, (*it)->p_matched->kp.pt.x)))) // erase the weakest kps
		{
			c++;
			q_T.push(*it); // add address of erased T to reuse memory
			it = l_T.erase(it); //remove from pointer from list
			it--;
		}
	}
	it = l_T.end();
	/*while (l_T.size() > maxNumber){
		c++;
		it--;
		q_T.push(*it);
		it = l_T.erase(it);
		it--;
	}*/
#if DEBUG_MODE
	std::cout << "Deleted : " << c << endl;
#endif
}

template<typename T>
void addModel_T(T* buffM, list<T*>& l_TModel, list<T*>& l_TFrame, queue<T*>& q_M , Mat& maskArea, Point position=Point(-1,-1), float w0 = 0.1, float phi0 = 1,bool isFor = true, Mat& frameUpdate = Mat())
{
	int c = 0;
	list<T*>::iterator it_frame;
	for (it_frame = l_TFrame.begin(); it_frame != l_TFrame.end(); it_frame++){
		bool SpikeAdded = false;
		if ((*it_frame)->p_matched == nullptr && maskArea.at<uchar>((*it_frame)->xy.y, (*it_frame)->xy.x)){
			for (int i = 0; i < (*it_frame)->v_branch.size(); i++){
				if (((*it_frame)->v_branch[i].p_kp2->p_matched != nullptr) && ((*it_frame)->v_branch[i].p_kp2->p_matched->isFor)){ //check if matching kp inside
					for (int j = 0; j < (*it_frame)->v_pixels.size(); j++){
						if ((Point)(*it_frame)->v_branch[i].p_kp2->kp.pt == (*it_frame)->v_pixels[j].xy){
							c++;
							T* add;
							if (q_M.empty()) add = &buffM[l_TModel.size()];
							else { add = q_M.front(); q_M.pop(); }
							*add = *(*it_frame); // copy spikes from frame to model
							l_TModel.push_back(add); // add to list
							l_TModel.back()->initVote(position, w0, phi0); 
							l_TModel.back()->state = Superpixel::FGND;
#if DEBUG_MODE
							l_TModel.back()->alight(frameUpdate, Vec3b(255, 0, 0)); //if debug
#endif
							SpikeAdded = true;
							break;
						}
					}
				}
				if (SpikeAdded) break;
			} 
		}
	}
#if DEBUG_MODE
	cout << " SpikeS Add : " << c << endl;
#endif
}

template<>
void addModel_T<KeyPoint2>(KeyPoint2* buffM, list<KeyPoint2*>& l_Kp2M, list<KeyPoint2*>& l_Kp2Frame, queue<KeyPoint2*>& q_M, Mat& maskArea, Point position , float w0 , float phi0, bool isFor, Mat& frameUpdate )
{
	int c = 0;
	for (auto kp2 : l_Kp2Frame){
		if (kp2->p_matched == nullptr && maskArea.at<uchar>(kp2->kp.pt.y, kp2->kp.pt.x)){
			c++;
			KeyPoint2* add;
			if (q_M.empty()) add = &buffM[l_Kp2M.size()];
			else { add = q_M.front(); q_M.pop(); }
			*add = *kp2; // copy kp2 frome frame to model
			l_Kp2M.push_back(add); // add to list
			l_Kp2M.back()->isFor = isFor;
			l_Kp2M.back()->w = w0;
#if DEBUG_MODE
			if (isFor)circle(frameUpdate, l_Kp2M.back()->kp.pt, 1, Scalar(0, 255, 0),1,CV_AA);
			else circle(frameUpdate, l_Kp2M.back()->kp.pt, 1, Scalar(0, 0, 255), 1, CV_AA);
#endif
		}
	}
#if DEBUG_MODE
	cout << " Kp Add : " << c << endl;
#endif
}

void updatePhi(list<SpikeS*>& l_spikesM, Point position)
{
	for (auto& spikes : l_spikesM){
		if (spikes->p_matched != nullptr){
			Point xloc = spikes->p_matched->xy + spikes->voteVector;
			spikes->phi += exp(-norm(xloc - position));
		}
	}
}
void updateVote(list<SpikeS*>& l_spikesM, Point position)
{
	for (auto& spikes : l_spikesM){
		if (spikes->p_matched != nullptr){
			spikes->voteVector = (1 - F_INTERP_VOTE)*spikes->voteVector + F_INTERP_VOTE*(position - spikes->xy);
		}
	}
}
void SpikeS_T::updateModel()
{
	Mat frameUpdate = Frame_t.clone(); //if debug
	Rect adapt_m_State = funUtils::giveAdaptRect(Frame0, m_State); //fit bounding box to frame size
	//Bouding Box mask
	maskSegment = Scalar(0);
	displayMatchSpikesMask(maskSegment); // all matching spikes
	maskBB.copyTo(maskSegment, maskSegment); // matching spikes in BB

	//Surrounding area of BoundingBox
	//maskAroundBB = funUtils::makeMask(adapt_m_State, maskBB.cols, maskBB.rows, 2, false);
	maskAroundBB = Mat(maskBB.size(), CV_8U, Scalar(0));
	Rect trueROI2 = funUtils::giveRegion(m_position, m_State.size(), 2);
	maskAroundBB(funUtils::giveAdaptRect(maskAroundBB, trueROI2)) = 1;
	maskAroundBB(funUtils::giveAdaptRect(maskAroundBB, m_State)) = 0;

	//Spikes Update
	updateModel_T<SpikeS>(l_spikesModel, ALPHA_FEAT_SPIKES, BETA_W_SPIKES, maskBB, v_rankedW_SpikeS);
#if PHI_UPDATE
	updatePhi(l_spikesModel, m_position);
#endif
#if VOTE_UPDATE
	updateVote(l_spikesModel, m_position);
#endif
	deleteModel_T<SpikeS>(l_spikesModel,m_maxSpikesModel,v_rankedW_SpikeS,q_spikesModel,Mat());
	addModel_T<SpikeS>(buffSpikeS_Model, l_spikesModel, l_spikesFrame, q_spikesModel, maskBB, m_position, W0_SPIKES, PHI0_SPIKES,true, frameUpdate);

	//Fgnd Kp2 Update
	updateModel_T<KeyPoint2>(l_kp2ModelF, ALPHA_FEAT_KP_FOR, BETA_W_FOR, maskBB, v_rankedW_Kp2F);
	deleteModel_T<KeyPoint2>(l_kp2ModelF, MAX_KP2_FGND, v_rankedW_Kp2F,q_kp2ModelF,maskBB);
	addModel_T<KeyPoint2>(buffKp2_Fgnd,l_kp2ModelF,l_kp2Frame,q_kp2ModelF,maskSegment,m_position,W0_KP_FGND,0,true, frameUpdate);
	//Bgnd KP2 Update
	updateModel_T<KeyPoint2>(l_kp2ModelB,ALPHA_FEAT_KP_BACK, BETA_W_BACK,Mat(),v_rankedW_Kp2B);
	deleteModel_T<KeyPoint2>(l_kp2ModelB, MAX_KP2_BGND, v_rankedW_Kp2B,q_kp2ModelB,Mat());
	addModel_T<KeyPoint2>(buffKp2_Bgnd,l_kp2ModelB, l_kp2Frame,q_kp2ModelB,maskAroundBB,m_position,W0_KP_BGND,0,false, frameUpdate);

	//Recreate branches with the updated Kp2 Model 
	for (auto& spikes : l_spikesModel) spikes->createBranchesKp2(l_kp2ModelF);

#if DEBUG_MODE
	imshow("updateFrame", frameUpdate);
#endif
}