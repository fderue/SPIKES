#include "SpikeS_T.h"


void SpikeS_T::setFBSpikes(const Mat& fgndMask, const Mat& bgndMask, const Mat& grabSegMask, const list<SpikeS*>& l_sf_in, list<SpikeS*>& l_sf_out)
{
	vector<SpikeS*> v_SpikesInROI;
	vector<SpikeS*> v_SpikesInGrabCut;
	for (auto& it = l_sf_in.begin(); it != l_sf_in.end(); it++){
		if (bgndMask.at<uchar>((int)(*it)->xy.y, (int)(*it)->xy.x)){ (*it)->state = Superpixel::BGND; l_sf_out.push_back(*it); }
		else if (fgndMask.at<uchar>((int)(*it)->xy.y, (int)(*it)->xy.x)) {
			v_SpikesInROI.push_back(*it);
			int nPxInSpx = 0;
			for (int i = 0; i < (*it)->v_pixels.size(); i++){
				if (grabSegMask.at<uchar>((int)(*it)->v_pixels[i].xy.y, (int)(*it)->v_pixels[i].xy.x)){
					nPxInSpx++;
				}
			}
			if (nPxInSpx> PERC_PX_IN_SPX*(*it)->v_pixels.size()){
				v_SpikesInGrabCut.push_back(*it);
			}
		}
	}
	if (v_SpikesInGrabCut.size() > PERC_SPX_IN_ROI*v_SpikesInROI.size()){ //if enough Fgnd Spx
		for (int i = 0; i < v_SpikesInGrabCut.size(); i++){
			v_SpikesInGrabCut[i]->state = Superpixel::FGND;
			l_sf_out.push_back(v_SpikesInGrabCut[i]);
		}
	}
	else{ // if not enough Fgnd, take all spx in ROI
		for (int i = 0; i < v_SpikesInROI.size(); i++){
			v_SpikesInROI[i]->state = Superpixel::FGND;
			l_sf_out.push_back(v_SpikesInROI[i]);
		}
#if DEBUG_MODE
		cout << "grabSeg not used , too few spikes for model , pourcentage : " << v_SpikesInGrabCut.size() / (float)v_SpikesInROI.size() << endl;
#endif
	}
}

void SpikeS_T::extractSpikeS(Mat& frame)
{
	//extract Superpixel
	spxEngine->Segment(frame);
	Mat labels = spxEngine->getLabels();

	//extract Keypoint
	kpEngine.extrAndDescrKp(frame);
	kpEngine.getKpFromGpu(); // if KP on gpu (no waste of time even if not)
	kpEngine.getDescFromGpu();

	//create KeyPoint2 from kpEngine
	l_kp2Frame.clear();
	for (int i = 0; i < kpEngine.v_kp.size(); i++){
		buffKp2_Frame[i].set(kpEngine.v_kp[i], kpEngine.d_kp.row(i));
		l_kp2Frame.push_back(&buffKp2_Frame[i]);
	}

	// free memory for Nspx in buffSpikeS_Frame
	for (int i = 0; i < spxEngine->getNSpx(); i++) buffSpikeS_Frame[i].v_pixels.resize(0);

	//convert image to process in the chosen space color
	Mat frameConvert;
	switch (FEAT_COLORSPACE)
	{
	case Pixel::HSV:
		cvtColor(frame, frameConvert, CV_BGR2HSV); break;
	case Pixel::Lab:
		cvtColor(frame, frameConvert, CV_BGR2Lab); break;
	default:
		frameConvert = frame;
		break;
	}

	//Create SpikeS from labels
	l_spikesFrame.clear();
	for (int i = 0; i < labels.rows; i++){
		int* labels_ptr = labels.ptr<int>(i);		
		Vec3b* frame_ptr = frameConvert.ptr<Vec3b>(i);
		for (int j = 0; j < labels.cols; j++){
			buffSpikeS_Frame[labels_ptr[j]].v_pixels.push_back(Pixel(Point(j, i), Vec3f(frame_ptr[j]),FEAT_COLORSPACE));
		}
	}
	float diamSpx = sqrt((float)frame.cols*frame.rows / spxEngine->getNSpx());
	float searchR = F_SEARCHRADIUS_KP*diamSpx;
	for (int i = 0; i < spxEngine->getNSpx(); i++){
		if (!buffSpikeS_Frame[i].v_pixels.empty()){
			buffSpikeS_Frame[i].computeMean(); //if featType = color_mean
			if (FEAT_TYPE == Superpixel::HISTO3D) buffSpikeS_Frame[i].computeHisto(NBIN_HISTO_SPX);
			buffSpikeS_Frame[i].featType = FEAT_TYPE;
			buffSpikeS_Frame[i].diamSpx = diamSpx;
			buffSpikeS_Frame[i].m_searchR = searchR;
			buffSpikeS_Frame[i].p_matched = nullptr;
			buffSpikeS_Frame[i].colorSpace = FEAT_COLORSPACE;
			l_spikesFrame.push_back(&buffSpikeS_Frame[i]);
		}
	}

	//Add Keypoints to Superpixel to build the SPiKeS
	for (auto spikes : l_spikesFrame){
		spikes->createBranchesKp2(l_kp2Frame);
		spikes->state = Superpixel::BGND;
	}
}

void SpikeS_T::matchSpikeS()
{
	matchKp2();
	computeSpikesMatchMat();
	constraintMatch(); //analyze Match matrix and assign final matches
	cout << "#MatchSpikes : " << nMatchSpikeS << "/" << l_spikesModel.size() << endl;
}


void SpikeS_T::resetMatches()
{
	//reset Kp match
	for (auto& it : l_kp2ModelF) it->p_matched = nullptr;
	for (auto& it : l_kp2ModelB) it->p_matched = nullptr;
	for (auto& it : l_kp2Frame) it->p_matched = nullptr;
	for (auto& it : l_spikesModel) it->p_matched = nullptr;
	for (auto& it : l_spikesFrame) it->p_matched = nullptr;
}

void SpikeS_T::matchKp2()
{
	nMatchKp_for = 0;
	nMatchKp_back = 0;
	//--------------- create Mat descriptor from Kp2 Foreground and Background --------------
	CV_Assert(l_kp2ModelF.back()->mat_kpDescr.depth() == CV_32F);
	Mat descrFB((int)(l_kp2ModelB.size() + l_kp2ModelF.size()), l_kp2ModelF.back()->mat_kpDescr.cols, CV_32F, Scalar(0));
	int i; list<KeyPoint2*>::iterator it_kp_m, it_kp_f;
	float *mglob_ptr, *mloc_ptr;
	for (i = 0, it_kp_m = l_kp2ModelF.begin(); it_kp_m != l_kp2ModelF.end();it_kp_m++, i++){
		CV_Assert((*it_kp_m)->mat_kpDescr.isContinuous());
		mloc_ptr = (float*)(*it_kp_m)->mat_kpDescr.data;
		mglob_ptr = descrFB.ptr<float>(i);
		for (int j = 0; j < descrFB.cols; j++) mglob_ptr[j] = mloc_ptr[j];
	}
	for (i = (int)l_kp2ModelF.size(), it_kp_m = l_kp2ModelB.begin(); it_kp_m != l_kp2ModelB.end(); it_kp_m++, i++){
		CV_Assert((*it_kp_m)->mat_kpDescr.isContinuous());
		mloc_ptr = (float*)(*it_kp_m)->mat_kpDescr.data;
		mglob_ptr = descrFB.ptr<float>(i);
		for (int j = 0; j < descrFB.cols; j++) mglob_ptr[j] = mloc_ptr[j];
	}

	//--------------- choose query and train Descriptor ----------
#if GPU_MATCHER
	cuda::GpuMat queryDescr,trainDescr;
#else
	Mat queryDescr,trainDescr;
#endif
	// reverse order if more keypoint in model than in frame
	// #kp_modelFB < #kp_frame -> query = kp_model | train = kp_frame
	// #kp_modelFB > #kp_frame -> query = kp_frame | train = kp_model
	bool switchDirMatch = false;
	if (descrFB.rows>kpEngine.v_kp.size()) switchDirMatch = true;

	// send descriptorFB_model to gpu if MatcherGPU
	// if KP on GPU, no need to upload the frame descriptor 
#if GPU_MATCHER
	if (switchDirMatch){
		trainDescr.upload(descrFB);
#if KP_GPU
		queryDescr = kpEngine.d_kp_gpu;
#else
		queryDescr.upload(kpEngine.d_kp);
#endif
	}
	else{
		queryDescr.upload(descrFB);
#if KP_GPU
		trainDescr = kpEngine.d_kp_gpu;
#else
		trainDescr.upload(kpEngine.d_kp);
#endif
	}
#else
	if (switchDirMatch){
		queryDescr = kpEngine.d_kp;
		trainDescr = descrFB;
	}
	else{
		trainDescr = kpEngine.d_kp;
		queryDescr = descrFB;
	}
#endif

	//--------------------------- Match Kp ----------------------------------
#if GPU_MATCHER
	 matchEngine->match<cuda::GpuMat>(queryDescr, trainDescr, KNN);
#else 
	matchEngine->match<Mat>(queryDescr, trainDescr, KNN);
#endif

	//----------------------- Kp2 Model <-> Kp2 Frame : Pointer match assignement from v_DMatch -----------------
	
	int frameIdx, modelIdx;
	for (int i = 0; i < matchEngine->v_DMatch.size(); i++)
	{
		if (switchDirMatch){
			frameIdx = matchEngine->v_DMatch[i].queryIdx;
			modelIdx = matchEngine->v_DMatch[i].trainIdx;
		}
		else{
			modelIdx = matchEngine->v_DMatch[i].queryIdx;
			frameIdx = matchEngine->v_DMatch[i].trainIdx;
		}
		it_kp_f = next(l_kp2Frame.begin(), frameIdx);

		if (modelIdx < l_kp2ModelF.size()){

			it_kp_m = next(l_kp2ModelF.begin(), modelIdx); // foreground
			nMatchKp_for++;
		}
		else{

			it_kp_m = next(l_kp2ModelB.begin(), modelIdx - l_kp2ModelF.size()); // background
			nMatchKp_back++;
		}
		(*it_kp_m)->p_matched = *it_kp_f;
		(*it_kp_f)->p_matched = *it_kp_m;
	}

	cout << "#MatchKp_BACK : " <<nMatchKp_back <<"/"<< l_kp2ModelB.size() << endl;
	cout << "#NMatchKp_FOR : " << nMatchKp_for << "/" << l_kp2ModelF.size() << endl;
}


void SpikeS_T::resetTracker()
{
	m_State = StateInit;
	m_position.x = m_State.x + m_State.width / 2;
	m_position.y = m_State.y + m_State.height / 2;
	dPos.x = Frame_t.cols; dPos.y = Frame_t.rows;
}