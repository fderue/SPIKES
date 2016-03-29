#include "SpikeS_T.h"

SpikeS_T::SpikeS_T()
{
	switch (SPX_METHOD){
	case SLIC_CPU: spxEngine = make_unique<SlicEngine>(); break;
	case SLIC_GPU: spxEngine = make_unique<SlicEngine_CUDA>(); break;
	case SEEDS: spxEngine = make_unique<SeedsEngine>(); break;
	}

	matchEngine = make_unique<MatchEngine>(NORM_L2,CROSS_CHECK);

	//buffer allocation
	buffSpikeS_Model = new SpikeS[MAX_SPIKES_MODEL];
	buffSpikeS_Frame = new SpikeS[MAX_SPIKES_FRAME];
	buffKp2_Fgnd = new KeyPoint2[MAX_KP2_FGND*2];
	buffKp2_Bgnd = new KeyPoint2[MAX_KP2_BGND*2];
	buffKp2_Frame = new KeyPoint2[MAX_KP2_FRAME];

}

SpikeS_T::~SpikeS_T()
{
	delete[]buffSpikeS_Model;
	delete[]buffSpikeS_Frame;
	delete[]buffKp2_Fgnd;
	delete[]buffKp2_Bgnd;
	delete[]buffKp2_Frame;

}

void SpikeS_T::initialize(Mat& frame0, Rect ROI0)
{
	//State initialization
	timeT = 0; 
	Frame0 = frame0;
	Frame_t = frame0;
	dPos.x = 0; dPos.y = 0;
	dPosEst = dPos;
	m_State = ROI0;
	StateInit = ROI0;
	m_position.x = ROI0.x + ROI0.width / 2;
	m_position.y = ROI0.y + ROI0.height / 2;
	maskBB = Mat(frame0.size(), CV_8U, Scalar(0));
	maskSegment = Mat(frame0.size(), CV_8U, Scalar(0));

	//Superpixel Init Engine
#ifdef SPX_DIAM_0
	if ((ROI0.width*ROI0.height) / (SPX_DIAM_0*SPX_DIAM_0)<NSPX_ROI0){
		int NspxTmp = frame0.cols*frame0.rows / (ROI0.width*ROI0.height)*NSPX_ROI0; //choose NSPX wrt NSPX_ROI0 
		if (NspxTmp<MAX_SPIKES_FRAME && sqrt(frame0.cols*frame0.rows / NspxTmp)>SPX_DIAM_MIN) spxEngine->Init(frame0, NspxTmp, SPX_WC, Segmentor::NSPX);
		else {
			spxEngine->Init(frame0, SPX_DIAM_MIN, SPX_WC, Segmentor::SIZE); cout << "to much spx, limit with size then" << endl;
		}
	}else{
		spxEngine->Init(frame0, SPX_DIAM_0, SPX_WC, Segmentor::SIZE);
	}
#else
	int NspxTmp = frame0.cols*frame0.rows / (ROI0.width*ROI0.height)*NSPX_ROI0;

	if (NspxTmp<MAX_SPIKES_FRAME && sqrt(frame0.cols*frame0.rows/NspxTmp)>SPX_DIAM_MIN) spxEngine->Init(frame0, NspxTmp, SPX_WC, Segmentor::NSPX);
	else {
		spxEngine->Init(frame0, SPX_DIAM_MIN, SPX_WC, Segmentor::SIZE); cout << "too much spx for memory, limit size with SPX_DIAM_MIN or increase MAX_SPIKES_FRAME" << endl;
	}
#endif
	//----------- Make Fgnd/Bgnd Spikes samples -----------
	//Coarse Fgnd/Bgnd segmentation 
	Mat grabSegMask;
#if REFINE_INIT
	funUtils::getGrabCutSeg(frame0, grabSegMask, ROI0);
#if DEBUG_MODE
	imshow("grabSegMask", grabSegMask*255);
#endif
#else
	grabSegMask = Mat(frame0.size(),CV_8U,Scalar(255)); // cancel the grabcut
#endif
	extractSpikeS(frame0);

	//select Fgnd/Bgnd SpikeS
	list<SpikeS*> l_spikesFB;
	Mat fgndMask(frame0.size(), CV_8U, Scalar(0));
	fgndMask(ROI0) = 1;
	Mat bgndMaskFull = funUtils::makeMask(ROI0, frame0.cols, frame0.rows, 2,true); 
	setFBSpikes(fgndMask, bgndMaskFull,grabSegMask,l_spikesFrame,l_spikesFB);

	//Fill SpikeS model with the selected Fgnd SpikeS
	int i;
	list<SpikeS*>::iterator it;
	l_spikesModel.clear();
	for (i = 0, it = l_spikesFB.begin(); it != l_spikesFB.end(); it++){
		if ((*it)->state == Superpixel::FGND){
			buffSpikeS_Model[i] = *(*it);// copy from buffSpike_Frame;
			l_spikesModel.push_back(&buffSpikeS_Model[i]);
			i++;
		}
	}
	Mat bgndMask = funUtils::makeMask(ROI0, frame0.cols, frame0.rows, 2);

	//Fill Kp2_Foreground list And link with the spikes Model + Fill Kp2_Background list
	int ifor =0, iback = 0;
	for (auto& it = l_kp2Frame.begin(); it != l_kp2Frame.end(); it++){
		if (fgndMask.at<uchar>((int)(*it)->kp.pt.y, (int)(*it)->kp.pt.x)){
			buffKp2_Fgnd[ifor] = *(*it);// copy from buffKp_Frame;
			l_kp2ModelF.push_back(&buffKp2_Fgnd[ifor]);
			l_kp2ModelF.back()->isFor = true;
			ifor++;
		}
		else if (bgndMask.at<uchar>((int)(*it)->kp.pt.y,(int)(*it)->kp.pt.x)){
			buffKp2_Bgnd[iback] = *(*it);
			l_kp2ModelB.push_back(&buffKp2_Bgnd[iback]);
			l_kp2ModelB.back()->isFor = false;
			iback++;
		}
	}

	//Link Fgnd Keypoints to SpikeS model + assign voteVector to each Spikes_Model
	for (int i = 0; i < l_spikesModel.size(); i++){
		buffSpikeS_Model[i].createBranchesKp2(l_kp2ModelF);
		buffSpikeS_Model[i].initVote(m_position, W0_SPIKES_INIT, PHI0_SPIKES_INIT);
	}
#if DEBUG_MODE
	cout << "SPiKeS Initial model = " << l_spikesModel.size() <<" SPiKeS"<< endl;
#endif
	//Define maximum number of SpikeS in appearance model
	m_maxSpikesModel = F_NSPIKES_MAX_MODEL*l_spikesModel.size();
}


void SpikeS_T::track(Mat& frame)
{
	Frame_t = frame;
	extractSpikeS(frame);
	resetMatches();
	matchSpikeS();
	estimatePos();
	if (checkPosUpdate()){
		cout << "--- POSITION UPDATE : YES" << endl;
		updatePos();
	}
	else{
		cout << "--- POSITION UPDATE : NO" << endl;
		//m_position and dPos stay the same
	}

	//update the State 
	m_State.x = m_position.x - m_State.width / 2;
	m_State.y = m_position.y - m_State.height / 2;

	//scale Estimation (not implemented)
	float alpha = 0.1;
	m_State.width = (1 - alpha)*m_State.width + alpha*m_State.width;
	m_State.height = (1 - alpha)*m_State.height + alpha*m_State.height;
	//update the StateEstimation (for comparison purpose)
	StateEst.x = positionEst.x - m_State.width / 2;
	StateEst.y = positionEst.y - m_State.height / 2;

	//reset tracker if out of frame
	if (m_State.x >= Frame_t.cols || m_State.y >= Frame_t.rows || m_State.x + m_State.width < 0 || m_State.y+m_State.height<0) resetTracker();

	timeT++;
}

void SpikeS_T::update()
{
	cout << "--- MODEL UPDATE : ";
	checkOcclusion(m_State);
	if (noOcc){
		cout << "YES" << endl;
		updateModel();
	}
	else{
		cout << "NO" << endl;
	}
}
