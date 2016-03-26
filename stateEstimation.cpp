#include "SpikeS_T.h"


void SpikeS_T::estimatePos()
{
	positionEst.x = 0; positionEst.y = 0;
	float weightTot = 0;
	for (auto& spikes_m : l_spikesModel){
		if (spikes_m->p_matched != nullptr)
		{
			Point x_l = spikes_m->p_matched->xy + spikes_m->voteVector;
			float weight = spikes_m->w*spikes_m->phi;
			weightTot += weight;
			positionEst += weight*x_l;
		}
	}
	if (weightTot != 0){
		positionEst = positionEst / weightTot;
		dPosEst = positionEst - m_position;
	}
	else{ //if no match -> stay on old position
		positionEst = m_position;
	}
}


bool SpikeS_T::checkPosUpdate()
{
	if (nMatchSpikeS == 0) return posEstOk = false;
	nMatchSpikeSBBEst = 0;
	Rect actualPosOldScale(positionEst.x - m_State.width / 2, positionEst.y - m_State.height / 2, m_State.width, m_State.height);
	Rect adapt_m_State = funUtils::giveAdaptRect(Frame0, actualPosOldScale);

	//Bouding Box mask
	maskBB = Scalar(0);
	maskBB(adapt_m_State) = 1;
	for (auto& spikes : l_spikesModel){
		SpikeS* m = spikes->p_matched;
		if (m != nullptr && maskBB.at<uchar>(m->xy.y, m->xy.x))
			nMatchSpikeSBBEst++;
	}
	float confidence = nMatchSpikeSBBEst / (float)nMatchSpikeS;
#if DEBUG_MODE
	cout << "Confidence on Position = " << confidence << endl;
#endif
	return posEstOk = confidence >= THR_CONF_STATE;;
}

void SpikeS_T::updatePos()
{
	//update global Position 
	m_position = positionEst;
	dPos = dPosEst;
	dPosTotale += dPos;

	//update Spikes and Kp position
	for (auto& spikes : l_spikesModel) spikes->xy += dPos;
	for (auto& kp : l_kp2ModelF) kp->kp.pt += Point2f(dPos.x,dPos.y);
}