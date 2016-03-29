/*
@author Derue François-Xavier
francois.xavier.derue<at>gmail.com
Parameter of the tracker SPiKeS_T 
*/


#pragma once

#define DEBUG_MODE 1 //visual support for debug
#define NSPX_ROI0 30 //#initial spx in frame0's BB
//#define SPX_DIAM_0 12 //or choose a constant size of spx instead (if uncomment)
#define SPX_DIAM_MIN 4 //min size of spx allowed

//Superpixel parameters
//-> SLIC_CPU || SLIC_GPU || SEEDS
#define SPX_METHOD SEEDS//SLIC_CPU//SLIC_GPU // SLIC_CPU
#define SPX_WC 30.f // color weight if SLIC

#define FEAT_COLORSPACE Pixel::HSV 

#define USE_HISTO 1
#if USE_HISTO
#define NBIN_HISTO_SPX 6
#define FEAT_TYPE Superpixel::HISTO3D
#define FEAT_DIM  NBIN_HISTO_SPX*NBIN_HISTO_SPX*NBIN_HISTO_SPX //(6x6x6)
#else
#define FEAT_TYPE Superpixel::MEAN_COLOR
#define FEAT_DIM 3 
#endif
#define HIST_COMP CV_COMP_BHATTACHARYYA

//Keypoints parameters
//-> the method is defined in KpEngine.h : choose SURF or SIFT
#define LOWE_RATIO .75f // david lowe ratio threshold (kp matching)
#define WKP0_INIT 1  // initial weight of Kp from inital BB (should be strong)
#define ALPHA_FEAT_KP_FOR 0.1f //adaptation factor for kp foreground
#define ALPHA_FEAT_KP_BACK 0.1f //"					 " kp background
#define BETA_W_FOR 0.1f // interpolation factor weight for kp foreground
#define BETA_W_BACK 0.1f //interpolation factor weight for kp background
#define W0_KP_FGND 0.1f //initial weight for new foreground kp
#define W0_KP_BGND 1.f //initial weight for new background kp

#define MAX_KP2_FGND 1000 //max # of fgnd kp in model
#define MAX_KP2_BGND 1000
#define MAX_KP2_FRAME 10000 //buffersize of kp extracted in a new frame

//SpikeS parameters
#define W0_SPIKES_INIT 1 // initial weight of SpKp from initial BB
#define PHI0_SPIKES_INIT 1 // initial persistence of SpKp frop initial BB
#define MAX_SPIKES_FRAME 50000 //buffer size of spikes extracted in new frame 
#define MAX_SPIKES_MODEL 1000 // max buff size of spikes model
#define F_NSPIKES_MAX_MODEL 3 //F_NSPKP_MAX*#SpikesModel0 = max size of model Spikes
#define F_SEARCHRADIUS_KP 2 // F_SEARCHRADIUS_KP*diamSpx = search radius for Kp around spx
#define W0_SPIKES 0.1f // initial weight of SpikeS from initial BB
#define PHI0_SPIKES 1 // initial persistence of SpKp frop initial BB
#define ALPHA_FEAT_SPIKES 0.1f //adaptation factor for a SpikeS in model
#define BETA_W_SPIKES 0.1f // learning factor for persistence factor
#define THR_SPIKES_COLOR_HIST 0.7f 
#define THR_SPIKES_COLOR_MC 250

//Some Thresholds
#define PERC_PX_IN_SPX 0.5 //grabcut threshold
#define PERC_SPX_IN_ROI 0.5 //grabcut threshold
#define MAX_IT_CMM 50 //#max iteration to ensure one-to-one match
#define THR_CONF_STATE 0.01f // #match_in_BB/#totalMatch > THR_CONF_STATE to update position
#define THR_OCC 3 //occlusion if #matchingKpBgnd_in_BB > THR_OCC
#define THR_MOTION_FAC 4 //motion constraint

//Some FLAG
#define REFINE_INIT 0 //activate grabcut for attempt of segmentation inside initial BB
#define PHI_UPDATE 1 //activate predictive factor
#define VOTE_UPDATE 1 //activate adaptation of vote 
#define F_INTERP_VOTE 0.1f // adaptive factor for vote


enum spxMeth
{
	SLIC_CPU,
	SLIC_GPU,
	SEEDS
};