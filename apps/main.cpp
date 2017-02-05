/*
@author Derue François-Xavier
francois.xavier.derue<at>gmail.com

MAKE_EXE : make executable for Matlab benchmark evaluation

*/

#include <opencv2/opencv.hpp>
#include "spikes_t/SpikeS_T.h"


#define MAKE_EXE 0 // if need a command line executable

using namespace cv;
using namespace std;

bool roi_captured = false;
bool mouse_pressed = false;
Point pt1, pt2;
Rect selection; // rectangle selected

void mouse_click(int event, int x, int y, int flags, void *param)
{
	switch (event){
		case CV_EVENT_LBUTTONDOWN:
		{
			std::cout << "Mouse Pressed" << std::endl;

			if (!roi_captured)
			{
				selection.x = x;
				selection.y = y;
				mouse_pressed = true;
			}
			else
			{
				std::cout << "ROI Already Acquired" << std::endl;
			}
			break;
		}
		case CV_EVENT_LBUTTONUP:
		{
			if (!roi_captured)
			{
				Mat cl;
				std::cout << "Mouse LBUTTON Released" << std::endl;

				selection.width = abs(x - selection.x);
				selection.height = abs(y - selection.y);

				roi_captured = true;
			}
			else
			{
				std::cout << "ROI Already Acquired" << std::endl;
			}
			break;
		}
		case CV_EVENT_MOUSEMOVE:
		{
			if (!roi_captured&&mouse_pressed)
			{
				selection.width = abs(x - selection.x);
				selection.height = abs(y - selection.y);
			}
			break;
		}
	}
}



#if MAKE_EXE

int main(int argc, const char *argv[])
{
	if (argc > 1)
	{
		Rect ROI;
		string vname = argv[1];
		string vpath = argv[2];
		int startFrame = stoi(argv[3]);
		int endFrame = stoi(argv[4]);
		int nz = stoi(argv[5]);
		string ext = argv[6];
		ROI.x = stoi(argv[7]);
		ROI.y = stoi(argv[8]);
		ROI.width = stoi(argv[9]);
		ROI.height = stoi(argv[10]);



		ofstream outFile(vname + "_SPiKeS_T.txt", ios::out);
		vector<string> files = funUtils::get_all_files_names_within_folder(vpath);
		Mat frame0 = imread(files[startFrame - 1]);

		//===== initialization ========
		SpikeS_T tracker;
		tracker.initialize(frame0, ROI);

		//===== Tracking ==========
		Mat frame;
		int toWrite[4];
		int end = endFrame;
		for (int j = startFrame - 1; j < end; j++)
		{
			cout << "==========frame :" << j + 1 << "/" << end << " ============" << endl;
			frame = imread(files[j]);

			tracker.track(frame);

			Rect BB = tracker.getState();

			toWrite[0] = BB.x;
			toWrite[1] = BB.y;
			toWrite[2] = BB.width;
			toWrite[3] = BB.height;
			for (int k = 0; k < 3; k++)
			{
				outFile << toWrite[k] << ",";
			}
			outFile << toWrite[3] << endl;

			tracker.update(); // update model
		}
		return 1;
	}
	else
	{
		cerr << "missing argument" << endl;
		return -1;
	}
}

#else


int main(int argc, const char** argv)
{
	string nameVideo = "basketball";
	string pathV = "E:/Videos/CVPR_benchmark/" + nameVideo + "/img/";
	string pathGnd = "E:/Videos/CVPR_benchmark/" + nameVideo + "/groundtruth_rect.txt";

	vector<string> files = funUtils::get_all_files_names_within_folder(pathV);
	Rect ROI0 = funUtils::getGndT(pathGnd);
	Mat frame = imread(files[0]);
	CV_Assert(frame.data != nullptr);

	//---------- Manual init ------------
	/*namedWindow("frame0");
	cvSetMouseCallback("frame0", mouse_click, 0);
	Mat frame0;
	while (!roi_captured)     
	{
		frame.copyTo(frame0);
		rectangle(frame0, selection, Scalar(0, 0, 255));

		imshow("frame0", frame0);
		cv::waitKey(30);
	}
	ROI0 = selection;*/
	//------------------------------------
	Mat frame0 = frame.clone();
	Mat imModel, imFrame, imMatchKp, imMatchSpikeS, imState, imFgnd, spxSegmentation;
	imModel = Mat(frame.size(), CV_8UC3, Scalar(0));
	SpikeS_T tracker;
	tracker.initialize(frame, ROI0);
	tracker.displayModelSpikeS(imModel);
	imshow("modelSpikes", imModel);

	size_t start, end;
	int endFrame = files.size();
	for (int i = 0; i < endFrame; i++)
	{
		frame = imread(files[i]);
		CV_Assert(frame.data != nullptr);
		cout << "------------------- Frame # : " << i+1 <<" / "<<endFrame<<"--------------------"<< endl;
		start = getTickCount();
		tracker.track(frame);
		end = getTickCount();
		cout << "----> track runtime =  " << (end - start) / getTickFrequency() << endl;
#if DEBUG_MODE
		cout << "State = " << tracker.getState() << endl;
		imFrame = frame.clone();
		tracker.displayFrameSpikeS(imFrame);
		tracker.displayMatchKp(frame0, frame, imMatchKp);
		tracker.displayMatchSpikeS(frame0, frame, imMatchSpikeS);
		spxSegmentation = frame.clone();
		tracker.displaySegmentation(spxSegmentation);
		imshow("spxSegmentation", spxSegmentation);
		imshow("frameSpikes", imFrame);
		imshow("matchKp", imMatchKp);
		imshow("matchSpikeS", imMatchSpikeS);
		tracker.displayFgnd(imFgnd);
		imshow("Matching Mask Foreground Spx", imFgnd);
#endif

		imState = frame.clone();
		tracker.displayVote(imState);
		tracker.displayState(imState);
		imshow("Tracking", imState);
		start = getTickCount();
		tracker.update();
		end = getTickCount();
		cout << "----> update runtime =  " << (end - start) / getTickFrequency() << endl;
		waitKey(1);
	}

	return 1;
}

#endif