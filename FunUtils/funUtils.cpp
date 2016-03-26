#include "funUtils.h"

#ifdef _WIN32
#include <Windows.h>
#else
#include <glob.h>
#endif

using namespace std;
namespace funUtils{

	bool stringCompare(const string &left, const string &right){
		for (string::const_iterator lit = left.begin(), rit = right.begin(); lit != left.end() && rit != right.end(); ++lit, ++rit)
		if (tolower(*lit) < tolower(*rit))
			return true;
		else if (tolower(*lit) > tolower(*rit))
			return false;
		if (left.size() < right.size())
			return true;
		return false;
	}
	vector<string> get_all_files_names_within_folder(string folder)
	{
		vector<string> names;

#ifdef _WIN64
		char search_path[200];
		sprintf_s(search_path, "%s*.*", folder.c_str());
		WIN32_FIND_DATA fd;
		HANDLE hFind = ::FindFirstFile(search_path, &fd);
		if (hFind != INVALID_HANDLE_VALUE) {
			do {
				// read all (real) files in current folder
				// , delete '!' read other 2 default folder . and ..
				if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {

					names.push_back(folder + fd.cFileName);
				}
			} while (::FindNextFile(hFind, &fd));
			::FindClose(hFind);
		}

		std::sort(names.begin(), names.end(), stringCompare);
#else
		string search_path = folder + "*";
		glob_t glob_result;
		glob(search_path.c_str(), GLOB_TILDE, NULL, &glob_result);
		string name;

		for (int i = 0; i<glob_result.gl_pathc; i++)
		{
			name = glob_result.gl_pathv[i];
			if (name[name.size() - 1] == 'g')
			{
				names.push_back(name);
			}
			else
			{
				break;
			}
		}


#endif
		return names;
	}

	using namespace cv;

	cv::Rect getGndT(string f)
	{
		size_t start = 0;
		size_t end = 0;
		int value[4];
		ifstream inFile(f.c_str(), ios::in);
		if (!inFile)cerr << "error file not found" << endl;
		string line;
		getline(inFile, line);
		for (int i = 0; i < 3; i++)
		{
			end = line.find(",", start);
			value[i] = stoi(line.substr(start, end - start));
			start = end + 1;
		}
		value[3] = stoi(line.substr(start, end - start));

		Rect gnd(value[0] - 1, value[1] - 1, value[2], value[3]);
		return gnd;
	}

	void getGrabCutSeg(Mat& inIm, Mat& mask_fgnd, Rect ROI)
	{
		Mat mask_out;
		Rect rect = ROI;
		Mat fgnd, bgnd;
		grabCut(inIm, mask_out, rect, bgnd, fgnd, 5, GC_INIT_WITH_RECT);
		bitwise_and(mask_out, GC_FGD, mask_fgnd);
	}

	Mat makeMask(Rect ROIin, int wFrame, int hFrame, float scale, bool fullFrame)
	{
		if (fullFrame){ Mat mask(hFrame, wFrame, CV_8U, Scalar(1)); mask(ROIin) = 0; return mask; }
		Mat mask(hFrame, wFrame, CV_8U, Scalar(0));
		float facx = ROIin.width / 2 * (1 - scale);
		float facy = ROIin.height / 2 * (1 - scale);
		Rect ROIex(ROIin.x + facx, ROIin.y + facy, ROIin.width*scale, ROIin.height*scale);

		adaptROI(ROIex, wFrame, hFrame);

		mask(ROIex) = 1;
		mask(ROIin) = 0;

		return mask;
	}


	void hist3D(Mat& image, Mat& hist, int Nbin, funUtils::HistColor histColor)
	{
		
		int h_bins = Nbin; int s_bins = Nbin; int v_bins = Nbin;
		int histSize[] = { h_bins, s_bins, v_bins };

		// hue varies from 0 to 179, saturation from 0 to 255
		float x_ranges[2];
		float y_ranges[2];
		float z_ranges[2];
		switch (histColor){
		case funUtils::HSV:
			x_ranges[0] = 0; x_ranges[1] = 180;
			y_ranges[0] = 0; y_ranges[1] = 256;
			z_ranges[0] = 0; z_ranges[1] = 256;
			break;
		case funUtils::BGR:
			x_ranges[0] = 0; x_ranges[1] = 256;
			y_ranges[0] = 0; y_ranges[1] = 256;
			z_ranges[0] = 0; z_ranges[1] = 256;
			break;
		case funUtils::Lab:
			x_ranges[0] = 0; x_ranges[1] = 256;
			y_ranges[0] = 0; y_ranges[1] = 256;
			z_ranges[0] = 0; z_ranges[1] = 256;
			break;
		default:
			x_ranges[0] = 0; x_ranges[1] = 256;
			y_ranges[0] = 0; y_ranges[1] = 256;
			z_ranges[0] = 0; z_ranges[1] = 256;
			break;
		}

		const float* ranges[] = { x_ranges, y_ranges, z_ranges };

		int channels[] = { 0, 1, 2 };

		calcHist(&image, 1, channels, Mat(), hist, 3, histSize, ranges, true, false);

		//normalization 3D
		CV_Assert(hist.isContinuous());
		float* hist_ptr = (float*)hist.data;;
		int a, b;
		int col_row = hist.size[0] * hist.size[1];
		float facNorm = image.cols*image.rows;

		for (int k = 0; k < hist.size[2]; k++)
		{
			a = k*col_row;
			for (int i = 0; i < hist.size[0]; i++)
			{
				b = i*hist.size[1];
				for (int j = 0; j < hist.size[1]; j++)
				{
					hist_ptr[a + b + j] /= facNorm;
				}
			}
		}
	}

	void printHist3D(Mat& histo3d){
		cout << "{";
		for (int i = 0; i < histo3d.size[0]; i++){
			cout << "[";
			for (int j = 0; j < histo3d.size[1]; j++){
				for (int k = 0; k < histo3d.size[2]; k++){
					cout << histo3d.at<float>(i, j, k) <<", ";
				}
				cout << ";" << endl;
			}
			cout << "]" << endl;
		}
		cout << "}" << endl;
	}

	

	void adaptROI(Rect& ROI, int wFrame, int hFrame)
	{
		if (ROI.x >= wFrame || ROI.y >= hFrame) { cerr << "error adaptROI out of frame" << endl;  ROI.x = 0; ROI.y = 0; ROI.width = 0; ROI.height = 0; }
		if (ROI.x < 0){ ROI.width = ROI.width + ROI.x; ROI.x = 0; }
		if (ROI.y < 0){ ROI.height = ROI.height + ROI.y; ROI.y = 0; }
		int dX, dY;
		if ((dX = ROI.x + ROI.width - wFrame) > 0){ ROI.width -= dX; }
		if ((dY = ROI.y + ROI.height - hFrame)>0){ ROI.height -= dY; }
		if (ROI.width < 0 || ROI.height < 0) { ROI.width = 0; ROI.height = 0; }
	}

	Rect giveAdaptRect(const Mat& frame, const Rect& ROIo)
	{
		Rect ROI = ROIo;
		int wFrame = frame.cols;
		int hFrame = frame.rows;
		if (ROI.x >= wFrame || ROI.y >= hFrame) { cerr << "error adaptROI out of frame" << endl; ROI.x = 0; ROI.y = 0; ROI.width = 0; ROI.height = 0; }
		if (ROI.x < 0){ ROI.width = ROI.width + ROI.x; ROI.x = 0; }
		if (ROI.y < 0){ ROI.height = ROI.height + ROI.y; ROI.y = 0; }
		int dX, dY;
		if ((dX = ROI.x + ROI.width - wFrame) > 0){ ROI.width -= dX; }
		if ((dY = ROI.y + ROI.height - hFrame)>0){ ROI.height -= dY; }
		if (ROI.width < 0 || ROI.height < 0) { ROI.width = 0; ROI.height = 0; }
		return ROI;
	}

	cv::Rect giveScaleRect(const cv::Rect& ROIin, float scale)
	{
		float facx = ROIin.width / 2 * (1 - scale);
		float facy = ROIin.height / 2 * (1 - scale);
		Rect ROIex(ROIin.x + facx, ROIin.y + facy, ROIin.width*scale, ROIin.height*scale);
		return ROIex;
	}
	cv::Rect giveRegion(cv::Point center, cv::Size s, int factor)
	{
		Rect zone = Rect(center.x - s.width / 2 * factor, center.y - s.height / 2 * factor, s.width*factor, s.height*factor);
		return zone;
	}

	void genSubWindByPoint(Rect base, float scale, Point delta, vector<cv::Point>& ULP, vector<cv::Point>& BRP, Mat& frame)
	{
		Rect upperBase = giveScaleRect(base, scale);
		Rect lowerBase = giveScaleRect(base, 1 / scale);

		adaptROI(upperBase, frame.cols, frame.rows);
		adaptROI(lowerBase, frame.cols, frame.rows);


		int height = upperBase.height;
		for (int y = upperBase.y; y <= lowerBase.y && height>delta.y; y += delta.y){

			int width = upperBase.width;
			for (int x = upperBase.x; x <= lowerBase.x && width>delta.x; x += delta.x){
				ULP.push_back(Point(x, y));
				BRP.push_back(Point(x + width-1, y + height-1));
				width -= 2 * delta.x;
			}
			height -= 2 * delta.y;
		}
	}


}