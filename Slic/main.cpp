#include <iostream>
#include <opencv2/opencv.hpp>
#include "Slic.h"

#define NSPX 1200
#define SIZE_SPX 16
#define WC 35

using namespace std;
using namespace cv;
int main() {

    //Mat im = imread("/media/derue/4A30A96F30A962A5/Videos/Tiger1/img/0001.jpg");
	//Mat im = imread("D:/Pictures/test_pic/lena.jpg");
	Mat im = imread("D:/Videos/CVPR_benchmark/tiger1/img/0001.jpg");
	CV_Assert(im.data!=nullptr);
    Slic seg;
	seg.initialize(im, SIZE_SPX, WC, Slic::SLIC_SIZE);

	for(int i=0; i<5; i++) {
		auto start = getTickCount();
		seg.generateSpx(im);
		auto end = getTickCount();
		cout << "runtime = " << (end - start) / getTickFrequency() << endl;
		cout << "# iteration : " << MAXIT << endl;
	}
    seg.display_contours(im,Scalar(0,0,255));
    imshow("seg", im);
    waitKey();

    return 0;
}