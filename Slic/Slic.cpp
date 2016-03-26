
#include "Slic.h"


void Slic::resetVariables()
{
	m_allCenters.clear();
	m_labels.forEach<int>([](int &l, const int* position)->void{l = -1; });
	for (int j = 0; j < m_height; j++){
		std::for_each(m_allDist[j].begin(), m_allDist[j].end(), [](float& dist)->void{dist = FLT_MAX; });
	}
}

void Slic::initialize(Mat& frame, int nspx_size, float wc, InitType type)
{
	m_wc = wc;
	m_height = frame.rows;
	m_width = frame.cols;
	if (type == SLIC_NSPX){
		m_nSpx = nspx_size;	m_diamSpx = (int)sqrt(m_width*m_height / (float)m_nSpx);
	}
	else{
		m_nSpx = m_height*m_width / (nspx_size*nspx_size);
		m_diamSpx = nspx_size;
	}
	//initialize labels
	m_labels = Mat(m_height, m_width, CV_32S, Scalar(-1));
	m_allDist.resize(m_height);
	for (int j = 0; j < m_height; j++){
		m_allDist[j] = vector<float>(m_width, FLT_MAX);
	}

}

int dx_n[] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
int dy_n[] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };
void moveToLowGrad(Point& xy_out, Vec3f& minColor, Mat& frameLab)
{
	Point xy = xy_out;
	float minGrad = FLT_MAX;
	int xloc, yloc;
	for (int k = 1; k < 9; k++){
		yloc = xy.y + dy_n[k];
		if (yloc < frameLab.rows - 2&&yloc >= 0){


			for (int l = 1; l < 9; l++){
				xloc = xy.x + dx_n[l];
				if (xloc < frameLab.cols-2&&xloc>=0){
					Vec3f cc = frameLab.at<Vec3f>(yloc, xloc);
					Vec3f cd = frameLab.at<Vec3f>(yloc + 1, xloc);
					Vec3f cr = frameLab.at<Vec3f>(yloc, xloc + 1);

					float grad;
					if ((grad = pow(cc[0] - cd[0], 2) + pow(cc[0] - cr[0], 2)) < minGrad)
					{
						minGrad = grad;
						xy_out.x = xloc;
						xy_out.y = yloc;
						minColor = cc;
					}
				}
			}
		}
	}
}
void Slic::generateSpx(Mat & frame)
{
	resetVariables();
	Mat frameLab;
	cvtColor(frame, frameLab, CV_BGR2Lab);
	frameLab.convertTo(frameLab, CV_32FC3);
	//initializa clusters
	int diamSpx_d2 = m_diamSpx / 2;
	for (int y = diamSpx_d2 - 1; y < m_height; y += m_diamSpx)
	{
		for (int x = diamSpx_d2 - 1; x < m_width; x += m_diamSpx)
		{
				center c;
				c.xy = Point(x, y);
				Vec3f cLab;
				moveToLowGrad(c.xy, cLab, frameLab);
				c.Lab[0] = cLab[0];
				c.Lab[1] = cLab[1];
				c.Lab[2] = cLab[2];

				m_allCenters.push_back(c);
		}
	}
	m_nSpx = (int)m_allCenters.size(); //real number of spx

	// iterate
	for (int it = 0; it < MAXIT; it++)
	{
		findCenters(frameLab);
		updateCenters(frameLab);
	}
	enforceConnectivity(); 
}

inline float slicDistance(center& c, float x, float y, float L, float a, float b, float S2, float m2)
{
	float dc2 = pow(c.Lab[0] - L, 2) + pow(c.Lab[1] - a, 2) + pow(c.Lab[2] - b, 2);
	float ds2 = pow(c.xy.x - x, 2) + pow(c.xy.y - y, 2);

	return dc2 + ds2 / S2*m2;
	//return sqrt(dc2 + ds2 / S2*m2);
}
void Slic::findCenters(Mat& frame)
{
	float S2 = m_diamSpx*m_diamSpx;
	float m2 = m_wc*m_wc;
	int diamSpx3d2 = m_diamSpx;
	for (int c = 0; c < m_allCenters.size(); c++)
	{
		Point xy_c = m_allCenters[c].xy;
		if (xy_c.x != -1) {
			for (int i = xy_c.y - diamSpx3d2; i <= xy_c.y + diamSpx3d2; i++) {
				for (int j = xy_c.x - diamSpx3d2; j <= xy_c.x + diamSpx3d2; j++) {
					if (i >= 0 && i < m_height && j >= 0 && j < m_width) {
						Vec3f lab = frame.at<Vec3f>(i, j);
						float d = slicDistance(m_allCenters[c], j, i, lab.val[0], lab.val[1], lab.val[2], S2, m2);

						if (d < m_allDist[i][j]) {
							m_allDist[i][j] = d;
							m_labels.at<int>(i,j) = c;
						}
					}
				}
			}
		}
	}

}
void Slic::updateCenters(Mat& frame)
{
	//clear center value
	vector<int> counter(m_allCenters.size(), 0);
	for (int i = 0; i < m_allCenters.size(); i++)
	{
		m_allCenters[i].xy.x = m_allCenters[i].xy.y = m_allCenters[i].Lab[0] = m_allCenters[i].Lab[1] = m_allCenters[i].Lab[1] = 0;
	}
	for (int i = 0; i < m_height; i++)
	{
		int* m_labels_ptr = m_labels.ptr<int>(i);
		for (int j = 0; j < m_width; j++)
		{

			int idxC = m_labels_ptr[j];
			if (idxC != -1){
				Vec3f lab = frame.at<Vec3f>(i, j);
				m_allCenters[idxC].xy += Point(j, i);

				m_allCenters[idxC].Lab[0] += lab.val[0];
				m_allCenters[idxC].Lab[1] += lab.val[1];
				m_allCenters[idxC].Lab[2] += lab.val[2];

				counter[idxC]++;
			}
			else{
				cerr << "one label is -1 : impossible normally" << endl;
				cout << i << "," << j << endl;
			}
		}
	}
	for (int i = 0; i < m_allCenters.size(); i++)
	{
		if (counter[i] != 0)
		{
			m_allCenters[i].xy /= counter[i];

			m_allCenters[i].Lab[0] /= counter[i];
			m_allCenters[i].Lab[1] /= counter[i];
			m_allCenters[i].Lab[2] /= counter[i];
		}
		else
		{
			m_allCenters[i].xy.x = -1; // reject a center which accept no pixel
			m_nSpx--;
		}
	}
}


const int dx4[4] = { -1, 0, 1, 0 };
const int dy4[4] = { 0, -1, 0, 1 };
void Slic::enforceConnectivity()
{
	int label = 0, adjlabel = 0;
	int lims = (m_width * m_height) / (m_nSpx);
	lims = lims >> 2;
	if (lims < 2)return;
	vector<vector<int> >newLabels;
	for (int i = 0; i < m_height; i++)
	{
		vector<int> nv(m_width, -1);
		newLabels.push_back(nv);
	}

	for (int i = 0; i < m_height; i++)
	{
		int* m_labels_ptr = m_labels.ptr<int>(i);
		for (int j = 0; j < m_width; j++)
		{
			if (newLabels[i][j] == -1)
			{
				vector<Point> elements;
				elements.push_back(Point(j, i));
				for (int k = 0; k < 4; k++)
				{
					int x = elements[0].x + dx4[k], y = elements[0].y + dy4[k];
					if (x >= 0 && x < m_width && y >= 0 && y < m_height)
					{
						if (newLabels[y][x] >= 0)
						{
							adjlabel = newLabels[y][x];
						}
					}
				}
				int count = 1;
				for (int c = 0; c < count; c++)
				{
					for (int k = 0; k < 4; k++)
					{
						int x = elements[c].x + dx4[k], y = elements[c].y + dy4[k];
						if (x >= 0 && x < m_width && y >= 0 && y < m_height)
						{
							if (newLabels[y][x] == -1 && m_labels_ptr[j] == m_labels.at<int>(y,x))
							{
								elements.push_back(Point(x, y));
								newLabels[y][x] = label;//m_labels[i][j];
								count += 1;
							}
						}
					}
				}
				if (count <= lims) {
					for (int c = 0; c < count; c++) {
						newLabels[elements[c].y][elements[c].x] = adjlabel;
					}
					label -= 1;
				}
				label += 1;
			}
		}
	}
	m_nSpx = label;
	for (int i = 0; i < newLabels.size(); i++){
		int* m_labels_ptr = m_labels.ptr<int>(i);
		for (int j = 0; j < newLabels[i].size(); j++){
			m_labels_ptr[j] = newLabels[i][j];
		}
	}
	//"Note: index in m_allCenters does not correspond anymore to the right label, but we do not need then anymore

}
const int dx8[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };

void Slic::display_contours(Mat& image, Scalar colour) {

	/* Initialize the contour vector and the matrix detailing whether a pixel
	 * is already taken to be a contour. */
	vector<Point> contours;
	vector<vector<bool> > istaken;
	for (int i = 0; i < image.rows; i++) {
		vector<bool> nb;
		for (int j = 0; j < image.cols; j++) {
			nb.push_back(false);
		}
		istaken.push_back(nb);
	}
	/* Go through all the pixels. */
	for (int i = 0; i < image.rows; i++) {
		int* m_labels_ptr = m_labels.ptr<int>(i);
		for (int j = 0; j < image.cols; j++) {

			int nr_p = 0;

			/* Compare the pixel to its 8 neighbours. */
			for (int k = 0; k < 8; k++) {
				int x = j + dx8[k], y = i + dy8[k];

				if (x >= 0 && x < image.cols && y >= 0 && y < image.rows) {
					if (istaken[y][x] == false && m_labels_ptr[j] != m_labels.at<int>(y,x)) {
						nr_p += 1;
					}
				}
			}
			/* Add the pixel to the contour list if desired. */
			if (nr_p >= 2) {
				contours.push_back(Point(j, i));
				istaken[i][j] = true;
			}

		}
	}
	/* Draw the contour pixels. */
	for (int i = 0; i < (int)contours.size(); i++) {
		image.at<Vec3b>(contours[i].y, contours[i].x) = Vec3b(colour[0], colour[1], colour[2]);
	}
}

Mat Slic::getLabels(){ return m_labels;}
int Slic::getNspx(){ return m_nSpx; }
int Slic::getSspx(){ return m_diamSpx; }