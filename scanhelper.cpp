#include<opencv2\core.hpp>
#include<opencv2\imgproc.hpp>
#include<opencv2\\highgui.hpp>
#include<iostream>

using namespace cv;
using namespace std;

#define T 1

void gammaTrans(double gamma, int coff[]);
void contrast_brighten(Mat &src);
void adjustHSV(Mat &src);

int main(int argc, char* argv[])
{
	if (argc < 2){
		cout << "too less parameters" << endl;
		return 0;
	}

	Mat src = imread(argv[1], 1);
	if (src.empty()){
		cout << "image openned failed" << endl;
		return 0;
	}

	contrast_brighten(src);
	adjustHSV(src);
	
	imwrite(argv[2], src);
	return 0;
}

//function used to calculate the coefficients of gamma transformation
void gammaTrans(double gamma, int coff[])
{
	for (int i = 0; i < 256; ++i){
		coff[i] = 256*pow(i / 256.0, gamma);
	}
}

//function used to enchance the contrast and brighten the image
void contrast_brighten(Mat &src)
{
	int win_size = 50;//
	int nw = src.cols / win_size;
	int nh = src.rows / win_size;
	for (int i = 0; i <= nh; ++i)
	{

		for (int j = 0; j <= nw; ++j)
		{
			int rstart = i*win_size;
			int cstart = j*win_size;
			int rend = rstart + win_size < src.rows ? rstart + win_size : src.rows;
			int cend = cstart + win_size < src.cols ? cstart + win_size : src.cols;

			double sum = win_size*win_size;
			double bgray = 0, ggray = 0, rgray = 0;

			for (int r = rstart; r < rend; ++r){
				for (int c = cstart; c < cend; ++c){
					bgray += src.at<uchar>(r, 3 * c) / sum;
					ggray += src.at<uchar>(r, 3 * c + 1) / sum;
					rgray += src.at<uchar>(r, 3 * c + 2) / sum;
				}
			}

			double contrast = 2.0;
			double avr_gray = 0.3*rgray + 0.59*ggray + 0.11*bgray;
			double bright;

			for (int r = rstart; r < rend; ++r){
				for (int c = cstart; c < cend; ++c){

					double max_b_g = src.at<uchar>(r, 3 * c) > src.at<uchar>(r, 3 * c + 1) ?
						src.at<uchar>(r, 3 * c) : src.at<uchar>(r, 3 * c + 1);
					double min_b_g = src.at<uchar>(r, 3 * c) < src.at<uchar>(r, 3 * c + 1) ?
						src.at<uchar>(r, 3 * c) : src.at<uchar>(r, 3 * c + 1);
					double max_b_g_r = max_b_g > src.at<uchar>(r, 3 * c + 2) ? max_b_g : src.at<uchar>(r, 3 * c + 2);
					double min_b_g_r = min_b_g < src.at<uchar>(r, 3 * c + 2) ? min_b_g : src.at<uchar>(r, 3 * c + 2);

					double s = max_b_g_r > 0 ? (1 - min_b_g_r / max_b_g_r) : 0;
					double v = max_b_g_r / 255;

					if (s > 0.45*v&&v > 0.5)
						bright = 1.0;
					else
						bright = 250.0 / avr_gray;

					double new_bgray = (bgray + (src.at<uchar>(r, 3 * c) - bgray)*contrast)*bright;
					double new_ggray = (ggray + (src.at<uchar>(r, 3 * c + 1) - ggray)*contrast)*bright;
					double new_rgray = (rgray + (src.at<uchar>(r, 3 * c + 2) - rgray)*contrast)*bright;

					if (new_bgray > 255)
						new_bgray = 255;
					else if (new_bgray < 0)
						new_bgray = 0;

					if (new_ggray > 255)
						new_ggray = 255;
					else if (new_ggray < 0)
						new_ggray = 0;

					if (new_rgray > 255)
						new_rgray = 255;
					else if (new_rgray < 0)
						new_rgray = 0;

					src.at<uchar>(r, 3 * c) = new_bgray;
					src.at<uchar>(r, 3 * c + 1) = new_ggray;
					src.at<uchar>(r, 3 * c + 2) = new_rgray;
				}
			}

		}
	}
}

//adjust image in HSV coclr model
void adjustHSV(Mat &src)
{
	cvtColor(src, src, CV_BGR2HSV);
	vector<Mat> hsv_planes;
	split(src, hsv_planes);
	int gammaTran1[256];
	int gammaTran2[256];
	gammaTrans(0.5, gammaTran1);
	gammaTrans(0.8, gammaTran2);
	for (int r = 0; r < src.rows; ++r){
		for (int c = 0; c < src.cols; ++c){
			if (hsv_planes[2].at<uchar>(r, c)*T < hsv_planes[1].at<uchar>(r, c))
				hsv_planes[1].at<uchar>(r, c) = T*hsv_planes[2].at<uchar>(r, c);
			double t = ((double)hsv_planes[1].at<uchar>(r, c)) / hsv_planes[2].at<uchar>(r, c);
			if (t < 0.35&&hsv_planes[2].at<uchar>(r, c)>220){
				hsv_planes[1].at<uchar>(r, c) = 0;
				hsv_planes[2].at<uchar>(r, c) = 255;
			}

			hsv_planes[1].at<uchar>(r, c) = gammaTran1[hsv_planes[1].at<uchar>(r, c)];
			hsv_planes[2].at<uchar>(r, c) = gammaTran2[hsv_planes[2].at<uchar>(r, c)];

		}
	}

	merge(hsv_planes, src);
	cvtColor(src, src, CV_HSV2BGR);
}

