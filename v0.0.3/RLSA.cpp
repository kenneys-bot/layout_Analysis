/*--------------------------------------------------------------------------
* Copyright (c) 2020, Nanchang Hangkong University (NCHU)
* All rights reserved.
*
* File name: RLAS.cpp
* Desciption : functions
*
* Version: 1.0
* Author: Zhicheng Liu
* Completion date:
--------------------------------------------------------------------------*/
#include <json/json.h>
#include "get_lines.h"
#include "RLSA.h"
#include <math.h> 
#include<opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>
#include "rotateTransformation.h"
#include "findTableByLSD.h"
#include <time.h>
#include <random>
#include <json/json.h>
/*#include <json.hpp>*/
#pragma comment(lib,"opencv_world455.lib")

using namespace std;
using namespace cv;


/*----------------------------------------------------------------------
* Description  : Output the value of each pixel
*
* Parameters   : Mat input: Picture in Mat format
* Output values: the value of each pixel
* Return values: No return values
----------------------------------------------------------------------*/
void printFunc(Mat input)
{
	for (int row = 0; row < input.rows; row++)
	{
		for (int col = 0; col < input.cols; col++)
		{
			int value = input.at<uchar>(row, col);
			cout << value << " ";
		}
		cout << endl;
	}
}

/*--------------------------------------------------------------
* Description: Open picture file
* Parameters: filepath: Source picture file with full path
* Output values: If the picture does not exist,cout"Open failure"
* Return values: Image of type mat, return empty mat when
				 file is invalid.
--------------------------------------------------------------*/
Mat loadImage(string filepath)
{
	Mat image = imread(filepath);

	if (image.empty())
	{
		cout << "��ʧ��" << endl;
		return Mat();
	}
	return image;
}

/*--------------------------------------------------------------
* Description: Image to grayscale
* Parameters: Mat input: Picture in Mat format
* Return values: Gray image of Mat type
--------------------------------------------------------------*/
Mat convert2gray(Mat input)
{
	Mat grayImage(input.size(), CV_8UC1);
	cvtColor(input, grayImage, cv::COLOR_BGR2GRAY);
	return grayImage;
}

/*--------------------------------------------------------------
* Description: ͼ���ֵ��
* Parameters: Mat���͵ĵ�ͨ��ͼ��
* Return: Mat���͵Ķ�ֵ��ͼ�񣬵�ͨ����Ϊһʱ���ؿ�Mat 
--------------------------------------------------------------*/
Mat binaryzation(Mat grayImage, int value)
{
	if (grayImage.channels() != 1)
	{
		cout << "����Ϊ��ͨ��ͼ��" << endl;
		return Mat();
	}
	Mat binaryImage;

	//binaryImage = Sauvola(grayImage, 9, 0.5);

	threshold(grayImage, binaryImage, value, 255, THRESH_BINARY); //BINARYģʽ����Ϊ��������ֵ�Ķ�ֵͼ��
	Scalar meanofImg,dev;
	
	
	meanStdDev(binaryImage, meanofImg, dev); //�����ֵ�ͱ�׼��
	if (meanofImg.val[0] <= 40 )
	{	
		//��ȡ������Ϣ
		const type_info& expInfo = typeid(meanofImg.val[0]);
		cout << "hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh\n" << endl;   // �����
		cout << expInfo.name() << " | " << expInfo.raw_name() << " | " << expInfo.hash_code() << endl;
		//��һ����ֵ������
		threshold(grayImage, binaryImage, 0, 255, THRESH_BINARY | THRESH_OTSU);
	}
	//showPic(binaryImage,"mean");  

	return binaryImage;
}

Mat thresholdAdapt(Mat& img, float rate)
{
	int hh = img.rows;
	int ww = img.cols;
	Mat ret = img.clone();
	int height = sqrt(float(hh) / ww*(hh + ww));
	int width = float(ww) / hh*height;
	for (int h = 0; h<hh; h++) {
		for (int w = 0; w<ww; w++) {
			int h1 = max(1, h - height / 2);
			int h2 = max(1, h + height / 2);
			int w1 = min(w - width / 2, hh);
			int w2 = min(w + width / 2, ww);
			float avg = 0;
			for (int i = h1; i<h2; i++)
				for (int j = w1; j<w2; j++)
					avg += *(img.data + i*ww + j);
			*(ret.data + h*ww + w) = int(avg / ((h2 - h1)*(w2 - w1)));
		}
	}
	for (int i = 0; i<hh; i++) {
		uchar* pDst = ret.data + i*ww;
		for (int j = 0; j<ww; j++)
			*(pDst + j) = *(img.data + i*ww + j)<(*(pDst + j)*rate) ? 0 : 255;
	}
	return ret;
}

/*--------------------------------------------------------------
* Description: ����ֵ���й�һ��
* Parameters: Mat���͵ĻҶ�ͼ��
* Return: Mat���͵Ĺ�һ��ͼ��
--------------------------------------------------------------*/
Mat normalization(Mat binaryImage)
{
	Mat temp = ~binaryImage;
	//showPic(temp,"temp");    //ȡ��
	Mat digitImage = (1 / 255.0) * temp;  //��0-1
	//showPic(digitImage, "normal");
	return digitImage;
}

/*--------------------------------------------------------------
* Description: ˮƽ������γ�ƽ��
* Parameters: Mat���͵�ͼ���γ���ֵ
* Return: ˮƽ�γ�ƽ�����ͼ��
--------------------------------------------------------------*/
Mat lengthSmoothHor(Mat digitImage, int threshold)
{
	Mat digitImageHor = digitImage.clone();
	int cols = digitImage.cols;
	int rows = digitImage.rows;
	for (int row = 0; row < rows; row++)
	{
		int cursor1 = 0;
		int cursor2 = 0;
		for (int col = 0; col < digitImageHor.cols;)
		{
			if (digitImageHor.at<uchar>(row, cursor1) == 0)
			{
				cursor2 = cursor1 + 1;
				while ((cursor2 < cols) && (digitImageHor.at<uchar>(row, cursor2) != 1))
				{
					cursor2++;
				}
				if ((cursor2 - cursor1) <= threshold)
				{
					while (cursor1 < cursor2)
					{
						digitImageHor.at<uchar>(row, cursor1) = 1;
						cursor1++;
					}
				}
				cursor1 = cursor2;
			}
			else
			{
				cursor1++;
				cursor2++;
			}
			col = cursor2;
		}
	}
	return digitImageHor;
}

/*--------------------------------------------------------------
* Description: ��ֱ������γ�ƽ��
* Parameters: Mat���͵�ͼ���γ���ֵ
* Return: ��ֱ�γ�ƽ�����ͼ��
--------------------------------------------------------------*/
Mat lengthSmoothVer(Mat digitImage, int threshold)
{
	Mat digitImageVer = digitImage.clone();
	for (int col = 0; col < digitImageVer.cols; col++)
	{
		int cursor1 = 0;
		int cursor2 = 0;
		for (int row = 0; row < digitImageVer.rows;)
		{
			if ((int)digitImageVer.at<uchar>(cursor1, col) == 0)
			{
				cursor2 = cursor1 + 1;
				while ((cursor2 < digitImageVer.rows) && ((int)digitImageVer.at<uchar>(cursor2, col) != 1))
				{
					cursor2++;
				}
				if ((cursor2 - cursor1) <= threshold)
				{
					while (cursor1 < cursor2)
					{
						digitImageVer.at<uchar>(cursor1, col) = 1;
						cursor1++;
					}
				}
				cursor1 = cursor2;
			}
			else
			{
				cursor2++;
				cursor1++;
			}
			row = cursor2;
		}
	}
	return digitImageVer;
}

/*--------------------------------------------------------------
* Description: ���Ͳ���
* Parameters: Mat���͵�ͼ�����ʹ���
* Return: ���ͺ��ͼ��
--------------------------------------------------------------*/
Mat doDilation(Mat smoothImage, int times)
{
	Mat dilateImage;
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	int index = 0;
	Mat temp = smoothImage.clone();
	while (index < times)
	{
		dilate(temp, temp, element);
		index++;
	}
	dilateImage = temp.clone();
	return dilateImage;
}

/*--------------------------------------------------------------
* Description: ����ɾ��
* Parameters: Mat���͵�ͼ�񡢾���������Ϣ
* Return: ����ȥ�����ͼ��
--------------------------------------------------------------*/
Mat plotRect(Mat& input, std::vector<Rect> inputRects)
{
	Mat rectImage = input.clone();
	for (std::vector<Rect>::const_iterator it1 = inputRects.begin(); it1 != inputRects.end(); it1++)
	{
		int limitY = (it1->br().y < input.rows) ? it1->br().y : input.rows - it1->tl().y;
		int limitX = (it1->br().x < input.cols) ? it1->br().x : input.cols - it1->tl().x;
		for (int row = it1->tl().y; row < limitY; row++)
		{
			if (row < 0) {
				continue;
			}
			for (int col = it1->tl().x; col < limitX; col++)
			{
				if (col < 0) {
					continue;
				}
				rectImage.at<uchar>(row, col) = 0;
			}
		}
	}
	return rectImage;
}
Mat plotRect(Mat& input, std::vector<Rect> inputRects,double width)
{
	Mat rectImage = input.clone();
	for (std::vector<Rect>::const_iterator it1 = inputRects.begin(); it1 != inputRects.end(); it1++)
	{
		if (it1->width < width)
		{
			continue;
		}
		int limitY = (it1->br().y < input.rows) ? it1->br().y : input.rows - it1->tl().y;
		int limitX = (it1->br().x < input.cols) ? it1->br().x : input.cols - it1->tl().x;
		for (int row = it1->tl().y; row < limitY; row++)
		{
			if (row < 0) {
				continue;
			}
			for (int col = it1->tl().x; col < limitX; col++)
			{
				if (col < 0) {
					continue;
				}
				rectImage.at<uchar>(row, col) = 0;
			}
		}
	}
	return rectImage;
}


/*--------------------------------------------------------------
* Description: �����������
* Parameters: Mat���͵�ͼ��
* Return: ���ƺ��ͼ��
--------------------------------------------------------------*/
Mat getBlock(Mat& input, std::vector<Rect> textRects, std::vector<Rect> imageRects, std::vector<Rect> tableRects, int rate)
{
	Mat blockImage = input.clone();
	for (std::vector<Rect>::const_iterator it = textRects.begin(); it != textRects.end(); it++)
	{
		rectangle(blockImage, Point(it->tl().x * rate, it->tl().y * rate), Point(it->br().x * rate, it->br().y * rate), cv::Scalar(0, 0, 255), 3, 1, 0);
	}
	for (std::vector<Rect>::const_iterator it = imageRects.begin(); it != imageRects.end(); it++)
	{
		rectangle(blockImage, Point(it->tl().x * rate, it->tl().y * rate), Point(it->br().x * rate, it->br().y * rate), cv::Scalar(0, 255, 0), 3, 1, 0);
	}
	for (std::vector<Rect>::const_iterator it = tableRects.begin(); it != tableRects.end(); it++)
	{
		rectangle(blockImage, Point(it->tl().x * rate, it->tl().y * rate), Point(it->br().x * rate, it->br().y * rate), cv::Scalar(255, 0, 0), 3, 1, 0);
	}
	return blockImage;
}
// ����
Mat getBlock(Mat& input, std::vector<Rect> rects)
{
	Mat blockImage = input.clone();
	for (std::vector<Rect>::const_iterator it = rects.begin(); it != rects.end(); it++)
	{
		rectangle(blockImage, it->tl(), it->br(), cv::Scalar(0, 0, 255), 3, 1, 0);
	}
	return blockImage;
}
// ����2022_12_01	--by lww
Mat getBlock(Mat input, std::vector<Rect> textRects, int rate)
{
	Mat blockImage = input.clone();
	for (std::vector<Rect>::const_iterator it = textRects.begin(); it != textRects.end(); it++)
	{
		rectangle(blockImage, Point(it->tl().x * rate , it->tl().y * rate), Point(it->br().x * rate, it->br().y * rate), cv::Scalar(0, 0, 255), 3, 1, 0);
	}
	return blockImage;
}




/*--------------------------------------------------------------
* Description: ���ּ��
* Parameters: �γ�ƽ�����ͼ���ԭʼͼ��
* Return: ��������λ��������Ϣ����
--------------------------------------------------------------*/
std::vector<Rect> textDetect(Mat dilateImage, Mat input)
{
	Mat ori_image = input.clone();
	std::vector<std::vector<Point>> contours;

	findContours(dilateImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	drawContours(ori_image, contours, -1, CV_RGB(255, 0, 0), 2);
	//showPic(ori_image,"textDetectImage");
	//imwrite("D:\\VS_Projects\\Layout Analysis\\datatextDetectImage.jpg", ori_image);

	std::vector<std::vector<Point>> mergeContours;

	//Ԥ�����þ��ο�����
	std::vector<Rect> boundingRects;
	boundingRects.reserve(contours.size());

	for (std::vector<std::vector<Point>>::const_iterator it = contours.begin(); it != contours.end(); it++)
	{
		Rect bRect = boundingRect(*it);
		//���ο�  12<width<2000, height>8, ���<ͼ���80%  �����
		if ((bRect.width < 2000 && bRect.width > 12) && (bRect.height > 8) && (bRect.area() < (dilateImage.rows * dilateImage.cols * 0.8)))
		{
			boundingRects.push_back(bRect);
		}
	}

	return boundingRects;
}

std::vector<Rect> textDetect(Mat dilateImage)
{
	std::vector<std::vector<Point>> contours;
	findContours(dilateImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

	std::vector<std::vector<Point>> mergeContours;

	std::vector<Rect> boundingRects;
	boundingRects.reserve(contours.size());

	for (std::vector<std::vector<Point>>::const_iterator it = contours.begin(); it != contours.end(); it++)
	{
		Rect bRect = boundingRect(*it);
		if (bRect.width < 2000 && bRect.width > 12 && bRect.height > 8 && bRect.area() < (dilateImage.rows * dilateImage.cols * 0.8))
		{
			boundingRects.push_back(bRect);
		}
	}
	return boundingRects;
}

std::vector<Rect> textDetect(Mat dilateImage, Mat input, vector<vector<Point> > &bigAreaContours)
{
	//Ԥ�����þ��ο�����
	std::vector<Rect> boundingRects;


	std::vector<std::vector<Point>> contours;

	findContours(dilateImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	/*drawContours(input, contours, -1, CV_RGB(0, 0, 255), 8);
	showPic(input, "re_word");*/
	std::vector<std::vector<Point>> mergeContours;

		
	//boundingRects.reserve(contours.size());

	std::vector<std::vector<Point>>::const_iterator it = contours.begin();
	while (it != contours.end())
	{
		Rect bRect = boundingRect(*it);
		//���ο�  12<width<2000, height>8, ���<ͼ���80%  �����
		if ((bRect.width < 2000 && bRect.width > 12) && (bRect.height > 8) && (bRect.area() < (dilateImage.rows * dilateImage.cols * 0.8)))
		{
			//boundingRects.push_back(bRect);
			it++;
		}
		else
		{
			it = contours.erase(it);
		}
	}
	/*clock_t start, end;
	start = clock();
	image_word_MinDistance(bigAreaContours, contours);
	end = clock();
	cout << "ͼ�����ֺϲ���������ʱ��=" << double(end - start) / CLOCKS_PER_SEC << "s" << endl;*/

	for (std::vector<std::vector<Point>>::const_iterator it_fix = contours.begin(); it_fix != contours.end(); it_fix++)
	{
		Rect bRect = boundingRect(*it_fix);
		boundingRects.push_back(bRect);
	}
	return boundingRects;
}

std::vector<Rect> textDetect(Mat dilateImage, Mat input, vector<vector<Point> >& bigAreaContours, int flag)
{
	//Ԥ�����þ��ο�����
	std::vector<Rect> boundingRects;

	if (flag == 1)
	{
		std::vector<std::vector<Point>> contours;

		findContours(dilateImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
		/*drawContours(input, contours, -1, CV_RGB(0, 0, 255), 8);
		showPic(input, "re_word");*/
		std::vector<std::vector<Point>> mergeContours;


		//boundingRects.reserve(contours.size());

		std::vector<std::vector<Point>>::const_iterator it = contours.begin();
		while (it != contours.end())
		{
			Rect bRect = boundingRect(*it);
			//���ο�  12<width<2000, height>8, ���<ͼ���80%  �����
			if ((bRect.width < 2000 && bRect.width > 12) && (bRect.height > 8) && (bRect.area() < (dilateImage.rows * dilateImage.cols * 0.8)))
			{
				//boundingRects.push_back(bRect);
				it++;
			}
			else
			{
				it = contours.erase(it);
			}
		}
		/*clock_t start, end;
		start = clock();
		image_word_MinDistance(bigAreaContours, contours);
		end = clock();
		cout << "ͼ�����ֺϲ���������ʱ��=" << double(end - start) / CLOCKS_PER_SEC << "s" << endl;*/

		for (std::vector<std::vector<Point>>::const_iterator it_fix = contours.begin(); it_fix != contours.end(); it_fix++)
		{
			Rect bRect = boundingRect(*it_fix);
			boundingRects.push_back(bRect);
		}
	}
	else if (flag == 2)
	{
		std::vector<std::vector<Point>> contours;

		findContours(dilateImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
		/*drawContours(input, contours, -1, CV_RGB(0, 0, 255), 8);
		showPic(input, "re_word");*/
		std::vector<std::vector<Point>> mergeContours;


		//boundingRects.reserve(contours.size());

		std::vector<std::vector<Point>>::const_iterator it = contours.begin();
		while (it != contours.end())
		{
			Rect bRect = boundingRect(*it);
			//���ο�  12<width<2300, height>8, ���<ͼ���80%  �����
			if ((bRect.width < 2300 && bRect.width > 12) && (bRect.height > 8) && (bRect.area() < (dilateImage.rows * dilateImage.cols * 0.8)))
			{
				//boundingRects.push_back(bRect);
				it++;
			}
			else
			{
				it = contours.erase(it);
			}
		}
		/*clock_t start, end;
		start = clock();
		image_word_MinDistance(bigAreaContours, contours);
		end = clock();
		cout << "ͼ�����ֺϲ���������ʱ��=" << double(end - start) / CLOCKS_PER_SEC << "s" << endl;*/

		for (int i = 0; i < contours.size(); i++)
		{
			Rect bRect = boundingRect(contours[i]);
			if ((bRect.area() > (dilateImage.rows * dilateImage.cols * 0.01)))
			{
				bigAreaContours.push_back(contours[i]);
				continue;
			}
			boundingRects.push_back(bRect);
		}
	}
	return boundingRects;
}

/*--------------------------------------------------------------
* Description: ��ʾͼ��
* Parameters: Mat showpic, String namepi
* Return:  null
--------------------------------------------------------------*/
void showPic(Mat showpic, String namepic) {
	namedWindow(namepic, WINDOW_NORMAL);
	imshow(namepic, showpic);
}


/*--------------------------------------------------------------
* Description:ֱ�߼��
* Parameters: Mat image, Mat grayImage
* Return:vector<Vec4f>
* Writter: ��ǧ��
---------------------------------------------------------------*/
vector<Vec4f> ls_detect(Mat inputCopy, Mat grayImage) {
	Mat image = inputCopy.clone();
	//��������
	Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_ADV);

	vector<Vec4f> lines_std, draw_lines;
	Mat drawnLines(image);

	//���ֱ��
	ls->detect(grayImage, lines_std);
	//ɸѡֱ��
	for (int i = 0; i < lines_std.size(); i++) {
		if (lines_std[i][0] > lines_std[i][2])
			swap(lines_std[i][0], lines_std[i][2]);
		if (lines_std[i][1] > lines_std[i][3])
			swap(lines_std[i][1], lines_std[i][3]);
		double x1 = lines_std[i][0];
		double y1 = lines_std[i][1];
		double x2 = lines_std[i][2];
		double y2 = lines_std[i][3];
		double dx = abs(x2 - x1);
		double dy = abs(y2 - y1);
		double len = sqrt(dy * dy + dx * dx);
		if (len > 380) {
			draw_lines.push_back(lines_std[i]);
			ls->drawSegments(drawnLines, draw_lines);
			cout << lines_std[i][0] << " " << lines_std[i][1] << " "
				<< lines_std[i][2] << " " << lines_std[i][3] << endl;
		}
	}

	cout << "ͼ�Ϲ���" << draw_lines.size() << "��ֱ�ߡ�" << endl;
	//imwrite("D:\\VS_Projects\\LearnforCpp\\data\\draw_lines.jpg", drawnLines);
	//showPic(drawnLines, "lsd");

	return draw_lines;
}


/*--------------------------------------------------------------
* Description: ���ڴ�ֱͶӰ�����������жϡ������帨���ж�
* Parameters: ValArry[]����
* Return: ���������ж�
* writer:��ǧ��
--------------------------------------------------------------*/

bool wave_judge(int* ValArry, int width, int height) {
	int up_sum = 0;//һ�����¼�¼
	int down_sum = 0; //һ�����¼�¼
	int acc_up_sum = 0; //��ȷ���¼�¼
	int acc_down_sum = 0; //һ�����¼�¼
	int* min_p = new int[width]; //�µ���͵㣨����ֵ������
	int* max_p = new int[width]; //�µ���ߵ㼯��
	memset(min_p, 0, width * 4);
	memset(max_p, 0, width * 4);
	int pmax = 0; //�±ꡣ
	int pmin = 0;
	int up_flag = 0;//��ǰ����״̬��־
	int down_flag = 0;
	int start_flag = 0;// 1Ϊ�����£�2Ϊ������
	for (int i = 0; i < width; i++) {
		////Խ���ж�
		//if (ValArry[i + 1]) {
		//	continue;
		//}
		//�տ�ʼ��ƽ���׶�
		if ((up_flag == 0) && (down_flag == 0) && (ValArry[i] == ValArry[i + 1])) {
			max_p[pmax] = ValArry[i];
			min_p[pmin] = ValArry[i];
			continue;
		}
		//��ʼ���£���δ�¹��£���Ծʱ��
		if ((up_flag == 0) && (down_flag == 0) && (ValArry[i] < ValArry[i + 1])) {
			up_flag = 1;
			down_flag = 0;
			start_flag = 1; //�����µĿ�ʼ��־
			min_p[pmin] = ValArry[i];//�ʼ�����£��µ���͵���+1
			pmin = pmin + 1;
			continue;
		}
		//��ʼ���£���δ�Ϲ��£���Ծʱ��
		if ((up_flag == 0) && (down_flag == 0) && (ValArry[i] > ValArry[i + 1])) {
			down_flag = 1;
			up_flag = 0;
			start_flag = 2; //�����µĿ�ʼ��־
			max_p[pmax] = ValArry[i];//�ʼ�����£��µ���͵���+1
			pmax = pmax + 1;
			continue;
		}
		//�����У���Ծʱ��
		if ((up_flag == 1) && (down_flag == 0) && (ValArry[i] < ValArry[i + 1])) {
			max_p[pmax] = ValArry[i];
			continue;
		}
		//�����У�ƽ�ؽ׶�
		if ((up_flag == 1) && (down_flag == 0) && (ValArry[i] == ValArry[i + 1])) {
			max_p[pmax] = ValArry[i];
			continue;
		}
		//��ʼ����,����������,��Ծʱ��
		if ((down_flag == 0) && (up_flag == 1) && (ValArry[i] > ValArry[i + 1])) {
			down_flag = 1;
			up_flag = 0;
			max_p[pmax] = ValArry[i];//��¼����ǰһ�̵���ߵ㣬�µ���ߵ�+1
			////ȥ������Ҫ����
			//if (start_flag == 1) {
			//	if (abs(max_p[pmax] - min_p[pmin]) < 10) {
			//		//�������+1
			//		useless_up = useless_up + 1;
			//	}
			//}
			//else if (start_flag == 2) {
			//	if (abs(max_p[pmax - 1] - min_p[pmin]) < 10) {
			//		//�ұ�����+1
			//		useless_down = useless_down + 1;
			//	}
			//}
			min_p[pmin] = ValArry[i];
			pmax = pmax + 1;
			continue;
		}
		//������,��Ծʱ��
		if ((down_flag == 1) && (up_flag == 0) && (ValArry[i] > ValArry[i + 1])) {
			min_p[pmin] = ValArry[i];
			continue;
		}
		//�����У�ƽ�ؽ׶�
		if ((down_flag == 1) && (up_flag == 0) && (ValArry[i] == ValArry[i + 1])) {
			min_p[pmin] = ValArry[i];
			continue;
		}
		//��ʼ���£������¹��£���Ծʱ��
		if ((up_flag == 0) && (down_flag == 1) && (ValArry[i] < ValArry[i + 1])) {
			up_flag = 1;
			down_flag = 0;
			min_p[pmin] = ValArry[i];//��¼����ǰһ�̵���͵㣬�µ���͵�+1

			pmin = pmin + 1;
			max_p[pmax] = ValArry[i];
			continue;
		}
	}
	//��������ж�
	if (start_flag == 1) {
		//һ��ʼ�����µ����
		for (int i = 0; i < pmin; i++) {
			//���
			if (abs(min_p[i] - max_p[i]) > height * 0.65) {
				acc_up_sum = acc_up_sum + 1;
			}
			else if (abs(min_p[i] - max_p[i]) > height * 0.25) {
				up_sum = up_sum + 1;
			}
			//�ұ�
			if (abs(min_p[i + 1] - max_p[i]) > height * 0.65) {
				acc_down_sum = acc_down_sum + 1;
			}
			else if (abs(min_p[i + 1] - max_p[i]) > height * 0.25) {
				down_sum = down_sum + 1;
			}
		}
	}
	else if (start_flag == 2) {
		//һ��ʼ�����µ����
		for (int i = 0; i < pmax; i++) {
			//�ұ�
			if (abs(max_p[i] - min_p[i]) > height * 0.65) {
				acc_down_sum = acc_down_sum + 1;
			}
			else if (abs(max_p[i] - min_p[i]) > height * 0.25) {
				down_sum = down_sum + 1;
			}
			//���
			if (abs(max_p[i + 1] - min_p[i]) > height * 0.65) {
				acc_up_sum = acc_up_sum + 1;
			}
			else if (abs(max_p[i + 1] - min_p[i]) > height * 0.25) {
				up_sum = up_sum + 1;
			}
		}
	}
	//��Խ϶̵���at\to�ȣ�����ȷ�ж�
	if ((acc_up_sum >= 1) && (acc_down_sum >= 2)) {
		return true;
	}
	//���354��A
	else if ((acc_up_sum >= 1) && (acc_down_sum >= 1) && (up_sum >= 1) && (down_sum >= 1)) {
		return true;
	}
	//�����һ�����������»����¼�¼
	else if ((up_sum >= 4) || (down_sum >= 4)) {
		return true;//��Ϊ��������
	}
	else {
		return false;
	}
}

/*--------------------------------------------------------------
* Description: ���ڴ�ֱͶӰ�����������жϡ������帨���ж�
* Parameters: ValArry[]����
* Return: ���������ж�
* writer:��ǧ��
--------------------------------------------------------------*/
bool wordJudge(int* ValArry, int width, int height) {
	int up_sum = 0;//һ�����¼�¼
	int down_sum = 0; //һ�����¼�¼
	int acc_up_sum = 0; //��ȷ���¼�¼
	int acc_down_sum = 0; //һ�����¼�¼
	int* min_p = new int[width]; //�µ���͵㣨����ֵ������
	int* max_p = new int[width]; //�µ���ߵ㼯��
	memset(min_p, 0, width * 4);
	memset(max_p, 0, width * 4);
	int pmax = 0; //�±ꡣ
	int pmin = 0;
	int up_flag = 0;//��ǰ����״̬��־
	int down_flag = 0;
	int start_flag = 0;// 1Ϊ�����£�2Ϊ������
	for (int i = 0; i < width; i++) {
		////Խ���ж�
		//if (ValArry[i + 1]) {
		//	continue;
		//}
		//�տ�ʼ��ƽ���׶�
		if ((up_flag == 0) && (down_flag == 0) && (ValArry[i] == ValArry[i + 1])) {
			max_p[pmax] = ValArry[i];
			min_p[pmin] = ValArry[i];
			continue;
		}
		//��ʼ���£���δ�¹��£���Ծʱ��
		if ((up_flag == 0) && (down_flag == 0) && (ValArry[i] < ValArry[i + 1])) {
			up_flag = 1;
			down_flag = 0;
			start_flag = 1; //�����µĿ�ʼ��־
			min_p[pmin] = ValArry[i];//�ʼ�����£��µ���͵���+1
			pmin = pmin + 1;
			continue;
		}
		//��ʼ���£���δ�Ϲ��£���Ծʱ��
		if ((up_flag == 0) && (down_flag == 0) && (ValArry[i] > ValArry[i + 1])) {
			down_flag = 1;
			up_flag = 0;
			start_flag = 2; //�����µĿ�ʼ��־
			max_p[pmax] = ValArry[i];//�ʼ�����£��µ���͵���+1
			pmax = pmax + 1;
			continue;
		}
		//�����У���Ծʱ��
		if ((up_flag == 1) && (down_flag == 0) && (ValArry[i] < ValArry[i + 1])) {
			max_p[pmax] = ValArry[i];
			continue;
		}
		//�����У�ƽ�ؽ׶�
		if ((up_flag == 1) && (down_flag == 0) && (ValArry[i] == ValArry[i + 1])) {
			max_p[pmax] = ValArry[i];
			continue;
		}
		//��ʼ����,����������,��Ծʱ��
		if ((down_flag == 0) && (up_flag == 1) && (ValArry[i] > ValArry[i + 1])) {
			down_flag = 1;
			up_flag = 0;
			max_p[pmax] = ValArry[i];//��¼����ǰһ�̵���ߵ㣬�µ���ߵ�+1
			min_p[pmin] = ValArry[i];
			pmax = pmax + 1;
			continue;
		}
		//������,��Ծʱ��
		if ((down_flag == 1) && (up_flag == 0) && (ValArry[i] > ValArry[i + 1])) {
			min_p[pmin] = ValArry[i];
			continue;
		}
		//�����У�ƽ�ؽ׶�
		if ((down_flag == 1) && (up_flag == 0) && (ValArry[i] == ValArry[i + 1])) {
			min_p[pmin] = ValArry[i];
			continue;
		}
		//��ʼ���£������¹��£���Ծʱ��
		if ((up_flag == 0) && (down_flag == 1) && (ValArry[i] < ValArry[i + 1])) {
			up_flag = 1;
			down_flag = 0;
			min_p[pmin] = ValArry[i];//��¼����ǰһ�̵���͵㣬�µ���͵�+1

			pmin = pmin + 1;
			max_p[pmax] = ValArry[i];
			continue;
		}
	}
	//��������ж�
	if (start_flag == 1) {
		//һ��ʼ�����µ����
		for (int i = 0; i < pmin; i++) {
			//���
			if (abs(min_p[i] - max_p[i]) > height * 0.65) {
				acc_up_sum = acc_up_sum + 1;
			}
			else if (abs(min_p[i] - max_p[i]) > height * 0.25) {
				up_sum = up_sum + 1;
			}
			//�ұ�
			if (abs(min_p[i + 1] - max_p[i]) > height * 0.65) {
				acc_down_sum = acc_down_sum + 1;
			}
			else if (abs(min_p[i + 1] - max_p[i]) > height * 0.25) {
				down_sum = down_sum + 1;
			}
		}
	}
	else if (start_flag == 2) {
		//һ��ʼ�����µ����
		for (int i = 0; i < pmax; i++) {
			//�ұ�
			if (abs(max_p[i] - min_p[i]) > height * 0.65) {
				acc_down_sum = acc_down_sum + 1;
			}
			else if (abs(max_p[i] - min_p[i]) > height * 0.25) {
				down_sum = down_sum + 1;
			}
			//���
			if (abs(max_p[i + 1] - min_p[i]) > height * 0.65) {
				acc_up_sum = acc_up_sum + 1;
			}
			else if (abs(max_p[i + 1] - min_p[i]) > height * 0.25) {
				up_sum = up_sum + 1;
			}
		}
	}
	//��Խ϶̵���at\to�ȣ�����ȷ�ж�
	if ((acc_up_sum >= 1) && (acc_down_sum >= 2)) {
		return true;
	}
	//���354��A
	else if ((acc_up_sum >= 1) && (acc_down_sum >= 1) && (up_sum >= 1) && (down_sum >= 1)) {
		return true;
	}
	//�����һ�����������»����¼�¼
	else if ((up_sum >= 4) || (down_sum >= 4)) {
		return true;//��Ϊ��������
	}
	else {
		return false;
	}
}

/*--------------------------------------------------------------
* Description: ���ڴ�ֱͶӰ�����������ж�
* Parameters: Mat���͵�ͼ�񣬾��ο��������
* Return: ���������ж�
* writer:��ǧ��
--------------------------------------------------------------*/
bool textlinesJudge(Mat normalImage, int pic_height, int pic_width) {
	int width = normalImage.cols;
	int height = normalImage.rows;
	int perPixelValue = 0;

	//����ͶӰ������
	Mat verpaint(normalImage.size(), CV_8UC1, Scalar(255));
	//showPic(verpaint, "verpaint_unpaint");
	//�������ڴ洢ÿ�а�ɫ���ظ���������
	int* ValArry = new int[width];
	//�����ʼ��
	memset(ValArry, 0, width * 4);

	int flag = 0;
	//�ҹ۲�ͶӰ
	if ((width == 0) && (height == 0)) {
		flag = 1;
	}
	//��¼ÿһ�а�ɫ���ص�����
	for (int col = 0; col < width; col++) {
		for (int row = 0; row < height; row++) {
			perPixelValue = normalImage.at<uchar>(row, col);
			if (perPixelValue == 1) {
				ValArry[col] = ValArry[col] + 1;
			}
			else {
				continue;
			}
		}
		////����ͶӰ
		//for (int i = 0; i < ValArry[col]; i++) {
		//	verpaint.at<uchar>(height - i - 1, col) = 0;
		//}
	}
	int w_sum = 0;//��ɫ���ص��
	int val_col = 0;//��¼Valarry��0��
	//����ͶӰ
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < ValArry[i]; j++) {
			verpaint.at<uchar>(height - j - 1, i) = 0;
		}
		if (ValArry[i] > 0) {
			val_col = val_col + 1;
		}
		w_sum = w_sum + ValArry[i];//��¼��ɫ���ص��
	}
	if (flag == 1) {
		cout << "val_col:" << val_col << endl;
		//showPic(verpaint, "verpaint");
		//imwrite("D:\\VS_Projects\\Layout Analysis\\data\\pic276\\verpaint_45.jpg", verpaint);
		//exit(0);
	}

	//showPic(verpaint, "verpaint");
	//imwrite("D:\\VS_Projects\\Layout Analysis\\data\\zaodian_verpro.jpg", verpaint);

	//������СͶӰ
	if (w_sum < (width * height) * 0.11) {
		cout << "��СͶӰ" << endl;
		return false;
	}
	/*if (w_sum > (width * height) * 0.8) {
		cout << "����ͶӰ" << endl;
		return false;
	}*/
	//����С�����ı�����
	if ((width < 20) && (height < 20) || ((height < 15) && (width < 25))) {
		cout << "С��������" << endl;
		return false;
	}
	//��������Ĵ��ı�����
	if (height > pic_height * 0.6) {
		cout << "������ı�����" << endl;
		return false;
	}
	//��������ͶӰ
	if (val_col < width * 0.5) {
		cout << "����ͶӰ" << endl;
		return false;
	}
	bool result = wordJudge(ValArry, width, height);
	return result;
}


/*--------------------------------------------------------------
* Description: ����ˮƽͶӰ�����������ж�
* Parameters: Mat���͵�ͼ�񣬾��ο��������
* Return: ���������ж�
* writer:��ǧ��
--------------------------------------------------------------*/
bool pic_textlineJudge(Mat normalImage, int pic_height, int pic_width) {
	int width = normalImage.cols;
	int height = normalImage.rows;
	int perPixelValue = 0;

	//����ͶӰ������
	Mat verpaint(normalImage.size(), CV_8UC1, Scalar(255));
	//showPic(verpaint, "verpaint_unpaint");
	//�������ڴ洢ÿ�а�ɫ���ظ���������
	int* ValArry = new int[height];
	//�����ʼ��
	memset(ValArry, 0, height * 4);

	int flag = 0;
	////�ҹ۲�ͶӰ
	//if ((width == 0) && (height == 0)) {
	//	flag = 1;
	//}
	//��¼ÿһ�а�ɫ���ص�����
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			perPixelValue = normalImage.at<uchar>(row, col);
			if (perPixelValue == 1) {
				ValArry[row] = ValArry[row] + 1;
			}
			else {
				continue;
			}
		}

	}
	int w_sum = 0;//��ɫ���ص��
	int val_row = 0;//��¼Valarry��0��
	//����ͶӰ
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < ValArry[i]; j++) {
			verpaint.at<uchar>(i, j) = 0;
		}
		if (ValArry[i] > 0) {
			val_row = val_row + 1;
		}
		w_sum = w_sum + ValArry[i];//��¼��ɫ���ص��
	}
	//if (flag == 1) {
	//	cout << "val_row:" << val_row << endl;
	//	showPic(verpaint, "verpaint");
	//	//imwrite("D:\\VS_Projects\\Layout Analysis\\data\\pic276\\verpaint_45.jpg", verpaint);
	//	//exit(0);
	//}

	//showPic(verpaint, "verpaint");
	//imwrite("D:\\VS_Projects\\Layout Analysis\\data\\zaodian_verpro.jpg", verpaint);

	//������СͶӰ
	if (w_sum < (width * height) * 0.11) {
		cout << "��СͶӰ" << endl;
		return false;
	}
	if (w_sum > (width * height) * 0.5) {
		cout << "����ͶӰ" << endl;
		return false;
	}
	//������С�����ı�����
	if ((width > pic_width * 0.5) && (height > pic_height * 0.4) || ((height < 15) && (width < 25))) {
		cout << "��С��������" << endl;
		return false;
	}
	////��������Ĵ��ı�����
	//if (height > pic_height * 0.6) {
	//	cout << "������ı�����" << endl;
	//	return false;
	//}
	//��������ͶӰ
	if (val_row < height * 0.5) {
		cout << "����ͶӰ" << endl;
		return false;
	}

	//////�洢ÿ����ɫ�������ֵ
	int* max_b_index = new int[width];
	int max = 0;
	memset(max_b_index, 0, width * 4);

	//�ж��Ƿ���ڿհ�����
	int* b_index = new int[width];//��ɫ����ʼ�ͽ�������
	int b_end = 0; //��ɫ�������
	int* w_index = new int[width];//��ɫ����ʼ�ͽ�������
	int w_end = 0;  //��ɫ�������
	bool white = false; //�����ɫ�����־
	bool black = false;  //�����ɫ�����־
	memset(b_index, 0, width * 4);//��ʼ��
	memset(w_index, 0, width * 4);

	int K_num = 5;

	int w = 0; //�����±꣬���Ҽ�¼��ɫ�������������ʽΪ[start,end]
	int b = 0;
	for (int i = 0; i < width; i++) {
		//��������ֵ��С�������ֵ�������հ�����
		//��ʼ�����ɫ������δ������ɫ����
		if ((ValArry[i] <= K_num) && (white == false) && (black == false)) {
			w_index[w] = i;
			w = w + 1;
			white = true;

		}
		//��ʼ�����ɫ�����Ҿ�������ɫ����
		else if ((ValArry[i] <= K_num) && (white == false) && (black == true)) {
			if (b_end > i) {
				b_index[b] = b_end;
				b = b + 1;
				w_index[w] = i;
				w = w + 1;
				white = true;
				black = false;
			}
			else if (b_end < i) {
				b_index[b] = i - 1;
				b = b + 1;
				w_index[w] = i;
				w = w + 1;
				white = true;
				black = false;
			}
			max = max + 1;//��ȡ���ɫ��������ֵ��׼��������һ��

		}
		//��ɫ����ѭ��
		else if ((ValArry[i] <= 10) && (white == true)) {
			w_end = i;
			continue;
		}
		//��ʼ�����ɫ������δ������ɫ����
		else if ((ValArry[i] > K_num) && (black == false) && (white == false)) {
			b_index[b++] = i;
			black = true;
			if (ValArry[i] > max_b_index[max]) {
				max_b_index[max] = ValArry[i];
			}
		}
		//��ʼ�����ɫ�����Ҿ�������ɫ����
		else if ((ValArry[i] > K_num) && (black == false) && (white == true)) {
			if (w_end > i) {
				w_index[w] = w_end;
				w = w + 1;
				b_index[b] = i;
				b = b + 1;
				black = true;
				white = false;
			}
			else if (w_end < i) {
				w_index[w] = i - 1;
				w = w + 1;
				b_index[b] = i;
				b = b + 1;
				black = true;
				white = false;
			}
			if (ValArry[i] > max_b_index[max]) {
				max_b_index[max] = ValArry[i];
			}
		}
		//��ɫ����ѭ��
		else if ((ValArry[i] > K_num) && (black == true)) {
			b_end = i;
			if (ValArry[i] > max_b_index[max]) {
				max_b_index[max] = ValArry[i];
			}
		}
	}

	//if (flag == 1) {
	//	cout << "flag==1" << b << endl;
	//	exit(0);
	//}

	//���ȫ��
	if (b == 0) {
		return false;
	}
	//���ȫ��
	if (w == 0) {
		return false;
	}

	//����ɫ�������ϡ�٣���㾯��
	if (b <= 6) {
		cout << "��㾯��" << endl;
		int max_b = max_b_index[0];
		int min_b = max_b_index[0];
		//��ɫ��������ֵ�Ƚϲ�������ֵ����ı�����
		for (int i = 0; i < max; i++) {
			if (max_b_index[i] > max_b) {
				max_b = max_b_index[i];
			}
			else if (max_b_index[i] < min_b) {
				min_b = max_b_index[i];
			}
			else {
				continue;
			}
		}
		if ((max_b - min_b) > height * 0.5) {
			return false;
		}
		if ((w == 2) && (width > pic_width * 0.15)) {
			return true;
		}
		if ((w == 2) && (w_index[0] > b_index[0])) {
			return true;
		}
	}

	//�������г����Ҫ�� ��ȫ�׷�ȫ�ڣ��Һ�ɫ��������϶�
	int wb_connect_flag = 0;
	int bw_connect_flag = 0;
	//�жϺڰ������Ƿ����
	if (w_index[0] < b_index[0]) {
		//�Ȱ׺��
		for (int i = 1; i < w + 1; i = i + 2) {
			for (int j = 0; j < b; j = j + 2) {
				if (w_index[i] + 1 == b_index[j]) {
					wb_connect_flag++;
				}
				else {
					continue;
				}
			}
		}
	}
	else if (w_index[0] > b_index[0]) {
		//�Ⱥں��
		for (int i = 1; i < b + 1; i = i + 2) {
			for (int j = 0; j < w; j = j + 2) {
				if (b_index[i] + 1 == w_index[j]) {
					bw_connect_flag++;
				}
				else {
					continue;
				}
			}
		}
	}

	if ((wb_connect_flag >= 2) || (bw_connect_flag >= 2)) {
		//cout << "����һ����������" << endl;
		return true;
		if ((width < 20) || (height < 20)) {
			return false;
		}
	}
	else {
		//cout << "����һ���������" << endl;
		return false;
	}

}

/*--------------------------------------------------------------
* Description: �����������򡪡�������
* Parameters : �����������������򼯺ϣ����������ı���Ϣ
* Return : ����������vector<Rect>
* writer : ��ǧ��
--------------------------------------------------------------*/
void wordSort(vector<Rect>& sorted_wordRects, vector<Rect> word_componentRects, vector<Rect>& rowFirst_wordRects, int start, int end, int sum) {

	//�����쳣���棬start�±�ض�����end�±�
	if (start < end) {
		cout << "error!" << endl;
	}


	//���е�һԪ�����
	if (sum == 1) {
		sorted_wordRects.push_back(word_componentRects[start]);
		rowFirst_wordRects.push_back(word_componentRects[start]);
	}

	//�ǵ�һ�������������ð�����򣬴�С����
	int temp_x = 0;
	int temp_y = 0;
	int temp_width = 0;
	int temp_height = 0;

	for (int i = 0; i < sum - 1; i++) {
		if (word_componentRects[start - i].x > word_componentRects[start - i - 1].x) {
			temp_x = word_componentRects[start - i].x;
			temp_y = word_componentRects[start - i].y;
			temp_width = word_componentRects[start - i].width;
			temp_height = word_componentRects[start - i].height;

			word_componentRects[start - i].x = word_componentRects[start - i - 1].x;
			word_componentRects[start - i].y = word_componentRects[start - i - 1].y;
			word_componentRects[start - i].width = word_componentRects[start - i - 1].width;
			word_componentRects[start - i].height = word_componentRects[start - i - 1].height;

			word_componentRects[start - i - 1].x = temp_x;
			word_componentRects[start - i - 1].y = temp_y;
			word_componentRects[start - i - 1].width = temp_width;
			word_componentRects[start - i - 1].height = temp_height;

		}

	}
	//��ӵ��������ı���������
	for (int i = start; i > (start - sum); i--) {
		sorted_wordRects.push_back(word_componentRects[i]);
		if (i == start) {
			rowFirst_wordRects.push_back(word_componentRects[i]);
		}
	}
}


/*--------------------------------------------------------------
* Description: ������������
* Parameters: vector<Rect>��������
* Return: vector<Rect>������������
* writer:��ǧ��
--------------------------------------------------------------*/
vector<Rect> textlinesSorting(Mat final_picture, vector<Rect> word_componentRects, vector<Rect>& rowFirst_wordRects, vector<int>& row_num) {
	int Rectsize = word_componentRects.size();
	int x = 0;
	int y = 0;
	int x_next = 0;
	int y_next = 0;
	int start_flag = 0;//ͬһ����ʼ���±�
	int end_flag = 0;
	int sum_y = 0; //������������
	int ing_flag = 0; //�б�����־ 
	//������������������¼


	vector<Rect> sorted_wordRects;


	for (int i = Rectsize - 1; i >= 0; i--) {
		x = word_componentRects[i].x;
		y = word_componentRects[i].y;
		if (i - 1 >= 0) {
			// 2023.05.18�ж������� > ��Ϊ >=
			x_next = word_componentRects[i - 1].x;
			y_next = word_componentRects[i - 1].y;
		}
		else {
			x_next = 0;
			y_next = 0;
		}
		//�и߲����ƣ������б����У���һ��һԪ�����
		if ((abs(y - y_next) > 3) && (ing_flag == 0)) {
			start_flag = i;
			end_flag = i;
			sum_y = 1;
			wordSort(sorted_wordRects, word_componentRects, rowFirst_wordRects, start_flag, end_flag, sum_y);
			row_num.push_back(sum_y);
			//row_index = row_index + 1;
			sum_y = 0;
			ing_flag = 0;
			continue;
		}
		//�и߲����ƣ������б���,������������
		else if ((abs(y - y_next) > 3) && (ing_flag == 1)) {
			end_flag = i;
			sum_y = sum_y + 1;
			wordSort(sorted_wordRects, word_componentRects, rowFirst_wordRects, start_flag, end_flag, sum_y);
			row_num.push_back(sum_y);
			//row_index = row_index + 1;
			sum_y = 0;//��������������
			ing_flag = 0;//�б���״̬����
			continue;
		}
		//�տ�ʼ�����б���
		else if ((abs(y - y_next) <= 3) && (ing_flag == 0)) {
			start_flag = i;
			ing_flag = 1;//�����б���
			sum_y = sum_y + 1;
			continue;
		}
		//���б�����
		else if ((abs(y - y_next) <= 3) && (ing_flag == 1)) {
			end_flag = i;
			sum_y = sum_y + 1;
			continue;
		}
	}

	//cout << "sizeof(row_num)" << row_index << endl;
	////�鿴�����
	//for (int i = 0; i < row_index; i++) {
	//	cout << row_num[i] << endl;
	//}
	//�����ı����ϸ���
	//int size_rowFirst = rowFirst_wordRects.size();
	//cout << "size_rowFrist" << size_rowFirst << endl;

	//���������ı�����
	//for (int i = 0; i < rowFirst_wordRects.size(); i++)
	//{
	//	rectangle(final_picture, Point(rowFirst_wordRects[i].x, rowFirst_wordRects[i].y), Point(rowFirst_wordRects[i].x + rowFirst_wordRects[i].width, rowFirst_wordRects[i].y + rowFirst_wordRects[i].height), cv::Scalar(0, 255, 255), 5, 1, 0);
	//	//showPic(final_picture, "final_Picture");
	//}


	return sorted_wordRects;
}

/*--------------------------------------------------------------
* Description: �����ؿ���ȡ
* Parameters:
* Return: ����ֵ�б�
* writer:��ǧ��
--------------------------------------------------------------*/
vector<int> getBgPixelblock(Mat grayImage, vector<Rect> sorted_wordRects, vector<Rect> rowFirst_wordRects, vector<int> row_num) {
	int row_index = row_num.size();
	vector<int> pixelArry;
	int size_rowFirst = rowFirst_wordRects.size();  //�����ı����ϸ���
	int catch_blocks = 0;	//�����ؿ����
	int catch_width = 2;	//��ȡ���ؿ鳤��
	int catch_height = 2;
	int row_sum = 0;//�������ڼ����ı�����
	vector<int> tempArry;//��ʱ�洢��ȡ�������ص�
	for (int i = 0; i < row_index; i++) {
		int pix_x = rowFirst_wordRects[i].x + rowFirst_wordRects[i].width - 20;
		int pix_y = rowFirst_wordRects[i].y + rowFirst_wordRects[i].height / 2;
		int pix_value = 0;
		int value_sum = 0;
		int ave_value = 0;
		int min_value = 0;
		int delnum = 0;//����ֵ������־
		row_sum = row_num[i] + row_sum;
		tempArry.clear();//�����ʱ����
		if (catch_blocks == 4) {
			break;//��ȡ�㹻������ؿ������ѭ��
		}
		//���������������������ȡ
		if (row_num[i] == 1) {
			for (int u = 0; u < catch_width; u++) {
				for (int v = 0; v < catch_height; v++) {
					pix_value = grayImage.at<uchar>(pix_y + v, pix_x + u);
					tempArry.push_back(pix_value); //��ʱ����洢
					//temp = temp + 1;
					//cout << "pix_value = " << pix_value << endl;
					if (pix_value == 255) {
						// ����ͼƬ�ð�����
						delnum = delnum + 1;
						continue;
					}
					value_sum = pix_value + value_sum;
					//cout << "value_sum = " << value_sum << endl;
					//cout << "delnum = " << delnum << endl;
				}
			}
			//ave_value = value_sum / (catch_width * catch_height - delnum);//ȡƽ��ֵ
			//ȡ���������С����ֵ
			/*for (int w = 0; w < (catch_width * catch_height); w++) // 2023.05.19���������⣬��ʱ����Χ�綨�޸�*/
			for (int w = 0; w < tempArry.size(); w++)
			{
				min_value = tempArry[w];
				if (min_value > tempArry[w]) {
					min_value = tempArry[w];
				}
			}
			//cout << catch_width * catch_height - delnum << endl;
			catch_blocks = catch_blocks + 1;//��ȡ������1
			pixelArry.push_back(min_value); //������С����ֵ
			value_sum = 0;
			////cout << "��ȡһ�����ؿ飬ֵΪ��" << min_value << " .��ȡ��ʼ��Ϊ��" << pix_x << " " << pix_y << endl;
			//pixel_index = pixel_index + 1;
		}

		else if (row_num[i] >= 2) {
			for (int w = 1; w < 5; w++) {
				if (catch_blocks == 4) {
					break;
				}
				int number_row = row_sum - row_num[i] + w;  //��ǰ�еĵ�w���ı�����
				pix_x = rowFirst_wordRects[number_row - 1].x + rowFirst_wordRects[number_row - 1].width;
				pix_y = rowFirst_wordRects[number_row - 1].y;
				int pix_next_x = sorted_wordRects[number_row].x;
				int pix_next_y = sorted_wordRects[number_row].y;
				if ((pix_x - pix_next_x) > 10) {
					for (int u = 0; u < catch_width; u++) {
						for (int v = 0; v < catch_height; v++) {
							pix_value = grayImage.at<uchar>(pix_y + v, pix_x + u);
							tempArry.push_back(pix_value); //��ʱ����洢
							//temp = temp + 1;
							//cout << "pix_value" << pix_value << endl;
							if (pix_value == 255) {//����ͼƬ�ð�����
								delnum = delnum + 1;
								continue;
							}
							value_sum = value_sum + pix_value;
							//cout << "value_sum" << value_sum << endl;
						}
					}
					//ave_value = value_sum / (catch_width * catch_height - delnum);
					//ȡ���������С����ֵ
					/*for (int w = 0; w < (catch_width * catch_height); w++) // ���Ϸ�����һ��*/
					for (int w = 0; w < tempArry.size(); w++) {
						min_value = tempArry[w];
						if (min_value > tempArry[w]) {
							min_value = tempArry[w];
						}
					}
					//cout << catch_width * catch_height - delnum << endl;
					catch_blocks = catch_blocks + 1;
					pixelArry.push_back(min_value);
					value_sum = 0;
					//cout << "��ȡһ�����ؿ飬ֵΪ��" << min_value << " .��ȡ��ʼ��Ϊ��" << pix_x << " " << pix_y << endl;
					//pixel_index = pixel_index + 1;
				}
				else {
					continue;
				}
			}
		}
	}
	//cout << "���ؿ�����Ϊ��" << pixel_index << endl;
	//for (int i = 0; i < 20; i++) {
	//	cout << pixelArry[i] << endl;
	//}
	return pixelArry;
}



/*--------------------------------------------------------------
* Description:�ֲ�ͶӰ���и�
* Parameters: Rect rect, Mat binaryImage
* Return: vector<vector<int> >
* Writter: ����ΰ
---------------------------------------------------------------*/
vector<Rect> Local_Range_Projection(Rect rect, Mat binaryImage, string type)
{
	vector<Rect> wordRect;
	if (type == "vertical")
	{
		Mat src = binaryImage(rect);

		//step1. ������ֱͶӰ��ɫ������
		int w = src.cols;
		int h = src.rows;
		vector<int> project_val_arry;
		int per_pixel_value;
		for (int j = 0; j < w; j++)//��
		{
			Mat j_im = src.col(j);
			int num = countNonZero(j_im);//��ǰ�е�����ֵ�в�Ϊ0����Ŀ
			if (num < h)
			{
				/*��ǰ����ǰ�����ݺ�ɫ����*/
				project_val_arry.push_back(h-num);
			}
			else
			{
				/*��ǰ��ȫ���Ǳ�����ɫ����*/
				project_val_arry.push_back(0);
			}
			//project_val_arry.push_back(num);
		}

		//��ʾ
		Mat hist_im(h, w, CV_8UC1, Scalar(255));
		for (int i = 0; i < w; i++)
		{
			for (int j = 0; j < project_val_arry[i]; j++)
			{
				hist_im.ptr<unsigned char>(h - 1 - j)[i] = 0;
			}
		}
		//imshow("project", hist_im);
		

		//step2. �ַ��ָ�
		int start_Index = 0;
		int end_Index = 0;
		bool in_Block = false;//�Ƿ���������ַ�����
		int k = 0;
		for (int i = 0; i < w; ++i)
		{
			if (!in_Block && project_val_arry[i] > 20)//�����ַ�����
			{
				in_Block = true;
				start_Index = i;
				//cout << "startIndex is " << startIndex << endl;
			}
			else if (project_val_arry[i] <= 20 && in_Block)//����հ�����
			{
				end_Index = i;
				in_Block = false;
				wordRect.push_back(Rect(rect.x + start_Index, rect.y, end_Index - start_Index + 1, h ));
			}
			if (in_Block && i == w - 1)
			{
				end_Index = i;
				in_Block = false;
				wordRect.push_back(Rect(rect.x + start_Index + 1, rect.y, end_Index - start_Index + 1, h));
			}
		}
	}
	else if (type == "horizontal")
	{
		Mat src = binaryImage(rect);

		//step1. ����ˮƽͶӰ��ɫ������
		int w = src.cols;
		int h = src.rows;
		vector<int> project_val_arry;
		int per_pixel_value;
		for (int i = 0; i < h; i++)//��
		{
			Mat i_img = src.row(i);
			int num = countNonZero(i_img); //��ǰ�е�����ֵ�в�Ϊ0����Ŀ
			if (num < w)
			{
				/*��ǰ����ǰ�����ݺ�ɫ����*/
				project_val_arry.push_back(w-num);
			}
			else
			{
				/*��ǰ��ȫ���Ǳ�����ɫ����*/
				project_val_arry.push_back(0);
			}
		}

		//��ʾ
		Mat hist_im(h, w, CV_8UC1, Scalar(255));
		for (int i = 0; i < h; i++)
		{
			for (int j = 0; j < project_val_arry[i]; j++)
			{
				hist_im.at<uchar>(i, w - 1 - j) = 0;
			}
		}
		//imshow("project", hist_im);
		//waitKey(0);

		//step2. �ַ��ָ�
		int start_Index = 0;
		int end_Index = 0;
		bool in_Block = false;//�Ƿ���������ַ�����
		int k = 0;
		for (int i = 0; i < h; ++i)
		{
			if (!in_Block && project_val_arry[i] > 5)//�����ַ�����
			{
				in_Block = true;
				start_Index = i;
				//cout << "startIndex is " << startIndex << endl;
			}
			else if (project_val_arry[i] <= 5 && in_Block)//����հ�����
			{
				end_Index = i;
				in_Block = false;
				wordRect.push_back(Rect(rect.x, rect.y + start_Index, rect.width, end_Index + 1 - start_Index));
			}
			if (in_Block && i == h - 1)
			{
				end_Index = i;
				in_Block = false;
				wordRect.push_back(Rect(rect.x, rect.y + start_Index, rect.width, end_Index + 1 - start_Index));
			}
		}
	}
	return wordRect;
}

/*--------------------------------------------------------------
* Description: �ֲ��γ�ƽ��
* Parameters: Mat binaryImage
* Return:
* Writter: ����ΰ
--------------------------------------------------------------*/
void local_RLSA(Mat binaryImage, Mat input, vector<vector<Point> > bigAreaContours, vector<Rect>& wordRects)
{
	Mat reuniteImage = normalization(binaryImage);
	Mat digitImage = InterferenceImage(reuniteImage); //��Ե�������ͼ��
	Mat RLSAImage = digitImage;
	int threshold_RLSA1 = valueRLSA1(RLSAImage, digitImage, bigAreaContours);
	int threshold_RLSA2 = valueRLSA2(RLSAImage, digitImage, bigAreaContours, threshold_RLSA1);

	cout << "�ֲ���ֵ�������" << endl;
	cout << "threshold_RLSA1 :" << threshold_RLSA1 << "threshold_RLSA2:" << threshold_RLSA2 << endl;


	Mat afterSmoothHor = lengthSmoothHor(RLSAImage, threshold_RLSA2 * 0.25);
	Mat afterSmoothVer = lengthSmoothVer(RLSAImage, threshold_RLSA1);
	// ������
	Mat afterSmooth = afterSmoothVer & afterSmoothHor;
	Mat afterSmooth2Show_1 = afterSmooth * 255;
	//showPic(afterSmooth2Show_1,"afterSmooth2Show_1");

	//��һ��ˮƽƽ��
	//afterSmooth = lengthSmoothHor(afterSmooth, threshold_RLSA2 * 0.95);
	//Mat afterSmooth2Show_2 = afterSmooth * 255;
	//showPic(afterSmooth2Show_2,"afterSmooth2Show_2");

	Mat afterSmoothHor2Show = afterSmoothHor * 255;
	//showPic(afterSmoothHor2Show, "afterSmoothHor2Show");
	Mat afterSmoothVer2Show = afterSmoothVer * 255;
	//showPic(afterSmoothVer2Show, "afterSmoothVer2Show");


	//���ּ��
	std::vector<Rect> componentRects = textDetect(afterSmooth, input, bigAreaContours, 2);
	
	for (int i = 0; i < componentRects.size(); i++)
	{
		wordRects.push_back(componentRects[i]);
	}
	return;
}

/*--------------------------------------------------------------
* Description: ��©��ȱ
* Parameters: Mat binaryImage, vector<vector<Point> >& bigAreaContours, vector<Rect>& wordRects
* Return:
* Writter: ����ΰ
--------------------------------------------------------------*/
void Checking_for_gaps(Mat binaryImage, Mat input, vector<vector<Point> >& bigAreaContours, vector<Rect> bigArea, vector<Rect>& wordRects)
{
	Mat digitImage = normalization(binaryImage);
	// ��δ���з����������
	for (int i = 0; i < bigAreaContours.size(); i++)
	{
		// ������ͼ�������ڸ�
		Get_Irregular_Contours(binaryImage, bigAreaContours, bigArea, wordRects, 1);
		//drawContours(binaryImage, bigAreaContours, i, CV_RGB(255, 255, 255), -1);
	}
	for (int i = 0; i < bigArea.size(); i++)
	{
		rectangle(binaryImage, bigArea[i], CV_RGB(255, 255, 255), -1);
		rectangle(input, bigArea[i], CV_RGB(255, 255, 255), -1);
	}
	for (int i = 0; i < wordRects.size(); i++)
	{
		// ���������������ڸ�
		rectangle(binaryImage, wordRects[i], CV_RGB(255, 255, 255), -1);
		rectangle(input, wordRects[i], CV_RGB(255, 255, 255), -1);
	}
	//showPic(binaryImage, "masked_binaryImage");


	// �ֲ��γ�ƽ��
	local_RLSA(binaryImage, input, bigAreaContours, wordRects);
	return;
}

/*--------------------------------------------------------------
* Description: ����
* Parameters: Mat binaryImage, vector<vector<Point> >& bigAreaContours, vector<Rect>& wordRects
* Return:
* Writter: ����ΰ
--------------------------------------------------------------*/
void image_word_correction(Mat binaryImage, Mat input, vector<vector<Point> >& bigAreaContours, vector<Rect>& bigAreaRect, vector<Rect>& wordRects)
{
	Mat digitImage = normalization(binaryImage);
	// ˮƽͶӰ�ж�ͼ���������Ƿ��ж����ı�
	vector<vector<Point> >::const_iterator it_bigAreaContours = bigAreaContours.begin();
	while (it_bigAreaContours != bigAreaContours.end())
	{
		Rect bRect = boundingRect(*it_bigAreaContours);
		/*Mat pic_image = digitImage(bRect);
		bool is_word_horizonl = pic_textlineJudge(pic_image, input.rows, input.cols);
		if (is_word_horizonl) {
			cout << "11111111��ʵ���ı���" << endl;
			wordRects.push_back(bRect);
			it_bigAreaContours = bigAreaContours.erase(it_bigAreaContours);
			continue;
		}
		else {
			cout << "111111111��Ȼ��ͼƬ" << endl;
			it_bigAreaContours++;
			continue;
		}*/
		bool flag = false;
		int count = 0;
		for (int i = 0; i < wordRects.size(); i++)
		{
			if (IsOverLap(bRect, wordRects[i]))
			{
				if (count > 2)
				{
					flag = true;
					break;
				}
				count++;
			}
		}
		if (flag)
		{
			it_bigAreaContours++;
			continue;
		}
		vector<Rect> wordRect_add = Local_Range_Projection(bRect, binaryImage, "horizontal");
		if (wordRect_add.size() > 1)
		{
			// �������к��ж����ı�
			for (int i = 0; i < wordRect_add.size(); i++)
			{
				wordRects.push_back(wordRect_add[i]);
			}
			it_bigAreaContours = bigAreaContours.erase(it_bigAreaContours);
			continue;
		}
		else
		{
			// ��������ܾ������������1.�����ı� 2.�����ı� 3.��ͼ�� 4. ͼ��+�ı�
			wordRect_add.clear();
			wordRect_add = Local_Range_Projection(bRect, binaryImage, "vertical");
			if (wordRect_add.size() > 1)
			{
				// �ж�����
				for (int i = 0; i < wordRect_add.size(); i++)
				{
					Mat pic_image = digitImage(wordRect_add[i]);
					bool is_word_vertical = pic_textlineJudge(pic_image, input.rows, input.cols);
					if (is_word_vertical)
					{
						cout << "��������ʵ���ı���" << endl;
						wordRects.push_back(wordRect_add[i]);
						continue;
					}
					else
					{
						cout << "��������ʵ��ͼ��" << endl;
						bigAreaRect.push_back(wordRect_add[i]);
						continue;
					}
				}
				it_bigAreaContours = bigAreaContours.erase(it_bigAreaContours);
				continue;
			}
		////	//else
		////	//{
		////	//	/*��������ܾ������������1.���� 2.ͼ��*/
		////	//	Mat bigAreaCut = binaryImage(bRect);
		////	//	Mat cut = input(bRect);
		////	//	int num = 0;
		////	//	for (int w = 0; w < bigAreaCut.cols; w++)
		////	//	{
		////	//		Mat h = bigAreaCut.col(w);
		////	//		num += bigAreaCut.rows - countNonZero(h); // ��ǰ�����������к�ɫ���صĸ���
		////	//	}
		////	//	double rate = (num * 1.0) / (bigAreaCut.cols * bigAreaCut.rows); // ��Ƭ�ں�ɫ����ռ�����صı���
		////	//	if (rate < 0.25)
		////	//	{
		////	//		// ���rateС��0.25���ʾ��ͼ��ӦΪ���֣�������ͼ��
		////	//		// ��bigAreaContours��ɾ������ӽ�wordRects
		////	//		wordRects.push_back(bRect);
		////	//		it_bigAreaContours = bigAreaContours.erase(it_bigAreaContours);
		////	//		continue;
		////	//	}
		////	//	else
		////	//	{
		////	//		it_bigAreaContours++;
		////	//		continue;
		////	//	}
		////	//}
		}
		it_bigAreaContours++;
		continue;

		/*
		// ��Ѳ���ͼ���ж�Ϊ����
		vector<vector<Point> >::const_iterator it_bigAreaContours_vertical = bigAreaContours.begin();
		while (it_bigAreaContours_vertical != bigAreaContours.end())
		{
			Rect bRect = boundingRect(*it_bigAreaContours_vertical);
			Mat pic_image = digitImage(bRect);
			bool is_word_vertical = textlinesJudge(pic_image, input.rows, input.cols);
			if (is_word_vertical) {
				cout << "222222222��ʵ���ı���" << endl;
				wordRects.push_back(bRect);
				it_bigAreaContours_vertical = bigAreaContours.erase(it_bigAreaContours_vertical);
				continue;
			}
			else {
				cout << "2222222222��Ȼ��ͼƬ" << endl;
				it_bigAreaContours_vertical++;
				continue;
			}
		}*/
	}
	return;
}

/*--------------------------------------------------------------
* Description:��ͼ������ִ���json
* Parameters: vector<vector<Point> > bigAreaContours, vector<Rect> wordRects
* Return: 
* Writter: ����ΰ
---------------------------------------------------------------*/
void writeFileJson(string filePath, vector<vector<Point> > bigAreaContours, vector<Rect>bigAreaRects, vector<Rect> wordRects)
{
	ofstream fout;
	fout.open(filePath.c_str());
	assert(fout.is_open());
	
	// ���ڵ�
	Json::Value root;

	// �ӽڵ�
	Json::Value graphic;
	Json::Value textlines;
	Json::Value irregular_points;
	Json::Value Rectangle_points;
	
	// �ӽڵ�����
	for (int i = 0; i < bigAreaContours.size(); i++)
	{
		Json::Value array;
		for (int j = 0; j < bigAreaContours[i].size(); j++)
		{
			array[j][0] = bigAreaContours[i][j].x;
			array[j][1] = bigAreaContours[i][j].y;
		}
		irregular_points["points"].append(array);
	}
	irregular_points["shapetype"] = Json::Value("irregular");
	graphic["irregular_points"].append(irregular_points);

	for (int i = 0; i < bigAreaRects.size(); i++)
	{
		Json::Value array;
		array[0] = wordRects[i].x;
		array[1] = wordRects[i].y;
		array[2] = wordRects[i].width;
		array[3] = wordRects[i].height;
		Rectangle_points["points"].append(array);
	}
	Rectangle_points["shapetype"] = Json::Value("Rectangble_x, y, width, height");
	graphic["Rectangle_points"].append(Rectangle_points);
	root["graphic"] = graphic;

	textlines["shapetype"] = Json::Value("Rectangle_x, y, width, height");
	for (int i = 0; i < wordRects.size(); i++)
	{
		Json::Value array;
		array[0] = wordRects[i].x;
		array[1] = wordRects[i].y;
		array[2] = wordRects[i].width;
		array[3] = wordRects[i].height;
		textlines["points"].append(array);
	}
	root["textlines"] = textlines;

	string out = root.toStyledString();
	fout << out << endl;
	return;
}

/*--------------------------------------------------------------
* Description: �жϴ�������������Ƿ��ཻ
* Parameters: Rect rect1, Rect rect2
* Return: bool
* Writter: ����ΰ
---------------------------------------------------------------*/
bool IsOverLap(Rect rect1, Rect rect2)
{
	int xmin1 = rect1.x;
	int ymin1 = rect1.y;
	int xmax1 = rect1.x + rect1.width;
	int ymax1 = rect1.y + rect1.height;
	int xmin2 = rect2.x;
	int ymin2 = rect2.y;
	int xmax2 = rect2.x + rect2.width;
	int ymax2 = rect2.y + rect2.height;

	if ((((xmin1 >= xmin2 && xmin1 < xmax2) || (xmin2 >= xmin1 && xmin2 <= xmax1)) &&
		((ymin1 >= ymin2 && ymin1 < ymax2) || (ymin2 >= ymin1 && ymin2 <= ymax1))) == 1)
	{
		return true;
	}
	else
	{
		return false;
	}
}
/*--------------------------------------------------------------
* Description:ȥ��ͼ������,flag=1�Զ�ֵ��ͼ�������flag=2������ɫͼ�������� flage=3����ɫͼ������
* Parameters: Mat binaryImage, vector<vector<Point> > contours, vector<Rect> bigAreaRects, vector<Rect> boundingRects, int flag
* Return: Null
* Writter: ����ΰ
---------------------------------------------------------------*/
void Get_Irregular_Contours(Mat &binaryImage, vector<vector<Point> > contours, vector<Rect> bigAreaRects, vector<Rect> boundingRects, int flag)
{
	/*int contours_length = contours.size();
	for (int i = 0; i < contours_length; i++)
	{
		int it_contour_length = contours[i].size();
		Point xmin_ymin = Point(9999, 9999);
		Point xmin_ymax = Point(9999, -1);
		Point xmax_ymin = Point(-1, 9999);
		Point xmax_ymax = Point(-1, -1);
		for (int j = 0; j < it_contour_length; j++)
		{
			// ��ѯ���ϵ�
			if (contours[i][j].x < xmin_ymin.x && contours[i][j].y < xmin_ymin.y)
			{
				xmin_ymin = contours[i][j];
				continue;
			}
			// ��ѯ���µ�
			if (contours[i][j].x < xmin_ymax.x && contours[i][j].y > xmin_ymax.y)
			{
				xmin_ymax = contours[i][j];
				continue;
			}
			// ��ѯ���ϵ�
			if (contours[i][j].x > xmax_ymin.x && contours[i][j].y < xmax_ymin.y)
			{
				xmax_ymin = contours[i][j];
				continue;
			}
			// ��ѯ���µ�
			if (contours[i][j].x > xmax_ymax.x && contours[i][j].y > xmax_ymax.y)
			{
				xmax_ymax = contours[i][j];
				continue;
			}
		}
		// �ĸ������������㹹�ɵ�����ֱ�ߴ�ֱ������ƾ��Σ�ͬʱɾ��contours[i]�ĵ㼯
		if (abs(xmin_ymin.x - xmin_ymax.x) < 20 && abs(xmin_ymin.y - xmax_ymin.y) < 20)
		{
			// �����Ƿ�ֱ
			int xmin_1 = xmin_ymin.x;
			int ymin_1 = xmin_ymin.y;
			int xmin_2 = xmin_ymax.x;
			int ymin_2 = xmax_ymin.y;
			int xmax_1 = xmax_ymin.x;
			int xmax_2 = xmax_ymax.x;
			int ymax_1 = xmin_ymax.y;
			int ymax_2 = xmax_ymax.y;
			int xmin = min(xmin_1, xmin_2);
			int ymin = min(ymin_1, ymin_2);
			int xmax = max(xmax_1, xmax_2);
			int ymax = max(ymax_1, ymax_2);
			rectangle(binaryImage, Point(xmin, ymin), Point(xmax, ymax), CV_RGB(255, 255, 255), -1);
			showPic(binaryImage, "binaryImage");
			waitKey(0);
			contours[i].clear();
			continue;
		}
		if (abs(xmin_ymin.y - xmax_ymin.y) < 20 && abs(xmax_ymax.x - xmax_ymin.x) < 20)
		{
			// �����Ƿ�ֱ
			int xmin_1 = xmin_ymin.x;
			int ymin_1 = xmin_ymin.y;
			int xmin_2 = xmin_ymax.x;
			int ymin_2 = xmax_ymin.y;
			int xmax_1 = xmax_ymin.x;
			int xmax_2 = xmax_ymax.x;
			int ymax_1 = xmin_ymax.y;
			int ymax_2 = xmax_ymax.y;
			int xmin = min(xmin_1, xmin_2);
			int ymin = min(ymin_1, ymin_2);
			int xmax = max(xmax_1, xmax_2);
			int ymax = max(ymax_1, ymax_2);
			rectangle(binaryImage, Point(xmin, ymin), Point(xmax, ymax), CV_RGB(255, 255, 255), -1);
			showPic(binaryImage, "binaryImage");
			waitKey(0);
			contours[i].clear();
			continue;
		}
		if (abs(xmax_ymin.x - xmax_ymax.x) < 20 && abs(xmin_ymax.y - xmax_ymax.y) < 20)
		{
			// �����Ƿ�ֱ
			int xmin_1 = xmin_ymin.x;
			int ymin_1 = xmin_ymin.y;
			int xmin_2 = xmin_ymax.x;
			int ymin_2 = xmax_ymin.y;
			int xmax_1 = xmax_ymin.x;
			int xmax_2 = xmax_ymax.x;
			int ymax_1 = xmin_ymax.y;
			int ymax_2 = xmax_ymax.y;
			int xmin = min(xmin_1, xmin_2);
			int ymin = min(ymin_1, ymin_2);
			int xmax = max(xmax_1, xmax_2);
			int ymax = max(ymax_1, ymax_2);
			rectangle(binaryImage, Point(xmin, ymin), Point(xmax, ymax), CV_RGB(255, 255, 255), -1);
			showPic(binaryImage, "binaryImage");
			waitKey(0);
			contours[i].clear();
			continue;
		}
		if (abs(xmin_ymin.x - xmin_ymax.x) < 20 && abs(xmin_ymax.y - xmax_ymax.x) < 20)
		{
			// �����Ƿ�ֱ
			int xmin_1 = xmin_ymin.x;
			int ymin_1 = xmin_ymin.y;
			int xmin_2 = xmin_ymax.x;
			int ymin_2 = xmax_ymin.y;
			int xmax_1 = xmax_ymin.x;
			int xmax_2 = xmax_ymax.x;
			int ymax_1 = xmin_ymax.y;
			int ymax_2 = xmax_ymax.y;
			int xmin = min(xmin_1, xmin_2);
			int ymin = min(ymin_1, ymin_2);
			int xmax = max(xmax_1, xmax_2);
			int ymax = max(ymax_1, ymax_2);
			rectangle(binaryImage, Point(xmin, ymin), Point(xmax, ymax), CV_RGB(255, 255, 255), -1);
			showPic(binaryImage, "binaryImage");
			waitKey(0);
			contours[i].clear();
			continue;
		}

		// ��������ֱ������䲻����ͼ��
		drawContours(binaryImage, contours, i, CV_RGB(255, 255, 255), -1);
		showPic(binaryImage, "binaryImage");
		waitKey(0);
	}*/
	if (flag == 1)
	{
		// flag = 1�Զ�ֵ��ͼ�����
		for (int i = 0; i < contours.size(); i++)
		{
			int xmin = 9999; int ymin = 9999;
			int xmax = -1; int ymax = -1;
			for (int j = 0; j < contours[i].size(); j++)
			{
				xmin = min(xmin, contours[i][j].x);
				ymin = min(ymin, contours[i][j].y);
				xmax = max(xmax, contours[i][j].x);
				ymax = max(ymax, contours[i][j].y);
			}
			bool corner[4] = { false };
			for (int j = 0; j < contours[i].size(); j++)
			{
				if (abs(contours[i][j].x - xmin) < 10 && abs(contours[i][j].y - ymin) < 10) corner[0] = true;
				if (abs(contours[i][j].x - xmax) < 10 && abs(contours[i][j].y - ymin) < 10) corner[1] = true;
				if (abs(contours[i][j].x - xmin) < 10 && abs(contours[i][j].y - ymax) < 10) corner[2] = true;
				if (abs(contours[i][j].x - xmax) < 10 && abs(contours[i][j].y - ymax) < 10) corner[3] = true;
			}
			if ((corner[0] && corner[1] && corner[2] && corner[3]) || (corner[1] && corner[2] && corner[3]) || (corner[0] && corner[2] && corner[3]) || (corner[0] && corner[1] && corner[3]) || (corner[0] && corner[1] && corner[2]))
			{
				bool sign = false;
				for (int v = 0; v < boundingRects.size(); v++)
				{
					if (IsOverLap(Rect(xmin, ymin, xmax - xmin, ymax - ymin), boundingRects[v]))
					{
						// ���signΪfalse����˵���ཻ�������ཻ
						sign = true;
					}
				}
				if (sign)
				{
					rectangle(binaryImage, Point(xmin, ymin), Point(xmax, ymax), CV_RGB(255, 255, 255), -1);
				}
			}
			else
			{
				drawContours(binaryImage, contours, i, CV_RGB(255, 255, 255), -1);
			}
		}
		for (int i = 0; i < bigAreaRects.size(); i++)
		{
			// ����
			rectangle(binaryImage, bigAreaRects[i], CV_RGB(255, 255, 255), -1);
		}
	}
	else if (flag == 2)
	{
		// flag = 2�Բ�ɫͼ�����
		for (int i = 0; i < contours.size(); i++)
		{
			// ������
			int xmin = 9999; int ymin = 9999;
			int xmax = -1; int ymax = -1;
			for (int j = 0; j < contours[i].size(); j++)
			{
				xmin = min(xmin, contours[i][j].x);
				ymin = min(ymin, contours[i][j].y);
				xmax = max(xmax, contours[i][j].x);
				ymax = max(ymax, contours[i][j].y);
			}
			bool corner[4] = { false };
			for (int j = 0; j < contours[i].size(); j++)
			{
				if (abs(contours[i][j].x - xmin) < 10 && abs(contours[i][j].y - ymin) < 10) corner[0] = true;
				if (abs(contours[i][j].x - xmax) < 10 && abs(contours[i][j].y - ymin) < 10) corner[1] = true;
				if (abs(contours[i][j].x - xmin) < 10 && abs(contours[i][j].y - ymax) < 10) corner[2] = true;
				if (abs(contours[i][j].x - xmax) < 10 && abs(contours[i][j].y - ymax) < 10) corner[3] = true;
			}
			if ((corner[0] && corner[1] && corner[2] && corner[3]) || (corner[1] && corner[2] && corner[3]) || (corner[0] && corner[2] && corner[3]) || (corner[0] && corner[1] && corner[3]) || (corner[0] && corner[1] && corner[2]))
			{
				bool sign = false;
				for (int v = 0; v < boundingRects.size(); v++)
				{
					if (IsOverLap(Rect(xmin, ymin, xmax - xmin, ymax - ymin), boundingRects[v]))
					{
						// ���signΪtrue����˵���ཻ�������ཻ
						sign = true;
					}
				}
				if (!sign)
				{
					rectangle(binaryImage, Point(xmin, ymin), Point(xmax, ymax), CV_RGB(255, 0, 0), 8);
				}
				else
				{
					drawContours(binaryImage, contours, i, CV_RGB(255, 0, 0), 8);
				}
			}
			else
			{
				drawContours(binaryImage, contours, i, CV_RGB(255, 0, 0), 8);
			}
		}
		for (int i = 0; i < bigAreaRects.size(); i++)
		{
			// ����
			rectangle(binaryImage, bigAreaRects[i], CV_RGB(255, 0, 0), 8);
		}
	}
	else if (flag == 3)
	{
		// flag = 2�Բ�ɫͼ�����
		for (int i = 0; i < contours.size(); i++)
		{
			int xmin = 9999; int ymin = 9999;
			int xmax = -1; int ymax = -1;
			for (int j = 0; j < contours[i].size(); j++)
			{
				xmin = min(xmin, contours[i][j].x);
				ymin = min(ymin, contours[i][j].y);
				xmax = max(xmax, contours[i][j].x);
				ymax = max(ymax, contours[i][j].y);
			}
			bool corner[4] = { false };
			for (int j = 0; j < contours[i].size(); j++)
			{
				if (abs(contours[i][j].x - xmin) < 10 && abs(contours[i][j].y - ymin) < 10) corner[0] = true;
				if (abs(contours[i][j].x - xmax) < 10 && abs(contours[i][j].y - ymin) < 10) corner[1] = true;
				if (abs(contours[i][j].x - xmin) < 10 && abs(contours[i][j].y - ymax) < 10) corner[2] = true;
				if (abs(contours[i][j].x - xmax) < 10 && abs(contours[i][j].y - ymax) < 10) corner[3] = true;
			}
			//if ((corner[0] && corner[1] && corner[2] && corner[3]) || (corner[1] && corner[2] && corner[3]) || (corner[0] && corner[2] && corner[3]) || (corner[0] && corner[1] && corner[3]) || (corner[0] && corner[1] && corner[2]))
			//{
			//	bool sign = false;
			//	for (int v = 0; v < boundingRects.size(); v++)
			//	{
			//		if (IsOverLap(Rect(xmin, ymin, xmax - xmin, ymax - ymin), boundingRects[v]))
			//		{
			//			// ���signΪfalse����˵���ཻ�������ཻ
			//			sign = true;
			//		}
			//	}
			//	if (sign)
			//	{
			//		rectangle(binaryImage, Point(xmin, ymin), Point(xmax, ymax), CV_RGB(255, 0, 0), -1);
			//	}
			//}
			if ((corner[0] && corner[1] && corner[2] && corner[3]) || (corner[1] && corner[2] && corner[3]) || (corner[0] && corner[2] && corner[3]) || (corner[0] && corner[1] && corner[3]) || (corner[0] && corner[1] && corner[2]))
			{
				bool sign = false;
				for (int v = 0; v < boundingRects.size(); v++)
				{
					if (IsOverLap(Rect(xmin, ymin, xmax - xmin, ymax - ymin), boundingRects[v]))
					{
						// ���signΪtrue����˵���ཻ�������ཻ
						sign = true;
					}
				}
				if (!sign)
				{
					rectangle(binaryImage, Point(xmin, ymin), Point(xmax, ymax), CV_RGB(255, 0, 0), -1);
				}
				else
				{
					drawContours(binaryImage, contours, i, CV_RGB(255, 0, 0), -1);
				}
			}
			else
			{
				drawContours(binaryImage, contours, i, CV_RGB(255, 0, 0), -1);
			}
		}

		for (int i = 0; i < bigAreaRects.size(); i++)
		{
			// ����
			rectangle(binaryImage, bigAreaRects[i], CV_RGB(255, 0, 0), -1);
		}
	}
	return;
}

/*--------------------------------------------------------------
* Description:ȥ��ͼ������,�Զ�ֵ��ͼ�����
* Parameters: Mat binaryImage, vector<vector<Point> > contours, vector<Rect> bigAreaRects
* Return: Null
* Writter: ����ΰ
---------------------------------------------------------------*/
void Get_Irregular_Contours(Mat& binaryImage, vector<vector<Point> > contours, vector<Rect> bigAreaRects)
{
	// flag = 1�Զ�ֵ��ͼ�����
	for (int i = 0; i < contours.size(); i++)
	{
		int xmin = 9999; int ymin = 9999;
		int xmax = -1; int ymax = -1;
		for (int j = 0; j < contours[i].size(); j++)
		{
			xmin = min(xmin, contours[i][j].x);
			ymin = min(ymin, contours[i][j].y);
			xmax = max(xmax, contours[i][j].x);
			ymax = max(ymax, contours[i][j].y);
		}
		bool corner[4] = { false };
		for (int j = 0; j < contours[i].size(); j++)
		{
			if (abs(contours[i][j].x - xmin) < 10 && abs(contours[i][j].y - ymin) < 10) corner[0] = true;
			if (abs(contours[i][j].x - xmax) < 10 && abs(contours[i][j].y - ymin) < 10) corner[1] = true;
			if (abs(contours[i][j].x - xmin) < 10 && abs(contours[i][j].y - ymax) < 10) corner[2] = true;
			if (abs(contours[i][j].x - xmax) < 10 && abs(contours[i][j].y - ymax) < 10) corner[3] = true;
		}
		if ((corner[0] && corner[1] && corner[2] && corner[3]) || (corner[1] && corner[2] && corner[3]) || (corner[0] && corner[2] && corner[3]) || (corner[0] && corner[1] && corner[3]) || (corner[0] && corner[1] && corner[2]))
		{
			rectangle(binaryImage, Point(xmin, ymin), Point(xmax, ymax), CV_RGB(255, 255, 255), -1);
		}
		else
		{
			drawContours(binaryImage, contours, i, CV_RGB(255, 255, 255), -1);
		}
	}
	for (int i = 0; i < bigAreaRects.size(); i++)
	{
		// ����
		rectangle(binaryImage, bigAreaRects[i], CV_RGB(255, 255, 255), -1);
	}
}

/*--------------------------------------------------------------
* Description: ���ı�������ͼ�������ཻ��ɾ���ı�����
* Parameters:
* Return:
* Writter: ����ΰ
---------------------------------------------------------------*/
void remove_overlap_area(vector < vector<Point>> bigAreaContours, vector<Rect> bigAreaRects, vector<Rect> &wordRects)
{
	vector<Rect>::const_iterator it_wordRect = wordRects.begin();
	while (it_wordRect != wordRects.end())
	{
		for (int i = 0; i < bigAreaRects.size(); i++)
		{
			if (IsOverLap(bigAreaRects[i], Rect(it_wordRect->x, it_wordRect->y, it_wordRect->width, it_wordRect->height)))
			{
				it_wordRect = wordRects.erase(it_wordRect);
				continue;
			}
		}
		it_wordRect++;
	}
	return;
}
/*--------------------------------------------------------------
* Description: ���ڴ�ֱͶӰ�����������ж�
* Parameters: 
* Return:
* Writter: ��ǧ��
---------------------------------------------------------------*/
void vertical_projection_for_word(vector<Rect>& word_componentRects, vector<Rect> componentRects, Mat reuniteImage_opencv, vector<Rect>& zaodian_componentRects)
{
	int height = reuniteImage_opencv.rows;
	int width = reuniteImage_opencv.cols;

	// ���ڴ�ֱͶӰ�����������ж�
	for (int i = 0; i < componentRects.size(); i++)
	{
		if (componentRects[i].x < 0 || componentRects[i].y < 0 || componentRects[i].x + componentRects[i].width > width || componentRects[i].y + componentRects[i].height > height)
			cout << "(" << componentRects[i].x << ", " << componentRects[i].y << "," << componentRects[i].width << ", " << componentRects[i].height << ")" << endl;
		Mat rect_image = reuniteImage_opencv(componentRects[i]);
		//showPic(rect_image * 255,"rect_image");
		bool is_word = textlinesJudge(rect_image, height, width);

		if (is_word == true) {
			cout << componentRects[i].x << " " << componentRects[i].y << " " << componentRects[i].width << " " << componentRects[i].height << "����һ����������" << endl;
			word_componentRects.push_back(componentRects[i]);
		}
		else if (is_word == false) {
			cout << componentRects[i].x << " " << componentRects[i].y << " " << componentRects[i].width << " " << componentRects[i].height << "����һ���������" << endl;
			zaodian_componentRects.push_back(componentRects[i]);
		}
		else {
			cout << "some errors!" << endl;
		}
	}
	return;
}

/*--------------------------------------------------------------
* Description:ȥ��ֱ��
* Parameters: vector<Vec4f>
* Return:
* Writter: ����ΰ
---------------------------------------------------------------*/
Mat ls_remove(vector<Vec4f> linePoint, Mat imageDel)
{
	Mat lineDel = imageDel.clone();
	/*for (int i = 0; i < linePoint.size(); i++)
		line(lineDel, Point(linePoint[i][0], linePoint[i][1]), Point(linePoint[i][2], linePoint[i][3]), CV_RGB(255, 0, 0), 2);*/
	for (int i = 0; i < linePoint.size()-1; i++)
		for (int j = i + 1; j < linePoint.size(); j++)
		{
			//if (abs(linePoint[i][1] - linePoint[i][3]) <= 5 && 
			//	abs(linePoint[j][1] - linePoint[j][3]) <= 5)
			//{
			//	//����
			//	int xmin = min((int)linePoint[i][0], (int)linePoint[j][0]);
			//	int ymin = min((int)linePoint[i][1], (int)linePoint[j][1]);
			//	int xmax = min((int)linePoint[i][2], (int)linePoint[j][2]);
			//	int ymax = min((int)linePoint[i][2], (int)linePoint[j][2]);
			//	/*vector<Point> contour;
			//	contour.push_back(Point(xmin, ymin));
			//	contour.push_back(Point(xmax, ymin));
			//	contour.push_back(Point(xmax, ymax));
			//	contour.push_back(Point(xmin, ymax));
			//	fillConvexPoly(lineDel, contour, CV_RGB(0, 0, 0));*/
			//	//rectangle(lineDel, Rect(xmin, ymin, xmax - xmin, ymax - ymin), CV_RGB(255, 0, 0), 2);
			//}
			if (linePoint[i][2] - linePoint[i][0] <= 5 
				&& linePoint[j][2] - linePoint[j][0] <= 5 
				&& abs(linePoint[i][0] - linePoint[j][0]) < 20)
			{
				//����
				int xmin = min((int)linePoint[i][0], (int)linePoint[j][0]);
				int ymin = min((int)linePoint[i][1], (int)linePoint[j][1]);
				int xmax = max((int)linePoint[i][2], (int)linePoint[j][2]);
				int ymax = max((int)linePoint[i][3], (int)linePoint[j][3]);
				vector<Point> contour;
				contour.push_back(Point(xmin, ymin));
				contour.push_back(Point(xmax, ymin));
				contour.push_back(Point(xmax, ymax));
				contour.push_back(Point(xmin, ymax));
				fillConvexPoly(lineDel, contour, CV_RGB(0, 0, 0));
				//rectangle(lineDel, Rect(xmin, ymin, xmax - xmin, ymax - ymin), CV_RGB(255, 0, 0), 2);
				//line(lineDel, Point(xmin, ymin), Point(xmax, ymax), CV_RGB(255, 0, 0), 2);
			}
		}
	return lineDel;
}
/*--------------------------------------------------------------
* Description:���������ͼ��ȥ��
* Parameters: vector<Rect> table,vector<Rect> bigAreas.
* Return: null
---------------------------------------------------------------*/
void imageInsideTable(vector<Rect>tableAreas, vector<Rect>&imageAreas)
{
	if (tableAreas.size() == 0 || imageAreas.size() == 0)
	{
		return;
	}
	vector<Rect>::const_iterator it_tableAreas = tableAreas.begin();
	vector<Rect>::const_iterator it_imageAreas = imageAreas.begin();
	while (it_tableAreas != tableAreas.end())
	{
		Rect tableArea = Rect(*it_tableAreas);
		while (it_imageAreas != imageAreas.end())
		{
			Rect imageArea = Rect(*it_imageAreas);
			Rect intersection = imageArea&tableArea;
			if (imageArea == intersection)
			{
				it_imageAreas = imageAreas.erase(it_imageAreas);
			}
			else
			{
				it_imageAreas++;
			}
		}
		it_tableAreas++;
	}
}

/*--------------------------------------------------------------
* Description: ͼ����������
* Parameters: vector<vector<Point> > bigAreaContours, vector<vector<Point> > wordContours
* Return: 
* writter: ����ΰ
---------------------------------------------------------------*/
void mergeContours(vector<vector<Point> > contours, vector<vector<Point> > &bigAreaContours)
{
	vector<vector<Point> > allcontours;
	vector<Point> contourslist;

	for (int i = 0; i < contours.size(); i++) {
		vector<Point> vec_i;
		vec_i = contours[i];
		contourslist.insert(contourslist.end(), vec_i.begin(), vec_i.end());
	}
	bigAreaContours.emplace_back(contourslist);
}

void image_word_MinDistance(vector<vector<Point> > &bigAreaContours, vector<vector<Point> > &wordContours)
{
	for (int i_word = 0; i_word < wordContours.size(); i_word++)
	{
		bool flag_word = false;
		Rect wordRect = boundingRect(wordContours[i_word]);
		vector<vector<Point> >::const_iterator it_bigAreaContours = bigAreaContours.begin();
		while (it_bigAreaContours != bigAreaContours.end())
		{
			if (it_bigAreaContours->size() == 0)
				it_bigAreaContours = bigAreaContours.erase(it_bigAreaContours);
			else
			{
				it_bigAreaContours++;
			}
		}
		for (int i_bigArea = 0; i_bigArea < bigAreaContours.size(); i_bigArea++)
		{
			if (flag_word)
				break;
			bool flag_bigArea = false;
			for (int j_bigArea = 0; j_bigArea < bigAreaContours[i_bigArea].size(); j_bigArea++)
			{
				if (flag_bigArea)
					break;
				int up_left_distance = sqrt(pow(bigAreaContours[i_bigArea][j_bigArea].x - wordRect.x, 2) + pow(bigAreaContours[i_bigArea][j_bigArea].y - wordRect.y, 2));
				int down_right_distance = sqrt(pow(bigAreaContours[i_bigArea][j_bigArea].x - (wordRect.x + wordRect.width), 2) + pow(bigAreaContours[i_bigArea][j_bigArea].y - (wordRect.y + wordRect.height), 2));
				if (up_left_distance < 10 || down_right_distance < 10)
				{
					// �ò�����ͼ��bigAreaContours[i_bigArea]����ı���wordContours[i_word]Ӧ�ϲ��ɲ�����ͼ��
					flag_word = true;
					flag_bigArea = true;
					vector<vector<Point> > temp_contours;
					temp_contours.emplace_back(bigAreaContours[i_bigArea]);
					temp_contours.emplace_back(wordContours[i_word]);
					bigAreaContours[i_bigArea].clear();
					wordContours[i_word].clear();
					mergeContours(temp_contours, bigAreaContours);
				}
			}
		}
	}
	vector<vector<Point> >::const_iterator it_wordContours = wordContours.begin();
	while (it_wordContours != wordContours.end())
	{
		if (it_wordContours->size() == 0)
			it_wordContours = wordContours.erase(it_wordContours);
		else
		{
			it_wordContours++;
		}
	}

	vector<vector<Point> >::const_iterator it_bigAreaContours = bigAreaContours.begin();
	while (it_bigAreaContours != bigAreaContours.end())
	{
		if (it_bigAreaContours->size() == 0)
			it_bigAreaContours = bigAreaContours.erase(it_bigAreaContours);
		else
		{
			it_bigAreaContours++;
		}
	}
}

/*--------------------------------------------------------------
* Description: ��ȡ��ͨ���Ҷ�ͼ��ı�����ɫֵ
* Parameters: img����ͨ���Ҷ�ͼ��startX��2*2���ڵ���ʼX���꣬startY��2*2���ڵ���ʼY���꣬endX��2*2���ڵĽ���X���꣬endY��2*2���ڵĽ���Y����
* Return: ͼ��ı�����ɫֵ��ȡֵ��ΧΪ 0 �� 255
* writter: ����ΰ
---------------------------------------------------------------*/
int getGrayBgColor(Mat img, int startX, int startY, int endX, int endY) {
	if (img.type() != CV_8UC1)
	{
		cout << "ͼ�����ͱ���Ϊ CV_8UC1" << endl;
	}
	// ͼ��Ŀ�Ⱥ͸߶�
	int height = img.rows;
	int width = img.cols;
	// ��ʼ��������ɫֵ
	int bgColor = 0;

	// ���ѡ��һ�� 2*2 ���ڼ��㱳����ɫֵ
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> distribX(startX, endX - 2);
	std::uniform_int_distribution<> distribY(startY, endY - 2);
	int i = distribY(gen);
	int j = distribX(gen);
	int sum = 0;
	for (int x = 0; x < 2; x++) {
		for (int y = 0; y < 2; y++) {
			sum += img.at<uchar>(i + x, j + y);
		}
	}
	bgColor = sum / 4;

	// ���ر�����ɫֵ
	return bgColor;
}

/*--------------------------------------------------------------
* Description: ץȡ����ɫ��
* Parameters:
* Return:
* writter: ����ΰ
---------------------------------------------------------------*/
int catch_background_color_block(Mat inputCopy, std::vector<Rect> wordRects)
{
	int bgColor = -1;
	Mat image_Gray = convert2gray(inputCopy);
	for (int i = 0; i < wordRects.size() - 1; i++)
	{
		if (bgColor != -1)
			break;
		for (int j = i + 1; j < wordRects.size(); j++)
		{
			if (bgColor != -1)
				break;
			if (abs((wordRects[i].y + wordRects[i].height / 2) - (wordRects[j].y + wordRects[j].height / 2)) > 5) // �ж�wordRects[i]ˮƽ�����ұ��Ƿ������ֿ�
				continue;
			int right_distance = wordRects[j].x - (wordRects[i].x + wordRects[i].width);			
			cout << "right_distance = " << right_distance << endl;
			if (right_distance > 8 && right_distance < 60) // wordRects[i]�ұߴ������ֿ����м���ͼ����
			{
				bgColor = getGrayBgColor(image_Gray, wordRects[i].x + wordRects[i].width + 4, wordRects[i].y, wordRects[j].x - 4, wordRects[j].y + wordRects[j].height - 4);
			}
		}
	}
	cout << "��ȡ����ɫֵΪ = " << bgColor << endl;
	return bgColor;
}
/*--------------------------------------------------------------
* Description:ͼ����������
* Parameters: vector<Rect>&imageAreas ,vector<Rect>&textAreas.
* Return: null
---------------------------------------------------------------*/
void GraphTextIntegrate(int imgwidth,vector<Rect>&imageAreas, vector<Rect>&textAreas)
{
	if (imageAreas.size() == 0 || textAreas.size() == 0)
	{
		return;
	}
	// ͼ����������
	vector<Rect>::iterator it_image = imageAreas.begin();
	while (it_image != imageAreas.end())
	{
		vector<Rect>::iterator it_text = textAreas.begin();
		while (it_text != textAreas.end())
		{
			Rect imageArea = Rect(*it_image);
			Rect textArea = Rect(*it_text);
			Rect intersection = imageArea & textArea;
			if (textArea.width < (imgwidth*0.4)|| intersection.area() > 0)
			{
				
				if (intersection.area() > 0)
				{
					it_text = textAreas.erase(it_text);
					int startX = imageArea.x < textArea.x ? imageArea.x : textArea.x;
					int endX = (imageArea.x+imageArea.width) >(textArea.x + textArea.width) ? imageArea.x + imageArea.width : textArea.x + textArea.width;
					int startY = imageArea.y < textArea.y ? imageArea.y : textArea.y;
					int endY = (imageArea.y + imageArea.height) >(textArea.y + textArea.height) ? imageArea.y + imageArea.height : textArea.y + textArea.height;

					it_image->x = startX;
					it_image->y = startY;
					it_image->width = endX - startX;
					it_image->height = endY - startY;
				}
				else if (textArea.y > (imageArea.y - 20) && textArea.y < (imageArea.y + imageArea.height + 20))
				{
					if (abs(textArea.x + textArea.width - imageArea.x) < 10 || abs(imageArea.x + imageArea.width - textArea.x) < 10
						|| abs(textArea.x - imageArea.x) < 10 || abs(textArea.x + textArea.width - imageArea.x - imageArea.width) < 10
						|| (textArea.x > imageArea.x && (textArea.x+textArea.width)<(imageArea.x+imageArea.width)))
					{
						it_text = textAreas.erase(it_text);
						int startX = imageArea.x < textArea.x ? imageArea.x : textArea.x;
						int endX = (imageArea.x + imageArea.width) >(textArea.x + textArea.width) ? imageArea.x + imageArea.width : textArea.x + textArea.width;
						int startY = imageArea.y < textArea.y ? imageArea.y : textArea.y;
						int endY = (imageArea.y + imageArea.height) >(textArea.y + textArea.height) ? imageArea.y + imageArea.height : textArea.y + textArea.height;

						it_image->x = startX;
						it_image->y = startY;
						it_image->width = endX - startX;
						it_image->height = endY - startY;
					}
					else
					{
						it_text++;
					}
				}
				else
				{
					it_text++;
				}
			}
			else
			{
				it_text++;
			}
		}
		it_image++;
	}
	// ͼƬ��������
	it_image = imageAreas.begin();
	
	while (it_image != imageAreas.end())
	{
		Rect imageArea = Rect(*it_image);
		vector<Rect>::iterator next_image = it_image + 1;
		while (next_image != imageAreas.end())
		{
			Rect nextImageArea = Rect(*(next_image));
			Rect intersection = imageArea & nextImageArea;
			
			if (intersection.area() > 0)
			{
				next_image = imageAreas.erase(next_image);
				int startX = imageArea.x < nextImageArea.x ? imageArea.x : nextImageArea.x;
				int endX = (imageArea.x + imageArea.width) >(nextImageArea.x + nextImageArea.width) ? imageArea.x + imageArea.width : nextImageArea.x + nextImageArea.width;
				int startY = imageArea.y < nextImageArea.y ? imageArea.y : nextImageArea.y;
				int endY = (imageArea.y + imageArea.height) >(nextImageArea.y + nextImageArea.height) ? imageArea.y + imageArea.height : nextImageArea.y + nextImageArea.height;

				it_image->x = startX;
				it_image->y = startY;
				it_image->width = endX - startX;
				it_image->height = endY - startY;
			}
			else
			{
				next_image++;
			}
		}
		it_image++;
	}
}


/*--------------------------------------------------------------
* Description:�����������
* Parameters: vector<Rect>&tableAreas ,vector<Rect>&textAreas.
* Return: null
---------------------------------------------------------------*/
void tableTextIntegrate(int imgwidth, vector<Rect>&tableAreas, vector<Rect>&textAreas)
{
	if (tableAreas.size() == 0 || textAreas.size() == 0)
	{
		return;
	}
	// �����������
	vector<Rect>::iterator it_table = tableAreas.begin();
	while (it_table != tableAreas.end())
	{
		vector<Rect>::iterator it_text = textAreas.begin();
		while (it_text != textAreas.end())
		{
			Rect tableArea = Rect(*it_table);
			Rect textArea = Rect(*it_text);
			Rect intersection = tableArea & textArea;
			if (textArea.width < (imgwidth*0.8) || intersection.area() > 0)
			{

				if (intersection.area() > 0)
				{
					it_text = textAreas.erase(it_text);
					int startX = tableArea.x < textArea.x ? tableArea.x : textArea.x;
					int endX = (tableArea.x + tableArea.width) >(textArea.x + textArea.width) ? tableArea.x + tableArea.width : textArea.x + textArea.width;
					int startY = tableArea.y < textArea.y ? tableArea.y : textArea.y;
					int endY = (tableArea.y + tableArea.height) >(textArea.y + textArea.height) ? tableArea.y + tableArea.height : textArea.y + textArea.height;

					it_table->x = startX;
					it_table->y = startY;
					it_table->width = endX - startX;
					it_table->height = endY - startY;
				}
				else if (textArea.y > (tableArea.y - 20) && textArea.y < (tableArea.y + tableArea.height + 20))
				{
					if (abs(textArea.x + textArea.width - tableArea.x) < 50 || abs(tableArea.x + tableArea.width - textArea.x) < 50
						|| abs(textArea.x - tableArea.x) < 50 || abs(textArea.x + textArea.width - tableArea.x - tableArea.width) < 50
						|| (textArea.x > tableArea.x && (textArea.x + textArea.width)<(tableArea.x + tableArea.width)))
					{
						it_text = textAreas.erase(it_text);
						int startX = tableArea.x < textArea.x ? tableArea.x : textArea.x;
						int endX = (tableArea.x + tableArea.width) >(textArea.x + textArea.width) ? tableArea.x + tableArea.width : textArea.x + textArea.width;
						int startY = tableArea.y < textArea.y ? tableArea.y : textArea.y;
						int endY = (tableArea.y + tableArea.height) >(textArea.y + textArea.height) ? tableArea.y + tableArea.height : textArea.y + textArea.height;

						it_table->x = startX;
						it_table->y = startY;
						it_table->width = endX - startX;
						it_table->height = endY - startY;
					}
					else
					{
						it_text++;
					}
				}
				else if (abs(textArea.y + textArea.height - tableArea.y) < 10 )
				{
					it_text = textAreas.erase(it_text);
					int startX = tableArea.x < textArea.x ? tableArea.x : textArea.x;
					int endX = (tableArea.x + tableArea.width) >(textArea.x + textArea.width) ? tableArea.x + tableArea.width : textArea.x + textArea.width;
					int startY = tableArea.y < textArea.y ? tableArea.y : textArea.y;
					int endY = (tableArea.y + tableArea.height) >(textArea.y + textArea.height) ? tableArea.y + tableArea.height : textArea.y + textArea.height;

					it_table->x = startX;
					it_table->y = startY;
					it_table->width = endX - startX;
					it_table->height = endY - startY;
				}
				else
				{
					it_text++;
				}
			}
			else
			{
				it_text++;
			}
		}
		it_table++;
	}
	// �����������
	it_table = tableAreas.begin();

	while (it_table != tableAreas.end())
	{
		Rect tableArea = Rect(*it_table);
		vector<Rect>::iterator next_table = it_table + 1;
		while (next_table != tableAreas.end())
		{
			Rect nextTableArea = Rect(*(next_table));
			Rect intersection = tableArea & nextTableArea;

			if (intersection.area() > 0)
			{
				next_table = tableAreas.erase(next_table);
				int startX = tableArea.x < nextTableArea.x ? tableArea.x : nextTableArea.x;
				int endX = (tableArea.x + tableArea.width) >(nextTableArea.x + nextTableArea.width) ? tableArea.x + tableArea.width : nextTableArea.x + nextTableArea.width;
				int startY = tableArea.y < nextTableArea.y ? tableArea.y : nextTableArea.y;
				int endY = (tableArea.y + tableArea.height) >(nextTableArea.y + nextTableArea.height) ? tableArea.y + tableArea.height : nextTableArea.y + nextTableArea.height;

				it_table->x = startX;
				it_table->y = startY;
				it_table->width = endX - startX;
				it_table->height = endY - startY;
			}
			else
			{
				next_table++;
			}
		}
		it_table++;
	}
	
}


/*--------------------------------------------------------------
* Description: ͼƬ���
* Parameters: Mat���͵�ͼ��
* Return: ͼƬλ��������Ϣ����
--------------------------------------------------------------*/
std::vector<Rect> imageDetect(Mat dilateImage)
{
	std::vector<std::vector<Point>> contours;
	findContours(dilateImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	std::vector<std::vector<Point>> imageAreas;
	for (int i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		if (area > (dilateImage.cols * dilateImage.rows * 0.002))
		{
			imageAreas.push_back(contours[i]);
		}
	}
	std::vector<Rect> boundingRects;
	boundingRects.reserve(imageAreas.size());

	for (std::vector<std::vector<Point>>::const_iterator it = imageAreas.begin(); it != imageAreas.end(); it++)
	{
		Rect bRect = boundingRect(*it);
		if (bRect.height > 10)
		{
			boundingRects.push_back(bRect);
		}
	}
	return boundingRects;
}

/*--------------------------------------------------------------
* Description: ͼ��ü�
* Parameters: Mat���͵�ͼ�񡢾���������Ϣ
* Return: �ü����ͼ�񼯺�
--------------------------------------------------------------*/
std::vector<Mat> cropImage(Mat& input, std::vector<Rect>& cropRects)
{
	std::vector<Mat> cropImages;
	cropImages.reserve(cropRects.size());
	for (std::vector<Rect>::const_iterator itr = cropRects.begin(); itr != cropRects.end(); itr++)
	{
		Mat cutImage = Mat(input, *itr);
		cropImages.push_back(cutImage);
	}
	return cropImages;
}
/*--------------------------------------------------------------
* Description: ��һͼ��ü�
* Parameters: Mat���͵�ͼ�񡢾���������Ϣ
* Return: �ü����ͼ�񼯺�
--------------------------------------------------------------*/
Mat cropSingleImage(Mat& input, Rect cropRects)
{
	Mat cutImage = Mat(input, cropRects);
	return cutImage;
}

/*--------------------------------------------------------------
* Description:�ж������Ƿ�����
* Parameters:Rect &�� Rect &
* Return:������ڷ���true�����򷵻�false
--------------------------------------------------------------*/
bool isOverlap(const Rect &rc1, const Rect &rc2,int threshold)
{
	if (abs(rc1.x + rc1.width - rc2.x) < threshold || abs(rc2.x + rc2.width - rc1.x) < threshold)
	{
		if (abs(rc1.y - rc2.y) < 40)
		{
			return true;
		}
		else
		{
			return false;
		}
	}
	else
	{
		return false;
	}
}

/*--------------------------------------------------------------
* Description:������������
* Parameters:Rect :��׼����vector<Rect>:��������ļ���
* Return:����������ļ���
--------------------------------------------------------------*/
vector<Rect> getNearRegion(Rect r, vector<Rect> &rects,int threshold)
{
	vector<Rect> res;
	std::vector<Rect>::const_iterator itr = rects.begin();
	while (itr != rects.end())
	{
		Rect irect = Rect(*itr);
		if (isOverlap(r, irect,threshold))
		{
			// ������ڣ��ʹ���
			res.push_back(irect);
			itr = rects.erase(itr);
		}
		else
		{
			itr++;
		}
	}
	return res;
}

/*--------------------------------------------------------------
* Description: �����򼯺��л��Ԥѡ��
* Parameters: vector<Rect>:�������򼯺�
* Return: ���Ԥѡ��
--------------------------------------------------------------*/
vector<Rect> getRegionFromRects(vector<Rect> rects,int threshold)
{
	// ������ս��
	vector<Rect> nRect;
	while (rects.size() > 0)
	{
		// ��ʱ���
		vector<Rect> temp;
		// ���vector�е����һ��Ԫ��
		Rect last = rects.at(rects.size() - 1);
		//ɾ�����һ��Ԫ��
		rects.pop_back();
		// ����temp��
		temp.push_back(last);

		// ����һ������
		queue<Rect> q;
		// ��ʱtemp��ֻ��һ��Ԫ�أ����
		q.push(temp.at(0));

		// ���в��������
		while (q.size())
		{
			// ��¼��ͷԪ��
			Rect rect_head = q.front();
			//����
			q.pop();
			// ����rect_head��ʣ�µ�rects�������ڵĿ�,����rects��ɾ����Щ��
			vector<Rect> nearRegion_rects = getNearRegion(rect_head, rects,threshold);
			// ������ڣ������
			if (nearRegion_rects.size() > 0)
			{
				for (int i = 0; i < nearRegion_rects.size(); i++)
				{
					// ����Щ�ص��Ŀ鱣�浽temp�У�ÿ����ѭ��temp�оʹ����һ�����ڵĿ飬���ȡ��Щ�����Сx��y�����x��y�ͻ��һ����ס��Щ��Ĵ��
					temp.push_back(nearRegion_rects.at(i));
					// �������
					q.push(nearRegion_rects.at(i));
				}
			}
		}
		// ȫ�������꣬temp�оͱ�����һ�ѱ˴����ڵĿ飬�Ƚ����е�Rect��x��y����Сֵ�����ֵ��
		int min_x = 100000, max_x = 0, min_y = 100000, max_y = 0;
		int width = 0, maxHeight = 0;
		for (int i = 0; i < temp.size(); i++)
		{
			// ����Rect��x��yֵ
			int x = temp.at(i).x;
			int y = temp.at(i).y;
			int height = temp.at(i).height;

			if (x < min_x)
			{
				min_x = x;
			}
			if (x > max_x)
			{
				max_x = x;
				width = temp.at(i).width;
			}
			if (y < min_y)
			{
				min_y = y;
			}
			if (y > max_y)
			{
				max_y = y;
			}
			if (height > maxHeight)
			{
				maxHeight = height;
			}
		}
		// ���rect
		Rect big;
		big.x = min_x;
		big.y = min_y;
		big.width = max_x + width - min_x;
		big.height = maxHeight;
		// �����ս��������
		nRect.push_back(big);
	}
	return nRect;
}

/*--------------------------------------------------------------
* Description: ���߷���
* Parameters: Mat:���߼���
* Return: �����ĺ��߼�����
--------------------------------------------------------------*/
vector<vector<Rect>> groupingRowLines(vector<Rect>rowlines)
{
	vector<vector<Rect>> groupedLines;
	std::vector<Rect>::const_iterator itr = rowlines.begin();
	while (itr != rowlines.end())
	{
		vector<Rect> group;
		group.push_back(*itr);
		std::vector<Rect>::const_iterator itr_next = itr + 1;
		while (itr_next != rowlines.end())
		{
			if (abs(itr->x - itr_next->x) < 1 && abs(itr->width - itr_next->width) < (itr->width * 0.05) && abs(itr->y - itr_next->y) > 20000)
			//if (abs(itr->x - itr_next->x) < 20 && abs(itr->width - itr_next->width) < (itr->width * 0.05)&& abs(itr->y - itr_next->y) > 20)
			{
				// ֱ�߰�������ͶӰ����
				Rect table;
				table.x = min(itr->x, itr_next->x);
				table.y = min(itr->y, itr_next->y);
				table.height = abs(itr->y - itr_next->y);
				table.width = max(itr->width, itr_next->width);
				//Mat imageRegion = cropImage(digitimage, table);

				group.push_back(*itr_next);
				itr_next = rowlines.erase(itr_next);
			}
			else
			{
				break;
			}
		}
		if (group.size() > 1)
		{
			groupedLines.push_back(group);
		}
		itr++;
	}
	return groupedLines;
}

/*--------------------------------------------------------------
* Description: ���߷���
* Parameters: Mat:���߼���
* Return: �����ĺ��߼�����
--------------------------------------------------------------*/
vector<vector<Rect>> groupingColLines(vector<Rect>collines)
{
	vector<vector<Rect>> groupedLines;
	std::vector<Rect>::const_iterator itr = collines.begin();
	while (itr != collines.end())
	{
		vector<Rect> group;
		group.push_back(*itr);
		std::vector<Rect>::const_iterator itr_next = itr + 1;
		while (itr_next != collines.end())
		{
			if (abs(itr->y - itr_next->y) < 20 && abs(itr->height - itr_next->height) < 20)
			{
				group.push_back(*itr_next);
				itr_next = collines.erase(itr_next);
			}
			else
				itr_next++;
		}
		if (group.size() > 1)
		{
			groupedLines.push_back(group);
		}
		itr++;
	}
	return groupedLines;
}

/*--------------------------------------------------------------
* Description: Json�ļ���д��
* Parameters: �ļ�·�����ı����������Ϣ��ͼ�����������Ϣ
* Return: NULL
--------------------------------------------------------------*/
void writeJson(string fileName, int height, int width, int depth, vector<Rect> textRects, vector<Rect> imageRects, vector<Rect> tableRects, string jsonPath,int ratio)
{
	// ���ڵ�
	Json::Value root;
	int pos = fileName.find_last_of('\\');
	int lastPos = fileName.find_last_of('.');
	string name(fileName.substr(pos + 1, -1));
	string filename(fileName.substr(pos + 1, lastPos - pos - 1));
	// ��װ���ڵ�����
	root["fileName"] = name;
	Json::Value image_size;
	image_size["height"] = height;
	image_size["width"] = width;
	image_size["depth"] = depth;
	root["size"] = image_size;

	// �ӽڵ�
	Json::Value textRectangle;
	Json::Value imageRectangle;
	Json::Value tableRectangle;

	// �����ӽڵ�
	int index = 0;
	for (std::vector<Rect>::const_iterator it = textRects.begin(); it != textRects.end(); it++)
	{
		Json::Value partner;
		partner["id"] = Json::Value(index);
		Json::Value point_temp;
		point_temp["x"] = Json::Value(it->tl().x * ratio);
		point_temp["y"] = Json::Value(it->tl().y * ratio);
		partner["leftTop"] = point_temp;
		point_temp["x"] = Json::Value(it->tl().x * 1);
		point_temp["y"] = Json::Value((it->tl().y + it->height) * ratio);
		partner["leftDown"] = point_temp;
		point_temp["x"] = Json::Value((it->tl().x + it->width) * ratio);
		point_temp["y"] = Json::Value((it->tl().y + it->height) * ratio);
		partner["rightDown"] = point_temp;
		point_temp["x"] = Json::Value((it->tl().x + it->width) * ratio);
		point_temp["y"] = Json::Value((it->tl().y) * ratio);
		partner["rightTop"] = point_temp;
		// �����ӽڵ�ҵ��ӽڵ���
		textRectangle.append(partner);
		// textRectangle["textRectangle"+to_string(index)] = Json::Value(partner);
		index++;
	}
	root["textAreas"] = Json::Value(textRectangle);

	int index_2 = 0;
	for (std::vector<Rect>::const_iterator it = imageRects.begin(); it != imageRects.end(); it++)
	{
		Json::Value partner;
		partner["id"] = Json::Value(index_2);
		Json::Value point_temp;
		point_temp["x"] = Json::Value(it->tl().x * ratio);
		point_temp["y"] = Json::Value(it->tl().y * ratio);
		partner["leftTop"] = point_temp;
		point_temp["x"] = Json::Value(it->tl().x * ratio);
		point_temp["y"] = Json::Value((it->tl().y + it->height) * ratio);
		partner["leftDown"] = point_temp;
		point_temp["x"] = Json::Value((it->tl().x + it->width) * ratio);
		point_temp["y"] = Json::Value((it->tl().y + it->height) * ratio);
		partner["rightDown"] = point_temp;
		point_temp["x"] = Json::Value((it->tl().x + it->width) * ratio);
		point_temp["y"] = Json::Value(it->tl().y * ratio);
		partner["rightTop"] = point_temp;
		// �����ӽڵ�ҵ��ӽڵ���
		imageRectangle.append(partner);
		// imageRectangle["imageRectangle" + to_string(index_2)] = Json::Value(partner);
		index_2++;
	}
	root["imageAreas"] = Json::Value(imageRectangle);

	int index_3 = 0;
	for (std::vector<Rect>::const_iterator it = tableRects.begin(); it != tableRects.end(); it++)
	{
		Json::Value partner;
		partner["id"] = Json::Value(index_3);
		Json::Value point_temp;
		point_temp["x"] = Json::Value(it->tl().x * ratio);
		point_temp["y"] = Json::Value(it->tl().y * ratio);
		partner["leftTop"] = point_temp;
		point_temp["x"] = Json::Value(it->tl().x * ratio);
		point_temp["y"] = Json::Value((it->tl().y + it->height) * ratio);
		partner["leftDown"] = point_temp;
		point_temp["x"] = Json::Value((it->tl().x + it->width) * ratio);
		point_temp["y"] = Json::Value((it->tl().y + it->height) * ratio);
		partner["rightDown"] = point_temp;
		point_temp["x"] = Json::Value((it->tl().x + it->width) * ratio);
		point_temp["y"] = Json::Value(it->tl().y * ratio);
		partner["rightTop"] = point_temp;
		// �����ӽڵ�ҵ��ӽڵ���
		tableRectangle.append(partner);
		//tableRectangle["tableRectangle" + to_string(index)] = Json::Value(partner);
		index_3++;
	}
	root["tableAreas"] = Json::Value(tableRectangle);

	// ��json���ݣ�������ʽ��������ļ�
	Json::StreamWriterBuilder writerBuilder;
	if (_access(jsonPath.c_str(), 0) == -1)
	{
		_mkdir(jsonPath.c_str());
	}
	ofstream os(jsonPath + filename + ".json");
	// ofstream os("F:/Layout analysis dataset/companydata/noTableResultJson/" + name + ".json");
	std::unique_ptr<Json::StreamWriter> writer(writerBuilder.newStreamWriter());
	writer->write(root, &os);
	os.close();
}
/*--------------------------------------------------------------
* Description: ͼ����ǿ�㷨1
* Parameters: Mat:����ͼ��
* Return: NULL
--------------------------------------------------------------*/
cv::Mat contrastStretch1(cv::Mat srcImage)
{
	cv::Mat resultImage = srcImage.clone();
	int nRows = resultImage.rows;
	int nCols = resultImage.cols;
	// ͼ���������ж�
	if (resultImage.isContinuous()) {
		nCols = nCols * nRows;
		nRows = 1;
	}

	// ����ͼ��������Сֵ
	double pixMin, pixMax;
	cv::minMaxLoc(resultImage, &pixMin, &pixMax);
	//std::cout << "min_a=" << pixMin << " max_b=" << pixMax << std::endl;
	// �Աȶ�����ӳ��
	for (int j = 0; j < nRows; j++) {
		uchar *pDataMat = resultImage.ptr<uchar>(j);
		for (int i = 0; i < nCols; i++) {
			pDataMat[i] = (pDataMat[i] - pixMin) * 255 / (pixMax - pixMin);
		}
	}
	return resultImage;
}
/*--------------------------------------------------------------
* Description: ͼ����ǿ�㷨2
* Parameters: Mat:����ͼ��
* Return: NULL
--------------------------------------------------------------*/
void contrastStretch2(cv::Mat &srcImage)
{
	// ����ͼ��������Сֵ
	double pixMin, pixMax;
	cv::minMaxLoc(srcImage, &pixMin, &pixMax);
	//create lut table
	cv::Mat lut(1, 256, CV_8U);
	for (int i = 0; i < 256; i++) {
		if (i < pixMin) lut.at<uchar>(i) = 0;
		else if (i > pixMax) lut.at<uchar>(i) = 255;
		else lut.at<uchar>(i) = static_cast<uchar>(255.0*(i - pixMin) / (pixMax - pixMin) + 0.5);
	}
	// apply lut
	LUT(srcImage, lut, srcImage);
}

void findPeak(Mat srcImage, vector<int>& resultVec)
{
	Mat verMat;
	Mat resMat = srcImage.clone();// ��¡���󣬷�������ռ�
								  // ��ֵ������
	int thresh = 130;
	int threshType = 0;
	// Ԥ�����ֵ
	const int maxVal = 255;
	// �̶���ֵ������
	threshold(srcImage, srcImage, thresh, maxVal, threshType);
	srcImage.convertTo(srcImage, CV_32FC1);
	// ���㴹ֱͶӰ
	reduce(srcImage, verMat, 0, cv::REDUCE_SUM);// �ϲ����У����������������ܺͣ�ת���ɾ���
	cout << verMat << endl;
	// �������ַ��ź���
	float* iptr = ((float*)verMat.data) + 1;
	// ����һ������tempVec
	vector<int> tempVec(verMat.cols - 1, 0);
	// �Բ���������з����ж���
	for (int i = 0; i < verMat.cols - 1; ++i, ++iptr)
	{
		if (*(iptr + 1) - *iptr > 0)
			tempVec[i] = 1;
		else if (*(iptr + 1) - *iptr < 0)
			tempVec[i] = -1;
		else
			tempVec[i] = 0;
	}
	// �Է��ź������б���
	for (int i = tempVec.size() - 1; i >= 0; i--)
	{
		if (tempVec[i] == 0 && i == tempVec.size() - 1)
		{
			tempVec[i] = 1;
		}
		else if (tempVec[i] == 0)
		{
			if (tempVec[i + 1] >= 0)
				tempVec[i] = 1;
			else
				tempVec[i] = -1;
		}
	}
	// �����ж����
	for (vector<int>::size_type i = 0; i != tempVec.size() - 1; i++)
	{
		if (tempVec[i + 1] - tempVec[i] == -2)
			// ��Ϊi+1ΪͶӰ����S��һ������λ�õ㣬�ѵ���뵽resultVec��
			resultVec.push_back(i + 1);
	}
	// �������λ��
	for (int i = 0; i < resultVec.size(); i++)
	{
		cout << resultVec[i] << '\t';
		// ����λ��Ϊ255
		for (int ii = 0; ii < resMat.rows; ++ii)
		{
			resMat.at<uchar>(ii, resultVec[i]) = 255;
		}
	}
	//imshow("resMat", resMat);
}

/*--------------------------------------------------------------
* Description:�����ڰ�ɫ���ر���
* Parameters: vector<Rect>&imageAreas
* Return: Percentage of white pixels
---------------------------------------------------------------*/
double whitepixelsCount(Mat image,Rect imageAreas)
{
	double count = imageAreas.width * imageAreas.height;
	int whitepixels = 0;
	double percentage = 0.0;
	Mat areas = Mat(image, imageAreas)*255;
	//namedWindow("��ֵ��", 2);
	//imshow("��ֵ��", areas);
	//waitKey();

	for (int row = 0; row < areas.rows; row++)
	{
		for (int col = 0; col < areas.cols; col++ )
		{
			if(areas.at<uchar>(row, col) == 255)
			{
				whitepixels++;
			}
		}
	}
	percentage = whitepixels*1.0 / count;
	cout << "��ɫ���أ�" << whitepixels << endl;
	cout << "���أ�" << count << endl;
	cout << "��ɫ���ذٷֱȣ�" << percentage << endl;
	return percentage;
}



/*------------------------������----------------------------*/
int valueRLSA1(Mat imageDel, Mat srcImage, std::vector<std::vector<Point>> contours)
{
	int s = 0;
	int sum = 0, count = 0;
	int minY = imageDel.rows;
	for (std::vector<std::vector<Point>>::const_iterator it = contours.begin(); it != contours.end(); it++)
	{
		Rect bRect = boundingRect(*it);
		if (bRect.area() < (srcImage.rows * srcImage.cols * 0.1))
		{
			if (bRect.y < minY)
			{
				minY = bRect.y;
			}
			sum += bRect.height;
			count += 1;
		}
		s += 1;
	}

	int threshold_RLSA;
	if (count != 0) {
		threshold_RLSA = (sum / count);
		//return threshold_RLSA * 1.2;
		return threshold_RLSA;
	}
	else
	{
		return 0;
	}
}


int valueRLSA2(Mat imageDel, Mat srcImage, std::vector<std::vector<Point>> contours, int threshold_RLSA1)
{
	int t = 0;
	int sum = 0, count = 0;
	for (std::vector<std::vector<Point>>::const_iterator it1 = contours.begin(); it1 != contours.end(); it1++)
	{
		std::vector<std::vector<Point>>::const_iterator it2 = it1 + 1;
		if (it2 != contours.end())
		{
			Rect bRect1 = boundingRect(*it1);
			Rect bRect2 = boundingRect(*it2);
			if (bRect1.area() < (srcImage.rows * srcImage.cols * 0.1))
			{
				t = fabs(bRect2.x - bRect1.x + bRect1.width);
				if (t < threshold_RLSA1 * 3)
				{
					sum += t;
					count += 1;
				}
			}
		}
	}

	int threshold_RLSA;
	if (count != 0) {
		threshold_RLSA = (sum / count);
		//return threshold_RLSA * 1.2857;
		return threshold_RLSA;
	}
	else
	{
		return 0;
	}
}


/*------------------------������----------------------------*/


//void main()
//{
//	string dirPath = "F:\\Layout analysis dataset\\Newdata\\data\\";
//	//string dirPath = "F:\\Layout analysis dataset\\companydata\\1\\";
//	//string dirPath = "F:\\Layout analysis dataset\\companydata\\noTableData\\";
//
//	vector<String> file_vec;
//	glob(dirPath + "*.jpg",file_vec,false);
//	int index = 0;
//	for (string fileName : file_vec)
//	{
//		clock_t start, end;
//		start = clock();
//		Mat imageResize;
//		Mat input = loadImage(fileName);
//		Mat roteInput = ScannedImageRectify(input);
//		resize(roteInput, imageResize, Size(0, 0), 0.5, 0.5, INTER_AREA);
//		
//		Rect roi(5, 5, imageResize.cols - 10, imageResize.rows - 10);
//		Mat cropImage = cropSingleImage(imageResize,roi);
//
//		//Mat inputCopy = input.clone();
//		//�������ȡ
//		//Mat grayImage = convert2gray(imageResize);
//		vector<Rect>rowLine;
//		vector<Vec4i>colLine;
//		findTableByLSD(cropImage,rowLine,colLine);
//		//��ñ��Ԥ��߶�
//		vector<double> tableHeight = getTableHeight(colLine);
//		vector<Rect> TableRects = getTableRects(rowLine, tableHeight);
//
//		//Mat result = getBlock(roteInput, /*textRects, processedImageRects,*/TableRects);
//		//imwrite("F:/Layout analysis dataset/Newdata/tableResult/" + to_string(index) + ".jpg", result);
//		/*vector<Vec4i> linesHorizonal;
//		vector<Vec4i> linesVertical;
//		GetLines(input, linesHorizonal, linesVertical);*/
//		//Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
//		//erode(roteInput, inputErode, element, Point(-1, -1), 1, 0);
//		//// ת�Ҷ�ͼ��
//		//Mat grayImage = convert2gray(inputErode);
//		//contrastStretch2(grayImage);
//		//resize(grayImage,imageResize,Size(0,0), 0.5,0.5, INTER_AREA);
//		// ��ֵ��
//		/*namedWindow("gray", 2);
//		cv::imshow("gray", grayImage);
//		cv::Mat resultImage = contrastStretch1(grayImage);
//		
//		namedWindow("result1", 2);
//		namedWindow("result2", 2);
//		cv::imshow("result1", resultImage);
//		cv::imshow("result2", grayImage);
//		cv::waitKey(0);*/
//		//cv::Mat resultImage = contrastStretch1(grayImage);
//		//Mat binaryImage = binaryzation(resultImage);
//		//imwrite("F:/Layout analysis dataset/companydata/noTableResult/" + to_string(index) + ".jpg", binaryImage);
//		// ��һ��
//		//Mat digitImage = normalization(imageResize);
//		//���������
//		//std::vector<Rect> tableRects = findTableByLSD(roteInput);
//		//Mat tableDel = plotRect(digitImage, tableRects);
//		///*Mat reTableDel = tableDel * 255;
//		//namedWindow("tableDel", 2);
//		//imshow("tableDel", reTableDel);*/
//		////ͼƬ������
//		//std::vector<Rect> processedImageRects = imageDetect(tableDel);
//		////ͼƬ����ȥ��
//		//Mat imageDel = plotRect(tableDel, processedImageRects);
//		///*Mat reImageDel = imageDel * 255;
//		//namedWindow("imageDel", 2);
//		//imshow("imageDel", reImageDel);*/
//		//// RSLA
//		//Mat afterSmoothHor = lengthSmoothHor(imageDel, imageResize.cols * 0.3);
//		//Mat afterSmoothVer = lengthSmoothVer(imageDel, imageResize.rows * 0.001);
//		//// ������
//		//Mat afterSmooth = afterSmoothVer & afterSmoothHor;
//		//Mat dilateImage = doDilation(afterSmooth, 5);
//
//		//Mat afterSmooth2show = 255 * afterSmooth;
//		//Mat dilateImage2show = 255 * dilateImage;
//		std::vector<Rect> componentRects = textDetect(dilateImage);
//		std::vector<Rect> textRects = getRegionFromRects(componentRects);
//		Mat result = getBlock(input, textRects, processedImageRects,tableRects);
//		writeJson(fileName,componentRects,processedImageRects);
//		imwrite("F:/Layout analysis dataset/Newdata/result/"+ to_string(index) + ".jpg", result);
//		//imwrite("F:/Layout analysis dataset/companydata/lsdResult/" + to_string(index) + ".jpg", result);
//		////imwrite("F:/Layout analysis dataset/companydata/noTableResultByMerge/" + to_string(index) + ".jpg", result);
//		////imwrite("F:/Layout analysis dataset/companydata/noTableResult/" + to_string(index) + ".jpg", result);
//		cout << "done!" << index << endl;
//		end = clock();
//		printf("%d\n", end - start);
//		index++;
//		waitKey(0);
//	}
//}

//int main()
//{
//	string srcPath = "F:\\Layout analysis dataset\\Newdata\\problem\\";
//	string dstPath = srcPath + "result\\";
//
//	// Check whether source path exists
//	if ((_access(srcPath.data(), 0)) != 0)
//	{
//		printf("Input path does not exist! \n\n");
//		system("pause");
//		return -1;
//	}
//
//	vector<String> file_vec;
//	glob(srcPath + "*.jpg", file_vec, false);
//	int index = 0;
//	int totalTime = 0;
//
//
//	// Create destination directory if it does not exist
//	if ((_access(dstPath.data(), 0)) != 0)
//	{
//		_mkdir(dstPath.data());
//	}
//
//	for (string fileName : file_vec)
//	{
//		clock_t start, end;
//		start = clock();
//		Mat imageResize;
//		Mat input = loadImage(fileName);
//		//Mat roteInput = ScannedImageRectify(input);
//		resize(input, imageResize, Size(0, 0), 0.5, 0.5, INTER_AREA);
//		Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
//		erode(imageResize, imageResize, element, Point(-1, -1), 1, 0);
//		/*Rect roi(5, 5, imageResize.cols - 10, imageResize.rows - 10);
//		Mat cropImage = cropSingleImage(imageResize, roi);*/
//	    Mat grayImage = convert2gray(imageResize);
//		Mat binaryImage = binaryzation(grayImage);
//		namedWindow("��ֵ��", 2);
//		imshow("��ֵ��", binaryImage);
//		waitKey(0);
//
//		Mat digitImage = normalization(binaryImage);
//		// ���ֱ��
//		vector<Rect> allRowLines, allColLines;
//		findTableByLSD(imageResize, allRowLines, allColLines);
//		// �߶εĽ�һ���ϲ�
//		vector<Rect> rowLines = getLineFromShortLines(allRowLines, 1);
//		vector<Rect> colLines = getLineFromShortLines(allColLines, 2);
//
//		// ��ýϴ���ͨ��
//		vector<Rect>bigArea = bigAreaDetect(digitImage);
//		vector<Rect> TableRects;
//		std::vector<Rect>::const_iterator itr = bigArea.begin();
//		while (itr != bigArea.end())
//		{
//			Rect rect = Rect(*itr);
//			if (ifTableLineInside(rect,rowLines) || ifTableLineInside(rect, colLines))
//			{
//				TableRects.push_back(rect);
//				itr = bigArea.erase(itr);
//			}
//			else {
//				itr++;
//			}
//		}
//		// �������ȥ��
//		Mat tableDel = plotRect(digitImage, TableRects);
//		// ͼƬ����ȥ��
//		Mat imageDel = plotRect(tableDel, bigArea);
//		// RSLA
//		Mat afterSmoothHor = lengthSmoothHor(imageDel, imageResize.cols * 0.3);
//		Mat afterSmoothVer = lengthSmoothVer(imageDel, imageResize.rows * 0.001);
//		// ������
//		Mat afterSmooth = afterSmoothVer & afterSmoothHor;
//		Mat dilateImage = doDilation(afterSmooth, 5);
//
//		std::vector<Rect> componentRects = textDetect(dilateImage);
//		std::vector<Rect> textRects = getRegionFromRects(componentRects);
//		Mat result = getBlock(input, textRects, bigArea, TableRects);
//		writeJson(fileName, textRects, bigArea,TableRects);
//		namedWindow("result", 2);
//		imshow("result", result);
//		int pos = fileName.find_last_of('\\');
//		int lastPos = fileName.find_last_of('.');
//		string name(fileName.substr(pos + 1, lastPos - pos - 1));
//		imwrite(dstPath + name + ".jpg", result);
//		cout << "done!" << index << endl;
//		end = clock();
//		printf("%d\n", end - start);
//		index++;
//		waitKey(0);
//	}
//}


//�������ھ�ֵ integral��Ϊ����ͼ
float fastMean(cv::Mat& integral, int x, int y, int window)
{

	int min_y = std::max(0, y - window / 2);
	int max_y = std::min(integral.rows - 1, y + window / 2);
	int min_x = std::max(0, x - window / 2);
	int max_x = std::min(integral.cols - 1, x + window / 2);

	int topright = integral.at<int>(max_y, max_x);
	int botleft = integral.at<int>(min_y, min_x);
	int topleft = integral.at<int>(max_y, min_x);
	int botright = integral.at<int>(min_y, max_x);

	float res = (float)((topright + botleft - topleft - botright) / (float)((max_y - min_y) *(max_x - min_x)));

	return res;
}




Mat Sauvola(cv::Mat& inpImg, int window, float k)
{
	Mat resImg = inpImg.clone();
	cv::Mat integral;
	int nYOffSet = 3;
	int nXOffSet = 3;
	cv::integral(inpImg, integral);  //�������ͼ��
	for (int y = 0; y < inpImg.rows; y += nYOffSet)
	{
		for (int x = 0; x < inpImg.cols; x += nXOffSet)
		{

			float fmean = fastMean(integral, x, y, window); float fthreshold = (float)(fmean*(1.0 - k));

			int nNextY = y + nYOffSet;
			int nNextX = x + nXOffSet;
			int nCurY = y;
			while (nCurY < nNextY && nCurY < inpImg.rows)
			{
				int nCurX = x;
				while (nCurX < nNextX && nCurX < inpImg.cols)
				{
					uchar val = inpImg.at<uchar>(nCurY, nCurX) < fthreshold;
					resImg.at<uchar>(nCurY, nCurX) = (val == 0 ? 0 : 255);
					nCurX++;
				}
				nCurY++;
			}

		}
	}

	return resImg;
}

/*--------------------------------------------------------------
* Description:����y����������������
* Parameters:vector<Rect>:Ԥѡ����ļ���
* Return:NULL
--------------------------------------------------------------*/
void reckRankY(vector<Rect>& srcRects)
{
	sort(srcRects.begin(), srcRects.end(), [](const Rect& a, const Rect& b)
	{
		return a.y > b.y;
	});
}


vector<Rect> TDRAR(Mat srcImage)
{
	Mat uimage;
	medianBlur(srcImage, uimage, 3);
	Mat grayImage = convert2gray(srcImage);

	Mat binaryImage;
	threshold(grayImage, binaryImage, 0, 255, THRESH_BINARY | THRESH_OTSU);
	Mat digitImage = normalization(binaryImage);
	// RSLA
	Mat afterSmoothHor = lengthSmoothHor(digitImage, srcImage.cols * 0.06);
	//Mat afterSmoothVer = lengthSmoothVer(digitImage, image.rows * 0.02);
	// ������
	//Mat afterSmooth = afterSmoothVer & afterSmoothHor;
	//Mat dilateImage = doDilation(afterSmooth, 3);
	vector<Rect> componentRects = textDetect(afterSmoothHor);
	vector<Rect> textRects = getRegionFromRects(componentRects, srcImage.cols*0.05);
	// ɸѡ�϶����ֿ�
	textRectSelect(textRects, srcImage.cols);
	// ��Y������
	reckRankY(textRects);
	// �����Ͻǵ�Y���������
	vector<vector<Rect>>groupedTextRects = groupingRectByY(textRects);
	//����ϲ�
	std::vector<vector<Rect>>::iterator itr = groupedTextRects.begin();
	std::vector<vector<Rect>>::iterator itr_next = groupedTextRects.begin();
	int val = groupedTextRects.at(0).at(0).y;
	while (itr_next != groupedTextRects.end())
	{
		itr_next = itr + 1;
		if (itr_next != groupedTextRects.end())
		{
			vector<Rect>oneGroup = vector<Rect>(*itr_next);
			if (abs(val - oneGroup.at(0).y)<srcImage.rows*0.05)
			{
				val = oneGroup.at(0).y;
				itr->insert(itr->end(), itr_next->begin(), itr_next->end());
				itr_next = groupedTextRects.erase(itr_next);
			}
			else
			{
				itr = itr_next;
				vector<Rect>group = vector<Rect>(*itr);
				val = group.at(0).y;
			}
		}
	}
	vector<Rect>tableRect;

	for (int groupNum = 0; groupNum<groupedTextRects.size(); groupNum++)
	{
		if (groupedTextRects.at(groupNum).size()<1)
		{
			continue;
		}
		int min_x = 10000, min_y = 10000, max_x = 0, max_y = 0;
		for (int textRectNum = 0; textRectNum<groupedTextRects.at(groupNum).size(); textRectNum++)
		{
			min_x = (min_x>groupedTextRects.at(groupNum).at(textRectNum).x) ? groupedTextRects.at(groupNum).at(textRectNum).x : min_x;
			min_y = (min_y>groupedTextRects.at(groupNum).at(textRectNum).y) ? groupedTextRects.at(groupNum).at(textRectNum).y : min_y;
			max_x = (max_x<(groupedTextRects.at(groupNum).at(textRectNum).x + groupedTextRects.at(groupNum).at(textRectNum).width)) ? groupedTextRects.at(groupNum).at(textRectNum).x + groupedTextRects.at(groupNum).at(textRectNum).width : max_x;
			max_y = (max_y<(groupedTextRects.at(groupNum).at(textRectNum).y + groupedTextRects.at(groupNum).at(textRectNum).height)) ? groupedTextRects.at(groupNum).at(textRectNum).y + groupedTextRects.at(groupNum).at(textRectNum).height : max_y;
		}
		Rect proTableRect;
		proTableRect.x = min_x;
		proTableRect.y = min_y;
		proTableRect.height = max_y - min_y;
		proTableRect.width = max_x - min_x;
		if (proTableRect.width > srcImage.cols*0.5)
		{
			tableRect.push_back(proTableRect);
		}
	}
	//Mat result = srcImage.clone();
	//getBlock(result,textRects,cv::Scalar(0,255,0));
	//getBlock(result, tableRect, cv::Scalar(0, 0, 255));
	return tableRect;
}

void textRectSelect(vector<Rect> &textRect, int img_width)
{
	sort(textRect.begin(), textRect.end(), [&](Rect a, Rect b)
	{
		return a.width < b.width;
	});
	int pos = textRect.size();
	for (int i = 0; i<textRect.size(); i++)
	{
		if (textRect.at(i).width>(img_width*0.4))
		{
			pos = i;
			break;
		}
	}
	textRect.resize(pos);
}

/*--------------------------------------------------------------
* Description:grouping Rect
* Parameters: vector<Rect>Rect
* Return: the group of Rect
---------------------------------------------------------------*/
vector<vector<Rect>> groupingRectByY(vector<Rect>rect)
{
	vector<vector<Rect>> groupedRect;
	std::vector<Rect>::const_iterator itr = rect.begin();
	while (itr != rect.end())
	{
		vector<Rect> group;
		group.push_back(*itr);
		std::vector<Rect>::const_iterator itr_next = itr + 1;
		while (itr_next != rect.end())
		{
			if (abs(itr->y - itr_next->y) < 20)
			{
				group.push_back(*itr_next);
				itr_next = rect.erase(itr_next);
			}
			else
				itr_next++;
		}
		if (group.size() > 0)
		{
			groupedRect.push_back(group);
		}
		itr++;
	}
	return groupedRect;
}

void showImage(String windowName, Mat image, int showTime)
{
	namedWindow(windowName, 2);
	imshow(windowName, image);
	waitKey(showTime);
}