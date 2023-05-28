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
		cout << "打开失败" << endl;
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
* Description: 图像二值化
* Parameters: Mat类型的单通道图像
* Return: Mat类型的二值化图像，当通道不为一时返回空Mat 
--------------------------------------------------------------*/
Mat binaryzation(Mat grayImage, int value)
{
	if (grayImage.channels() != 1)
	{
		cout << "必须为单通道图像" << endl;
		return Mat();
	}
	Mat binaryImage;

	//binaryImage = Sauvola(grayImage, 9, 0.5);

	threshold(grayImage, binaryImage, value, 255, THRESH_BINARY); //BINARY模式处理为仅有两个值的二值图像
	Scalar meanofImg,dev;
	
	
	meanStdDev(binaryImage, meanofImg, dev); //求出均值和标准差
	if (meanofImg.val[0] <= 40 )
	{	
		//获取类型信息
		const type_info& expInfo = typeid(meanofImg.val[0]);
		cout << "hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh\n" << endl;   // 无输出
		cout << expInfo.name() << " | " << expInfo.raw_name() << " | " << expInfo.hash_code() << endl;
		//又一次阈值处理？？
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
* Description: 像素值进行归一化
* Parameters: Mat类型的灰度图像
* Return: Mat类型的归一化图像
--------------------------------------------------------------*/
Mat normalization(Mat binaryImage)
{
	Mat temp = ~binaryImage;
	//showPic(temp,"temp");    //取反
	Mat digitImage = (1 / 255.0) * temp;  //置0-1
	//showPic(digitImage, "normal");
	return digitImage;
}

/*--------------------------------------------------------------
* Description: 水平方向的游程平滑
* Parameters: Mat类型的图像、游程阈值
* Return: 水平游程平滑后的图像
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
* Description: 垂直方向的游程平滑
* Parameters: Mat类型的图像、游程阈值
* Return: 垂直游程平滑后的图像
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
* Description: 膨胀操作
* Parameters: Mat类型的图像、膨胀次数
* Return: 膨胀后的图像
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
* Description: 区域删除
* Parameters: Mat类型的图像、矩形区域信息
* Return: 矩形去除后的图像
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
* Description: 矩形区域绘制
* Parameters: Mat类型的图像
* Return: 绘制后的图像
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
// 重载
Mat getBlock(Mat& input, std::vector<Rect> rects)
{
	Mat blockImage = input.clone();
	for (std::vector<Rect>::const_iterator it = rects.begin(); it != rects.end(); it++)
	{
		rectangle(blockImage, it->tl(), it->br(), cv::Scalar(0, 0, 255), 3, 1, 0);
	}
	return blockImage;
}
// 重载2022_12_01	--by lww
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
* Description: 文字检测
* Parameters: 游程平滑后的图像和原始图像
* Return: 所需文字位置区域信息集合
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

	//预先设置矩形框容器
	std::vector<Rect> boundingRects;
	boundingRects.reserve(contours.size());

	for (std::vector<std::vector<Point>>::const_iterator it = contours.begin(); it != contours.end(); it++)
	{
		Rect bRect = boundingRect(*it);
		//矩形框  12<width<2000, height>8, 面积<图像的80%  则加入
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
	//预先设置矩形框容器
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
		//矩形框  12<width<2000, height>8, 面积<图像的80%  则加入
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
	cout << "图边文字合并程序运行时间=" << double(end - start) / CLOCKS_PER_SEC << "s" << endl;*/

	for (std::vector<std::vector<Point>>::const_iterator it_fix = contours.begin(); it_fix != contours.end(); it_fix++)
	{
		Rect bRect = boundingRect(*it_fix);
		boundingRects.push_back(bRect);
	}
	return boundingRects;
}

std::vector<Rect> textDetect(Mat dilateImage, Mat input, vector<vector<Point> >& bigAreaContours, int flag)
{
	//预先设置矩形框容器
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
			//矩形框  12<width<2000, height>8, 面积<图像的80%  则加入
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
		cout << "图边文字合并程序运行时间=" << double(end - start) / CLOCKS_PER_SEC << "s" << endl;*/

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
			//矩形框  12<width<2300, height>8, 面积<图像的80%  则加入
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
		cout << "图边文字合并程序运行时间=" << double(end - start) / CLOCKS_PER_SEC << "s" << endl;*/

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
* Description: 显示图像
* Parameters: Mat showpic, String namepi
* Return:  null
--------------------------------------------------------------*/
void showPic(Mat showpic, String namepic) {
	namedWindow(namepic, WINDOW_NORMAL);
	imshow(namepic, showpic);
}


/*--------------------------------------------------------------
* Description:直线检测
* Parameters: Mat image, Mat grayImage
* Return:vector<Vec4f>
* Writter: 李千红
---------------------------------------------------------------*/
vector<Vec4f> ls_detect(Mat inputCopy, Mat grayImage) {
	Mat image = inputCopy.clone();
	//创建对象
	Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_ADV);

	vector<Vec4f> lines_std, draw_lines;
	Mat drawnLines(image);

	//检测直线
	ls->detect(grayImage, lines_std);
	//筛选直线
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

	cout << "图上共有" << draw_lines.size() << "条直线。" << endl;
	//imwrite("D:\\VS_Projects\\LearnforCpp\\data\\draw_lines.jpg", drawnLines);
	//showPic(drawnLines, "lsd");

	return draw_lines;
}


/*--------------------------------------------------------------
* Description: 基于垂直投影的文字区域判断——波峰辅助判断
* Parameters: ValArry[]数组
* Return: 布尔类型判断
* writer:李千红
--------------------------------------------------------------*/

bool wave_judge(int* ValArry, int width, int height) {
	int up_sum = 0;//一般爬坡记录
	int down_sum = 0; //一般下坡记录
	int acc_up_sum = 0; //精确爬坡记录
	int acc_down_sum = 0; //一般下坡记录
	int* min_p = new int[width]; //坡的最低点（像素值）集合
	int* max_p = new int[width]; //坡的最高点集合
	memset(min_p, 0, width * 4);
	memset(max_p, 0, width * 4);
	int pmax = 0; //下标。
	int pmin = 0;
	int up_flag = 0;//当前爬坡状态标志
	int down_flag = 0;
	int start_flag = 0;// 1为先爬坡，2为先下坡
	for (int i = 0; i < width; i++) {
		////越界判断
		//if (ValArry[i + 1]) {
		//	continue;
		//}
		//刚开始的平缓阶段
		if ((up_flag == 0) && (down_flag == 0) && (ValArry[i] == ValArry[i + 1])) {
			max_p[pmax] = ValArry[i];
			min_p[pmin] = ValArry[i];
			continue;
		}
		//开始爬坡，且未下过坡，跳跃时刻
		if ((up_flag == 0) && (down_flag == 0) && (ValArry[i] < ValArry[i + 1])) {
			up_flag = 1;
			down_flag = 0;
			start_flag = 1; //先爬坡的开始标志
			min_p[pmin] = ValArry[i];//最开始是上坡，坡的最低点先+1
			pmin = pmin + 1;
			continue;
		}
		//开始下坡，且未上过坡，跳跃时刻
		if ((up_flag == 0) && (down_flag == 0) && (ValArry[i] > ValArry[i + 1])) {
			down_flag = 1;
			up_flag = 0;
			start_flag = 2; //先下坡的开始标志
			max_p[pmax] = ValArry[i];//最开始是上坡，坡的最低点先+1
			pmax = pmax + 1;
			continue;
		}
		//爬坡中，跳跃时刻
		if ((up_flag == 1) && (down_flag == 0) && (ValArry[i] < ValArry[i + 1])) {
			max_p[pmax] = ValArry[i];
			continue;
		}
		//爬坡中，平地阶段
		if ((up_flag == 1) && (down_flag == 0) && (ValArry[i] == ValArry[i + 1])) {
			max_p[pmax] = ValArry[i];
			continue;
		}
		//开始下坡,且已爬过坡,跳跃时刻
		if ((down_flag == 0) && (up_flag == 1) && (ValArry[i] > ValArry[i + 1])) {
			down_flag = 1;
			up_flag = 0;
			max_p[pmax] = ValArry[i];//记录下坡前一刻的最高点，坡的最高点+1
			////去除不重要数对
			//if (start_flag == 1) {
			//	if (abs(max_p[pmax] - min_p[pmin]) < 10) {
			//		//左边无用+1
			//		useless_up = useless_up + 1;
			//	}
			//}
			//else if (start_flag == 2) {
			//	if (abs(max_p[pmax - 1] - min_p[pmin]) < 10) {
			//		//右边无用+1
			//		useless_down = useless_down + 1;
			//	}
			//}
			min_p[pmin] = ValArry[i];
			pmax = pmax + 1;
			continue;
		}
		//下坡中,跳跃时刻
		if ((down_flag == 1) && (up_flag == 0) && (ValArry[i] > ValArry[i + 1])) {
			min_p[pmin] = ValArry[i];
			continue;
		}
		//下坡中，平地阶段
		if ((down_flag == 1) && (up_flag == 0) && (ValArry[i] == ValArry[i + 1])) {
			min_p[pmin] = ValArry[i];
			continue;
		}
		//开始爬坡，且已下过坡，跳跃时刻
		if ((up_flag == 0) && (down_flag == 1) && (ValArry[i] < ValArry[i + 1])) {
			up_flag = 1;
			down_flag = 0;
			min_p[pmin] = ValArry[i];//记录上坡前一刻的最低点，坡的最低点+1

			pmin = pmin + 1;
			max_p[pmax] = ValArry[i];
			continue;
		}
	}
	//波峰情况判断
	if (start_flag == 1) {
		//一开始先爬坡的情况
		for (int i = 0; i < pmin; i++) {
			//左边
			if (abs(min_p[i] - max_p[i]) > height * 0.65) {
				acc_up_sum = acc_up_sum + 1;
			}
			else if (abs(min_p[i] - max_p[i]) > height * 0.25) {
				up_sum = up_sum + 1;
			}
			//右边
			if (abs(min_p[i + 1] - max_p[i]) > height * 0.65) {
				acc_down_sum = acc_down_sum + 1;
			}
			else if (abs(min_p[i + 1] - max_p[i]) > height * 0.25) {
				down_sum = down_sum + 1;
			}
		}
	}
	else if (start_flag == 2) {
		//一开始先下坡的情况
		for (int i = 0; i < pmax; i++) {
			//右边
			if (abs(max_p[i] - min_p[i]) > height * 0.65) {
				acc_down_sum = acc_down_sum + 1;
			}
			else if (abs(max_p[i] - min_p[i]) > height * 0.25) {
				down_sum = down_sum + 1;
			}
			//左边
			if (abs(max_p[i + 1] - min_p[i]) > height * 0.65) {
				acc_up_sum = acc_up_sum + 1;
			}
			else if (abs(max_p[i + 1] - min_p[i]) > height * 0.25) {
				up_sum = up_sum + 1;
			}
		}
	}
	//针对较短单词at\to等，做精确判断
	if ((acc_up_sum >= 1) && (acc_down_sum >= 2)) {
		return true;
	}
	//针对354的A
	else if ((acc_up_sum >= 1) && (acc_down_sum >= 1) && (up_sum >= 1) && (down_sum >= 1)) {
		return true;
	}
	//如果有一定次数的爬坡或下坡记录
	else if ((up_sum >= 4) || (down_sum >= 4)) {
		return true;//则为文字区域
	}
	else {
		return false;
	}
}

/*--------------------------------------------------------------
* Description: 基于垂直投影的文字区域判断——波峰辅助判断
* Parameters: ValArry[]数组
* Return: 布尔类型判断
* writer:李千红
--------------------------------------------------------------*/
bool wordJudge(int* ValArry, int width, int height) {
	int up_sum = 0;//一般爬坡记录
	int down_sum = 0; //一般下坡记录
	int acc_up_sum = 0; //精确爬坡记录
	int acc_down_sum = 0; //一般下坡记录
	int* min_p = new int[width]; //坡的最低点（像素值）集合
	int* max_p = new int[width]; //坡的最高点集合
	memset(min_p, 0, width * 4);
	memset(max_p, 0, width * 4);
	int pmax = 0; //下标。
	int pmin = 0;
	int up_flag = 0;//当前爬坡状态标志
	int down_flag = 0;
	int start_flag = 0;// 1为先爬坡，2为先下坡
	for (int i = 0; i < width; i++) {
		////越界判断
		//if (ValArry[i + 1]) {
		//	continue;
		//}
		//刚开始的平缓阶段
		if ((up_flag == 0) && (down_flag == 0) && (ValArry[i] == ValArry[i + 1])) {
			max_p[pmax] = ValArry[i];
			min_p[pmin] = ValArry[i];
			continue;
		}
		//开始爬坡，且未下过坡，跳跃时刻
		if ((up_flag == 0) && (down_flag == 0) && (ValArry[i] < ValArry[i + 1])) {
			up_flag = 1;
			down_flag = 0;
			start_flag = 1; //先爬坡的开始标志
			min_p[pmin] = ValArry[i];//最开始是上坡，坡的最低点先+1
			pmin = pmin + 1;
			continue;
		}
		//开始下坡，且未上过坡，跳跃时刻
		if ((up_flag == 0) && (down_flag == 0) && (ValArry[i] > ValArry[i + 1])) {
			down_flag = 1;
			up_flag = 0;
			start_flag = 2; //先下坡的开始标志
			max_p[pmax] = ValArry[i];//最开始是上坡，坡的最低点先+1
			pmax = pmax + 1;
			continue;
		}
		//爬坡中，跳跃时刻
		if ((up_flag == 1) && (down_flag == 0) && (ValArry[i] < ValArry[i + 1])) {
			max_p[pmax] = ValArry[i];
			continue;
		}
		//爬坡中，平地阶段
		if ((up_flag == 1) && (down_flag == 0) && (ValArry[i] == ValArry[i + 1])) {
			max_p[pmax] = ValArry[i];
			continue;
		}
		//开始下坡,且已爬过坡,跳跃时刻
		if ((down_flag == 0) && (up_flag == 1) && (ValArry[i] > ValArry[i + 1])) {
			down_flag = 1;
			up_flag = 0;
			max_p[pmax] = ValArry[i];//记录下坡前一刻的最高点，坡的最高点+1
			min_p[pmin] = ValArry[i];
			pmax = pmax + 1;
			continue;
		}
		//下坡中,跳跃时刻
		if ((down_flag == 1) && (up_flag == 0) && (ValArry[i] > ValArry[i + 1])) {
			min_p[pmin] = ValArry[i];
			continue;
		}
		//下坡中，平地阶段
		if ((down_flag == 1) && (up_flag == 0) && (ValArry[i] == ValArry[i + 1])) {
			min_p[pmin] = ValArry[i];
			continue;
		}
		//开始爬坡，且已下过坡，跳跃时刻
		if ((up_flag == 0) && (down_flag == 1) && (ValArry[i] < ValArry[i + 1])) {
			up_flag = 1;
			down_flag = 0;
			min_p[pmin] = ValArry[i];//记录上坡前一刻的最低点，坡的最低点+1

			pmin = pmin + 1;
			max_p[pmax] = ValArry[i];
			continue;
		}
	}
	//波峰情况判断
	if (start_flag == 1) {
		//一开始先爬坡的情况
		for (int i = 0; i < pmin; i++) {
			//左边
			if (abs(min_p[i] - max_p[i]) > height * 0.65) {
				acc_up_sum = acc_up_sum + 1;
			}
			else if (abs(min_p[i] - max_p[i]) > height * 0.25) {
				up_sum = up_sum + 1;
			}
			//右边
			if (abs(min_p[i + 1] - max_p[i]) > height * 0.65) {
				acc_down_sum = acc_down_sum + 1;
			}
			else if (abs(min_p[i + 1] - max_p[i]) > height * 0.25) {
				down_sum = down_sum + 1;
			}
		}
	}
	else if (start_flag == 2) {
		//一开始先下坡的情况
		for (int i = 0; i < pmax; i++) {
			//右边
			if (abs(max_p[i] - min_p[i]) > height * 0.65) {
				acc_down_sum = acc_down_sum + 1;
			}
			else if (abs(max_p[i] - min_p[i]) > height * 0.25) {
				down_sum = down_sum + 1;
			}
			//左边
			if (abs(max_p[i + 1] - min_p[i]) > height * 0.65) {
				acc_up_sum = acc_up_sum + 1;
			}
			else if (abs(max_p[i + 1] - min_p[i]) > height * 0.25) {
				up_sum = up_sum + 1;
			}
		}
	}
	//针对较短单词at\to等，做精确判断
	if ((acc_up_sum >= 1) && (acc_down_sum >= 2)) {
		return true;
	}
	//针对354的A
	else if ((acc_up_sum >= 1) && (acc_down_sum >= 1) && (up_sum >= 1) && (down_sum >= 1)) {
		return true;
	}
	//如果有一定次数的爬坡或下坡记录
	else if ((up_sum >= 4) || (down_sum >= 4)) {
		return true;//则为文字区域
	}
	else {
		return false;
	}
}

/*--------------------------------------------------------------
* Description: 基于垂直投影的文字区域判断
* Parameters: Mat类型的图像，矩形框区域参数
* Return: 布尔类型判断
* writer:李千红
--------------------------------------------------------------*/
bool textlinesJudge(Mat normalImage, int pic_height, int pic_width) {
	int width = normalImage.cols;
	int height = normalImage.rows;
	int perPixelValue = 0;

	//设置投影背景布
	Mat verpaint(normalImage.size(), CV_8UC1, Scalar(255));
	//showPic(verpaint, "verpaint_unpaint");
	//创建用于存储每列白色像素个数的数组
	int* ValArry = new int[width];
	//数组初始化
	memset(ValArry, 0, width * 4);

	int flag = 0;
	//找观察投影
	if ((width == 0) && (height == 0)) {
		flag = 1;
	}
	//记录每一列白色像素点数量
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
		////绘制投影
		//for (int i = 0; i < ValArry[col]; i++) {
		//	verpaint.at<uchar>(height - i - 1, col) = 0;
		//}
	}
	int w_sum = 0;//白色像素点和
	int val_col = 0;//记录Valarry非0列
	//绘制投影
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < ValArry[i]; j++) {
			verpaint.at<uchar>(height - j - 1, i) = 0;
		}
		if (ValArry[i] > 0) {
			val_col = val_col + 1;
		}
		w_sum = w_sum + ValArry[i];//记录白色像素点和
	}
	if (flag == 1) {
		cout << "val_col:" << val_col << endl;
		//showPic(verpaint, "verpaint");
		//imwrite("D:\\VS_Projects\\Layout Analysis\\data\\pic276\\verpaint_45.jpg", verpaint);
		//exit(0);
	}

	//showPic(verpaint, "verpaint");
	//imwrite("D:\\VS_Projects\\Layout Analysis\\data\\zaodian_verpro.jpg", verpaint);

	//抛弃极小投影
	if (w_sum < (width * height) * 0.11) {
		cout << "极小投影" << endl;
		return false;
	}
	/*if (w_sum > (width * height) * 0.8) {
		cout << "极大投影" << endl;
		return false;
	}*/
	//抛弃小方块文本区域
	if ((width < 20) && (height < 20) || ((height < 15) && (width < 25))) {
		cout << "小方块区域" << endl;
		return false;
	}
	//抛弃错误的大文本区域
	if (height > pic_height * 0.6) {
		cout << "错误大文本区域" << endl;
		return false;
	}
	//抛弃独峰投影
	if (val_col < width * 0.5) {
		cout << "独峰投影" << endl;
		return false;
	}
	bool result = wordJudge(ValArry, width, height);
	return result;
}


/*--------------------------------------------------------------
* Description: 基于水平投影的文字区域判断
* Parameters: Mat类型的图像，矩形框区域参数
* Return: 布尔类型判断
* writer:李千红
--------------------------------------------------------------*/
bool pic_textlineJudge(Mat normalImage, int pic_height, int pic_width) {
	int width = normalImage.cols;
	int height = normalImage.rows;
	int perPixelValue = 0;

	//设置投影背景布
	Mat verpaint(normalImage.size(), CV_8UC1, Scalar(255));
	//showPic(verpaint, "verpaint_unpaint");
	//创建用于存储每列白色像素个数的数组
	int* ValArry = new int[height];
	//数组初始化
	memset(ValArry, 0, height * 4);

	int flag = 0;
	////找观察投影
	//if ((width == 0) && (height == 0)) {
	//	flag = 1;
	//}
	//记录每一行白色像素点数量
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
	int w_sum = 0;//白色像素点和
	int val_row = 0;//记录Valarry非0行
	//绘制投影
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < ValArry[i]; j++) {
			verpaint.at<uchar>(i, j) = 0;
		}
		if (ValArry[i] > 0) {
			val_row = val_row + 1;
		}
		w_sum = w_sum + ValArry[i];//记录白色像素点和
	}
	//if (flag == 1) {
	//	cout << "val_row:" << val_row << endl;
	//	showPic(verpaint, "verpaint");
	//	//imwrite("D:\\VS_Projects\\Layout Analysis\\data\\pic276\\verpaint_45.jpg", verpaint);
	//	//exit(0);
	//}

	//showPic(verpaint, "verpaint");
	//imwrite("D:\\VS_Projects\\Layout Analysis\\data\\zaodian_verpro.jpg", verpaint);

	//抛弃极小投影
	if (w_sum < (width * height) * 0.11) {
		cout << "极小投影" << endl;
		return false;
	}
	if (w_sum > (width * height) * 0.5) {
		cout << "极大投影" << endl;
		return false;
	}
	//抛弃大小方块文本区域
	if ((width > pic_width * 0.5) && (height > pic_height * 0.4) || ((height < 15) && (width < 25))) {
		cout << "大小方块区域" << endl;
		return false;
	}
	////抛弃错误的大文本区域
	//if (height > pic_height * 0.6) {
	//	cout << "错误大文本区域" << endl;
	//	return false;
	//}
	//抛弃独峰投影
	if (val_row < height * 0.5) {
		cout << "独峰投影" << endl;
		return false;
	}

	//////存储每个黑色区域最大值
	int* max_b_index = new int[width];
	int max = 0;
	memset(max_b_index, 0, width * 4);

	//判断是否存在空白区域
	int* b_index = new int[width];//黑色区域开始和结束索引
	int b_end = 0; //黑色区域结束
	int* w_index = new int[width];//白色区域开始和结束索引
	int w_end = 0;  //白色区域结束
	bool white = false; //进入白色区域标志
	bool black = false;  //进入黑色区域标志
	memset(b_index, 0, width * 4);//初始化
	memset(w_index, 0, width * 4);

	int K_num = 5;

	int w = 0; //遍历下标，并且记录白色区域数组个数格式为[start,end]
	int b = 0;
	for (int i = 0; i < width; i++) {
		//设置了阈值，小于这个阈值，当作空白区间
		//开始进入白色区域，且未经历黑色区域
		if ((ValArry[i] <= K_num) && (white == false) && (black == false)) {
			w_index[w] = i;
			w = w + 1;
			white = true;

		}
		//开始进入白色区域，且经历过黑色区域
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
			max = max + 1;//已取完黑色区域的最大值，准备进入下一个

		}
		//白色区域循环
		else if ((ValArry[i] <= 10) && (white == true)) {
			w_end = i;
			continue;
		}
		//开始进入黑色区域，且未经历白色区域
		else if ((ValArry[i] > K_num) && (black == false) && (white == false)) {
			b_index[b++] = i;
			black = true;
			if (ValArry[i] > max_b_index[max]) {
				max_b_index[max] = ValArry[i];
			}
		}
		//开始进入黑色区域，且经历过白色区域
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
		//黑色区域循环
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

	//如果全白
	if (b == 0) {
		return false;
	}
	//如果全黑
	if (w == 0) {
		return false;
	}

	//若黑色区域个数稀少，噪点警惕
	if (b <= 6) {
		cout << "噪点警惕" << endl;
		int max_b = max_b_index[0];
		int min_b = max_b_index[0];
		//黑色区域的最大值比较并抛弃差值大的文本区域
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

	//进入下列程序的要求： 非全白非全黑，且黑色区域个数较多
	int wb_connect_flag = 0;
	int bw_connect_flag = 0;
	//判断黑白区域是否相接
	if (w_index[0] < b_index[0]) {
		//先白后黑
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
		//先黑后白
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
		//cout << "这是一个文字区域" << endl;
		return true;
		if ((width < 20) || (height < 20)) {
			return false;
		}
	}
	else {
		//cout << "这是一个噪点区域" << endl;
		return false;
	}

}

/*--------------------------------------------------------------
* Description: 文字区域排序——排序函数
* Parameters : 排序容器，文字区域集合，部分排序文本信息
* Return : 部分已排序vector<Rect>
* writer : 李千红
--------------------------------------------------------------*/
void wordSort(vector<Rect>& sorted_wordRects, vector<Rect> word_componentRects, vector<Rect>& rowFirst_wordRects, int start, int end, int sum) {

	//设置异常警告，start下标必定大于end下标
	if (start < end) {
		cout << "error!" << endl;
	}


	//行中单一元素情况
	if (sum == 1) {
		sorted_wordRects.push_back(word_componentRects[start]);
		rowFirst_wordRects.push_back(word_componentRects[start]);
	}

	//非单一情况，部分区域冒泡排序，从小到大
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
	//添加到已排序文本区域容器
	for (int i = start; i > (start - sum); i--) {
		sorted_wordRects.push_back(word_componentRects[i]);
		if (i == start) {
			rowFirst_wordRects.push_back(word_componentRects[i]);
		}
	}
}


/*--------------------------------------------------------------
* Description: 文字区域排序
* Parameters: vector<Rect>文字区域
* Return: vector<Rect>排序文字区域
* writer:李千红
--------------------------------------------------------------*/
vector<Rect> textlinesSorting(Mat final_picture, vector<Rect> word_componentRects, vector<Rect>& rowFirst_wordRects, vector<int>& row_num) {
	int Rectsize = word_componentRects.size();
	int x = 0;
	int y = 0;
	int x_next = 0;
	int y_next = 0;
	int start_flag = 0;//同一行起始点下标
	int end_flag = 0;
	int sum_y = 0; //行中轮廓数量
	int ing_flag = 0; //行遍历标志 
	//行中文字区域数量记录


	vector<Rect> sorted_wordRects;


	for (int i = Rectsize - 1; i >= 0; i--) {
		x = word_componentRects[i].x;
		y = word_componentRects[i].y;
		if (i - 1 >= 0) {
			// 2023.05.18判断条件由 > 改为 >=
			x_next = word_componentRects[i - 1].x;
			y_next = word_componentRects[i - 1].y;
		}
		else {
			x_next = 0;
			y_next = 0;
		}
		//行高不相似，不在行遍历中，即一行一元素情况
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
		//行高不相似，跳出行遍历,进行区间排序
		else if ((abs(y - y_next) > 3) && (ing_flag == 1)) {
			end_flag = i;
			sum_y = sum_y + 1;
			wordSort(sorted_wordRects, word_componentRects, rowFirst_wordRects, start_flag, end_flag, sum_y);
			row_num.push_back(sum_y);
			//row_index = row_index + 1;
			sum_y = 0;//行中轮廓数清零
			ing_flag = 0;//行遍历状态清零
			continue;
		}
		//刚开始进入行遍历
		else if ((abs(y - y_next) <= 3) && (ing_flag == 0)) {
			start_flag = i;
			ing_flag = 1;//进入行遍历
			sum_y = sum_y + 1;
			continue;
		}
		//在行遍历中
		else if ((abs(y - y_next) <= 3) && (ing_flag == 1)) {
			end_flag = i;
			sum_y = sum_y + 1;
			continue;
		}
	}

	//cout << "sizeof(row_num)" << row_index << endl;
	////查看行情况
	//for (int i = 0; i < row_index; i++) {
	//	cout << row_num[i] << endl;
	//}
	//行首文本集合个数
	//int size_rowFirst = rowFirst_wordRects.size();
	//cout << "size_rowFrist" << size_rowFirst << endl;

	//绘制行首文本轮廓
	//for (int i = 0; i < rowFirst_wordRects.size(); i++)
	//{
	//	rectangle(final_picture, Point(rowFirst_wordRects[i].x, rowFirst_wordRects[i].y), Point(rowFirst_wordRects[i].x + rowFirst_wordRects[i].width, rowFirst_wordRects[i].y + rowFirst_wordRects[i].height), cv::Scalar(0, 255, 255), 5, 1, 0);
	//	//showPic(final_picture, "final_Picture");
	//}


	return sorted_wordRects;
}

/*--------------------------------------------------------------
* Description: 后景像素块提取
* Parameters:
* Return: 像素值列表
* writer:李千红
--------------------------------------------------------------*/
vector<int> getBgPixelblock(Mat grayImage, vector<Rect> sorted_wordRects, vector<Rect> rowFirst_wordRects, vector<int> row_num) {
	int row_index = row_num.size();
	vector<int> pixelArry;
	int size_rowFirst = rowFirst_wordRects.size();  //行首文本集合个数
	int catch_blocks = 0;	//后景像素块个数
	int catch_width = 2;	//提取像素块长宽
	int catch_height = 2;
	int row_sum = 0;//计数，第几个文本区域
	vector<int> tempArry;//临时存储提取框内像素点
	for (int i = 0; i < row_index; i++) {
		int pix_x = rowFirst_wordRects[i].x + rowFirst_wordRects[i].width - 20;
		int pix_y = rowFirst_wordRects[i].y + rowFirst_wordRects[i].height / 2;
		int pix_value = 0;
		int value_sum = 0;
		int ave_value = 0;
		int min_value = 0;
		int delnum = 0;//废弃值数量标志
		row_sum = row_num[i] + row_sum;
		tempArry.clear();//清空临时数组
		if (catch_blocks == 4) {
			break;//提取足够多的像素块就跳出循环
		}
		//根据行中区域个数进行提取
		if (row_num[i] == 1) {
			for (int u = 0; u < catch_width; u++) {
				for (int v = 0; v < catch_height; v++) {
					pix_value = grayImage.at<uchar>(pix_y + v, pix_x + u);
					tempArry.push_back(pix_value); //临时数组存储
					//temp = temp + 1;
					//cout << "pix_value = " << pix_value << endl;
					if (pix_value == 255) {
						// 丢弃图片置白区域
						delnum = delnum + 1;
						continue;
					}
					value_sum = pix_value + value_sum;
					//cout << "value_sum = " << value_sum << endl;
					//cout << "delnum = " << delnum << endl;
				}
			}
			//ave_value = value_sum / (catch_width * catch_height - delnum);//取平均值
			//取捕获框内最小像素值
			/*for (int w = 0; w < (catch_width * catch_height); w++) // 2023.05.19疑似有问题，暂时将范围界定修改*/
			for (int w = 0; w < tempArry.size(); w++)
			{
				min_value = tempArry[w];
				if (min_value > tempArry[w]) {
					min_value = tempArry[w];
				}
			}
			//cout << catch_width * catch_height - delnum << endl;
			catch_blocks = catch_blocks + 1;//提取块数加1
			pixelArry.push_back(min_value); //存入最小像素值
			value_sum = 0;
			////cout << "提取一个像素块，值为：" << min_value << " .提取起始点为：" << pix_x << " " << pix_y << endl;
			//pixel_index = pixel_index + 1;
		}

		else if (row_num[i] >= 2) {
			for (int w = 1; w < 5; w++) {
				if (catch_blocks == 4) {
					break;
				}
				int number_row = row_sum - row_num[i] + w;  //当前行的第w个文本区域
				pix_x = rowFirst_wordRects[number_row - 1].x + rowFirst_wordRects[number_row - 1].width;
				pix_y = rowFirst_wordRects[number_row - 1].y;
				int pix_next_x = sorted_wordRects[number_row].x;
				int pix_next_y = sorted_wordRects[number_row].y;
				if ((pix_x - pix_next_x) > 10) {
					for (int u = 0; u < catch_width; u++) {
						for (int v = 0; v < catch_height; v++) {
							pix_value = grayImage.at<uchar>(pix_y + v, pix_x + u);
							tempArry.push_back(pix_value); //临时数组存储
							//temp = temp + 1;
							//cout << "pix_value" << pix_value << endl;
							if (pix_value == 255) {//丢弃图片置白区域
								delnum = delnum + 1;
								continue;
							}
							value_sum = value_sum + pix_value;
							//cout << "value_sum" << value_sum << endl;
						}
					}
					//ave_value = value_sum / (catch_width * catch_height - delnum);
					//取捕获框内最小像素值
					/*for (int w = 0; w < (catch_width * catch_height); w++) // 与上方问题一致*/
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
					//cout << "提取一个像素块，值为：" << min_value << " .提取起始点为：" << pix_x << " " << pix_y << endl;
					//pixel_index = pixel_index + 1;
				}
				else {
					continue;
				}
			}
		}
	}
	//cout << "像素块总数为：" << pixel_index << endl;
	//for (int i = 0; i < 20; i++) {
	//	cout << pixelArry[i] << endl;
	//}
	return pixelArry;
}



/*--------------------------------------------------------------
* Description:局部投影法切割
* Parameters: Rect rect, Mat binaryImage
* Return: vector<vector<int> >
* Writter: 梁文伟
---------------------------------------------------------------*/
vector<Rect> Local_Range_Projection(Rect rect, Mat binaryImage, string type)
{
	vector<Rect> wordRect;
	if (type == "vertical")
	{
		Mat src = binaryImage(rect);

		//step1. 计算竖直投影白色点数量
		int w = src.cols;
		int h = src.rows;
		vector<int> project_val_arry;
		int per_pixel_value;
		for (int j = 0; j < w; j++)//列
		{
			Mat j_im = src.col(j);
			int num = countNonZero(j_im);//当前列的像素值中不为0的数目
			if (num < h)
			{
				/*当前列有前景内容黑色像素*/
				project_val_arry.push_back(h-num);
			}
			else
			{
				/*当前列全部是背景白色像素*/
				project_val_arry.push_back(0);
			}
			//project_val_arry.push_back(num);
		}

		//显示
		Mat hist_im(h, w, CV_8UC1, Scalar(255));
		for (int i = 0; i < w; i++)
		{
			for (int j = 0; j < project_val_arry[i]; j++)
			{
				hist_im.ptr<unsigned char>(h - 1 - j)[i] = 0;
			}
		}
		//imshow("project", hist_im);
		

		//step2. 字符分割
		int start_Index = 0;
		int end_Index = 0;
		bool in_Block = false;//是否遍历到了字符区内
		int k = 0;
		for (int i = 0; i < w; ++i)
		{
			if (!in_Block && project_val_arry[i] > 20)//进入字符区了
			{
				in_Block = true;
				start_Index = i;
				//cout << "startIndex is " << startIndex << endl;
			}
			else if (project_val_arry[i] <= 20 && in_Block)//进入空白区了
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

		//step1. 计算水平投影白色点数量
		int w = src.cols;
		int h = src.rows;
		vector<int> project_val_arry;
		int per_pixel_value;
		for (int i = 0; i < h; i++)//行
		{
			Mat i_img = src.row(i);
			int num = countNonZero(i_img); //当前行的像素值中不为0的数目
			if (num < w)
			{
				/*当前行有前景内容黑色像素*/
				project_val_arry.push_back(w-num);
			}
			else
			{
				/*当前行全部是背景白色像素*/
				project_val_arry.push_back(0);
			}
		}

		//显示
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

		//step2. 字符分割
		int start_Index = 0;
		int end_Index = 0;
		bool in_Block = false;//是否遍历到了字符区内
		int k = 0;
		for (int i = 0; i < h; ++i)
		{
			if (!in_Block && project_val_arry[i] > 5)//进入字符区了
			{
				in_Block = true;
				start_Index = i;
				//cout << "startIndex is " << startIndex << endl;
			}
			else if (project_val_arry[i] <= 5 && in_Block)//进入空白区了
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
* Description: 局部游程平滑
* Parameters: Mat binaryImage
* Return:
* Writter: 梁文伟
--------------------------------------------------------------*/
void local_RLSA(Mat binaryImage, Mat input, vector<vector<Point> > bigAreaContours, vector<Rect>& wordRects)
{
	Mat reuniteImage = normalization(binaryImage);
	Mat digitImage = InterferenceImage(reuniteImage); //边缘处理过的图像
	Mat RLSAImage = digitImage;
	int threshold_RLSA1 = valueRLSA1(RLSAImage, digitImage, bigAreaContours);
	int threshold_RLSA2 = valueRLSA2(RLSAImage, digitImage, bigAreaContours, threshold_RLSA1);

	cout << "局部二值化结果：" << endl;
	cout << "threshold_RLSA1 :" << threshold_RLSA1 << "threshold_RLSA2:" << threshold_RLSA2 << endl;


	Mat afterSmoothHor = lengthSmoothHor(RLSAImage, threshold_RLSA2 * 0.25);
	Mat afterSmoothVer = lengthSmoothVer(RLSAImage, threshold_RLSA1);
	// 并操作
	Mat afterSmooth = afterSmoothVer & afterSmoothHor;
	Mat afterSmooth2Show_1 = afterSmooth * 255;
	//showPic(afterSmooth2Show_1,"afterSmooth2Show_1");

	//又一次水平平滑
	//afterSmooth = lengthSmoothHor(afterSmooth, threshold_RLSA2 * 0.95);
	//Mat afterSmooth2Show_2 = afterSmooth * 255;
	//showPic(afterSmooth2Show_2,"afterSmooth2Show_2");

	Mat afterSmoothHor2Show = afterSmoothHor * 255;
	//showPic(afterSmoothHor2Show, "afterSmoothHor2Show");
	Mat afterSmoothVer2Show = afterSmoothVer * 255;
	//showPic(afterSmoothVer2Show, "afterSmoothVer2Show");


	//文字检测
	std::vector<Rect> componentRects = textDetect(afterSmooth, input, bigAreaContours, 2);
	
	for (int i = 0; i < componentRects.size(); i++)
	{
		wordRects.push_back(componentRects[i]);
	}
	return;
}

/*--------------------------------------------------------------
* Description: 查漏补缺
* Parameters: Mat binaryImage, vector<vector<Point> >& bigAreaContours, vector<Rect>& wordRects
* Return:
* Writter: 梁文伟
--------------------------------------------------------------*/
void Checking_for_gaps(Mat binaryImage, Mat input, vector<vector<Point> >& bigAreaContours, vector<Rect> bigArea, vector<Rect>& wordRects)
{
	Mat digitImage = normalization(binaryImage);
	// 对未进行分类的区域处理
	for (int i = 0; i < bigAreaContours.size(); i++)
	{
		// 将现有图像区域掩盖
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
		// 将现有文字区域掩盖
		rectangle(binaryImage, wordRects[i], CV_RGB(255, 255, 255), -1);
		rectangle(input, wordRects[i], CV_RGB(255, 255, 255), -1);
	}
	//showPic(binaryImage, "masked_binaryImage");


	// 局部游程平滑
	local_RLSA(binaryImage, input, bigAreaContours, wordRects);
	return;
}

/*--------------------------------------------------------------
* Description: 纠错
* Parameters: Mat binaryImage, vector<vector<Point> >& bigAreaContours, vector<Rect>& wordRects
* Return:
* Writter: 梁文伟
--------------------------------------------------------------*/
void image_word_correction(Mat binaryImage, Mat input, vector<vector<Point> >& bigAreaContours, vector<Rect>& bigAreaRect, vector<Rect>& wordRects)
{
	Mat digitImage = normalization(binaryImage);
	// 水平投影判断图像区域中是否有多行文本
	vector<vector<Point> >::const_iterator it_bigAreaContours = bigAreaContours.begin();
	while (it_bigAreaContours != bigAreaContours.end())
	{
		Rect bRect = boundingRect(*it_bigAreaContours);
		/*Mat pic_image = digitImage(bRect);
		bool is_word_horizonl = pic_textlineJudge(pic_image, input.rows, input.cols);
		if (is_word_horizonl) {
			cout << "11111111其实是文本行" << endl;
			wordRects.push_back(bRect);
			it_bigAreaContours = bigAreaContours.erase(it_bigAreaContours);
			continue;
		}
		else {
			cout << "111111111仍然是图片" << endl;
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
			// 该区域中含有多行文本
			for (int i = 0; i < wordRect_add.size(); i++)
			{
				wordRects.push_back(wordRect_add[i]);
			}
			it_bigAreaContours = bigAreaContours.erase(it_bigAreaContours);
			continue;
		}
		else
		{
			// 该区域可能具有三种情况：1.单字文本 2.多字文本 3.单图像 4. 图像+文本
			wordRect_add.clear();
			wordRect_add = Local_Range_Projection(bRect, binaryImage, "vertical");
			if (wordRect_add.size() > 1)
			{
				// 判断类型
				for (int i = 0; i < wordRect_add.size(); i++)
				{
					Mat pic_image = digitImage(wordRect_add[i]);
					bool is_word_vertical = pic_textlineJudge(pic_image, input.rows, input.cols);
					if (is_word_vertical)
					{
						cout << "该区域其实是文本行" << endl;
						wordRects.push_back(wordRect_add[i]);
						continue;
					}
					else
					{
						cout << "该区域其实是图像" << endl;
						bigAreaRect.push_back(wordRect_add[i]);
						continue;
					}
				}
				it_bigAreaContours = bigAreaContours.erase(it_bigAreaContours);
				continue;
			}
		////	//else
		////	//{
		////	//	/*该区域可能具有两种情况：1.单字 2.图像*/
		////	//	Mat bigAreaCut = binaryImage(bRect);
		////	//	Mat cut = input(bRect);
		////	//	int num = 0;
		////	//	for (int w = 0; w < bigAreaCut.cols; w++)
		////	//	{
		////	//		Mat h = bigAreaCut.col(w);
		////	//		num += bigAreaCut.rows - countNonZero(h); // 当前列像素序列中黑色像素的个数
		////	//	}
		////	//	double rate = (num * 1.0) / (bigAreaCut.cols * bigAreaCut.rows); // 切片内黑色像素占总像素的比率
		////	//	if (rate < 0.25)
		////	//	{
		////	//		// 如果rate小于0.25则表示该图像应为文字，而不是图像
		////	//		// 从bigAreaContours中删除，添加进wordRects
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
		// 会把部分图像判定为文字
		vector<vector<Point> >::const_iterator it_bigAreaContours_vertical = bigAreaContours.begin();
		while (it_bigAreaContours_vertical != bigAreaContours.end())
		{
			Rect bRect = boundingRect(*it_bigAreaContours_vertical);
			Mat pic_image = digitImage(bRect);
			bool is_word_vertical = textlinesJudge(pic_image, input.rows, input.cols);
			if (is_word_vertical) {
				cout << "222222222其实是文本行" << endl;
				wordRects.push_back(bRect);
				it_bigAreaContours_vertical = bigAreaContours.erase(it_bigAreaContours_vertical);
				continue;
			}
			else {
				cout << "2222222222仍然是图片" << endl;
				it_bigAreaContours_vertical++;
				continue;
			}
		}*/
	}
	return;
}

/*--------------------------------------------------------------
* Description:将图像和文字存入json
* Parameters: vector<vector<Point> > bigAreaContours, vector<Rect> wordRects
* Return: 
* Writter: 梁文伟
---------------------------------------------------------------*/
void writeFileJson(string filePath, vector<vector<Point> > bigAreaContours, vector<Rect>bigAreaRects, vector<Rect> wordRects)
{
	ofstream fout;
	fout.open(filePath.c_str());
	assert(fout.is_open());
	
	// 根节点
	Json::Value root;

	// 子节点
	Json::Value graphic;
	Json::Value textlines;
	Json::Value irregular_points;
	Json::Value Rectangle_points;
	
	// 子节点属性
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
* Description: 判断传入的两个区域是否相交
* Parameters: Rect rect1, Rect rect2
* Return: bool
* Writter: 梁文伟
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
* Description:去除图像区域,flag=1对二值化图像操作，flag=2画出彩色图像轮廓， flage=3填充彩色图像轮廓
* Parameters: Mat binaryImage, vector<vector<Point> > contours, vector<Rect> bigAreaRects, vector<Rect> boundingRects, int flag
* Return: Null
* Writter: 梁文伟
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
			// 查询左上点
			if (contours[i][j].x < xmin_ymin.x && contours[i][j].y < xmin_ymin.y)
			{
				xmin_ymin = contours[i][j];
				continue;
			}
			// 查询左下点
			if (contours[i][j].x < xmin_ymax.x && contours[i][j].y > xmin_ymax.y)
			{
				xmin_ymax = contours[i][j];
				continue;
			}
			// 查询右上点
			if (contours[i][j].x > xmax_ymin.x && contours[i][j].y < xmax_ymin.y)
			{
				xmax_ymin = contours[i][j];
				continue;
			}
			// 查询右下点
			if (contours[i][j].x > xmax_ymax.x && contours[i][j].y > xmax_ymax.y)
			{
				xmax_ymax = contours[i][j];
				continue;
			}
		}
		// 四个点若有三个点构成的两条直线垂直，则绘制矩形，同时删除contours[i]的点集
		if (abs(xmin_ymin.x - xmin_ymax.x) < 20 && abs(xmin_ymin.y - xmax_ymin.y) < 20)
		{
			// 左上是否垂直
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
			// 右上是否垂直
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
			// 右下是否垂直
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
			// 左下是否垂直
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

		// 若都不垂直，则填充不规则图像
		drawContours(binaryImage, contours, i, CV_RGB(255, 255, 255), -1);
		showPic(binaryImage, "binaryImage");
		waitKey(0);
	}*/
	if (flag == 1)
	{
		// flag = 1对二值化图像操作
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
						// 如果sign为false，则说明相交；否则不相交
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
			// 规则
			rectangle(binaryImage, bigAreaRects[i], CV_RGB(255, 255, 255), -1);
		}
	}
	else if (flag == 2)
	{
		// flag = 2对彩色图像操作
		for (int i = 0; i < contours.size(); i++)
		{
			// 不规则
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
						// 如果sign为true，则说明相交；否则不相交
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
			// 规则
			rectangle(binaryImage, bigAreaRects[i], CV_RGB(255, 0, 0), 8);
		}
	}
	else if (flag == 3)
	{
		// flag = 2对彩色图像操作
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
			//			// 如果sign为false，则说明相交；否则不相交
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
						// 如果sign为true，则说明相交；否则不相交
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
			// 规则
			rectangle(binaryImage, bigAreaRects[i], CV_RGB(255, 0, 0), -1);
		}
	}
	return;
}

/*--------------------------------------------------------------
* Description:去除图像区域,对二值化图像操作
* Parameters: Mat binaryImage, vector<vector<Point> > contours, vector<Rect> bigAreaRects
* Return: Null
* Writter: 梁文伟
---------------------------------------------------------------*/
void Get_Irregular_Contours(Mat& binaryImage, vector<vector<Point> > contours, vector<Rect> bigAreaRects)
{
	// flag = 1对二值化图像操作
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
		// 规则
		rectangle(binaryImage, bigAreaRects[i], CV_RGB(255, 255, 255), -1);
	}
}

/*--------------------------------------------------------------
* Description: 若文本区域与图像区域相交，删除文本区域
* Parameters:
* Return:
* Writter: 梁文伟
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
* Description: 基于垂直投影的文字区域判断
* Parameters: 
* Return:
* Writter: 李千红
---------------------------------------------------------------*/
void vertical_projection_for_word(vector<Rect>& word_componentRects, vector<Rect> componentRects, Mat reuniteImage_opencv, vector<Rect>& zaodian_componentRects)
{
	int height = reuniteImage_opencv.rows;
	int width = reuniteImage_opencv.cols;

	// 基于垂直投影的文字区域判断
	for (int i = 0; i < componentRects.size(); i++)
	{
		if (componentRects[i].x < 0 || componentRects[i].y < 0 || componentRects[i].x + componentRects[i].width > width || componentRects[i].y + componentRects[i].height > height)
			cout << "(" << componentRects[i].x << ", " << componentRects[i].y << "," << componentRects[i].width << ", " << componentRects[i].height << ")" << endl;
		Mat rect_image = reuniteImage_opencv(componentRects[i]);
		//showPic(rect_image * 255,"rect_image");
		bool is_word = textlinesJudge(rect_image, height, width);

		if (is_word == true) {
			cout << componentRects[i].x << " " << componentRects[i].y << " " << componentRects[i].width << " " << componentRects[i].height << "这是一个文字区域" << endl;
			word_componentRects.push_back(componentRects[i]);
		}
		else if (is_word == false) {
			cout << componentRects[i].x << " " << componentRects[i].y << " " << componentRects[i].width << " " << componentRects[i].height << "这是一个噪点区域" << endl;
			zaodian_componentRects.push_back(componentRects[i]);
		}
		else {
			cout << "some errors!" << endl;
		}
	}
	return;
}

/*--------------------------------------------------------------
* Description:去除直线
* Parameters: vector<Vec4f>
* Return:
* Writter: 梁文伟
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
			//	//横线
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
				//竖线
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
* Description:表格区域内图像去除
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
* Description: 图边文字整合
* Parameters: vector<vector<Point> > bigAreaContours, vector<vector<Point> > wordContours
* Return: 
* writter: 梁文伟
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
					// 该不规则图像bigAreaContours[i_bigArea]与该文本框wordContours[i_word]应合并成不规则图像
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
* Description: 提取单通道灰度图像的背景颜色值
* Parameters: img：单通道灰度图像，startX：2*2窗口的起始X坐标，startY：2*2窗口的起始Y坐标，endX：2*2窗口的结束X坐标，endY：2*2窗口的结束Y坐标
* Return: 图像的背景颜色值，取值范围为 0 到 255
* writter: 梁文伟
---------------------------------------------------------------*/
int getGrayBgColor(Mat img, int startX, int startY, int endX, int endY) {
	if (img.type() != CV_8UC1)
	{
		cout << "图像类型必须为 CV_8UC1" << endl;
	}
	// 图像的宽度和高度
	int height = img.rows;
	int width = img.cols;
	// 初始化背景颜色值
	int bgColor = 0;

	// 随机选择一个 2*2 窗口计算背景颜色值
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

	// 返回背景颜色值
	return bgColor;
}

/*--------------------------------------------------------------
* Description: 抓取后景颜色块
* Parameters:
* Return:
* writter: 梁文伟
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
			if (abs((wordRects[i].y + wordRects[i].height / 2) - (wordRects[j].y + wordRects[j].height / 2)) > 5) // 判断wordRects[i]水平方向右边是否有文字框
				continue;
			int right_distance = wordRects[j].x - (wordRects[i].x + wordRects[i].width);			
			cout << "right_distance = " << right_distance << endl;
			if (right_distance > 8 && right_distance < 60) // wordRects[i]右边存在文字框且中间无图像间隔
			{
				bgColor = getGrayBgColor(image_Gray, wordRects[i].x + wordRects[i].width + 4, wordRects[i].y, wordRects[j].x - 4, wordRects[j].y + wordRects[j].height - 4);
			}
		}
	}
	cout << "提取后景颜色值为 = " << bgColor << endl;
	return bgColor;
}
/*--------------------------------------------------------------
* Description:图边文字整合
* Parameters: vector<Rect>&imageAreas ,vector<Rect>&textAreas.
* Return: null
---------------------------------------------------------------*/
void GraphTextIntegrate(int imgwidth,vector<Rect>&imageAreas, vector<Rect>&textAreas)
{
	if (imageAreas.size() == 0 || textAreas.size() == 0)
	{
		return;
	}
	// 图边文字整合
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
	// 图片区域整合
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
* Description:表边文字整合
* Parameters: vector<Rect>&tableAreas ,vector<Rect>&textAreas.
* Return: null
---------------------------------------------------------------*/
void tableTextIntegrate(int imgwidth, vector<Rect>&tableAreas, vector<Rect>&textAreas)
{
	if (tableAreas.size() == 0 || textAreas.size() == 0)
	{
		return;
	}
	// 表边文字整合
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
	// 表格区域整合
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
* Description: 图片检测
* Parameters: Mat类型的图像
* Return: 图片位置区域信息集合
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
* Description: 图像裁剪
* Parameters: Mat类型的图像、矩形区域信息
* Return: 裁剪后的图像集合
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
* Description: 单一图像裁剪
* Parameters: Mat类型的图像、矩形区域信息
* Return: 裁剪后的图像集合
--------------------------------------------------------------*/
Mat cropSingleImage(Mat& input, Rect cropRects)
{
	Mat cutImage = Mat(input, cropRects);
	return cutImage;
}

/*--------------------------------------------------------------
* Description:判断区域是否相邻
* Parameters:Rect &， Rect &
* Return:如果相邻返回true，否则返回false
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
* Description:获得相近的区域
* Parameters:Rect :基准区域，vector<Rect>:所有区域的集合
* Return:获得相近区域的集合
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
			// 如果相邻，就存入
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
* Description: 从区域集合中获得预选框
* Parameters: vector<Rect>:所有区域集合
* Return: 获得预选框
--------------------------------------------------------------*/
vector<Rect> getRegionFromRects(vector<Rect> rects,int threshold)
{
	// 存放最终结果
	vector<Rect> nRect;
	while (rects.size() > 0)
	{
		// 临时存放
		vector<Rect> temp;
		// 获得vector中的最后一个元素
		Rect last = rects.at(rects.size() - 1);
		//删除最后一个元素
		rects.pop_back();
		// 存入temp中
		temp.push_back(last);

		// 声明一个队列
		queue<Rect> q;
		// 此时temp中只有一个元素，入队
		q.push(temp.at(0));

		// 队列不空则出队
		while (q.size())
		{
			// 记录队头元素
			Rect rect_head = q.front();
			//出队
			q.pop();
			// 根据rect_head在剩下的rects中找相邻的块,并从rects中删除这些块
			vector<Rect> nearRegion_rects = getNearRegion(rect_head, rects,threshold);
			// 如果存在，就入队
			if (nearRegion_rects.size() > 0)
			{
				for (int i = 0; i < nearRegion_rects.size(); i++)
				{
					// 将这些重叠的块保存到temp中，每个大循环temp中就存放了一堆相邻的块，最后取这些块的最小x、y，最大x、y就获得一个框住这些块的大框
					temp.push_back(nearRegion_rects.at(i));
					// 重新入队
					q.push(nearRegion_rects.at(i));
				}
			}
		}
		// 全部出队完，temp中就保存了一堆彼此相邻的块，比较所有的Rect的x和y的最小值，最大值。
		int min_x = 100000, max_x = 0, min_y = 100000, max_y = 0;
		int width = 0, maxHeight = 0;
		for (int i = 0; i < temp.size(); i++)
		{
			// 遍历Rect的x、y值
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
		// 大框rect
		Rect big;
		big.x = min_x;
		big.y = min_y;
		big.width = max_x + width - min_x;
		big.height = maxHeight;
		// 将最终结果存起来
		nRect.push_back(big);
	}
	return nRect;
}

/*--------------------------------------------------------------
* Description: 横线分组
* Parameters: Mat:横线集合
* Return: 分组后的横线集合组
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
				// 直线包含区域投影分析
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
* Description: 竖线分组
* Parameters: Mat:横线集合
* Return: 分组后的横线集合组
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
* Description: Json文件的写入
* Parameters: 文件路径、文本区域矩形信息、图像区域矩形信息
* Return: NULL
--------------------------------------------------------------*/
void writeJson(string fileName, int height, int width, int depth, vector<Rect> textRects, vector<Rect> imageRects, vector<Rect> tableRects, string jsonPath,int ratio)
{
	// 根节点
	Json::Value root;
	int pos = fileName.find_last_of('\\');
	int lastPos = fileName.find_last_of('.');
	string name(fileName.substr(pos + 1, -1));
	string filename(fileName.substr(pos + 1, lastPos - pos - 1));
	// 组装根节点属性
	root["fileName"] = name;
	Json::Value image_size;
	image_size["height"] = height;
	image_size["width"] = width;
	image_size["depth"] = depth;
	root["size"] = image_size;

	// 子节点
	Json::Value textRectangle;
	Json::Value imageRectangle;
	Json::Value tableRectangle;

	// 二级子节点
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
		// 二级子节点挂到子节点上
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
		// 二级子节点挂到子节点上
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
		// 二级子节点挂到子节点上
		tableRectangle.append(partner);
		//tableRectangle["tableRectangle" + to_string(index)] = Json::Value(partner);
		index_3++;
	}
	root["tableAreas"] = Json::Value(tableRectangle);

	// 将json内容（缩进格式）输出到文件
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
* Description: 图像增强算法1
* Parameters: Mat:输入图像
* Return: NULL
--------------------------------------------------------------*/
cv::Mat contrastStretch1(cv::Mat srcImage)
{
	cv::Mat resultImage = srcImage.clone();
	int nRows = resultImage.rows;
	int nCols = resultImage.cols;
	// 图像连续性判断
	if (resultImage.isContinuous()) {
		nCols = nCols * nRows;
		nRows = 1;
	}

	// 计算图像的最大最小值
	double pixMin, pixMax;
	cv::minMaxLoc(resultImage, &pixMin, &pixMax);
	//std::cout << "min_a=" << pixMin << " max_b=" << pixMax << std::endl;
	// 对比度拉伸映射
	for (int j = 0; j < nRows; j++) {
		uchar *pDataMat = resultImage.ptr<uchar>(j);
		for (int i = 0; i < nCols; i++) {
			pDataMat[i] = (pDataMat[i] - pixMin) * 255 / (pixMax - pixMin);
		}
	}
	return resultImage;
}
/*--------------------------------------------------------------
* Description: 图像增强算法2
* Parameters: Mat:输入图像
* Return: NULL
--------------------------------------------------------------*/
void contrastStretch2(cv::Mat &srcImage)
{
	// 计算图像的最大最小值
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
	Mat resMat = srcImage.clone();// 克隆矩阵，分配独立空间
								  // 阈值化操作
	int thresh = 130;
	int threshType = 0;
	// 预设最大值
	const int maxVal = 255;
	// 固定阈值化操作
	threshold(srcImage, srcImage, thresh, maxVal, threshType);
	srcImage.convertTo(srcImage, CV_32FC1);
	// 计算垂直投影
	reduce(srcImage, verMat, 0, cv::REDUCE_SUM);// 合并成列，计算所有向量的总和，转化成矩阵
	cout << verMat << endl;
	// 遍历求差分符号函数
	float* iptr = ((float*)verMat.data) + 1;
	// 生成一列向量tempVec
	vector<int> tempVec(verMat.cols - 1, 0);
	// 对差分向量进行符号判定，
	for (int i = 0; i < verMat.cols - 1; ++i, ++iptr)
	{
		if (*(iptr + 1) - *iptr > 0)
			tempVec[i] = 1;
		else if (*(iptr + 1) - *iptr < 0)
			tempVec[i] = -1;
		else
			tempVec[i] = 0;
	}
	// 对符号函数进行遍历
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
	// 波峰判断输出
	for (vector<int>::size_type i = 0; i != tempVec.size() - 1; i++)
	{
		if (tempVec[i + 1] - tempVec[i] == -2)
			// 认为i+1为投影向量S的一个波峰位置点，把点加入到resultVec中
			resultVec.push_back(i + 1);
	}
	// 输出波峰位置
	for (int i = 0; i < resultVec.size(); i++)
	{
		cout << resultVec[i] << '\t';
		// 波峰位置为255
		for (int ii = 0; ii < resMat.rows; ++ii)
		{
			resMat.at<uchar>(ii, resultVec[i]) = 255;
		}
	}
	//imshow("resMat", resMat);
}

/*--------------------------------------------------------------
* Description:区域内白色像素比例
* Parameters: vector<Rect>&imageAreas
* Return: Percentage of white pixels
---------------------------------------------------------------*/
double whitepixelsCount(Mat image,Rect imageAreas)
{
	double count = imageAreas.width * imageAreas.height;
	int whitepixels = 0;
	double percentage = 0.0;
	Mat areas = Mat(image, imageAreas)*255;
	//namedWindow("二值化", 2);
	//imshow("二值化", areas);
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
	cout << "白色像素：" << whitepixels << endl;
	cout << "像素：" << count << endl;
	cout << "白色像素百分比：" << percentage << endl;
	return percentage;
}



/*------------------------王茜茜----------------------------*/
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


/*------------------------王茜茜----------------------------*/


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
//		//表格线提取
//		//Mat grayImage = convert2gray(imageResize);
//		vector<Rect>rowLine;
//		vector<Vec4i>colLine;
//		findTableByLSD(cropImage,rowLine,colLine);
//		//获得表格预测高度
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
//		//// 转灰度图像
//		//Mat grayImage = convert2gray(inputErode);
//		//contrastStretch2(grayImage);
//		//resize(grayImage,imageResize,Size(0,0), 0.5,0.5, INTER_AREA);
//		// 二值化
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
//		// 归一化
//		//Mat digitImage = normalization(imageResize);
//		//表格区域检测
//		//std::vector<Rect> tableRects = findTableByLSD(roteInput);
//		//Mat tableDel = plotRect(digitImage, tableRects);
//		///*Mat reTableDel = tableDel * 255;
//		//namedWindow("tableDel", 2);
//		//imshow("tableDel", reTableDel);*/
//		////图片区域检测
//		//std::vector<Rect> processedImageRects = imageDetect(tableDel);
//		////图片区域去除
//		//Mat imageDel = plotRect(tableDel, processedImageRects);
//		///*Mat reImageDel = imageDel * 255;
//		//namedWindow("imageDel", 2);
//		//imshow("imageDel", reImageDel);*/
//		//// RSLA
//		//Mat afterSmoothHor = lengthSmoothHor(imageDel, imageResize.cols * 0.3);
//		//Mat afterSmoothVer = lengthSmoothVer(imageDel, imageResize.rows * 0.001);
//		//// 并操作
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
//		namedWindow("二值化", 2);
//		imshow("二值化", binaryImage);
//		waitKey(0);
//
//		Mat digitImage = normalization(binaryImage);
//		// 获得直线
//		vector<Rect> allRowLines, allColLines;
//		findTableByLSD(imageResize, allRowLines, allColLines);
//		// 线段的进一步合并
//		vector<Rect> rowLines = getLineFromShortLines(allRowLines, 1);
//		vector<Rect> colLines = getLineFromShortLines(allColLines, 2);
//
//		// 获得较大连通域
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
//		// 表格区域去除
//		Mat tableDel = plotRect(digitImage, TableRects);
//		// 图片区域去除
//		Mat imageDel = plotRect(tableDel, bigArea);
//		// RSLA
//		Mat afterSmoothHor = lengthSmoothHor(imageDel, imageResize.cols * 0.3);
//		Mat afterSmoothVer = lengthSmoothVer(imageDel, imageResize.rows * 0.001);
//		// 并操作
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


//求区域内均值 integral即为积分图
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
	cv::integral(inpImg, integral);  //计算积分图像
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
* Description:依据y坐标对区域进行排序
* Parameters:vector<Rect>:预选区域的集合
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
	// 并操作
	//Mat afterSmooth = afterSmoothVer & afterSmoothHor;
	//Mat dilateImage = doDilation(afterSmooth, 3);
	vector<Rect> componentRects = textDetect(afterSmoothHor);
	vector<Rect> textRects = getRegionFromRects(componentRects, srcImage.cols*0.05);
	// 筛选较短文字块
	textRectSelect(textRects, srcImage.cols);
	// 按Y轴排序
	reckRankY(textRects);
	// 按左上角点Y轴坐标分组
	vector<vector<Rect>>groupedTextRects = groupingRectByY(textRects);
	//分组合并
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