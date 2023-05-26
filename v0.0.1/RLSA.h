/*--------------------------------------------------------------
* Copyright (c) 2018, Corporation name. All rights reserved.
*
* File name: RLSA.h
* Description:
* Version:0.1
* Author:Zhicheng Liu
* Completion date:2020.03.05
--------------------------------------------------------------*/
#ifndef RLSA_H
#define RLSA_H

#include <io.h>  
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp> 
#include <vector>
#include <string>
#include <direct.h>

using namespace std;
using namespace cv;

/*----------------------------------------------------------------------
* Description  : Output the value of each pixel
*
* Parameters   : Mat input: Picture in Mat format
* Output values: the value of each pixel
* Return values: No return values
----------------------------------------------------------------------*/
void printFunc(Mat input);

/*--------------------------------------------------------------
* Description: Open picture file
* Parameters: filepath: Source picture file with full path
* Output values: If the picture does not exist,cout"Open failure"
* Return values: Image of type mat, return empty mat when
file is invalid.
--------------------------------------------------------------*/
Mat loadImage(string filepath);

/*--------------------------------------------------------------
* Description: Image to grayscale
* Parameters: Mat input: Picture in Mat format
* Return values: Gray image of Mat type
--------------------------------------------------------------*/
Mat convert2gray(Mat input);

/*--------------------------------------------------------------
* Description: 图像二值化
* Parameters: Mat类型的单通道图像
* Return: Mat类型的二值化图像，当通道不为一时返回空Mat
--------------------------------------------------------------*/
Mat binaryzation(Mat grayImage, int value);
Mat thresholdAdapt(Mat& img, float rate);
/*--------------------------------------------------------------
* Description: 像素值进行归一化
* Parameters: Mat类型的灰度图像
* Return: Mat类型的归一化图像
--------------------------------------------------------------*/
Mat normalization(Mat binaryImage);

/*--------------------------------------------------------------
* Description: 水平方向的游程平滑
* Parameters: Mat类型的图像、游程阈值
* Return: 水平游程平滑后的图像
--------------------------------------------------------------*/
Mat lengthSmoothHor(Mat digitImage, int threshold);

/*--------------------------------------------------------------
* Description: 垂直方向的游程平滑
* Parameters: Mat类型的图像、游程阈值
* Return: 垂直游程平滑后的图像
--------------------------------------------------------------*/
Mat lengthSmoothVer(Mat digitImage, int threshold);

/*--------------------------------------------------------------
* Description: 膨胀操作
* Parameters: Mat类型的图像、膨胀次数
* Return: 膨胀后的图像
--------------------------------------------------------------*/
Mat doDilation(Mat smoothImage, int times);

/*--------------------------------------------------------------
* Description: 区域删除
* Parameters: Mat类型的图像、矩形区域信息
* Return: 矩形去除后的图像
--------------------------------------------------------------*/
Mat plotRect(Mat& input, std::vector<Rect> inputRects);
Mat plotRect(Mat& input, std::vector<Rect> inputRects, double width);
/*--------------------------------------------------------------
* Description: 矩形区域绘制
* Parameters: Mat类型的图像
* Return: 绘制后的图像
--------------------------------------------------------------*/
Mat getBlock(Mat& input, std::vector<Rect> textRects, std::vector<Rect> imageRects, std::vector<Rect> tableRects, int rate);
Mat getBlock(Mat& input, std::vector<Rect> rects);
Mat getBlock(Mat input, std::vector<Rect> textRects, int rate);

/*--------------------------------------------------------------
* Description: 文字检测
* Parameters: Mat类型的图像
* Return: 文字位置区域信息集合
--------------------------------------------------------------*/
std::vector<Rect> textDetect(Mat dilateImage);
std::vector<Rect> textDetect(Mat dilateImage, Mat input);
std::vector<Rect> textDetect(Mat dilateImage, Mat input, vector<vector<Point> >& bigAreaContours);
std::vector<Rect> textDetect(Mat dilateImage, Mat input, vector<vector<Point> >& bigAreaContours, int flag);
/*--------------------------------------------------------------
* Description: 图片检测
* Parameters: Mat类型的图像
* Return: 图片位置区域信息集合
--------------------------------------------------------------*/
std::vector<Rect> imageDetect(Mat dilateImage);

/*--------------------------------------------------------------
* Description: 图像裁剪
* Parameters: Mat类型的图像、矩形区域信息
* Return: 裁剪后的图像集合
--------------------------------------------------------------*/
std::vector<Mat> cropImage(Mat& input, std::vector<Rect>& cropRects);

/*--------------------------------------------------------------
* Description:判断区域是否相邻
* Parameters:Rect &， Rect &
* Return:如果相邻返回true，否则返回false
--------------------------------------------------------------*/
bool isOverlap(const Rect &rc1, const Rect &rc2,int threshold);

/*--------------------------------------------------------------
* Description:获得相近的区域
* Parameters:Rect :基准区域，vector<Rect>:所有区域的集合
* Return:获得相近区域的集合
--------------------------------------------------------------*/
vector<Rect> getNearRegion(Rect r, vector<Rect> &rects,int threshold);

/*--------------------------------------------------------------
* Description: 从区域集合中获得预选框
* Parameters: vector<Rect>:所有区域集合
* Return: 获得预选框
--------------------------------------------------------------*/
vector<Rect> getRegionFromRects(vector<Rect> rects,int threshould);

/*--------------------------------------------------------------
* Description: Json文件的写入
* Parameters: 文件路径、文本区域矩形信息、图像区域矩形信息
* Return: NULL
--------------------------------------------------------------*/
void writeJson(string fileName, int height, int width, int depth, vector<Rect> textRects, vector<Rect> imageRects, vector<Rect> tableRects, string jsonPath,int ratio);

/*--------------------------------------------------------------
* Description: 图像增强算法1
* Parameters: Mat:输入图像
* Return: NULL
--------------------------------------------------------------*/
cv::Mat contrastStretch1(cv::Mat srcImage);

/*--------------------------------------------------------------
* Description: 图像增强算法2
* Parameters: Mat:输入图像
* Return: NULL
--------------------------------------------------------------*/
void contrastStretch2(cv::Mat &srcImage);

/*--------------------------------------------------------------
* Description: 横线分组
* Parameters: Mat:横线集合
* Return: 分组后的横线集合组
--------------------------------------------------------------*/
vector<vector<Rect>> groupingRowLines(vector<Rect>rowlines);

/*--------------------------------------------------------------
* Description: 竖线分组
* Parameters: Mat:竖线集合
* Return: 分组后的竖线集合组
--------------------------------------------------------------*/
vector<vector<Rect>> groupingColLines(vector<Rect>collines);


void findPeak(Mat srcImage, vector<int>& resultVec);

/*--------------------------------------------------------------
* Description:表格区域内图像去除
* Parameters: vector<Rect> tableAreas,表格区域；vector<Rect> imageAreas图片区域
* Return: null
---------------------------------------------------------------*/
void imageInsideTable(vector<Rect>tableAreas, vector<Rect>&imageAreas);

/*--------------------------------------------------------------
* Description:图边文字整合
* Parameters: vector<Rect>&imageAreas ,vector<Rect>&textAreas.
* Return: null
---------------------------------------------------------------*/
void GraphTextIntegrate(int imgwidth,vector<Rect>&imageAreas, vector<Rect>&textAreas);

/*--------------------------------------------------------------
* Description:表边文字整合
* Parameters: vector<Rect>&imageAreas ,vector<Rect>&textAreas.
* Return: null
---------------------------------------------------------------*/
void tableTextIntegrate(int imgwidth, vector<Rect>&tableAreas, vector<Rect>&textAreas);

/*--------------------------------------------------------------
* Description:区域内白色像素比例
* Parameters: vector<Rect>&imageAreas
* Return: Percentage of white pixels
---------------------------------------------------------------*/
double whitepixelsCount(Mat image,Rect imageAreas);


Mat Sauvola(cv::Mat& inpImg, int window, float k);
float fastMean(cv::Mat& integral, int x, int y, int window);

void reckRankY(vector<Rect>& srcRects);
vector<vector<Rect>> groupingRectByY(vector<Rect>rect);
vector<Rect> TDRAR(Mat srcImage);
void textRectSelect(vector<Rect> &textRect, int img_width);

void showImage(String windowName, Mat image, int showTime);


/*------------------------王茜茜----------------------------*/
int valueRLSA1(Mat imageDel, Mat srcImage, std::vector<std::vector<Point>> contours);
int valueRLSA2(Mat imageDel, Mat srcImage, std::vector<std::vector<Point>> contours, int threshold_RLSA1);

/*------------------------王茜茜----------------------------*/

vector<Rect> removeExtraTextArea(vector<Rect>data, Mat imageResize, Mat binaryImage, Mat afterSmooth);
void Get_Irregular_Contours(Mat& binaryImage, vector<vector<Point> > contours, vector<Rect> bigAreaRects);
void Get_Irregular_Contours(Mat& binaryImage, vector<vector<Point> > contours, vector<Rect> bigAreaRects, vector<Rect> boundingRects, int flag);
void showPic(Mat showpic, String namepic);
Mat ls_remove(vector<Vec4f> linePoint, Mat imageDel);
void image_word_MinDistance(vector<vector<Point> >& bigAreaContours, vector<vector<Point> >& wordContours);
int catch_background_color_block(Mat inputCopy, std::vector<Rect> wordRects);
void image_word_correction(Mat binaryImage, Mat input, vector<vector<Point> >& bigAreaContours, vector<Rect>& bigAreaRect, vector<Rect>& wordRects);
void writeFileJson(string filePath, vector<vector<Point> > bigAreaContours, vector<Rect>bigAreaRects, vector<Rect> wordRects);
void Checking_for_gaps(Mat binaryImage, Mat input, vector<vector<Point> >& bigAreaContours, vector<Rect> bigArea, vector<Rect>& wordRects);
bool textlinesJudge(Mat normalImage, int pic_height, int pic_width);
vector<Rect> textlinesSorting(Mat final_picture, vector<Rect>word_componentRects, vector<Rect>& rowFirst_wordRects, vector<int>& row_num);
bool pic_textlineJudge(Mat normalImage, int pic_height, int pic_width);
vector<int> getBgPixelblock(Mat grayImage, vector<Rect>sorted_wordRects, vector<Rect>rowFirst_wordRects, vector<int> row_num);
void vertical_projection_for_word(vector<Rect>& word_componentRects, vector<Rect> componentRects, Mat reuniteImage_opencv, vector<Rect>& zaodian_componentRects);
bool IsOverLap(Rect rect1, Rect rect2);
#endif