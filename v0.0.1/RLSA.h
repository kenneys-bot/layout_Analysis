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
* Description: ͼ���ֵ��
* Parameters: Mat���͵ĵ�ͨ��ͼ��
* Return: Mat���͵Ķ�ֵ��ͼ�񣬵�ͨ����Ϊһʱ���ؿ�Mat
--------------------------------------------------------------*/
Mat binaryzation(Mat grayImage, int value);
Mat thresholdAdapt(Mat& img, float rate);
/*--------------------------------------------------------------
* Description: ����ֵ���й�һ��
* Parameters: Mat���͵ĻҶ�ͼ��
* Return: Mat���͵Ĺ�һ��ͼ��
--------------------------------------------------------------*/
Mat normalization(Mat binaryImage);

/*--------------------------------------------------------------
* Description: ˮƽ������γ�ƽ��
* Parameters: Mat���͵�ͼ���γ���ֵ
* Return: ˮƽ�γ�ƽ�����ͼ��
--------------------------------------------------------------*/
Mat lengthSmoothHor(Mat digitImage, int threshold);

/*--------------------------------------------------------------
* Description: ��ֱ������γ�ƽ��
* Parameters: Mat���͵�ͼ���γ���ֵ
* Return: ��ֱ�γ�ƽ�����ͼ��
--------------------------------------------------------------*/
Mat lengthSmoothVer(Mat digitImage, int threshold);

/*--------------------------------------------------------------
* Description: ���Ͳ���
* Parameters: Mat���͵�ͼ�����ʹ���
* Return: ���ͺ��ͼ��
--------------------------------------------------------------*/
Mat doDilation(Mat smoothImage, int times);

/*--------------------------------------------------------------
* Description: ����ɾ��
* Parameters: Mat���͵�ͼ�񡢾���������Ϣ
* Return: ����ȥ�����ͼ��
--------------------------------------------------------------*/
Mat plotRect(Mat& input, std::vector<Rect> inputRects);
Mat plotRect(Mat& input, std::vector<Rect> inputRects, double width);
/*--------------------------------------------------------------
* Description: �����������
* Parameters: Mat���͵�ͼ��
* Return: ���ƺ��ͼ��
--------------------------------------------------------------*/
Mat getBlock(Mat& input, std::vector<Rect> textRects, std::vector<Rect> imageRects, std::vector<Rect> tableRects, int rate);
Mat getBlock(Mat& input, std::vector<Rect> rects);
Mat getBlock(Mat input, std::vector<Rect> textRects, int rate);

/*--------------------------------------------------------------
* Description: ���ּ��
* Parameters: Mat���͵�ͼ��
* Return: ����λ��������Ϣ����
--------------------------------------------------------------*/
std::vector<Rect> textDetect(Mat dilateImage);
std::vector<Rect> textDetect(Mat dilateImage, Mat input);
std::vector<Rect> textDetect(Mat dilateImage, Mat input, vector<vector<Point> >& bigAreaContours);
std::vector<Rect> textDetect(Mat dilateImage, Mat input, vector<vector<Point> >& bigAreaContours, int flag);
/*--------------------------------------------------------------
* Description: ͼƬ���
* Parameters: Mat���͵�ͼ��
* Return: ͼƬλ��������Ϣ����
--------------------------------------------------------------*/
std::vector<Rect> imageDetect(Mat dilateImage);

/*--------------------------------------------------------------
* Description: ͼ��ü�
* Parameters: Mat���͵�ͼ�񡢾���������Ϣ
* Return: �ü����ͼ�񼯺�
--------------------------------------------------------------*/
std::vector<Mat> cropImage(Mat& input, std::vector<Rect>& cropRects);

/*--------------------------------------------------------------
* Description:�ж������Ƿ�����
* Parameters:Rect &�� Rect &
* Return:������ڷ���true�����򷵻�false
--------------------------------------------------------------*/
bool isOverlap(const Rect &rc1, const Rect &rc2,int threshold);

/*--------------------------------------------------------------
* Description:������������
* Parameters:Rect :��׼����vector<Rect>:��������ļ���
* Return:����������ļ���
--------------------------------------------------------------*/
vector<Rect> getNearRegion(Rect r, vector<Rect> &rects,int threshold);

/*--------------------------------------------------------------
* Description: �����򼯺��л��Ԥѡ��
* Parameters: vector<Rect>:�������򼯺�
* Return: ���Ԥѡ��
--------------------------------------------------------------*/
vector<Rect> getRegionFromRects(vector<Rect> rects,int threshould);

/*--------------------------------------------------------------
* Description: Json�ļ���д��
* Parameters: �ļ�·�����ı����������Ϣ��ͼ�����������Ϣ
* Return: NULL
--------------------------------------------------------------*/
void writeJson(string fileName, int height, int width, int depth, vector<Rect> textRects, vector<Rect> imageRects, vector<Rect> tableRects, string jsonPath,int ratio);

/*--------------------------------------------------------------
* Description: ͼ����ǿ�㷨1
* Parameters: Mat:����ͼ��
* Return: NULL
--------------------------------------------------------------*/
cv::Mat contrastStretch1(cv::Mat srcImage);

/*--------------------------------------------------------------
* Description: ͼ����ǿ�㷨2
* Parameters: Mat:����ͼ��
* Return: NULL
--------------------------------------------------------------*/
void contrastStretch2(cv::Mat &srcImage);

/*--------------------------------------------------------------
* Description: ���߷���
* Parameters: Mat:���߼���
* Return: �����ĺ��߼�����
--------------------------------------------------------------*/
vector<vector<Rect>> groupingRowLines(vector<Rect>rowlines);

/*--------------------------------------------------------------
* Description: ���߷���
* Parameters: Mat:���߼���
* Return: ���������߼�����
--------------------------------------------------------------*/
vector<vector<Rect>> groupingColLines(vector<Rect>collines);


void findPeak(Mat srcImage, vector<int>& resultVec);

/*--------------------------------------------------------------
* Description:���������ͼ��ȥ��
* Parameters: vector<Rect> tableAreas,�������vector<Rect> imageAreasͼƬ����
* Return: null
---------------------------------------------------------------*/
void imageInsideTable(vector<Rect>tableAreas, vector<Rect>&imageAreas);

/*--------------------------------------------------------------
* Description:ͼ����������
* Parameters: vector<Rect>&imageAreas ,vector<Rect>&textAreas.
* Return: null
---------------------------------------------------------------*/
void GraphTextIntegrate(int imgwidth,vector<Rect>&imageAreas, vector<Rect>&textAreas);

/*--------------------------------------------------------------
* Description:�����������
* Parameters: vector<Rect>&imageAreas ,vector<Rect>&textAreas.
* Return: null
---------------------------------------------------------------*/
void tableTextIntegrate(int imgwidth, vector<Rect>&tableAreas, vector<Rect>&textAreas);

/*--------------------------------------------------------------
* Description:�����ڰ�ɫ���ر���
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


/*------------------------������----------------------------*/
int valueRLSA1(Mat imageDel, Mat srcImage, std::vector<std::vector<Point>> contours);
int valueRLSA2(Mat imageDel, Mat srcImage, std::vector<std::vector<Point>> contours, int threshold_RLSA1);

/*------------------------������----------------------------*/

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