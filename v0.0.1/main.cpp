
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
#include "json/json.h"
#include <time.h>
#include "rotateTransformation.h"
#include "findTableByLSD.h"
#include "get_lines.h"
#include "RLSA.h"
#include <direct.h>
#include "DirtyDocUnet.h"

#include "draw.h"
#include<algorithm>
//#include "RLSA.cpp"


using namespace std;
using namespace cv;



int main()
{
	cout << "1" << endl;
	// ԭʼ·��
	//string srcPath = "F:\\Layout analysis dataset\\Newdata\\data\\JPEGImages\\data\\";
	//string srcPath = "E:\\Y�ĵ�\\Y2\\�������\\Layout analysis dataset\\data\\";
	//string srcPath = "D:\\Data\\���ݼ�\\data\\";
	//string srcPath = "D:\\Data\\���ݼ�\\compress_double\\";
	//string srcPath = "E:\\Y�ĵ�\\Y1\\���ݼ�\\ori_img\\ori_img\\DocBank_500K_ori_img\\";
	//string srcPath = "E:\\Y�ĵ�\\Y2\\�������\\Layout analysis dataset\\data1\\PRImA Layout Analysis Dataset\\Images\\";
	//string srcPath = "E:\\Data\\CDLA_DATASET\\train\\";
	//string srcPath = "E:\\Dataset\\PRImA Layout Analysis Dataset\\Images\\";
	//string srcPath = "E:\\Dataset\\compress_double\\";
	//string path = "F:\\works\\Layout Analysis\\data1\\JPEGImages\\data\\result\\result_pre_4\\";
	string path = "F:\\works\\Layout Analysis\\data\\JPEGImages\\data\\result\\result_pre_7\\";

	// ԭʼͼ��·��
	string srcPath = path + "image\\";
	// ͼ����·��
	string dstPath = path + "result\\";
	// ��������ṹ�ļ�����·��
	string jsonPath = path + "json\\";
	// �¾ɶ�ֵ��ͼ��ԱȽ��·��
	string binaryResultPath = path + "binaryResult\\";
	// program_image���ͼ��·��
	string programImagePath = path + "program_image\\";
	
	// Check whether source path exists
	if ((_access(srcPath.data(), 0)) != 0)
	{
		printf("Input path does not exist! \n\n");
		system("pause");
		return -1;
	}
    // ����ͼ�������б�
	vector<String> file_vec;
	// ��srcPath�����е�jpg�ļ�
	glob(srcPath + "*.tif", file_vec, false);
	//glob(srcPath + "*.tif", file_vec, false);


	int index = 0;
	int totalTime = 0;


	// Create destination directory if it does not exist
	if ((_access(dstPath.data(), 0)) != 0)
	{
		_mkdir(dstPath.data());
	}
	// model
	string model_path = "models/unetv2.onnx";
	DirtyDocUnet doc_bin(model_path);

	// �������
	for (string fileName : file_vec)
	{
		clock_t start, end;
		start = clock();
		cout << fileName << endl;

		Mat input = loadImage(fileName);
		int width = input.cols;
		int height = input.rows;
		int depth = input.channels();

		Mat inputCopy = input.clone();
		

		//ͼ���ƫ
		//Mat roteInput = ScannedImageRectify(input);

		//ͼ��ȥ��
		 Mat imageWithBlur;
		// ˫���˲�ȥ��
		//bilateralFilter(inputCopy, imageWithBlur, 3, 100, 100);
		blur(inputCopy, imageWithBlur, Size(5,5), Point(-1, -1));

		//��̬ѧ����
		/*Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
		erode(imageWithBlur, imageWithBlur, element, Point(-1, -1), 1, 0);*/

		//ͼ����ǿ���Աȶ�����
		//imageWithBlur = contrastStretch1(imageWithBlur);


		//ͼ����С
		/*Mat imageResize;
		resize(imageWithBlur, imageResize, Size(0, 0), 0.5, 0.5, INTER_AREA);*/

		//binaryImage by openCV
		//�Ҷ�
		Mat grayImage = convert2gray(imageWithBlur);
		//showPic(grayImage, "gray");

		//��ֵ����
		Mat binaryImage_opencv = binaryzation(grayImage, 195);  //������ֵ��������
		//showPic(binaryImage_opencv, "thres_200");

		Mat binaryImage_opencv_make_border = image_make_border(binaryImage_opencv);
		//showPic(binaryImage_opencv_make_border, "binaryImage_opencv_make_border");
		// 
		//��һ������0-1
		Mat reuniteImage_opencv = normalization(binaryImage_opencv);
		//Mat reuniteImage_opencvshow = reuniteImage_opencv * 255;
		//showPic(reuniteImage_opencvshow, "reuniteImage_opencv");
		//imwrite("D:\\VS_Projects\\Layout Analysis\\data\\reuniteImage_opencvshow.jpg", reuniteImage_opencvshow);

		Mat reuniteImage_opencv_make_border = normalization(binaryImage_opencv_make_border);
		
		
		
		//binaryImage by model
		Mat binaryImage_model;
		doc_bin.docBin(inputCopy, binaryImage_model);




		//�������һ�����ű�Ե����
		Mat digitImage_opencv = InterferenceImage(reuniteImage_opencv);
		//showPic(digitImage_opencv,"digitImage_opencv");

		Mat digitImage_opencv_make_border = InterferenceImage(reuniteImage_opencv_make_border);



		// ���ֱ��
		/*vector<Rect> allRowLines, allColLines;
		//findLine(digitImage, allRowLines, allColLines);
		findTableByLSD(input, allRowLines, allColLines);*/


		// opencvԤ�����µĽϴ���ͨ���ȡ
		std::vector<std::vector<Point>> contours;
		vector<Rect>bigArea = bigAreaDetect(digitImage_opencv, contours); //���ء�1 ��60��С����ͨ��
		/*Mat bigArea2Show = input.clone();
		for (int i = 0; i < bigArea.size(); i++)
		{
			//�Է���Ҫ�����ͨ���þ��ο�
			rectangle(bigArea2Show, Point(bigArea[i].x, bigArea[i].y), Point(bigArea[i].x + bigArea[i].width, bigArea[i].y + bigArea[i].height), cv::Scalar(255, 0, 0), 5, 1, 0);
		}
		showPic(bigArea2Show, "bigArea2Show");*/

		//vector<vector<Point> > contours_fix = bigAreaDetect(digitImage_opencv);//������һ���ϴ���ͨ����������
		vector<vector<Point> > contours_fix = bigAreaDetect(digitImage_opencv_make_border);

		vector<Rect> bigAreaRects;


		/*---------ȥ���������---------*/
		//vector<Rect> TableRects;
		//if (allRowLines.size() > 0)
		//{
		//	std::vector<Rect>::const_iterator itr = bigArea.begin();
		//	// �ϴ����������ж�(�������)
		//	while (itr != bigArea.end())
		//	{
		//		Rect rect = Rect(*itr);
		//		if (ifTableLineInside(input.cols, rect, allRowLines) && whitepixelsCount(digitImage, rect) < 0.7 || ifTableColLineInside(input.rows, rect, allColLines))
		//		{
		//			TableRects.push_back(rect);
		//			itr = bigArea.erase(itr);
		//		}
		//		else
		//		{
		//			itr++;
		//		}
		//	}
		//}
		//// ʣ��ֱ�߷���(�������)
		//if (allRowLines.size() > 0)
		//{
		//	vector<vector<Rect>>possibleTableLines = groupingRowLines(allRowLines);
		//	if (possibleTableLines.size() > 0)
		//	{
		//		for (int groupId = 0; groupId < possibleTableLines.size(); groupId++)
		//		{
		//			vector<Rect> combinedTables = getTableRects(possibleTableLines.at(groupId), digitImage);
		//			if (combinedTables.size() > 0) {
		//				TableRects.insert(TableRects.end(), combinedTables.begin(), combinedTables.end());
		//			}
		//		}
		//	}
		//}
		//if (TableRects.size() > 1)
		//{
		//	recRankArea(TableRects);
		//	std::vector<Rect>::const_iterator itr = TableRects.begin();
		//	while (itr != TableRects.end())
		//	{
		//		Rect rect = Rect(*itr);
		//		std::vector<Rect>::const_iterator itr_next = itr + 1;
		//		while (itr_next != TableRects.end())
		//		{
		//			Rect nextRect = Rect(*itr_next);
		//			if (isTableInside(nextRect, rect))
		//			{
		//				itr_next = TableRects.erase(itr_next);//���������ɾ��
		//			}
		//			else
		//				itr_next++;
		//		}
		//		itr++;
		//	}
		//}
		//// �������ȥ��
		//Mat tableDel = plotRect(digitImage, TableRects);






		//�Խϴ���ͨ��������
		Get_Irregular_Contours(binaryImage_model, contours_fix, bigAreaRects);
		//drawContours(binaryImage_model, contours_fix, -1, CV_RGB(255, 255, 255), -1);
		//showPic(binaryImage_model, "binaryImage_model");

		Mat reuniteImage_model = normalization(binaryImage_model);
		Mat digitImage_model = InterferenceImage(reuniteImage_model); //��Ե�������ͼ��
		Mat imageDel_model = digitImage_model;
		drawContours(imageDel_model, contours_fix, -1, CV_RGB(0, 0, 0), -1);


		//ȥֱ�߸���
		vector<Vec4f> newLines = ls_detect(inputCopy, grayImage);
		Mat lineDel = ls_remove(newLines, imageDel_model);
		Mat lineDel2Show = lineDel * 255;
		//showPic(lineDel2Show, "lineDel2Show");



		int threshold_RLSA1 = valueRLSA1(lineDel, digitImage_model, contours);
		int threshold_RLSA2 = valueRLSA2(lineDel, digitImage_model, contours, threshold_RLSA1);

		cout << "threshold_RLSA1 :"<<threshold_RLSA1<<"threshold_RLSA2:" << threshold_RLSA2 << endl;

  		
		Mat afterSmoothHor = lengthSmoothHor(lineDel, threshold_RLSA2 * 0.25);
		Mat afterSmoothVer = lengthSmoothVer(lineDel, threshold_RLSA1);
		/*Mat afterSmoothHor = lengthSmoothHor(digitImage, threshold_RLSA2);
		Mat afterSmoothVer = lengthSmoothVer(digitImage, threshold_RLSA1);*/
		// ������
		Mat afterSmooth = afterSmoothVer & afterSmoothHor;
		Mat afterSmooth2Show_1 = afterSmooth * 255;
		//showPic(afterSmooth2Show_1,"afterSmooth2Show_1");
		
		//��һ��ˮƽƽ��
		afterSmooth = lengthSmoothHor(afterSmooth, threshold_RLSA2 * 0.95);
		Mat afterSmooth2Show_2 = afterSmooth * 255;
		//showPic(afterSmooth2Show_2,"afterSmooth2Show_2");
		

		//Mat afterSmoothHor = lengthSmoothHor(lineDel, imageResize.cols * 0.2);
		//Mat afterSmoothVer = lengthSmoothVer(lineDel, imageResize.rows * 0.1);
		//// ������
		//Mat afterSmooth = afterSmoothVer & afterSmoothHor;
		////Mat dilateImage = doDilation(afterSmooth, 3);

		Mat afterSmoothHor2Show = afterSmoothHor * 255;
		//showPic(afterSmoothHor2Show, "afterSmoothHor2Show");
		Mat afterSmoothVer2Show = afterSmoothVer * 255;
		//showPic(afterSmoothVer2Show, "afterSmoothVer2Show");

		
		//���ּ��
		/*std::vector<Rect> componentRects = textDetect(afterSmooth, inputCopy, contours_fix);*/
		std::vector<Rect> componentRects = textDetect(afterSmooth, input, contours_fix);
		/*Mat temp_img = input.clone();
		for (int i = 0; i < componentRects.size(); i++)
		{
			rectangle(temp_img, Point(componentRects[i].x, componentRects[i].y), Point(componentRects[i].x + componentRects[i].width, componentRects[i].y + componentRects[i].height), cv::Scalar(255, 0, 0), 8);
		}
		showPic(temp_img, "temp_img");
		waitKey(0);*/
		vector<Rect> zaodian_componentRects;
		std::vector<Rect> word_componentRects;
		vertical_projection_for_word(word_componentRects, componentRects, reuniteImage_opencv, zaodian_componentRects);
		
		// ������������û�������⣩
		vector<Rect> rowFirst_wordRects;
		vector<int> row_num;
		Mat final_Picture = input.clone();//������ʾ�����ı���
		vector<Rect> sorted_wordRects = textlinesSorting(final_Picture, word_componentRects, rowFirst_wordRects, row_num);
		cout << "����������ϣ�" << endl;

		// ��ȡ�����ؿ飨vectorԽ�磩 
		/*vector<int> bgColor_array = getBgPixelblock(grayImage, sorted_wordRects, rowFirst_wordRects, row_num);
		int bgColor_sum = 0;
		for (int k = 0; k < bgColor_array.size(); k++)
		{
			bgColor_sum += bgColor_array[k];
		}
		int bgColor_value = bgColor_sum / bgColor_array.size();
		cout << "bgcolor_value" << bgColor_value << endl;*/

		
		///*��ȡ����ɫ��--����ΰ*/
		////int bgColor_value = catch_background_color_block(inputCopy, sorted_wordRects);

		// �ٴζ�ֵ��ͼƬ
		//Mat re_binaryImage = binaryzation(grayImage, bgColor_value);
		//showPic(re_binaryImage, "�ٴζ�ֵ��ͼ��");

		// ��ֵ195��ֵ��ͼ���¶�ֵ��ͼƴ�ӡ���binaryResult
		/*Mat binaryResult;
		hconcat(binaryImage_opencv, re_binaryImage, binaryResult);
		int pos_0 = fileName.find_last_of('\\');
		int lastPos_0 = fileName.find_last_of('.');
		string name_0(fileName.substr(pos_0 + 1, lastPos_0 - pos_0 - 1));
		imwrite(binaryResultPath + name_0 + ".jpg", binaryResult);*/

		// ������©��ȱ
		image_word_correction(binaryImage_opencv, inputCopy, contours_fix, bigAreaRects, word_componentRects);
		Checking_for_gaps(binaryImage_opencv, inputCopy, contours_fix, bigAreaRects, word_componentRects);
		vertical_projection_for_word(word_componentRects, word_componentRects, reuniteImage_opencv, zaodian_componentRects);

		// �鿴���ο���������
		//for (vector<Rect>::iterator it = word_componentRects.begin(); it != word_componentRects.end(); it++) {
		//	cout << *it << endl;
		//	//[288 x 27 from (1095, 3021)]  width,height from(x,y)
		//}

		
		Mat final_Picture2Show = input.clone();
		//����ͼƬ����
		//drawContours(final_Picture2Show, contours_fix, -1, CV_RGB(255, 0, 0), 2);
		Get_Irregular_Contours(final_Picture2Show, contours_fix, bigAreaRects, word_componentRects, 2);
		//���봹ֱͶӰ����ı��п�
		for (int i = 0; i < word_componentRects.size(); i++)
		{
			rectangle(final_Picture2Show, Point(word_componentRects[i].x, word_componentRects[i].y), Point(word_componentRects[i].x + word_componentRects[i].width, word_componentRects[i].y + word_componentRects[i].height), CV_RGB(0, 0, 255), 8);
			//showPic(final_Picture2Show, "final_Picture2Show");
		}
		//���봹ֱͶӰ��ķ��ı��п�
		/*for (int i = 0; i < zaodian_componentRects.size(); i++)
		{
			rectangle(final_Picture2Show, Point(zaodian_componentRects[i].x, zaodian_componentRects[i].y), Point(zaodian_componentRects[i].x + zaodian_componentRects[i].width, zaodian_componentRects[i].y + zaodian_componentRects[i].height), cv::Scalar(0, 255, 0), 5, 1, 0);

		}*/

		//�����ı�������
		//for (int i = 0; i < componentRects.size(); i++)
		//{
		//	rectangle(final_Picture2Show, Point(componentRects[i].x, componentRects[i].y), Point(componentRects[i].x + componentRects[i].width, componentRects[i].y + componentRects[i].height), CV_RGB(255, 0, 0), 8);
		//	//showPic(final_Picture2Show, "final_Picture2Show");
		//}


		/*showPic(final_Picture2Show, "final_Picture2Show");
		waitKey(0);
		imwrite("D:\\VS_Projects\\Layout Analysis\\data\\" + fileName, final_Picture2Show);*/

		/*
		//�����������ͼƬ�����ཻ����Ĵ���
		std::vector<Rect> textRects = componentRects;
		std::vector<Rect> deletebigArea = imageInterText(bigArea, textRects);
		//��ȡȥ������һ��ͼƬ����������򣬶���ȥ���֣�������ͼ�μ������ּ��
		std::vector<std::vector<Point>> irRegularArea;
		if(!deletebigArea.empty())
		{
			test(deletebigArea, irRegularArea, textRects, input);
		}


		Mat textDel = plotRect(afterSmooth, textRects, width*0.4);

		
		recRankYMTS(textRects);
		
		// �����������
		//tableTextIntegrate(imageResize.cols, TableRects, textRects);
		
		//����ͼƬȥ��
		//imageInsideTable(TableRects, bigArea);
		
		//��������ͶӰ�ж�
		text_ver_projection(digitImage_model, textRects, bigArea);

		//ͼ������ȥ��
		Graph_Text_remove(textRects, bigArea);
		
		//ͼ������ͶӰ�ж�
		textFeatureDetect(binaryImage_model, textRects, bigArea);

		//�����������򵥶���ȡͼ����������
		vector<vector<Point> > IrRegularArea = getirRegularArea(input, textRects, deletebigArea, bigArea);



		// �����
		Mat mark = getBlock(input, textRects, 1);


		for (int j = 0; j < IrRegularArea.size(); j++) {
			if (IrRegularArea[j].size() > 100)
				drawContours(mark, IrRegularArea, j, CV_RGB(0, 255, 0), 3);
		}



		
		namedWindow("input'", WINDOW_NORMAL);
		imshow("input'", mark);
		waitKey(0);

		//imwrite("F:\\works\\Layout Analysis\\sd\\" + to_string(rand()) + ".png", mark);
		*/


		
		// �������ս���Ա�ͼ
		Mat result;
		hconcat(input, final_Picture2Show,result);
		int pos = fileName.find_last_of('\\');
		int lastPos = fileName.find_last_of('.');
		string name(fileName.substr(pos + 1, lastPos - pos - 1));
		imwrite(dstPath + name + ".jpg", result);
		cout << "done!" << index << ":" << name << endl;
		end = clock();
		printf("%d\n", end - start);
		index++;
		

		// ��json��ʽ����
		writeFileJson(jsonPath + name + ".json", contours_fix, bigAreaRects, word_componentRects);

		// ����program_image
		Mat program_image = input.clone();
		for (int i = 0; i < word_componentRects.size(); i++)
		{
			rectangle(program_image, Point(word_componentRects[i].x, word_componentRects[i].y), Point(word_componentRects[i].x + word_componentRects[i].width, word_componentRects[i].y + word_componentRects[i].height), CV_RGB(0, 0, 255), -1);
		}
		Get_Irregular_Contours(program_image, contours_fix, bigAreaRects, word_componentRects, 3);
		imwrite(programImagePath + name + ".bmp", program_image);
	}
}


