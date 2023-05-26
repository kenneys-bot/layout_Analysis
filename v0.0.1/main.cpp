
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
	// 原始路径
	//string srcPath = "F:\\Layout analysis dataset\\Newdata\\data\\JPEGImages\\data\\";
	//string srcPath = "E:\\Y文档\\Y2\\版面分析\\Layout analysis dataset\\data\\";
	//string srcPath = "D:\\Data\\数据集\\data\\";
	//string srcPath = "D:\\Data\\数据集\\compress_double\\";
	//string srcPath = "E:\\Y文档\\Y1\\数据集\\ori_img\\ori_img\\DocBank_500K_ori_img\\";
	//string srcPath = "E:\\Y文档\\Y2\\版面分析\\Layout analysis dataset\\data1\\PRImA Layout Analysis Dataset\\Images\\";
	//string srcPath = "E:\\Data\\CDLA_DATASET\\train\\";
	//string srcPath = "E:\\Dataset\\PRImA Layout Analysis Dataset\\Images\\";
	//string srcPath = "E:\\Dataset\\compress_double\\";
	//string path = "F:\\works\\Layout Analysis\\data1\\JPEGImages\\data\\result\\result_pre_4\\";
	string path = "F:\\works\\Layout Analysis\\data\\JPEGImages\\data\\result\\result_pre_7\\";

	// 原始图像路径
	string srcPath = path + "image\\";
	// 图像结果路径
	string dstPath = path + "result\\";
	// 版面分析结构文件保存路径
	string jsonPath = path + "json\\";
	// 新旧二值化图像对比结果路径
	string binaryResultPath = path + "binaryResult\\";
	// program_image结果图像路径
	string programImagePath = path + "program_image\\";
	
	// Check whether source path exists
	if ((_access(srcPath.data(), 0)) != 0)
	{
		printf("Input path does not exist! \n\n");
		system("pause");
		return -1;
	}
    // 创建图像名称列表
	vector<String> file_vec;
	// 找srcPath下所有的jpg文件
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

	// 版面分析
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
		

		//图像纠偏
		//Mat roteInput = ScannedImageRectify(input);

		//图像去噪
		 Mat imageWithBlur;
		// 双边滤波去噪
		//bilateralFilter(inputCopy, imageWithBlur, 3, 100, 100);
		blur(inputCopy, imageWithBlur, Size(5,5), Point(-1, -1));

		//形态学操作
		/*Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
		erode(imageWithBlur, imageWithBlur, element, Point(-1, -1), 1, 0);*/

		//图像增强，对比度拉伸
		//imageWithBlur = contrastStretch1(imageWithBlur);


		//图像缩小
		/*Mat imageResize;
		resize(imageWithBlur, imageResize, Size(0, 0), 0.5, 0.5, INTER_AREA);*/

		//binaryImage by openCV
		//灰度
		Mat grayImage = convert2gray(imageWithBlur);
		//showPic(grayImage, "gray");

		//阈值处理
		Mat binaryImage_opencv = binaryzation(grayImage, 195);  //两次阈值操作处理？
		//showPic(binaryImage_opencv, "thres_200");

		Mat binaryImage_opencv_make_border = image_make_border(binaryImage_opencv);
		//showPic(binaryImage_opencv_make_border, "binaryImage_opencv_make_border");
		// 
		//归一化，置0-1
		Mat reuniteImage_opencv = normalization(binaryImage_opencv);
		//Mat reuniteImage_opencvshow = reuniteImage_opencv * 255;
		//showPic(reuniteImage_opencvshow, "reuniteImage_opencv");
		//imwrite("D:\\VS_Projects\\Layout Analysis\\data\\reuniteImage_opencvshow.jpg", reuniteImage_opencvshow);

		Mat reuniteImage_opencv_make_border = normalization(binaryImage_opencv_make_border);
		
		
		
		//binaryImage by model
		Mat binaryImage_model;
		doc_bin.docBin(inputCopy, binaryImage_model);




		//大概做了一个干扰边缘处理
		Mat digitImage_opencv = InterferenceImage(reuniteImage_opencv);
		//showPic(digitImage_opencv,"digitImage_opencv");

		Mat digitImage_opencv_make_border = InterferenceImage(reuniteImage_opencv_make_border);



		// 获得直线
		/*vector<Rect> allRowLines, allColLines;
		//findLine(digitImage, allRowLines, allColLines);
		findTableByLSD(input, allRowLines, allColLines);*/


		// opencv预处理下的较大连通域获取
		std::vector<std::vector<Point>> contours;
		vector<Rect>bigArea = bigAreaDetect(digitImage_opencv, contours); //返回》1 《60大小的连通域
		/*Mat bigArea2Show = input.clone();
		for (int i = 0; i < bigArea.size(); i++)
		{
			//对符合要求的连通域用矩形框定
			rectangle(bigArea2Show, Point(bigArea[i].x, bigArea[i].y), Point(bigArea[i].x + bigArea[i].width, bigArea[i].y + bigArea[i].height), cv::Scalar(255, 0, 0), 5, 1, 0);
		}
		showPic(bigArea2Show, "bigArea2Show");*/

		//vector<vector<Point> > contours_fix = bigAreaDetect(digitImage_opencv);//重载了一个较大连通域搜索函数
		vector<vector<Point> > contours_fix = bigAreaDetect(digitImage_opencv_make_border);

		vector<Rect> bigAreaRects;


		/*---------去除表格区域---------*/
		//vector<Rect> TableRects;
		//if (allRowLines.size() > 0)
		//{
		//	std::vector<Rect>::const_iterator itr = bigArea.begin();
		//	// 较大区域类型判断(表格区域)
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
		//// 剩余直线分组(表格区域)
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
		//				itr_next = TableRects.erase(itr_next);//如果包含就删除
		//			}
		//			else
		//				itr_next++;
		//		}
		//		itr++;
		//	}
		//}
		//// 表格区域去除
		//Mat tableDel = plotRect(digitImage, TableRects);






		//对较大连通域进行填充
		Get_Irregular_Contours(binaryImage_model, contours_fix, bigAreaRects);
		//drawContours(binaryImage_model, contours_fix, -1, CV_RGB(255, 255, 255), -1);
		//showPic(binaryImage_model, "binaryImage_model");

		Mat reuniteImage_model = normalization(binaryImage_model);
		Mat digitImage_model = InterferenceImage(reuniteImage_model); //边缘处理过的图像
		Mat imageDel_model = digitImage_model;
		drawContours(imageDel_model, contours_fix, -1, CV_RGB(0, 0, 0), -1);


		//去直线干扰
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
		// 并操作
		Mat afterSmooth = afterSmoothVer & afterSmoothHor;
		Mat afterSmooth2Show_1 = afterSmooth * 255;
		//showPic(afterSmooth2Show_1,"afterSmooth2Show_1");
		
		//又一次水平平滑
		afterSmooth = lengthSmoothHor(afterSmooth, threshold_RLSA2 * 0.95);
		Mat afterSmooth2Show_2 = afterSmooth * 255;
		//showPic(afterSmooth2Show_2,"afterSmooth2Show_2");
		

		//Mat afterSmoothHor = lengthSmoothHor(lineDel, imageResize.cols * 0.2);
		//Mat afterSmoothVer = lengthSmoothVer(lineDel, imageResize.rows * 0.1);
		//// 并操作
		//Mat afterSmooth = afterSmoothVer & afterSmoothHor;
		////Mat dilateImage = doDilation(afterSmooth, 3);

		Mat afterSmoothHor2Show = afterSmoothHor * 255;
		//showPic(afterSmoothHor2Show, "afterSmoothHor2Show");
		Mat afterSmoothVer2Show = afterSmoothVer * 255;
		//showPic(afterSmoothVer2Show, "afterSmoothVer2Show");

		
		//文字检测
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
		
		// 文字区域排序（没出现问题）
		vector<Rect> rowFirst_wordRects;
		vector<int> row_num;
		Mat final_Picture = input.clone();//用于显示行首文本框
		vector<Rect> sorted_wordRects = textlinesSorting(final_Picture, word_componentRects, rowFirst_wordRects, row_num);
		cout << "文字排序完毕！" << endl;

		// 提取后景像素块（vector越界） 
		/*vector<int> bgColor_array = getBgPixelblock(grayImage, sorted_wordRects, rowFirst_wordRects, row_num);
		int bgColor_sum = 0;
		for (int k = 0; k < bgColor_array.size(); k++)
		{
			bgColor_sum += bgColor_array[k];
		}
		int bgColor_value = bgColor_sum / bgColor_array.size();
		cout << "bgcolor_value" << bgColor_value << endl;*/

		
		///*提取后景颜色块--梁文伟*/
		////int bgColor_value = catch_background_color_block(inputCopy, sorted_wordRects);

		// 再次二值化图片
		//Mat re_binaryImage = binaryzation(grayImage, bgColor_value);
		//showPic(re_binaryImage, "再次二值化图像");

		// 阈值195二值化图与新二值化图拼接——binaryResult
		/*Mat binaryResult;
		hconcat(binaryImage_opencv, re_binaryImage, binaryResult);
		int pos_0 = fileName.find_last_of('\\');
		int lastPos_0 = fileName.find_last_of('.');
		string name_0(fileName.substr(pos_0 + 1, lastPos_0 - pos_0 - 1));
		imwrite(binaryResultPath + name_0 + ".jpg", binaryResult);*/

		// 纠错、查漏补缺
		image_word_correction(binaryImage_opencv, inputCopy, contours_fix, bigAreaRects, word_componentRects);
		Checking_for_gaps(binaryImage_opencv, inputCopy, contours_fix, bigAreaRects, word_componentRects);
		vertical_projection_for_word(word_componentRects, word_componentRects, reuniteImage_opencv, zaodian_componentRects);

		// 查看矩形框容器内容
		//for (vector<Rect>::iterator it = word_componentRects.begin(); it != word_componentRects.end(); it++) {
		//	cout << *it << endl;
		//	//[288 x 27 from (1095, 3021)]  width,height from(x,y)
		//}

		
		Mat final_Picture2Show = input.clone();
		//绘制图片轮廓
		//drawContours(final_Picture2Show, contours_fix, -1, CV_RGB(255, 0, 0), 2);
		Get_Irregular_Contours(final_Picture2Show, contours_fix, bigAreaRects, word_componentRects, 2);
		//加入垂直投影后的文本行框定
		for (int i = 0; i < word_componentRects.size(); i++)
		{
			rectangle(final_Picture2Show, Point(word_componentRects[i].x, word_componentRects[i].y), Point(word_componentRects[i].x + word_componentRects[i].width, word_componentRects[i].y + word_componentRects[i].height), CV_RGB(0, 0, 255), 8);
			//showPic(final_Picture2Show, "final_Picture2Show");
		}
		//加入垂直投影后的非文本行框定
		/*for (int i = 0; i < zaodian_componentRects.size(); i++)
		{
			rectangle(final_Picture2Show, Point(zaodian_componentRects[i].x, zaodian_componentRects[i].y), Point(zaodian_componentRects[i].x + zaodian_componentRects[i].width, zaodian_componentRects[i].y + zaodian_componentRects[i].height), cv::Scalar(0, 255, 0), 5, 1, 0);

		}*/

		//绘制文本行轮廓
		//for (int i = 0; i < componentRects.size(); i++)
		//{
		//	rectangle(final_Picture2Show, Point(componentRects[i].x, componentRects[i].y), Point(componentRects[i].x + componentRects[i].width, componentRects[i].y + componentRects[i].height), CV_RGB(255, 0, 0), 8);
		//	//showPic(final_Picture2Show, "final_Picture2Show");
		//}


		/*showPic(final_Picture2Show, "final_Picture2Show");
		waitKey(0);
		imwrite("D:\\VS_Projects\\Layout Analysis\\data\\" + fileName, final_Picture2Show);*/

		/*
		//对文字区域和图片区域相交情况的处理
		std::vector<Rect> textRects = componentRects;
		std::vector<Rect> deletebigArea = imageInterText(bigArea, textRects);
		//截取去掉的那一个图片区域横向区域，对其去文字，不规则图形检测和文字检测
		std::vector<std::vector<Point>> irRegularArea;
		if(!deletebigArea.empty())
		{
			test(deletebigArea, irRegularArea, textRects, input);
		}


		Mat textDel = plotRect(afterSmooth, textRects, width*0.4);

		
		recRankYMTS(textRects);
		
		// 表边文字整合
		//tableTextIntegrate(imageResize.cols, TableRects, textRects);
		
		//表内图片去除
		//imageInsideTable(TableRects, bigArea);
		
		//文字区域投影判定
		text_ver_projection(digitImage_model, textRects, bigArea);

		//图内文字去除
		Graph_Text_remove(textRects, bigArea);
		
		//图像区域投影判定
		textFeatureDetect(binaryImage_model, textRects, bigArea);

		//屏蔽文字区域单独获取图像区域轮廓
		vector<vector<Point> > IrRegularArea = getirRegularArea(input, textRects, deletebigArea, bigArea);



		// 区域框定
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


		
		// 绘制最终结果对比图
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
		

		// 以json形式储存
		writeFileJson(jsonPath + name + ".json", contours_fix, bigAreaRects, word_componentRects);

		// 生成program_image
		Mat program_image = input.clone();
		for (int i = 0; i < word_componentRects.size(); i++)
		{
			rectangle(program_image, Point(word_componentRects[i].x, word_componentRects[i].y), Point(word_componentRects[i].x + word_componentRects[i].width, word_componentRects[i].y + word_componentRects[i].height), CV_RGB(0, 0, 255), -1);
		}
		Get_Irregular_Contours(program_image, contours_fix, bigAreaRects, word_componentRects, 3);
		imwrite(programImagePath + name + ".bmp", program_image);
	}
}


