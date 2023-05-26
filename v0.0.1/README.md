#v0.0.1版本
##更新项：
RLSA.cpp新增IsOverLap函数，用于解决两个Rect类型区域是否相交问题。
***image_word_correction函数：***在规则中添加IsOverLap函数，以期避免如果图像区域内同时包含图像和文本被进行水平投影切割。
***Get_Irregular_Contours函数:***新增一个仅针对二值化图像的重定义函数。
***Checking_for_gaps函数：***更改调用参数和规则。