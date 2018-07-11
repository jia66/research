import cv2;
import numpy as np;
import matplotlib.pyplot as plt;
import os;
import sys;
 
def func(img,pic_path,pic):
	#把图片转换为灰度图
	gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
	#获取灰度图矩阵的行数和列数
	r,c = gray_img.shape[:2];
	dark_sum=0;	#偏暗的像素 初始化为0个
	dark_prop=0;	#偏暗像素所占比例初始化为0
	piexs_sum=r*c;	#整个弧度图的像素个数为r*c
	
	#遍历灰度图的所有像素
	for row in gray_img:
		for colum in row:
			if colum<30:	#人为设置的超参数,表示0~39的灰度值为暗
				dark_sum+=1;
	dark_prop=dark_sum/(piexs_sum);	
	#print("dark_sum:"+str(dark_sum));
	#print("piexs_sum:"+str(piexs_sum));
	#print("dark_prop=dark_sum/piexs_sum:"+str(dark_prop));
	if dark_prop >=0.75:	#人为设置的超参数:表示若偏暗像素所占比例超过0.78,则这张图被认为整体环境黑暗的图片
		print(pic_path+" is dark!");
		cv2.imwrite("./DarkPicDir/"+pic,img);#把被认为黑暗的图片保存
	#else:
		#print(pic_path+" is bright!")
	#hist(pic_path);  #若要查看图片的灰度值分布情况,可以这个注释解除
 
#用于显示图片的灰度直方图
def hist(pic_path):
	img=cv2.imread(pic_path,0);
	hist = cv2.calcHist([img],[0],None,[256],[0,256])
	plt.subplot(121)
	plt.imshow(img,'gray')
	plt.xticks([])
	plt.yticks([])
	plt.title("Original")
	plt.subplot(122)
	plt.hist(img.ravel(),256,[0,256])
	plt.show()
	
#读取给定目录的所有图片
def readAllPictures(pics_path):
	if not os.path.exists(pics_path):
		print("路径错误，路径不存在！")
		return;
	allPics = [];
	pics = os.listdir(pics_path);
	for pic in pics:
		pic_path = os.path.join(pics_path,pic);
		if os.path.isfile(pic_path):
			allPics.append(pic_path);
			img=cv2.imread(pic_path);
			func(img,pic_path,pic);
	return allPics;
 
#创建用于存放黑暗图片的目录
def createDarkDir():
	DarkDirPath = "./DarkPicDir";
	isExists = os.path.exists(DarkDirPath);
	if not isExists:
		os.makedirs(DarkDirPath);
		print("dark pics dir is created successfully!");
		return True;
	else:
		return False;
 
if __name__ =='__main__':
	#pics_path = sys.argv[1];#获取所给图片目录
	pics_path = './train'
	createDarkDir();
	allPics=readAllPictures(pics_path);
