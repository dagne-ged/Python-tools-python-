import numpy as np
from PIL import Image
def mtx_similar1(arr1:np.ndarray, arr2:np.ndarray) ->float:
    '''
    计算矩阵相似度的一种方法。将矩阵展平成向量，计算向量的乘积除以模长。
    注意有展平操作。
    :param arr1:矩阵1
    :param arr2:矩阵2
    :return:实际是夹角的余弦值，ret = (cos+1)/2
    '''
    farr1 = arr1.ravel()
    farr2 = arr2.ravel()
    len1 = len(farr1)
    len2 = len(farr2)
    if len1 > len2:
        farr1 = farr1[:len2]
    else:
        farr2 = farr2[:len1]
    #~如果矩阵的维度、向量的模长不一样。
    numer = np.sum(farr1 * farr2)
    denom = np.sqrt(np.sum(farr1**2) * np.sum(farr2**2))
    similar = numer / denom # 这实际是夹角的余弦值
    return  (similar+1) / 2     # 姑且把余弦函数当线性

def mtx_similar2(arr1:np.ndarray, arr2:np.ndarray) ->float:
    '''
    计算对矩阵1的相似度。相减之后对元素取平方再求和。因为如果越相似那么为0的会越多。
    如果矩阵大小不一样会在左上角对齐，截取二者最小的相交范围。
    :param arr1:矩阵1
    :param arr2:矩阵2
    :return:相似度（0~1之间）
    '''
    if arr1.shape != arr2.shape:
        minx = min(arr1.shape[0],arr2.shape[0])
        miny = min(arr1.shape[1],arr2.shape[1])
        differ = arr1[:minx,:miny] - arr2[:minx,:miny]
    else:
        differ = arr1 - arr2
    numera = np.sum(differ**2)
    denom = np.sum(arr1**2)
    similar = 1 - (numera / denom)
    return similar


def mtx_similar3(arr1:np.ndarray, arr2:np.ndarray) ->float:
    '''
    From CS231n: There are many ways to decide whether
    two matrices are similar; one of the simplest is the Frobenius norm. In case
    you haven't seen it before, the Frobenius norm of two matrices is the square
    root of the squared sum of differences of all elements; in other words, reshape
    the matrices into vectors and compute the Euclidean distance between them.
    difference = np.linalg.norm(dists - dists_one, ord='fro')
    :param arr1:矩阵1
    :param arr2:矩阵2
    :return:相似度（0~1之间）
    '''
    if arr1.shape != arr2.shape:
        minx = min(arr1.shape[0],arr2.shape[0])
        miny = min(arr1.shape[1],arr2.shape[1])
        differ = arr1[:minx,:miny] - arr2[:minx,:miny]
        #~如果矩阵的大小不一样
    else:
        differ = arr1 - arr2
    dist = np.linalg.norm(differ, ord='fro')
    len1 = np.linalg.norm(arr1)
    len2 = np.linalg.norm(arr2)     # 普通模长
    denom = (len1 + len2) / 2
    similar = 1 - (dist / denom)
    return similar


import random
def find_position_process(page:str,pic:str):
    '''
    问题：在一个大图中查找一个小图/模板图
    这是基础但低效的算法，效率之低而几乎不会用——在我的电脑上一张一千多像素宽度的图查一个400*400图用了超过20min。
    :param page:大图
    :param pic:小图
    :return:打印匹配度较高的位置
    '''
    pageArr = np.array(Image.open(page).convert('L'))
    picArr = np.array(Image.open(pic).convert('L'))     # 首先转换成灰度图像
    hei,wid = picArr.shape
    # if hei > 600:
    #     #太大了,裁一个400*400以内的小块，用之预比较
    #     hStart = int(random.uniform(0,hei-400))
    #     if wid > 400:
    #         wStart = int(random.uniform(0,wid-400))
    #         smallArr = picArr[hStart:hStart + 400, wStart:wStart + 400]
    #     else:
    #         smallArr = picArr[hStart:hStart + 400,:]
    # elif wid > 600:
    #     wStart = int(random.uniform(0,wid-400))
    #     smallArr = picArr[:, wStart:wStart+400]     # 好吧，实际上保证是比较块一定在600*600以内
    # else:
    #……发现可能大的子图比较的反而快
    smallArr = picArr
    sHei, sWid = smallArr.shape
    for i in range(0,hei - sHei):
        if i%100 == 0:
            print('do i:', i)
        for k in range(0, wid - sWid):
            sim = mtx_similar2(smallArr, pageArr[i: i+sHei, k: k+sWid])
            if sim > 0.95:
                print(i,k)

#：： 一段网上找到的openCV代码，也是滑动窗口法，慢。
# # 简单定位图片
# import cv2
# # import numpy as np
#
# def showpiclocation(img, findimg):  # 定义定位函数
#     # 定位图片
#     w = img.shape[1]  # 返回img的第二维度长度---宽度
#     h = img.shape[0]  # 返回img的第一维度长度---高度
#     fw = findimg.shape[1]
#     fh = findimg.shape[0]
#     findpt = None
#     for now_h in range(0, h - fh):
#         for now_w in range(0, w - fw):
#             comp_tz = img[now_h:now_h + fh, now_w:now_w + fw, :] - findimg
#             if np.sum(comp_tz) < 1:
#                 findpt = now_w, now_h
#
#     if findpt != None:
#         cv2.rectangle(img, findpt, (findpt[0] + fw, findpt[1] + fh), (0, 0, 255))  # opencv函数画矩形
#     return img
#
# fn = '00000033.tif'
# fn1 = '000000331.jpg'
# # fn2 = 'pictestt2.png'
# myimg = cv2.imread(fn)
# myimg1 = cv2.imread(fn1)
# # myimg2 = cv2.imread(fn2)
# myimg = showpiclocation(myimg, myimg1)
# # myimg = showpiclocation(myimg, myimg2)
# cv2.namedWindow('img')
# cv2.imshow('img', myimg)
# cv2.waitKey()
# cv2.destroyAllWindows()


#：KMP字符串匹配算法
def KMP(bigStr:str, smallStr:str, all = False)->(list, bool, int):
    '''
    KMP算法，用于查找后字符串在前面字符串的匹配位置。
    :param bigStr: 包含的字符串
    :param smallStr: 要检测是否含有的字串
    :param all: 是否要找到所有的位置。默认为false
    :return: 匹配（或最佳匹配）的位置的列表，从0开始,-1为完全没有找到。
        bool变量是否是精确匹配。最后第三个返回值是匹配的长度。
    '''
    def cal_next(string:str)->np.ndarray:
        '''
        KMP算法所需要的子函数
        :param string:
        :return:
        '''
        k = -1
        length = len(string)
        nextArr = -np.ones(length,dtype=int) #全赋值为-1，不过算法里只要【0】赋值为-1即可
        for qq in range(1,length):
            while string[k + 1] != string[qq] and k>-1: # 第一个放过
                k = nextArr[k]      # 向前回溯。qq一定大于k，故安全
            if string[k+1] == string[qq]:
                k += 1
            nextArr[qq] = k
        return nextArr

    nextA = cal_next(smallStr)
    kk = -1
    exact = False
    bestPos = list()
    bestK = -1  # store the answer
    bigLen = len(bigStr)
    smaLen = len(smallStr)
    for ii in range(0,bigLen):
        while bigStr[ii] != smallStr[kk+1] and kk>-1:
            kk = nextA[kk]
        if bigStr[ii] == smallStr[kk+1]:
            kk += 1
            if kk > bestK:
                bestK = kk
                bestPos = [ii-kk,]
            elif kk == bestK:
                bestPos.append(ii - kk)
            else:
                pass
        if kk == smaLen - 1:
            exact = True
            if not all:     #只要一个就好的话
                # return ii - kk, exact
                return bestPos, exact, bestK+1
            else:
                kk = -1   # 找到了一个完全匹配，继续开始
    return bestPos, exact, bestK+1


def KMP_int_near(bigStr:str, smallStr:str, all = False, near = 0)->(list,bool, int):
    '''
    查找后字符串在前面字符串的匹配位置。
    :param bigStr: 包含的字符串
    :param smallStr: 要检测是否含有的字串
    :param all: 是否要找到所有的位置。默认为false
    :return: 匹配（或最佳匹配）的位置的列表，从0开始,-1为完全没有找到。
        bool变量是否是精确匹配。最后第三个返回值是匹配长度。
    '''
    nearEqua = lambda a,b: a<b+near and a>b-near

    def cal_next(string:str)->np.ndarray:
        k = -1
        length = len(string)
        nextArr = -np.ones(length,dtype=int) #全赋值为-1，不过算法里只要【0】赋值为-1即可
        for qq in range(1,length):

            while not nearEqua(string[k + 1], string[qq]) and k>-1: # 第一个放过
                k = nextArr[k]      # 向前回溯。qq一定大于k，故安全
            if nearEqua(string[k + 1], string[qq]):
                k += 1
            nextArr[qq] = k
        return nextArr

    nextA = cal_next(smallStr)
    kk = -1
    exact = False
    bestPos = list()
    bestK = -1  # store the answer
    bigLen = len(bigStr)
    smaLen = len(smallStr)
    for ii in range(0,bigLen):
        while not nearEqua(bigStr[ii], smallStr[kk+1]) and kk>-1:
            kk = nextA[kk]
        if nearEqua(bigStr[ii], smallStr[kk+1]):
            kk += 1
            if kk > bestK:
                bestK = kk
                bestPos = [ii-kk,]
            elif kk == bestK:
                bestPos.append(ii - kk)
            else:
                pass
        if kk == smaLen - 1:
            exact = True
            if not all:     #只要一个就好的话
                # return ii - kk, exact
                return bestPos, exact, bestK+1
            else:
                kk = -1   # 找到了一个完全匹配，继续开始
    return bestPos, exact, bestK+1


def testKMP():
    a = '100100001111010101011111100'
    b = '11110'
    poiList,ex,le = KMP(a,b,True)
    print(poiList, ex, le)
    a = 's ixdsrvsnkjandsoicnzxjdskcasdajnc'
    b = 'sxzihih'
    poiList, ex, le = KMP(a, b)
    print(poiList,ex,le)
    a = 'bacbababadababacambabacaddababacasdsd'
    b = 'ababaca'
    poiList,ex,le = KMP(a,b,True)
    print(poiList, ex, le)
    a = 'bacbababadababacambabacaddababacasdsd'
    b = '13323'
    poiList,ex,le = KMP(a,b,True)
    print(poiList, ex, le)


#---------------------------------------------------------------

def find_poi_by_deltaString(picB, picS):
    '''
    在一个图片中查找一个子图/模板图的位置。将图像投影到一维形成“字符串”，使用字符串匹配算法进行快速查找。
    首先转换为灰度矩阵。再将图像0-1化了（这里的粒度可以修改）
    对小图求差分，得到小图字符串。对大图切成和小图一样宽的行，找到在这一行中的最佳匹配，
    循环对所有行（注意行是相互重叠的，相差一个像素行一个像素行），找到最佳的位置。
    :param picB:大图，
    :param picS:小图，模板图，用于查找的图
    :return:打印最佳匹配的位置（可能有多个）
    参照论文：孙远,周刚慧,赵立初,施鹏飞  灰度图像匹配的快速算法 上海交通大学学报
    '''
    arr11 = np.array(Image.open(picB).convert('L'))
    arr22 = np.array(Image.open(picS).convert('L'))     # 转换为灰度矩阵。
    arr1 = np.round(arr11/128)
    arr2 = np.round(arr22/128)   # 转成0-1图，小于128的像素点都被当作0。为了更鲁棒。
    Hei, Wid = arr1.shape
    sHei, sWid = arr2.shape
    tem1 = np.sum(arr2,axis=0,dtype=np.longlong)   #投影（简单求和）.axis0为上下方向，形成一行
    smaStr = np.zeros(sWid-1, dtype=np.longlong)    # 矩阵数据类型要选longlong，防止溢出
    for k in range(0, sWid-1):
        smaStr[k] = tem1[k] - tem1[k+1] # 求差分，长度变为Swid-1
    #~ 这里也可以用roll函数。求差分是为了更加鲁棒。一般像素之间的差分变动的要小。

    bestFitLen = 0
    bestWidth = list()
    bestHeight = list()
    for i in range(0, Hei-sHei+1):
        tem = np.sum(arr1[i:i+sHei,:],axis=0,dtype=np.longlong)  #投影（求和）
        bigStr = np.zeros(Wid-1,dtype=np.longlong)
        for k in range(0, Wid - 1):
            bigStr[k] = tem[k] - tem[k + 1]  # 求差分，长度变为wid-1
        #:use Kmp
        temws, fit, teml = KMP(bigStr,smaStr,True)    # 为了更鲁棒,可用KPM_int_near
        if teml > bestFitLen:
            bestWidth = [temws,]
            bestFitLen = teml
            bestHeight = [i]
        elif teml == bestFitLen:
            bestWidth.append(temws)     # 注意是嵌套的两个列表
            bestHeight.append(i)
        else:
            pass
    print('Ans is:', bestHeight)
    print(bestWidth)
    print('fit length vs sma length:',bestFitLen, sWid-1)
    #~



from scipy.ndimage import filters
def find_poi_by_GuassianString(picB, picS):
    '''
    在一个图片中查找一个子图/模板图的位置。这是一个实验版本。
    result：效果不佳
    将图像投影到一维形成“字符串”，使用字符串匹配算法进行快速查找。
    对灰度图进行了高斯模糊，试图增强其鲁棒性。但反而并没有。
    原因可能是出现在边缘。因为对大图进行高斯模糊再切成小块，大图的小块的边缘受到了其周围的像素的干扰。
    :param picB:大图，
    :param picS:小图，模板图，用于查找的图
    :return:打印最佳匹配的位置（可能有多个）
    参照论文：孙远,周刚慧,赵立初,施鹏飞  灰度图像匹配的快速算法 上海交通大学学报
    '''
    arr11 = np.array(Image.open(picB).convert('L'))/16
    arr22 = np.array(Image.open(picS).convert('L'))/16
    arr1 = filters.gaussian_filter(arr11,sigma=1)
    arr2 = filters.gaussian_filter(arr22,sigma=1)  # 用高斯滤波增强鲁棒
    Hei, Wid = arr1.shape
    sHei, sWid = arr2.shape
    smaStr = np.sum(arr2,axis=0,dtype=np.longlong)   #向x轴投影（求和），不再求差分

    bestFitLen = 0
    bestWidth = list()
    bestHeight = list()
    for i in range(0, Hei-sHei+1):
        bigStr = np.sum(arr1[i:i+sHei,:],axis=0,dtype=np.longlong)  # 向x轴投影（求和）

        #:use Kmp
        temws, fit, teml = KMP(bigStr,smaStr,True)
        if teml > bestFitLen:
            bestWidth = [temws,]
            bestFitLen = teml
            bestHeight = [i]
        elif teml == bestFitLen:
            bestWidth.append(temws)     # 注意是嵌套的两个列表
            bestHeight.append(i)
        else:
            pass
    print('Height , width is:', bestHeight)
    print(bestWidth)
    print('fit length vs sma length:',bestFitLen, sWid)
    # ~ 发现高斯滤波的效果反而不好


def find_poi_by_String(picB, picS):
    '''
    在一个图片中查找一个子图/模板图的位置。但这里没有求图像的差分。
    result：看情况，只要噪声不大用这个版本就好
    将图像投影到一维形成“字符串”，使用字符串匹配算法进行快速查找。
    首先转换为灰度矩阵。再将图像0-1化了（这里的粒度可以修改）
    得到小图字符串。对大图切成和小图一样宽的行，找到在这一行中的最佳匹配，
    循环对所有行（注意行是相互重叠的，相差一个像素行一个像素行），找到最佳的位置。
    :param picB:大图，
    :param picS:小图，模板图，用于查找的图
    :return:打印最佳匹配的位置（可能有多个）
    参照论文：孙远,周刚慧,赵立初,施鹏飞  灰度图像匹配的快速算法 上海交通大学学报
    '''
    arr11 = np.array(Image.open(picB).convert('L'))
    arr22 = np.array(Image.open(picS).convert('L'))
    arr1 = np.round(arr11/128)
    arr2 = np.round(arr22/128)   # 为了更鲁棒。数值可以调。
    Hei, Wid = arr1.shape
    sHei, sWid = arr2.shape
    smaStr = np.sum(arr2,axis=0,dtype=np.longlong)   #向x轴投影（求和）

    bestFitLen = 0
    bestWidth = list()
    bestHeight = list()
    for i in range(0, Hei-sHei+1):
        bigStr = np.sum(arr1[i:i+sHei,:],axis=0,dtype=np.longlong)  # 向x轴投影（求和）

        #:use Kmp
        temws, fit, teml = KMP(bigStr,smaStr,True)    # 为了更鲁棒,可用KPM_near
        if teml > bestFitLen:
            bestWidth = [temws,]
            bestFitLen = teml
            bestHeight = [i]
        elif teml == bestFitLen:
            bestWidth.append(temws)
            bestHeight.append(i)
        else:
            pass
    print('Ans is:', bestHeight)
    print(bestWidth)
    print('fit length vs sma length:',bestFitLen, sWid)


def find_poi_by_StringMean(picB, picS):
    '''
    在一个图片中查找一个子图/模板图的位置。不求差分，用平均值增强鲁棒。
    同时还可以考虑使用KPM—near
    result：效果可以。
    将图像投影到一维形成“字符串”，使用字符串匹配算法进行快速查找。
    首先转换为灰度矩阵。再将图像0-1化了（这里的粒度可以修改）
    得到小图字符串。对大图切成和小图一样宽的行，找到在这一行中的最佳匹配，
    循环对所有行（注意行是相互重叠的，相差一个像素行一个像素行），找到最佳的位置。
    :param picB:大图，
    :param picS:小图，模板图，用于查找的图
    :return:打印最佳匹配的位置（可能有多个）
    参照论文：孙远,周刚慧,赵立初,施鹏飞  灰度图像匹配的快速算法 上海交通大学学报
    '''
    arr1 = np.array(Image.open(picB).convert('L'))
    arr2 = np.array(Image.open(picS).convert('L'))      # 转灰度矩阵，不再进行粒度模糊。
    Hei, Wid = arr1.shape
    sHei, sWid = arr2.shape
    smaStr = np.mean(arr2,axis=0,dtype=np.longlong)   #向x轴投影（求均值）
    bestFitLen = 0
    bestWidth = list()
    bestHeight = list()
    for i in range(0, Hei-sHei+1):
        bigStr = np.mean(arr1[i:i+sHei,:],axis=0,dtype=np.longlong)  # 向x轴投影（求均值）
        #:use Kmp
        temws, fit, teml = KMP_near(bigStr,smaStr,True,2)    # 为了更鲁棒,可用KPM_near
        if teml > bestFitLen:
            bestWidth = [temws,]
            bestFitLen = teml
            bestHeight = [i]
        elif teml == bestFitLen:
            bestWidth.append(temws)
            bestHeight.append(i)
        else:
            pass
    print('Ans is:', bestHeight)
    print(bestWidth)
    print('fit length vs sma length:',bestFitLen, sWid)


def find_poi_by_StringMean2(picB, picS):
    '''
    增加了方向选择，增强效率。因为KMP算法是相对很快的。
    在一个图片中查找一个子图/模板图的位置。不求差分，用平均值增强鲁棒。
    同时还可以考虑使用KPM—near
    将图像投影到一维形成“字符串”，使用字符串匹配算法进行快速查找。
    首先转换为灰度矩阵。再将图像0-1化了（这里的粒度可以修改）
    得到小图字符串。对大图切成和小图一样宽的行，找到在这一行中的最佳匹配，
    循环对所有行（注意行是相互重叠的，相差一个像素行一个像素行），找到最佳的位置。
    :param picB:大图，
    :param picS:小图，模板图，用于查找的图
    :return:打印最佳匹配的位置（可能有多个）
    参照论文：孙远,周刚慧,赵立初,施鹏飞  灰度图像匹配的快速算法 上海交通大学学报
    '''
    arr1 = np.array(Image.open(picB).convert('L'))
    arr2 = np.array(Image.open(picS).convert('L'))
    Hei, Wid = arr1.shape
    sHei, sWid = arr2.shape
    # sWid = int(sWid/4)
    # 发现子图变小后，搜索时间反而变长了—— Wid次的字符串比较才是影响时间效率的主要因素
    if Hei > Wid:
        dirc = 1    # 向y轴投影更好
        smaStr = np.mean(arr2[:,:sWid], axis=dirc, dtype=np.longlong)  # 向x\y轴投影（求和）
        bestFitLen = 0
        bestWidth = list()
        bestHeight = list()
        for i in range(0, Wid - sWid + 1):
            bigStr = np.mean(arr1[:, i:i + sWid], axis=dirc, dtype=np.longlong)  # 向x轴投影（求和）

            #:use Kmp
            temHeis, fit, temlen = KMP_int_near(bigStr, smaStr, True, 2)  # 为了更鲁棒,可用KPM_near
            if temlen > bestFitLen:
                bestWidth = [i]
                bestFitLen = temlen
                bestHeight = [temHeis, ]
            elif temlen == bestFitLen:
                bestHeight.append(temHeis)
                bestWidth.append(i)
            else:
                pass
    else:# 向x轴投影
        dirc = 0
        smaStr = np.mean(arr2[:,:sWid], axis=dirc, dtype=np.longlong)  # 向x\y轴投影（求和）
        bestFitLen = 0
        bestWidth = list()
        bestHeight = list()
        for i in range(0, Hei - sHei + 1):
            bigStr = np.mean(arr1[i:i + sHei, :], axis=0, dtype=np.longlong)  # 向x轴投影（求和）
            #:use Kmp
            temws, fit, temlen = KMP_near(bigStr, smaStr, True, 2)  # 为了更鲁棒,可用KPM_near
            if temlen > bestFitLen:
                bestWidth = [temws, ]
                bestFitLen = temlen
                bestHeight = [i]
            elif temlen == bestFitLen:
                bestWidth.append(temws)
                bestHeight.append(i)
            else:
                pass
    print('bestHeight is:', bestHeight)
    print('bestWidth is:', bestWidth)
    print('fit length vs sma length:',bestFitLen, sWid)



# import time
# c1 = time.clock()
# # find_poi_by_StringMean('big.jpg','inbig2.jpg')
# find_poi_by_StringMean3('00000033.tif','000000331.jpg')
# c2 = time.clock()
# print(c1,c2)
