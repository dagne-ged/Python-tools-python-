#：KMP字符串匹配算法
def KMP(bigStr:str, smallStr:str, all = False)->(list,bool, int):
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


def KMP_int_near(bigStr: str, smallStr: str, all=False, near=0) -> (list, bool, int):
    '''
    KMP算法另一修改版，用于查找后字符串在前面字符串的匹配位置。
    这里计算的可以不是字符串，只要是可迭代序列即可（支持[]操作）。但要求是数值变量（支持+->=<操作）。
    判定某两个字符/元素相等的比较是近似比较，相互差值在near范围内即认为是相等。near默认值是0。
    :param bigStr: 包含的字符串
    :param smallStr: 要检测是否含有的字串
    :param all: 是否要找到所有的位置。默认为false
    :param near: 认为相等的范围域，默认是0。
    :return: 匹配（或最佳匹配）的位置的列表，从0开始,-1为完全没有找到。
        bool变量是否是精确匹配。最后第三个返回值是匹配长度。
    '''
    nearEqua = lambda a, b: a < b + near and a > b - near
    #~ 相等比较函数
    def cal_next(string: str) -> np.ndarray:
        k = -1
        length = len(string)
        nextArr = -np.ones(length, dtype=int)  # 全赋值为-1，不过算法里只要【0】赋值为-1即可
        for qq in range(1, length):

            while not nearEqua(string[k + 1], string[qq]) and k > -1:  # 第一个放过
                k = nextArr[k]  # 向前回溯。qq一定大于k，故安全
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
    for ii in range(0, bigLen):
        while not nearEqua(bigStr[ii], smallStr[kk + 1]) and kk > -1:
            kk = nextA[kk]
        if nearEqua(bigStr[ii], smallStr[kk + 1]):
            kk += 1
            if kk > bestK:
                bestK = kk
                bestPos = [ii - kk, ]
            elif kk == bestK:
                bestPos.append(ii - kk)
            else:
                pass
        if kk == smaLen - 1:
            exact = True
            if not all:  # 只要一个就好的话
                # return ii - kk, exact
                return bestPos, exact, bestK + 1
            else:
                kk = -1  # 找到了一个完全匹配，继续开始
    return bestPos, exact, bestK + 1

