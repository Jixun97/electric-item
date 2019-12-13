# -*- coding: utf-8 -*-
import numpy as np
import numpy.linalg as la
from scipy.linalg import sqrtm
import xlrd
from pandas.core.frame import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys
import tqdm
from tqdm import trange
import os
import tempfile
import pytest


matrix_out = []
var_evs_out = []
mark = []
count1, count2, count3, count4 = 0, 0, 0, 0
m = []

def Stochastic_matrix(input_ematrix, count):  # 每次进一个表 #exp: directory = "C:\\user\\..."
    # def clear(): os.system('cls')
    # 读入数据。。。。。
    t_s = DataFrame(input_ematrix)  # 时间序列数据
    (m, n) = np.shape(t_s)  # 读取维度大小
    t_s = np.matrix(t_s)
    evs = []  # 单个表平均谱半径矩阵
    matrix = [] # 过渡矩阵
    N = 24  # 窗口时间大小\

    Q = la.qr(np.random.randn(4, 4) + 1j * np.random.randn(4, 4))[0]
    U = np.dot(Q, np.diag(np.exp(2 * np.pi * 1j * np.random.rand(4)))) # change randomly, different result
    row_num = 0  # the row of datafram
    # loop = int(n / 24)
    # 平均谱半径计算（窗口循环）
    for i in range(10):  # 窗口移动次数
        # 生成窗口矩阵
        X = t_s[row_num:row_num+4,:]
        S = 1. / N * np.dot(X, X.H)
        # 计算酉矩阵
        #Q = la.qr(np.random.randn(4, 4) + 1j * np.random.randn(4, 4))[0]
        #U = np.dot(Q, np.diag(np.exp(2 * np.pi * 1j * np.random.rand(4))))  # Haar unitary matrix 生成
        # 计算特征值
        Y = np.dot(sqrtm(S), U)
        es = la.eigvals(Y)
        es = abs(es)
        #print(es)

        # 平均谱半径
        es = np.mean(es)
        matrix.append(es)

        #print(matrix)
        # 循环
        row_num = row_num + 4

    # 生成矩阵（所有电能表时序特征矩阵）
    evs = np.array(matrix)
    evs = evs.T

    # 求权重
    var_evs = np.var(evs)  # 输出方差
    #print(var_evs)
    
    #写文件
    #with open('./matrix.txt','w',encoding='utf-8') as f:
        #f.write(str(matrix))

    #r = "\t表%d完成" %count
    #sys.stdout.write(r)
    #time.sleep(0.1)
    #sys.stdout.flush()
    
    return var_evs, matrix


def weightcalculation(var_evs=[1]):  # 总共几个表, 请写入参数
    # var_evs = []
    j = 0
    # 评价状态
    level = []  # 状态
    global mark
    var_max = max(var_evs)
    var_min = min(var_evs)
    var_mean = np.mean(var_evs)
    var_levelthree = (var_max + var_mean) / 2
    var_levelfour = (var_levelthree + var_max) / 2
    with open('./mark.txt','a',encoding='utf-8') as f:
        for j in range(len(var_evs)):  # 可能要改
            if ( var_evs[j]>0.7):  # 4
                mark.append(70 - (var_evs[j] - 0.7) / ((var_max - 0.7) / 10))
                print("第", j, "号表状态: 异常,得分: ", mark[len(mark)-1])
                f.write("第%d号表状态: 异常,得分: %d"%(j,mark[len(mark)-1]))
                #f.write("%d"%mark[len(mark)-1])
                level.append(4)
            elif ( var_evs[j]>0.2 and var_evs[j]<0.7):  # 3
                mark.append(80 - abs((var_evs[j] - 0.2) / ((var_levelfour - 0.2) / 10)))
                print("第", j, "号表状态:预警 ,得分: ", mark[len(mark)-1])
                f.write("第%d号表状态: 预警,得分: %d"%(j,mark[len(mark)-1]))
                #f.write("%d"%mark[len(mark)-1])
                level.append(3)
            elif (var_evs[j] >= var_mean and var_evs[j] < var_levelthree):  # 2
                mark.append(90 - (var_evs[j] - var_mean) / ((var_levelthree - var_mean) / 10))
                print("第", j, "号表状态: 正常,得分: ", mark[len(mark)-1])
                f.write("第%d号表状态: 正常,得分: %d"%(j,mark[len(mark)-1]))
                #f.write("%d"%mark[len(mark)-1])
                level.append(2)
            elif (var_evs[j] >= var_min and var_evs[j] < var_mean):  # 1
                mark.append(100 - (var_evs[j] - var_min) / ((var_mean - var_min) / 10))
                print("第", j, "号表状态: 良好,得分: ", mark[len(mark)-1])
                f.write("第%d号表状态: 良好,得分: %d"%(j,mark[len(mark)-1]))
                #f.write("%d"%mark[len(mark)-1])
                level.append(1)
    f.close()
        
    # 输出权重
    i = 0
    global m
    m, m_1 , m_2 , m_3, m_4 = [],[],[],[],[]  #不能连等！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    
    for i in range(len(var_evs)):
        weight1 = np.random.uniform(0.25, 0.35)
        weight2 = (float)(np.random.uniform(0.25, 0.28, 1))
        weight3 = (float)(np.random.uniform(0.15, 0.25, 1))
        weight4 = (float)(1 - weight1 - weight2- weight3)
        print("权重为",weight1,weight2,weight3,weight4)
        temp_mark = mark[i]
        m_1.append(temp_mark * weight1)
        m_2.append(temp_mark * weight2)
        m_3.append(temp_mark * weight3)
        m_4.append(temp_mark * weight4)
        print("权重分数为", m_1[i], m_2[i], m_3[i], m_4[i], "总分为", mark[i])
    
    m.append(m_1)
    m.append(m_2)
    m.append(m_3) 
    m.append(m_4)
    m.append(mark)
    m.append(level)
    
    m = DataFrame(m)
    writer = pd.ExcelWriter('./m.xlsx')
    m.to_excel(writer,'page_1',float_format='%.5f')
    writer.save()
    writer.close()
    return mark, level


def Statistic(level=[]):
    i = 0
    count1, count2, count3, count4 = 0, 0, 0, 0
    length = len(level)
    for i in range(length):
        if (level[i] == 1): count1 = count1 + 1
        if (level[i] == 2): count2 = count2 + 1
        if (level[i] == 3): count3 = count3 + 1
        if (level[i] == 4): count4 = count4 + 1
    print("正常状态：", count1, "\n", "关注状态：", count2, "\n", "预警状态：", count3, "\n", "异常状态：", count4)
    print("正常状态占比：", count1 / length, "\n", "关注状态占比：", count2 / length, "\n", "预警状态占比：", count3 / length, "\n",
          "异常状态占比：", count4 / length)
    return count1, count2, count3, count4

def read_data(file="D:\\Ji Xun\\数据\\Stochastic_matrix\\data\\合并数据.xlsx"):
    # file = "C:\\Users\\84494\\Desktop\\Stochastic_matrix\\data\\合并数据.xlsx"
    value = []

    data = xlrd.open_workbook(file)
    table = data.sheet_by_index(0)
    nrows = table.nrows
    ncols = table.ncols
    id_elec = []
    find_err(table)
    for i in range(1,nrows):  #数据不对
        value.append(table.row_values(i)[13:14 + 24 - 1])
        id_elec.append(table.row_values(i)[1])
    return nrows, ncols, value

def find_err(table):
    err = [] 
    i = 1
    while(i<=table.nrows-4):
         if table.row_values(i)[39] == table.row_values(i+1)[39] == table.row_values(i+2)[39] == table.row_values(i+3)[39]:
             i = i + 4
         else:
             err.append(i)
             if table.row_values(i)[39] == table.row_values(i+1)[39] == table.row_values(i+2)[39] :
                 i = i + 3
             elif table.row_values(i)[39] == table.row_values(i+1)[39] :
                 i = i + 2
             else:
                 i = i + 1
    print(err)
    return i
            
     
def save_plt():
    '''
    #存储路径
    if directory:
        filestr = "result.jpg"
        filename = os.path.join(directory,filestr)
    else:
        filename = tempfile.NamedTemporaryFile()

    #可视化
    plt.figure(figsize=(8, 8))
    plt.plot(evs)
    # encoding: utf-8
    plt.ylim((0.5, 2))
    plt.tick_params(labelsize=23)
    plt.title('Linear eigenvalue statistic', fontsize=18)
    plt.xlabel('Ti', fontsize=18,fontstyle='italic')
    plt.ylabel('Mean spectral radius', fontsize=18)
    plt.legend( labels=label, loc='upper left',ncol=7)
    plt.show()
    plt.savefig(filename,bbox_inches='tight',pad_inches=0)
    '''

def view_bar(num, total,count):
    rate = float(num) / float(total)
    rate_num = int(rate * 100)
    #r = '\r[%s%s]%d,%d\t表%d完成'%("="*rate_num, " "*(100-rate_num), rate_num, num,count)
    r = '\r[%s]%d,%d\t表%d完成'%("="*rate_num, rate_num, num,count)
    sys.stdout.write(r)
    #time.sleep(0.01)
    sys.stdout.flush()


def test():
    nrows, ncols, value = read_data()
    #times = 2
    global var_evs_out
    var_evs_out = []
    global matrix_out
    matrix_out = []
    global mark
    mark = []
    i = 1
    j = 0
    #for j in trange(100):
    while(i<=int(nrows-1)):  # 多少张表
        var_evs_temp, matrix_temp = Stochastic_matrix(value, i)
        var_evs_out.append(var_evs_temp)
        matrix_out.append(matrix_temp)
        i = i + 40
        #进度条 Description will be displayed on the left
        #j.set_description('数据处理进度 %j' % j)
        view_bar(j, 2964, i)
        j = j + 1
    print("var_evs_out:", var_evs_out)
    
    save_data=pd.DataFrame(data=matrix_out)
    save_data.to_csv('./matrix.csv',encoding='gbk')
    
    mark, level = weightcalculation(var_evs_out)
    #print(mark)
    count1, count2, count3, count4 = Statistic(level)
    
    

if __name__ == '__main__':
    #test()
    nrows, ncols, value = read_data("D:\\Ji Xun\\数据\\123合并数据.xlsx")
    #print("matrix")
    #print(matrix)
