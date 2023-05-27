# -*- coding: utf-8 -*-  
"""          
# @Author : Changsheng Wang
# @Time : 2023/5/27 21:07
# @Contact : wcsa23187@mail.ustc.edu.cn
"""
import numpy as np
import pandas as pd
df = pd.read_csv('.\RSlib\data_latest\ml1m\deal well\deal_well.csv')
print(df.groupby('rating').count())
x = np.zeros((50, 6))
for j in range(1,6):
    k = j + 1
    for i in range(2023, 2033):
        path = r'.\RSlib\data_latest\ml1m\MF_matrix\AUSHunpop' + '\\' + str(k) + '_' + str(i) + '.npy'
        # print(path)
        try:
            mydict = np.load(path, allow_pickle=True).item()
            order = 0
            for key in mydict.keys():
                if mydict[key] != 0 and mydict[key] <= 20:
                    x[order][k-1] += 1
                order += 1
        except:
            print(path)
            continue
new = np.load(r'.\RSlib\data_latest\ml1m\MF_matrix\a3path3116.npy')
final_csv =[]
for i in range(50):
    empty = []
    for j in range(6):
        print(j)
        empty.append(new[j][i])
    final_csv.append(empty)
data_index = list(mydict.keys())
df = pd.DataFrame(x / 10, index=data_index)
index = 0
weight = []
user_id = []
v = []
for user in data_index:
    temp_w = []
    temp_v = []

    for i in range(6):
        if x[index][i] != 0:
            temp_w.append(i)
            temp_v.append(x[index][i] / 10)

    if len(temp_w) != 0:
        weight.append(temp_w)
        v.append(temp_v)
        user_id.append(user)
    index += 1
print(user_id)
print(weight)
print(v)
lenth = []
for i in weight:
   lenth.append(len(i))
print(lenth)
def pack5(w, v, n, c):
    rec = []
    for i in range(len(n)):
        rec.append([])
    sum = 0
    for i in n:
        for j in range(i):
            rec[sum].append(0)
        sum += 1
    # print(rec)
    mydict = {}
    dict_list = []
    dp = [0 for _ in range(c + 1)]
    for i in range(1, len(w) + 1):
        for j in reversed(range(1, c + 1)):
            for k in range(n[i - 1]):
                if j - w[i - 1][k] >= 0:

                    dp[j] = max(dp[j], dp[j - w[i - 1][k]] + v[i - 1][k])
                    if dp[j - w[i - 1][k]] + v[i - 1][k] >= dp[j]:
                        if (j,round(dp[j],2)) in mydict.keys():
                            mydict[(j,round(dp[j],2))].append((i,k))
                        else:
                            mydict[(j,round(dp[j],2))] = []
                            mydict[(j,round(dp[j],2))].append((i,k))
                        print((i,k,j,round(dp[j],2)))
                        dict_list.append((i,k,j,round(dp[j],2)))
        print(dp)
    print(dp[c])
    # print(mydict)
    return mydict,dict_list
c = 100
w = weight
v = v
n = lenth
mydict,dict_list = pack5(w, v, n, c)
for i in dict_list:
    if i[2]==100 and i[3] == 7223.8:
        print(i)
value = 7223.8
c = 100
team = 50
order = 5
dict_final = {}
sum_list = []
while (team,order,c,value) in dict_list:
    print(((team,order,c,value),w[team-1][order],v[team-1][order],user_id[team-1]))
    dict_final[user_id[team-1]] = w[team-1][order]
    sum_list.append(v[team-1][order])
    c_d = w[team-1][order]
    value_d = v[team-1][order]
    c = c - c_d
    value=value- value_d
    c = round(c,2)
    value = round(value,2)
    temp = []
    for i in dict_list:
        if i[2]==c and i[3] == value:
            temp.append(i)
    sign = 0
    for j in range(len(temp)):
        if sign == 0:
            if temp[j][0] != team:
                team = temp[j][0]
                order = temp[j][1]
                sign = 1
user_list = []
for key in dict_final.keys():
    for j in range(dict_final[key]):
        user_list.append(key)

print(user_list)
print(len(set(user_list)))
