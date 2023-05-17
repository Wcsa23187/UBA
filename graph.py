import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data_path_before = 'results/before_after_attack/before.csv'
data_before = pd.read_csv(data_path_before)

# 删除符合条件的指定行，并替换原始df
data_before.drop(data_before[(data_before.rating < 4)].index, inplace=True)

data = data_before.groupby(["item_id"]).agg({"item_id": "count"})

data.to_csv('results/before_after_attack/data.csv', encoding='utf_8_sig', index=True)
data_path = 'results/before_after_attack/data.csv'
da = pd.read_csv(data_path)
da.columns = ['item_id', 'count']
da = da.sort_values(by='count', ascending=False)
numpyData=da.to_numpy()
numpyData = numpyData.T
x_label = list(numpyData[0])
y_value = list(numpyData[1])
print(x_label.index(62))
print(y_value[x_label.index(62)])
y = y_value[x_label.index(62)]
x_value = range(len(x_label))
x_value = [i+1 for i in x_value]
print(x_value)
sns.set(style="whitegrid")  # 这是seaborn默认的风格
# 使用标记而不是破折号来识别组
plt.xlabel('items')  # 添加x轴的名称
plt.ylabel('count')
sns.lineplot(x=x_value, y=y_value,
             markers=True, dashes=False)
plt.plot(x_label.index(62)+1,y,'ks')
# plt.xticks(x_value, x_label)  ## 可以设置坐标字
plt.title('62 , randomattack , MF ')
plt.show()


data_path_after = 'results/before_after_attack/after.csv'
data_before = pd.read_csv(data_path_after)

# 删除符合条件的指定行，并替换原始df
data_before.drop(data_before[(data_before.rating < 4)].index, inplace=True)

data = data_before.groupby(["item_id"]).agg({"item_id": "count"})

data.to_csv('results/before_after_attack/data.csv', encoding='utf_8_sig', index=True)
data_path = 'results/before_after_attack/data.csv'
da = pd.read_csv(data_path)
da.columns = ['item_id', 'count']
da = da.sort_values(by='count', ascending=False)
numpyData=da.to_numpy()
numpyData = numpyData.T
x_label = list(numpyData[0])
y_value = list(numpyData[1])
print(x_label.index(62))
print(y_value[x_label.index(62)])
y = y_value[x_label.index(62)]
x_value = range(len(x_label))
x_value = [i+1 for i in x_value]
sns.set(style="whitegrid")  # 这是seaborn默认的风格
# 使用标记而不是破折号来识别组
plt.xlabel('items')  # 添加x轴的名称
plt.ylabel('count')
sns.lineplot(x=x_value, y=y_value,
             markers=True, dashes=False)
plt.plot(x_label.index(62)+1,y,'ks')
# plt.xticks(x_value, x_label)  ## 可以设置坐标字
plt.title('62 , randomattack , MF ')
plt.show()