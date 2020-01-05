from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import KNeighborsClassifier ,RadiusNeighborsClassifier # 导入knn算法类
from sklearn.neighbors import KNeighborsRegressor
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
"""
#分类模型
centers = [[-2,2],[2,2],[0,4]]
X,y = make_blobs(n_samples=60,centers=centers,random_state=0,cluster_std=0.6)  # 生成聚类使用的数据
plt.figure(figsize=(16,10),dpi=144)
c = np.array(centers)   # 将数据转为np array
plt.scatter(X[:,0],X[:,1],c=y,s=100,cmap="cool")
plt.scatter(c[:,0],c[:,1],s=100,marker="^",c="orange")  # 画出散点图
k= 5 # 设置k参数为5
clf = KNeighborsClassifier(n_neighbors=k)   # 定义knn的算法模型
clf.fit(X,y)
X_sample = np.array([0,2]).reshape(1, -1)   # 新版本的sklearn中所有的输入必须为2D的数据，所以这里使用了np reshape
y_sample = clf.predict(X_sample)  # 预测
neighbors = clf.kneighbors(X_sample,return_distance=False)  # 取出最邻近的k个数据样本
plt.scatter(X_sample[:,0],X_sample[:,1],marker="x",c=y_sample,s=100,cmap="cool")
for i in neighbors[0]:
    plt.plot([X[i][0],X_sample[:,0]],[X[i][1],X_sample[:,1]],"k--",linewidth = 0.6)
plt.show()"""
"""
# 回归模型
n_dots =40
X = 5*np.random.rand(n_dots,1)
y = np.cos(X).ravel()
y+= 0.2*np.random.rand(n_dots)-1
k = 5
knn=KNeighborsRegressor(n_neighbors=k)
knn.fit(X,y)
T = np.linspace(0,5,500)[:,np.newaxis]
y_pred = knn.predict(T)
knn.score(X,y)
plt.figure(figsize=(16,10),dpi=144)
plt.scatter(X,y,c="g",label="data",s=100)
plt.plot(T,y_pred,c="k",label="pred",lw=4)
plt.axis("tight")
plt.title("KNN")
plt.show()"""


data = pd.read_csv("diabetes.csv")
X = data.iloc[:,0:8]
Y = data.iloc[:,-1]
from sklearn.model_selection import train_test_split

# X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2) # 将数据集分为测试集和训练集
# models = []
# models.append(("KNN",KNeighborsClassifier(n_neighbors=2)))
# models.append(("KNN with weights",KNeighborsClassifier(n_neighbors=2,weights="distance")))
# models.append(("KNN RadiusNeighborsClassifier",RadiusNeighborsClassifier(n_neighbors=2,radius=500)))
# result = []
# from sklearn.model_selection import KFold
# from sklearn.model_selection import cross_val_score
# for name,model in models:
#     kflod = KFold(n_splits=10)
#     cv_result = cross_val_score(model,X,Y,cv=kflod)
#     # model.fit(X_train,Y_train)
#     result.append((name,cv_result))
# for i in range(len(result)):
#     print("name:{} score:{}".format(result[i][0],result[i][1].mean()))
from sklearn.feature_selection import SelectKBest
selector = SelectKBest(k=2)
X_new = selector.fit_transform(X,Y)
print(X_new[0:5])

