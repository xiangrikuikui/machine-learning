#-*-coding:utf-8 -*-
import matplotlib
matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.datasets import make_hastie_10_2
import numpy as np 
import matplotlib.pyplot as plt
from itertools import islice
from sklearn.grid_search import GridSearchCV

"""
大多数的GBRT的应用效果可以用一条简单的拟合曲线来展示，如下图中用一个只有一个特征x和相应变量y的回归问题来举例。
我们随机从数据集中均匀抽取100个训练数据，用ground truth (sinoid函数; 淡蓝色线) 拟合，加入一些随机噪音。
100个训练数据之外（蓝色），再用100个测试数据（红色）来评估模型的效果。
"""
def ground_truth(x):
	"""Ground truth -- function to approximate"""
	return x*np.sin(x)*np.sin(2*x)

def gen_data(n_samples=200):
	"""generate training and testing data"""
	np.random.seed(13)
	x = np.random.uniform(0,10,size=n_samples) #从一个均匀分布[0,10)中随机采样
	x.sort()
	y = ground_truth(x) + 0.75*np.random.normal(size=n_samples) #均值为0，标准差为1的正态分布
	#print("x, y: ", x[:10], y[:10])
	train_mask = np.random.randint(0,2,size=n_samples).astype(np.bool) #返回值为True or False
	#print("train_mask: ",train_mask[:10])
	x_train, y_train = x[train_mask, np.newaxis], y[train_mask] #如果train_mask是1，就将x,y作为训练集，np.newaxis表示将行向量转为列向量
	#print("x_train, y_train: ", x_train[:10], y_train[:10])
	x_test, y_test = x[~train_mask, np.newaxis], y[~train_mask] #train_mask为False，将x,y作为测试集
	#print("x_test, y_test: ", x_test[:10], y_test[:10])
	return x_train, x_test, y_train, y_test

X_train,X_test,y_train,y_test = gen_data(200)
#plot ground truth
x_plot = np.linspace(0,10,500) #从0-10之间等间隔采500个数

def plot_data(figsize=(8, 5)): #figsize：指定figure的宽高
	fig = plt.figure(figsize=figsize)
	gt = plt.plot(x_plot,ground_truth(x_plot),alpha=0.4,label='ground_truth') #alpha代表透明度

	#plot training and testing data
	plt.scatter(X_train, y_train, s=10, alpha=0.4)
	plt.scatter(X_test, y_test, s=10, alpha=0.4, color='red')
	plt.xlim((1,10)) #x轴坐标的范围
	plt.ylabel('y')
	plt.xlabel('x')
 
plot_data(figsize=(8,5))

"""
如果对以上数据仅使用一棵独立的回归树，就只能得到区域内稳定的近似。树的深度越深，数据分割的越细致，那么能够解决的差异就越多。
"""
plot_data()
est = DecisionTreeRegressor(max_depth=1).fit(X_train, y_train)
plt.plot(x_plot, est.predict(x_plot[:, np.newaxis]), label='RT max_depth=1', color='g', alpha=0.9, linewidth=2)

est = DecisionTreeRegressor(max_depth=3).fit(X_train, y_train)
plt.plot(x_plot, est.predict(x_plot[:, np.newaxis]), label='RT max_depth=3', color='g', alpha=0.7, linewidth=1)
plt.legend(loc='upper left')
plt.savefig("GradientBoostingRegressor1")



"""
我们可以使用gradient boosting模型来你和训练数据，然后看看随着添加更多的树，预测值与实际值的近似度是如何
提升的。Scikit-learn的gradient boosting Estimator可以通过staged_(predict|predict_proba) 方法，评估
模型预测效果，该方法返回一个生成器可以随着添加越来越多的树，迭代评估预测结果。

下述代码将生成中50条红线，每条代表GBRT模型增加20棵树后的效果。可以看到，刚开始预测近似度非常粗，但随着添加
更多的树，模型可以覆盖到更多的偏差，最终产生紧密的红线。

可以看到，向GBRT添加的更多的树以及更深的深度，可以捕获更多的偏差，因此我们模型也更复杂。但和以往一样，机器
学习模型的复杂度是以“过拟合”为代价的。
"""
est = GradientBoostingRegressor(n_estimators=2000, max_depth=1, learning_rate=1.0)
est.fit(X_train, y_train)

ax = plt.gca()
first = True
#step over prediction as we added 20 more trees.
for pred in islice(est.staged_predict(x_plot[:, np.newaxis]), 0, 2000, 10):
	plt.plot(x_plot, pred, color='r', alpha=0.2)
	if first:
		ax.annotate('High bias - low variance', xy=(x_plot[x_plot.shape[0]//2], pred[x_plot.shape[0]//2]),
			xycoords='data', xytext=(3,4), textcoords='data', arrowprops=dict(arrowstyle="->", connectionstyle="arc"))
		first = False

pred = est.predict(x_plot[:, np.newaxis])
plt.plot(x_plot, pred, color='r', label='GBRT max_depth=1')
ax.annotate('Low bias - high variance', xy=(x_plot[x_plot.shape[0]//2], pred[x_plot.shape[0]//2]),
			xycoords='data', xytext=(5,-5), textcoords='data', arrowprops=dict(arrowstyle="->", connectionstyle="arc"))
#xy表示箭头尖端的位置；xytext表示文字的坐标；xycoords=data和textcoords=data都表示使用轴域数据坐标系。
plt.legend(loc='upper right')
plt.savefig("GradientBoostingRegressor2.png")



"""
GBRT实战中重要的诊断方法是使用异常坐标图来展示训练集/测试集的错误（或异常），以树的数量为横坐标。

下述代码生成的图中蓝线是指训练集的预测偏差：可以看到开始阶段快速下降，之后随着添加更多的树而逐步降低。测试集
预测偏差（红线）同样在开始阶段快速下降，但是之后速度降低很快达到了最小值（50棵树左右），之后甚至
开始上升。这就是我们所指的“过拟合”：在一定阶段，模型能够非常好的拟合训练数据的特点（这个例子里是
我们随机生成的噪音）但是对于新的未知数据其能力受到限制。图中在训练数据与测试数据的预测偏差中存在
的巨大的差异，就是“过拟合”的一个信号。

"""
n_estimators = len(est.estimators_)
#print("n_estimators: ", n_estimators)

def deviance_plot(est, X_test, y_test, ax=None, label='',train_color='#2c7bb6', test_color='#d7191c', alpha=1.0):
	"""Deviance plot for"est", use  X_test and y_test for test error."""
	test_dev = np.empty(n_estimators) #创建数组
	for i, pred in enumerate(est.staged_predict(X_test)):
		test_dev[i] = est.loss_(y_test, pred)
	if ax is None:
		fig = plt.figure(figsize=(8,5))
		ax = plt.gca();
	ax.plot(np.arange(n_estimators)+1, test_dev, color=test_color, label='Test Error max_depth=1 %s' % label, linewidth=2, alpha=alpha)
	ax.plot(np.arange(n_estimators)+1, est.train_score_, color=train_color, label='Train Error max_depth=1 %s' % label, linewidth=2, alpha=alpha)
	ax.set_ylabel('Error')
	ax.set_xlabel('n_estimators')
	ax.set_ylim((0,2))
	return test_dev, ax

test_dev, ax = deviance_plot(est, X_test, y_test)
ax.legend(loc='upper right')

#add some annotations
ax.annotate('Lowest test error', xy=(test_dev.argmin()+1, test_dev.min()+0.02), xycoords='data',
	xytext=(150, 1.0), textcoords='data', arrowprops=dict(arrowstyle="->", connectionstyle="arc"))

ann = ax.annotate('', xy=(1500,test_dev[1499]), xycoords='data', xytext=(1500, est.train_score_[1499]), 
	textcoords='data', arrowprops=dict(arrowstyle="<->"))
ax.text(1510, 0.4, 'train-test gap')
plt.savefig("GradientBoostingRegressor3.png")



"""
Regularization

GBRT提供三个“把手”来控制“过拟合”：树结构（tree structure），收敛（shrinkage）， 随机性（randomization）。

###树结构（tree structure）
单棵树的深度是模型复杂度的一方面。树的深度基本上控制了特征相互作用的成都。例如，如果想覆盖维度特征和精度特征
之间的交叉关系特征，需要深度至少为2的树来覆盖。不幸的是，特征相互作用的程度是预先未知的，但通常设置的比较低
较好–实战中，深度4-6常得到最佳结果。在scikit-learn中，可以通过max_depth参数来限制树的深度。

另一个控制树的深度的方法是在叶节点的样例数量上使用较低的边界：这样可以避免不均衡的划分，出现一个叶节点仅有一个
数据点构成。在scikit-learn中可以使用min_samples_leaf参数来实现。这是一个有效的方法来减少偏差，如下例所示.
"""
def fmt_params(params):
	return ", ".join("{0}={1}".format(key, val) for key, val in params.iteritems())

fig = plt.figure(figsize=(8, 5))
ax = plt.gca()
for params, (test_color, train_color) in [({}, ('#d7191c', '#2c7bb6')), ({'min_samples_leaf': 3}, 
	('#fdae61', '#abd9e9'))]:
	est = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=1, learning_rate=1.0)
	est.set_params(**params)
	est.fit(X_train, y_train)

	test_dev, ax = deviance_plot(est, X_test, y_test, ax=ax, label=fmt_params(params), 
		train_color=train_color, test_color=test_color)

ax.annotate('Higher bias', xy=(900, est.train_score_[899]), xycoords='data', xytext=(800, 0.4), 
	textcoords='data',arrowprops=dict(arrowstyle="->", connectionstyle="arc"))
ax.annotate('Lower variance', xy=(900, test_dev[899]), xycoords='data', xytext=(800, 0.6), 
	textcoords='data',arrowprops=dict(arrowstyle="->", connectionstyle="arc"))

plt.legend(loc='upper right')
plt.savefig('GradientBoostingRegressor4.png')



"""
###收敛（Shrinkage）
GBRT调参的技术最重要的就是收敛：基本想法是进行通过收敛每棵树预测值进行缓慢学习，通过learning_rate来控制。
较低的学习速率需要更高数量的n_estimators，以达到相同程度的训练集误差–用时间换准确度的。
"""
fig = plt.figure(figsize=(8, 5))
ax = plt.gca()
for params, (test_color, train_color) in [({}, ('#d7191c', '#2c7bb6')), ({'learning_rate': 0.1}, 
	('#fdae61', '#abd9e9'))]:
	est = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=1, learning_rate=1.0)
	est.set_params(**params)
	est.fit(X_train, y_train)

	test_dev, ax = deviance_plot(est, X_test, y_test, ax=ax, label=fmt_params(params), 
		train_color=train_color, test_color=test_color)

ax.annotate('Require more trees', xy=(750, est.train_score_[749]), xycoords='data', xytext=(300, 0.25), 
	textcoords='data',arrowprops=dict(arrowstyle="->", connectionstyle="arc"))
ax.annotate('Lower test error', xy=(1750, test_dev[1749]), xycoords='data', xytext=(1600, 0.4), 
	textcoords='data',arrowprops=dict(arrowstyle="->", connectionstyle="arc"))

plt.legend(loc='upper right')
plt.savefig('GradientBoostingRegressor5.png')



"""
###随机梯度推进（Stochastic Gradient Boosting）
与随机森林相似，在构建树的过程中引入随机性导致更高的准确率。Scikit-learn提供了两种方法引入随机性：
a)在构建树之前对训练集进行随机取样（subsample）；b)在找到最佳划分节点前对所有特征取样(max_features)。
经验表明，如果有充足的特征（大于30个）后者效果更佳。值得强调的是两种选择都会降低运算时间。
下面以subsample=0.5来展示效果，即使用50%的训练集来训练每棵树：
"""
fig = plt.figure(figsize=(8, 5))
ax = plt.gca()
for params, (test_color, train_color) in [({}, ('#d7191c', '#2c7bb6')), ({'learning_rate': 0.1, 'subsample':0.5}, 
	('#fdae61', '#abd9e9'))]:
	est = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=1, learning_rate=1.0,random_state=1)
	est.set_params(**params)
	est.fit(X_train, y_train)
	test_dev, ax = deviance_plot(est, X_test, y_test, ax=ax, label=fmt_params(params), 
		train_color=train_color, test_color=test_color)

ax.annotate('Even lower test error', xy=(1750, test_dev[1749]), xycoords='data', xytext=(1500, 0.3), 
	textcoords='data',arrowprops=dict(arrowstyle="->", connectionstyle="arc"))
est = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=1, learning_rate=1.0, subsample=0.4)
est.fit(X_train, y_train)
test_dev, ax = deviance_plot(est, X_test, y_test, ax=ax, label=fmt_params({'subsample':0.5}), 
		train_color=train_color, test_color=test_color)
ax.annotate('Subsample alone does poorly', xy=(750, test_dev[749]), xycoords='data', xytext=(600, 0.6), 
	textcoords='data',arrowprops=dict(arrowstyle="->", connectionstyle="arc"))

plt.legend(loc='upper right', fontsize='small')
plt.savefig('GradientBoostingRegressor6.png')

 

#超参数调优（Hyperparameter tuning）
"""
我们已经介绍了一系列参数，在机器学习中参数优化工作非常单调，尤其是参数之间相互影响，比如learning_rate和
n_estimators， learning_rate和subsample， max_depth和max_features）。

对于gradient boosting模型我们通常使用以下“秘方”来优化参数：

1.根据要解决的问题选择损失函数
2.n_estimators尽可能大（如3000）
3.通过grid search方法对max_depth, learning_rate, min_samples_leaf, 及max_features进行寻优
4.增加n_estimators，保持其它参数不变，再次对learning_rate调优

Scikit-learn提供了方便的API进行参数调优及grid search:
"""
param_grid = {'learning_rate': [0.1,0.05,0.02,0.01],
'max_depth': [4,6],
'min_samples_leaf': [3,5,9,13], 
#'max_feature': [1.0, 0.3, 0.1]  #not possible in our example(only 1 fx)
}
est = GradientBoostingRegressor(n_estimators=5000)
#this may take some minutes
gs_cv = GridSearchCV(est, param_grid, n_jobs=4).fit(X_train, y_train)

#best hyperparameter setting
print(gs_cv.best_params_)
#{'learning_rate': 0.1, 'max_depth': 4, 'min_samples_leaf': 13}