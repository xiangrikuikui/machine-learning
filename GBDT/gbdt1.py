#-*- coding: utf-8 -*-
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.datasets import make_hastie_10_2
from sklearn.cross_validation import train_test_split
#generate synthetic data from SELII 
X, y = make_hastie_10_2(n_samples = 5000)
#X.shape (5000,10)
#y.shape (5000,1)
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1)
#print("x,y: ",X_train[:5],y_train[:5],X_test[:5],y_test[:5])
print("x,y: ",X_test[:5],y_test[:5])
#print X_train.shape (3750,10)
#print y_train.shape (3750,1)
#print X_test.shape (1250,10)
#print y_test.shape (1250,1)

#fit estimator
est = GradientBoostingClassifier(n_estimators=200, max_depth=3)
est.fit(X_train,y_train)

#predict class labels
pred = est.predict(X_test)

#score on test data(accuracy)
acc = est.score(X_test,y_test)
print('ACC: %.4f' % acc)

#predict class probabilities
print est.predict_proba(X_test)[0] #返回标签为-1和1的概率 eg.array([ 0.86593187,  0.13406813])