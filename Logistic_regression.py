import sklearn as sl
from sklearn import datasets
from sklearn import model_selection

data,target = sl.datasets.load_iris(return_X_y=True,as_frame=True)
x_train,x_test,y_train,y_test = model_selection.train_test_split(data,target,test_size=0.3)

from sklearn import svm
svc = svm.SVC(kernel = 'linear', C = 1)
svc.fit(x_train,y_train)
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test,svc.predict(x_test))
print(score)


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
cvs = cross_val_score(svc,x_train,y_train, cv=2)
cvp = cross_val_predict(svc,x_train,y_train)


