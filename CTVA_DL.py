import sklearn as sl
from sklearn import datasets

data, target = datasets.load_diabetes(return_X_y=True,as_frame=False)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(data,target,test_size=0.3)

""" from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler

pp = make_pipeline(StandardScaler(),GradientBoostingRegressor())
pp.fit(x_train,y_train)

from sklearn.metrics import mean_absolute_error
mae=mean_absolute_error(y_test,pp.predict(x_test))

from sklearn.metrics import r2_score
r2=r2_score(y_test,pp.predict(x_test)) """