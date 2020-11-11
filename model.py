import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pickle
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("diabetes.csv")

#splitting into dependent and independent data, x & y

X = df.drop(['Outcome'],axis = 1)
y = df['Outcome']

# Normalizing Data

from sklearn.preprocessing import StandardScaler
SCALE = StandardScaler()
SCALE.fit(X)
X_scale = SCALE.transform(X)
X_scale = pd.DataFrame(X_scale)

#Splitting into train-test sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X_scale,y,test_size = 0.2,random_state = 0)

param_grid = [{'C' : [0.001,0.01,0.1,1,10,100], 'kernel' : ['linear', 'rbf']}]
svc = SVC()
clf = GridSearchCV(svc, param_grid, cv = 5)
clf.fit(x_train, y_train)
pickle.dump(clf,open('model.pkl','wb'))
pickle.dump(SCALE,open('model1.pkl','wb'))