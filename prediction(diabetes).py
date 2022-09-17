import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
#loading the diabetes dataset to a pandas dataframe
diabetes_dataset=pd.read_csv('/content/drive/MyDrive/diabetes.csv')
pd.read_csv
diabetes_dataset.head()
# number of raws and column in thid dataset
diabetes_dataset.shape
#gettin the statistical measures of the data 
diabetes_dataset.describe()
diabetes_dataset['Outcome'].value_counts()
diabetes_dataset.groupby('Outcome').mean()
#seperating the data and lables
x=diabetes_dataset.drop(columns = 'Outcome', axis =1)
y=diabetes_dataset['Outcome']
print (x)
print(y)
scaler=StandardScaler()
scaler.fit(x)
standardized_data=scaler.transform(x)
print(standardized_data)
x=standardized_data
print(x)
print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)
print(x.shape,x_train.shape,x_test.shape)
classifier=svm.SVC(kernel="linear")
#training the support vector machine 
classifier.fit(x_train,y_train)
#accuracy score
x_train_prediction=classifier.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)
print(training_data_accuracy)
#accuracy score
x_test_prediction=classifier.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)
print(test_data_accuracy)
input_data=(4,110,92,0,0,37,6,0)

#changing the input_data to numpy array
input_data_as_numpy_array=np.asarray(input_data)

#reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#standardize the input_data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction=classifier.predict(std_data)
print(prediction)
if (prediction[0]==0):
  print('the person is not diabetic')
else:
  print('the person is diabetic')
