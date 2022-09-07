# If cannot run, then you need to install pytorch-tabnet to you local machine: pip install pytorch-tabnet 

import time
initial_time = time.time() 

print("Importing the modules...")
# 0.Import the needed modules/packages 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sklearn.datasets as skdatasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import os, datetime
import warnings
warnings.filterwarnings('ignore')

# 1.Upload the dataset from your local machine.(You may download the dataset from: #
# https://archive.ics.uci.edu/ml/datasets/Productivity+Prediction+of+Garment+Employees

print("Read the csv.file : garments_worker_productivity.csv as raw_data.")
print()
# 2. Read the csv.file 
raw_data = pd.read_csv(r"C:\Users\INTEL\Desktop\Deep learning Exercise\garments_worker_productivity.csv", header=0) 

# 3. Doing data cleaning and Extract new features.
print("The following are done during Data Cleaning and Data Preprocessing.")
raw_data.loc[raw_data.actual_productivity>1.00,'actual_productivity']=1.00
print("1. Set the actual productivity that exceeds 1 to 1.")

raw_data.loc[633,"targeted_productivity"]=0.7
print("2. Fix the typo error in targeted productivity.")
raw_data['department']=raw_data['department'].replace(['finishing ','sweing'],['finishing','sewing'])
print("3. Fix the typo error in department:",raw_data['department'].unique())

day_dictionary = {"Saturday":0, "Sunday":1, "Monday":2, "Tuesday":3, "Wednesday":4, "Thursday":5}
for i,day in enumerate(raw_data["day"]):
  raw_data["day"][i] = day_dictionary[day]
raw_data['day'] = pd.to_numeric(raw_data['day'])
print("4. Transform the weekday in the ascending order of consecutive working day in a week.")

day_no_list = []
month_list = []
for date in raw_data["date"]:
  date_listform = date.split("/")
  month, day_no = date_listform[:2]
  day_no_list.append(day_no)
  month_list.append(month)
raw_data['month'] = pd.to_numeric(month_list)
raw_data['day_no'] = pd.to_numeric(day_no_list)
print("5. Extract month and day from date.")

label=LabelEncoder()
for item in ['quarter','date']:
  raw_data[item]=label.fit_transform(raw_data[item])
print("6. Convert quarter and date into numeric form.")

raw_data["wip"].fillna(np.exp(1), inplace=True)
raw_data["wip"] = np.log(raw_data["wip"])
print("7. Fill NAN value of 'wip' as np.exp(1) and transform all wip values by using natural logarithm to reduce the skewness of the data.")

raw_data["Average_smv"] = raw_data["smv"]*100/raw_data["no_of_workers"] 
raw_data["Average_ot"] = raw_data["over_time"]/60/raw_data["no_of_workers"]
print('8. Add two extra features, "Average_smv" and "Average_ot" by divided the original value by the number of workers.' )

finishing_data = raw_data[raw_data['department']=='finishing']
sewing_data = raw_data[raw_data['department']=='sewing']
print("9. Segmented the data into Finishing_data and Sewing_data.")

finishing_features = finishing_data.iloc[:,np.r_[0:2,3:14,15:19]]
finishing_labels = finishing_data.iloc[:,14]

# use list comprehension and dictionary comprehension to make dictionaries to transform teams grouped by department. 
list_finishing = [raw_data.loc[(raw_data['department']=='finishing') & (raw_data['team']==(i+1)),'actual_productivity'].mean() for i in range(12)]
dict_finishing = {i+1 : list_finishing[i] for i in range(12)}
list_sewing = [raw_data.loc[(raw_data['department']=='sewing') & (raw_data['team']==(i+1)),'actual_productivity'].mean() for i in range(12)]
dict_sewing = {i+1 : list_sewing[i] for i in range(12)}

# Use the dict_finishing and dict_sewing to transfrom the data 
for i in range(12):
    sewing_data['team'] = sewing_data['team'].replace([i+1], dict_sewing[i+1])
    finishing_data['team'] = finishing_data['team'].replace([i+1], dict_finishing[i+1])
print("10. Tranform the value of 'team' from number to average productivity of the corresponding team.")

# Define the features and the labels
# In feature part, we will exclude department, which is redundant after grouping, and also productivity, which is used as the label.
# we will also employs exclude the wip columns for finishing data.
finishing_features = finishing_data.iloc[:,np.r_[0:2,3:14,15:19]]
finishing_labels = finishing_data.iloc[:,14]
sewing_features = sewing_data.iloc[:,np.r_[0:2,3:14,15:19]]
sewing_labels = sewing_data.iloc[:,14]

# Applying train_test split
SEED = 1
x_train, x_test, y_train, y_test = train_test_split(finishing_features, finishing_labels, test_size=0.1, random_state=SEED)
print("11. Use train_test_split to split finishing department data into train data and test data.")

# Add sewing data to give further information
x_train = x_train.append(sewing_features)
y_train = y_train.append(sewing_labels)
print("12. Append all sewing department data to the train data.")

# We also make a all_features data to standardize all features 
raw_features = raw_data.iloc[:,np.r_[0:2,3:14,15:19]]

standardizer = StandardScaler()
standardizer.fit(raw_features)
x_train = standardizer.transform(x_train)
x_test = standardizer.transform(x_test)
print("13. Standardize the features by using standard scaler.")
print("Data preprocessing done!!!")
print()
print("Final train_features:")
print(x_train)
print()
from pytorch_tabnet.tab_model import TabNetRegressor as TR
y_train = np.array(y_train).reshape(-1,1)
y_test = np.array(y_test).reshape(-1,1)
model_TR = TR(verbose=0,seed=137)
print('Training the data by using the "TabNetRegressor" model...')
model_TR.fit(X_train=x_train, y_train=y_train, eval_set=[(x_test, y_test)], patience=300, max_epochs=2000, eval_metric=['mae'])

print()
print("Performance of the model:")
predictions = model_TR.predict(x_test).flatten()
y_test = y_test.flatten()
model_mae = mae(predictions, y_test)
correlation = np.corrcoef(predictions, y_test)
print("Model's mean absolute erro ={:.4f}".format(model_mae))
print("correlation = {:.4f}".format(correlation[0,1]))

print()
print("Comparison between actual labels and predicted labels.") 
predictions = predictions.flatten() 
y_test = y_test.flatten()
Comparison = pd.DataFrame({"Actual_labels": y_test, "Predicted labels": predictions})
print(Comparison)

final_time = time.time()
total_time_used = final_time - initial_time
print()
print("total_time_used = {}".format(total_time_used))
print()
