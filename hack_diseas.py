# IMportIng The LIbrary
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

# Processing and ReaDing THE DAta sets

tt = pd.read_csv(r"C:\Users\HP\Desktop\train.csv")
ts =pd.read_csv(r"C:\Users\HP\Desktop\test.csv")

# MAke It CompatiAble

tt['gender'].replace(['M','F'],[0,1],inplace=True)
ts['gender'].replace(['M','F'],[0,1],inplace=True)

train_1 = tt.iloc[:,1:4]
test_1 = ts.iloc[:,1:4]

target = tt.detected

# Choose and Create THE object of ML-MODEL

obj = GaussianNB()
obj = obj.fit(train_1,target)
pred = obj.predict(test_1)

# Now makeing a .CSV file for USer

submit = pd.DataFrame({'row_id':ts['row_id'],'detected':pred})
submit = submit[['row_id','detected']]
submit.to_csv('submit_file.csv',index=False)
