#importing all files for use
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
#read the csv file
df=pd.read_csv('Fertilizer.csv')
x=df.drop('Fertilizer Name',axis=1)
y=df['Fertilizer Name']
#mapping the file
target_mapping = {'DAP': 0, 'Fourteen-Thirty Five-Fourteen': 1, 'Seventeen-Seventeen-Seventeen': 2, 'Ten-Twenty Six-Twenty Six': 3, 'Twenty Eight-Twenty Eight': 4, 'Twenty-Twenty': 5, 'Urea': 6}
y = np.vectorize(target_mapping.get)(y)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)
model = xgb.XGBClassifier()
model.fit(xtrain,ytrain)
ypred = model.predict(xtest)
accuracy = accuracy_score(ytest, ypred)
print("Accuracy:", accuracy)