# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


#def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    #print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
#if __name__ == '__main__':
  #  print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/



import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

data=pd.read_csv("training_dataset.csv")

#print(data.shape)
data.drop_duplicates()
#print(data.shape)

for col in data.columns :
    check_nan = data[col].isnull().values.any()
    if(check_nan) :
        mean_value = data[col].mean()
        data[col].fillna(value=mean_value, inplace=True)

#for col in data.columns :
#    fig = px.box(data, y=col, points="all")
#    fig.show()

for col in data.columns :

    df_boston = data
    df_boston.columns=df_boston.columns
    df_boston.head()

    Q1=np.percentile(df_boston[col],25,interpolation='midpoint')
    Q3=np.percentile(df_boston[col],75,interpolation='midpoint')

    IQR = Q3-Q1
    print("Old Shape: ",df_boston.shape)

    upper = np.where(df_boston[col] >= (Q3+1.5*IQR))
    lower = np.where(df_boston[col] <= (Q1-1.5*IQR))

    df_boston.drop(upper[0],inplace=True)
    df_boston.drop(lower[0],inplace=True)

    print("New Shape: ",df_boston.shape)


col_names=data.columns
feature_cols = col_names.copy().tolist()
feature_cols.remove("label")
feature_cols=np.array(feature_cols)
print(feature_cols)
X=data[feature_cols]
y=data.label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

clf = DecisionTreeClassifier()

clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

input =[68,58,38,23.2,82,6.2,221.5]
input = np.reshape(input,(1,-1))
result=clf.predict(input)
print(result)



#print(data.describe())

#print(data.nunique())

#print(data['label'].unique())



#print(data['label'].value_counts())

#print(data.head(5))
