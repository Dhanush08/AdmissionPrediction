
# %%
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as mp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as DataSplit
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

sb.set(style='whitegrid', color_codes=True)


# %%
data = pd.read_csv("./Admission_Predict.csv")
data.head(10)


# %%
data.describe()


# %%
data.rename(columns={'Chance of Admit ': 'Chance of Admit',
                     'LOR ': 'LOR'}, inplace=True)
data.drop(labels='Serial No.', axis=1, inplace=True)


# %%
data.corr()


# %%
mp.subplots(figsize=(10, 10))
sb.heatmap(data.corr(), annot=True, cmap='Reds')


# %%
mp.figure(figsize=(20, 6))
mp.subplot(1, 2, 1)
sb.distplot(data['CGPA'])
mp.title('CGPA Distribution of Applicants')

mp.subplot(1, 2, 2)
sb.regplot(data['CGPA'], data['Chance of Admit'])
mp.title('CGPA vs Chance of Admit')


# %%
mp.figure(figsize=(20, 6))
mp.subplot(1, 2, 1)
sb.distplot(data['GRE Score'])
mp.title('GRE Scores Distribution of Applicants')

mp.subplot(1, 2, 2)
sb.regplot(data['GRE Score'], data['Chance of Admit'])
mp.title('GRE Scores vs Chance of Admit')


# %%
mp.figure(figsize=(20, 6))
mp.subplot(1, 2, 1)
sb.distplot(data['TOEFL Score'])
mp.title('TOEFL Scores Distribution of Applicants')

mp.subplot(1, 2, 2)
sb.regplot(data['TOEFL Score'], data['Chance of Admit'])
mp.title('TOEFL Scores vs Chance of Admit')


# %%
_, axis = mp.subplots(figsize=(8, 6))
sb.countplot(data['Research'])
mp.title('Research Experience')
mp.ylabel('Number of Applicants')
axis.set_xticklabels(['No Research Experience', 'Has Research Experience'])


# %%
mp.subplots(figsize=(8, 6))
sb.countplot(data['University Rating'])
mp.title('University Rating')
mp.ylabel('Number of Applicants')


# %%
targets = data['Chance of Admit']
features = data.drop(columns={'Chance of Admit'})

X_train, X_test, y_train, y_test = DataSplit(
    features, targets, test_size=0.2, random_state=42)


# %%
Normaliser = StandardScaler()
X_train = Normaliser.fit_transform(X_train)
X_test = Normaliser.fit_transform(X_test)


# %%
LinearReg = LinearRegression()
LinearReg.fit(X_train, y_train)
y_predict = LinearReg.predict(X_test)
LinearRegAccuracy = (LinearReg.score(X_test, y_test))*100
LinearRegAccuracy


# %%
DecisionTreeReg = DecisionTreeRegressor(random_state=0, max_depth=6)
DecisionTreeReg.fit(X_train, y_train)
y_predict = DecisionTreeReg.predict(X_test)
DecisionTreeRegAccuracy = (DecisionTreeReg.score(X_test, y_test))*100
DecisionTreeRegAccuracy


# %%
RandomForestReg = RandomForestRegressor(
    n_estimators=110, max_depth=6, random_state=0)
RandomForestReg.fit(X_train, y_train)
y_predict = RandomForestReg.predict(X_test)
RandomForestRegAccuracy = (RandomForestReg.score(X_test, y_test))*100
RandomForestRegAccuracy


# %%
Methods = ['Linear Regression', 'Decision Trees Regression',
           'Random Forests Regression']
Scores = [LinearRegAccuracy, DecisionTreeRegAccuracy, RandomForestRegAccuracy]

mp.subplots(figsize=(8, 6))
sb.barplot(Methods, Scores)
mp.title('Algorithm Prediction Accuracies')
mp.ylabel('Accuracy')
