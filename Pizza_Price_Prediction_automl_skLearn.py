from evalml.automl import AutoMLSearch
import evalml
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import category_encoders as ce
import plotly.express as px
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('pizza.csv')
df.head()
df.isnull().sum()
df.info()
df.columns
df.price = df.price.str.replace('Rp', '').str.replace(',', '').astype(int)
df['diameter'] = df['diameter'].str.extract('(\\d+\\.?\\d*)').astype(float)
df.head()
plt.figure(figsize=(6, 6))
plt.pie(df['company'].value_counts(), autopct='%.1f',
        labels=df['company'].unique())
plt.title('Companies Weigtage')
plt.show()
plt.figure(figsize=(6, 6))
plt.pie(df['diameter'].value_counts(), autopct='%.1f',
        labels=df['diameter'].unique())
plt.title('diameter Weigtage')
plt.show()
plt.figure(figsize=(6, 6))
plt.pie(df['topping'].value_counts(), autopct='%.1f',
        labels=df['topping'].unique())
plt.title('topping Weigtage')
plt.show()
plt.figure(figsize=(7, 7))
plt.pie(df['variant'].value_counts(), autopct='%.1f',
        labels=df['variant'].unique())
plt.title('variant Weigtage')
plt.show()
plt.figure(figsize=(6, 6))
plt.pie(df['size'].value_counts(), autopct='%.1f', labels=df['size'].unique())
plt.title('size Weigtage')
plt.show()
plt.figure(figsize=(6, 6))
plt.pie(df['extra_sauce'].value_counts(), autopct='%.1f',
        labels=df['extra_sauce'].unique())
plt.title('extra_sauce Weigtage')
plt.show()
plt.figure(figsize=(6, 6))
plt.pie(df['extra_cheese'].value_counts(), autopct='%.1f',
        labels=df['extra_cheese'].unique())
plt.title('extra_cheese Weigtage')
plt.show()
sns.histplot(data=df, x='company', hue='extra_sauce')
df_company = df.groupby('company').agg({'company': ['count']})
sns.kdeplot(df['company'].value_counts(), color='g', shade=True)
df_company = df.groupby('topping').agg({'topping': ['count']})
sns.kdeplot(df['topping'].value_counts(), color='r', shade=True)
sns.histplot(data=df, x='company', hue='extra_cheese')
sns.histplot(data=df, x='company', hue='extra_sauce')
sns.histplot(data=df, x='price', bins=30)
sns.pointplot(data=df.sort_values(by='diameter'), x='diameter', y='price')
plt.xticks(rotation=90)
plt.figure(figsize=(13, 3))
sns.boxplot(data=df, x='variant', y='price')
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize=(10, 3))
sns.boxplot(data=df, x='topping', y='price')
plt.xticks(rotation=90)
plt.show()
def one_hot_encoding(df, col): one_hot_encoder = ce.OneHotEncoder(
    cols=col, return_df=True, use_cat_names=True); df_final = one_hot_encoder.fit_transform(df); return df_final


def one_hot(df, column): df = one_hot_encoding(df, column); return df


df = one_hot(df, 'company')
df = one_hot(df, 'topping')
df = one_hot(df, 'variant')
df = one_hot(df, 'size')
df = one_hot(df, 'extra_sauce')
df = one_hot(df, 'extra_cheese')
df = one_hot(df, 'extra_mushrooms')
df.head()
df.dtypes
x = df.drop('price', axis=1)
y = df['price']
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)
param_grid = {'n_estimators': [2000, 4000, 6000], 'max_depth': [
    3, 4, 5, 6], 'learning_rate': [0.01, 0.1, 0.2, 0.3, 0.5]}
final = GridSearchCV(XGBRegressor(random_state=42),
                     param_grid=param_grid, scoring='r2')
X_train.head()
final.fit(X_train, Y_train)
final.best_params_
Best_param = {'learning_rate': 0.2, 'max_depth': 6, 'n_estimators': 2000}
xgb = XGBRegressor(learning_rate=0.2, max_depth=6, n_estimators=2000)
xgb.fit(X_train, Y_train)
pred = xgb.predict(X_test)
mae = metrics.mean_absolute_error(pred, Y_test)
print(f"MAE: {mae:.5f}")
rsme = metrics.mean_squared_error(pred, Y_test, squared=False)
print(f"RMSE: {rsme:.5f}")
r2_score = metrics.r2_score(pred, Y_test)
print(f"r2 score: {r2_score:.5f}")
svr = SVR()
svr.fit(X_train, Y_train)
pred1 = svr.predict(X_test)
mae = metrics.mean_absolute_error(pred1, Y_test)
print(f"MAE: {mae:.5f}")
rsme = metrics.mean_squared_error(pred1, Y_test, squared=False)
print(f"RMSE: {rsme:.5f}")
r2_score = metrics.r2_score(pred1, Y_test)
print(f"r2 score: {r2_score:.5f}")
evalml.problem_types.ProblemTypes.all_problem_types
X_train, X_test, y_train, y_test = evalml.preprocessing.split_data(
    x, y, problem_type='regression')
automl = AutoMLSearch(X_train=X_train, y_train=y_train,
                      problem_type='regression')
automl.search()
automl.rankings
automl.best_pipeline
best_pipeline = automl.best_pipeline
automl.describe_pipeline(automl.rankings.iloc[0]['id'])
best_pipeline.score(X_test, y_test, objectives=['R2'])
automl_r2 = AutoMLSearch(X_train=X_train, y_train=y_train, problem_type='regression',
                         objective='R2', max_batches=1, optimize_thresholds=True)
automl_r2.search()
automl_r2.rankings
automl_r2.describe_pipeline(automl_auc.rankings.iloc[0]['id'])
best_pipeline_r2 = automl_r2.best_pipeline
best_pipeline_r2.score(X_test, Y_test, objectives=['R2'])
best_pipeline.save('model.pkl')
final_model = automl.load('model.pkl')
final_model.predict(X_test)
