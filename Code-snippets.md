* PCA
``` python
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize the data to have a mean of ~0 and a variance of 1
X_std = StandardScaler().fit_transform(data.dropna())

# Create a PCA instance: pca
pca = PCA(n_components=26)
principalComponents = pca.fit_transform(X_std)
features = range(pca.n_components_)

pca_df = pd.DataFrame(pca.explained_variance_ratio_)

# Plot the explained variances
pca_df.iplot(kind='bar',title='PCA - Kidney Desease')
```
* count unique values
``` python
data.feature.value_count()
``` 
* Silhouette score
* plotly plot
* 6d plot
* Tsne
* elbow method
``` python
from sklearn.cluster import KMeans
from pandas import DataFrame

data_cluster =  data[['classification', 'hemoglobin']]
n_cluster = range(1, 6)

kmeans = [KMeans(n_clusters=i).fit(data_cluster) for i in n_cluster]
scores = [kmeans[i].score(data_cluster) for i in range(len(kmeans))]

scores_df = DataFrame(scores)
scores_df.iplot(kind="scatter", theme="white")
```
* Legend: 
``` python
# Legend and title
plt.legend(labels=['Mild Injuries', 'Serious Injuries'])
``` python
* Scatter plot with min max of certain days: 
``` python
accidents = df.groupby(df['date'].dt.date).count().date

accidents.plot(figsize=(13,8), color='blue')

# sunday accidents
sundays = df.groupby(df[df['date'].dt.dayofweek==6].date.dt.date).count().date
plt.scatter(sundays.index, sundays, color='green', label='sunday')

# friday accidents
friday = df.groupby(df[df['date'].dt.dayofweek==4].date.dt.date).count().date
plt.scatter(friday.index, friday, color='red', label='friday')

# Title, x label and y label
plt.title('Accidents in Barcelona in 2017', fontsize=20)
plt.xlabel('Date',fontsize=16)
plt.ylabel('Number of accidents per day',fontsize=16);
plt.legend()
```
* Workign with dates
```python
series = data[data.columns[11]].dropna()
series_dt = pd.to_datetime(pd.Series(series))
series_dt.dt.dayofweek
friday = series_dt.groupby(series_dt[series_dt.dt.dayofweek==4].dt.date).count()
```
* Timestamps
``` python
date = pd.DataFrame(data.last_review.dropna().str.split('-').tolist(), columns = ['year','month', 'day'])

ts = pd.Timestamp(year = int(date['year'][1]),  month = int(date['month'][1]), day = int(date['day'][1]),  
                  hour = 10, second = 49, tz = 'US/Central')  

ts
```
* XGBoostRegressor + hyperparameter tuning
``` python
from sklearn.metrics import mean_squared_error

# Instantiate an XGBRegressor
xgr = xgb.XGBRegressor(random_state=2)

# Fit the classifier to the training set
xgr.fit(X_train, y_train)

y_pred = xgr.predict(X_test)

mean_squared_error(y_test, y_pred)
```
``` python
from sklearn.model_selection import GridSearchCV

# Various hyper-parameters to tune
xgb1 = xgb.XGBRegressor()
parameters = {'nthread':[4], 
              'objective':['reg:linear'],
              'learning_rate': [.03, 0.05, .07], 
              'max_depth': [5, 6, 7],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [250]}

xgb_grid = GridSearchCV(xgb1,
                        parameters,
                        cv = 2,
                        n_jobs = 5,
                        verbose=True)

xgb_grid.fit(X_train, y_train)
```

* Train test split
``` python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)
```
* Feature importance, by name
``` python
for feature, importance in zip(list(X.columns), xgr.feature_importances_):
    print('Model weight for feature {}: {}'.format(feature, importance))
```
* Decode categorical string column to numbers
``` python
data.neighbourhood.astype('category').cat.codes
```