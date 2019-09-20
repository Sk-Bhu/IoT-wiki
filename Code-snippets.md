* PCA
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
```python
* Workign with dates
```
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