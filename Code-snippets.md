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
``` python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

print(__doc__)

X = data[['classification', 'hemoglobin']].to_numpy()

range_n_clusters = [2, 3, 4, 5, 6]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

plt.show()
```
* plotly plot
* 6d plot
* Tsne
``` python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
tsne_pca_results = tsne.fit_transform(principalComponents)

df = pd.DataFrame()


df['tsne-pca50-one'] = tsne_pca_results[:,0]
df['tsne-pca50-two'] = tsne_pca_results[:,1]
df['classification'] = data['classification']


plt.figure(figsize=(16,4))
sns.scatterplot(
    x="tsne-pca50-one", y="tsne-pca50-two",
    hue="classification",
    palette=sns.color_palette("hls", 2),
    data=df,
    legend="full",
    alpha=0.3
)
```
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
* Get columns by data type
``` python
data.dtypes[data.dtypes == ‘object’]
```
* Labelencoder
``` python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
```
* OneHotEncoding
``` python
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
```
* Scaling
``` python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```
* Confusion Matrix
``` python
from sklearn.metrics import confusion_matrix

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test, y_pred)
```
* Collaborative filtering recommender system
``` python
import graphlab
import pandas as pd

# Load up the data with pandas
r_cols = ['user_id', 'food_item', 'rating']
train_data_df = pd.read_csv('train_data.csv', sep='\t', names=r_cols)
test_data_df = pd.read_csv('test_data.csv', sep='\t', names=r_cols)

# Convert the pandas dataframes to graph lab SFrames
train_data = graphlab.SFrame(train_data_df)
test_data = graphlab.SFrame(test_data_df)

# Train the model
collab_filter_model = graphlab.item_similarity_recommender.create(train_data, 
                                                                  user_id='user_id', 
                                                                  item_id='food_item',                                                                 
                                                                  target='rating', 
                                                                  similarity_type='cosine')
                                                                  
# Make recommendations
which_user_ids = [1, 2, 3, 4]
how_many_recommendations = 5
item_recomendation = collab_filter_model.recommend(users=which_user_ids,
                                                   k=how_many_recommendations)
```
* Content based recommender system
``` python
import graphlab
import pandas as pd

# Load up the data with pandas
r_cols = ['user_id', 'food_item', 'rating']
train_data_df = pd.read_csv('train_data.csv', sep='\t', names=r_cols)
test_data_df = pd.read_csv('test_data.csv', sep='\t', names=r_cols)

# Convert the pandas dataframes to graph lab SFrames
train_data = graphlab.SFrame(train_data_df)
test_data = graphlab.SFrame(test_data_df)

# Train the model
cotent_filter_model = graphlab.item_content_recommender.create(train_data, 
                                                              user_id='user_id', 
                                                              item_id='food_item', 
                                                              target='rating')
                                                                  
# Make recommendations
which_user_ids = [1, 2, 3, 4]
how_many_recommendations = 5
item_recomendation = cotent_filter_model.recommend(users=which_user_ids,
                                                   k=how_many_recommendations)
``` 
* Find and drop duplicates
``` python
duplicate_rows_df = df[df.duplicated()]
print(“number of duplicate rows: “, duplicate_rows_df.shape)

df = df.drop_duplicates()
```
* Find and drop outliers
``` python

Exploratory data analysis in Python.
Let us understand how to explore the data in python.
Tanu N Prabhu
Tanu N Prabhu
Aug 10 · 9 min read

Image Credits: Morioh
Introduction
What is Exploratory Data Analysis?
Exploratory Data Analysis or (EDA) is understanding the data sets by summarizing their main characteristics often plotting them visually. This step is very important especially when we arrive at modeling the data in order to apply Machine learning. Plotting in EDA consists of Histograms, Box plot, Scatter plot and many more. It often takes much time to explore the data. Through the process of EDA, we can ask to define the problem statement or definition on our data set which is very important.
How to perform Exploratory Data Analysis?
This is one such question that everyone is keen on knowing the answer. Well, the answer is it depends on the data set that you are working. There is no one method or common methods in order to perform EDA, whereas in this tutorial you can understand some common methods and plots that would be used in the EDA process.
What data are we exploring today?
Since I am a huge fan of cars, I got a very beautiful data-set of cars from Kaggle. The data-set can be downloaded from here. To give a piece of brief information about the data set this data contains more of 10, 000 rows and more than 10 columns which contains features of the car such as Engine Fuel Type, Engine Size, HP, Transmission Type, highway MPG, city MPG and many more. So in this tutorial, we will explore the data and make it ready for modeling.
Let's get started !!!
1. Importing the required libraries for EDA
Below are the libraries that are used in order to perform EDA (Exploratory data analysis) in this tutorial. The complete code can be found on my GitHub.
# Importing required libraries.
import pandas as pd
import numpy as np
import seaborn as sns #visualisation
import matplotlib.pyplot as plt #visualisation
%matplotlib inline 
sns.set(color_codes=True)
2. Loading the data into the data frame.
Loading the data into the pandas data frame is certainly one of the most important steps in EDA, as we can see that the value from the data set is comma-separated. So all we have to do is to just read the CSV into a data frame and pandas data frame does the job for us.
To get or load the dataset into the notebook, all I did was one trivial step. In Google Colab at the left-hand side of the notebook, you will find a “>” (greater than symbol). When you click that you will find a tab with three options, you just have to select Files. Then you can easily upload your file with the help of the Upload option. No need to mount to the google drive or use any specific libraries just upload the data set and your job is done. One thing to remember in this step is that uploaded files will get deleted when this runtime is recycled. This is how I got the data set into the notebook.
df = pd.read_csv(“data.csv”)
# To display the top 5 rows
df.head(5)

Displaying the top 5 rows.
# To display the bottom 5 rows
df.tail(5) 

Displaying the last 10 rows.
3. Checking the types of data
Here we check for the datatypes because sometimes the MSRP or the price of the car would be stored as a string or object, if in that case, we have to convert that string to the integer data only then we can plot the data via a graph. Here, in this case, the data is already in integer format so nothing to worry.
# Checking the data type
df.dtypes

Checking the type of data.
4. Dropping irrelevant columns
This step is certainly needed in every EDA because sometimes there would be many columns that we never use in such cases dropping is the only solution. In this case, the columns such as Engine Fuel Type, Market Category, Vehicle style, Popularity, Number of doors, Vehicle Size doesn't make any sense to me so I just dropped for this instance.
# Dropping irrelevant columns
df = df.drop([‘Engine Fuel Type’, ‘Market Category’, ‘Vehicle Style’, ‘Popularity’, ‘Number of Doors’, ‘Vehicle Size’], axis=1)
df.head(5)

Dropping irrelevant columns.
5. Renaming the columns
In this instance, most of the column names are very confusing to read, so I just tweaked their column names. This is a good approach it improves the readability of the data set.
# Renaming the column names
df = df.rename(columns={“Engine HP”: “HP”, “Engine Cylinders”: “Cylinders”, “Transmission Type”: “Transmission”, “Driven_Wheels”: “Drive Mode”,”highway MPG”: “MPG-H”, “city mpg”: “MPG-C”, “MSRP”: “Price” })
df.head(5)

Renaming the column name.
6. Dropping the duplicate rows
This is often a handy thing to do because a huge data set as in this case contains more than 10, 000 rows often have some duplicate data which might be disturbing, so here I remove all the duplicate value from the data-set. For example prior to removing I had 11914 rows of data but after removing the duplicates 10925 data meaning that I had 989 of duplicate data.
# Total number of rows and columns
df.shape
(11914, 10)
# Rows containing duplicate data
duplicate_rows_df = df[df.duplicated()]
print(“number of duplicate rows: “, duplicate_rows_df.shape)
number of duplicate rows:  (989, 10)
Now let us remove the duplicate data because it's ok to remove them.
# Used to count the number of rows before removing the data
df.count() 
Make            11914 
Model           11914 
Year            11914 
HP              11845 
Cylinders       11884 
Transmission    11914 
Drive Mode      11914 
MPG-H           11914 
MPG-C           11914 
Price           11914 
dtype: int64
So seen above there are 11914 rows and we are removing 989 rows of duplicate data.
# Dropping the duplicates 
df = df.drop_duplicates()
df.head(5)

# Counting the number of rows after removing duplicates.
df.count()
Make            10925 
Model           10925 
Year            10925 
HP              10856 
Cylinders       10895 
Transmission    10925 
Drive Mode      10925 
MPG-H           10925 
MPG-C           10925 
Price           10925 
dtype: int64
7. Dropping the missing or null values.
This is mostly similar to the previous step but in here all the missing values are detected and are dropped later. Now, this is not a good approach to do so, because many people just replace the missing values with the mean or the average of that column, but in this case, I just dropped that missing values. This is because there is nearly 100 missing value compared to 10, 000 values this is a small number and this is negligible so I just dropped those values.
# Finding the null values.
print(df.isnull().sum())
Make             0 
Model            0 
Year             0 
HP              69 
Cylinders       30 
Transmission     0 
Drive Mode       0 
MPG-H            0 
MPG-C            0 
Price            0 
dtype: int64
This is the reason in the above step while counting both Cylinders and Horsepower (HP) had 10856 and 10895 over 10925 rows.
# Dropping the missing values.
df = df.dropna() 
df.count()
Make            10827 
Model           10827 
Year            10827 
HP              10827 
Cylinders       10827 
Transmission    10827 
Drive Mode      10827 
MPG-H           10827 
MPG-C           10827 
Price           10827 
dtype: int64
Now we have removed all the rows which contain the Null or N/A values (Cylinders and Horsepower (HP)).
# After dropping the values
print(df.isnull().sum()) 
Make            0 
Model           0 
Year            0 
HP              0 
Cylinders       0 
Transmission    0 
Drive Mode      0 
MPG-H           0 
MPG-C           0 
Price           0 
dtype: int64
8. Detecting Outliers
An outlier is a point or set of points that are different from other points. Sometimes they can be very high or very low. It’s often a good idea to detect and remove the outliers. Because outliers are one of the primary reasons for resulting in a less accurate model. Hence it’s a good idea to remove them. The outlier detection and removing that I am going to perform is called IQR score technique. Often outliers can be seen with visualizations using a box plot. Shown below are the box plot of MSRP, Cylinders, Horsepower and EngineSize. Herein all the plots, you can find some points are outside the box they are none other than outliers. The technique of finding and removing outlier that I am performing in this assignment is taken help of a tutorial from towards data science.
sns.boxplot(x=df[‘Price’])

Box plot of Price
sns.boxplot(x=df[‘HP’])

Box Plot of HP
sns.boxplot(x=df['Cylinders'])

Box Plot of Cylinders
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

df = df[~((df < (Q1–1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
df.shape
```