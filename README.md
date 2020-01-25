
# Module 4 Code Challenge

This code challenge is designed to test your understanding of the Module 4 material. It covers:

* Clustering
* Time Series
* Natural Language Processing
* Neural Networks

_Read the instructions carefully._ You will be asked both to write code and respond to a few short answer questions.

The goal here is to demonstrate your knowledge. Showing that you know things about certain concepts is more important than getting the best model. You can use any libraries you want to solve the problems in the assessment. 

You will have up to 90 minutes to complete this assessments
### Note on the short answer questions

For the short answer questions, _please use your own words._ The expectation is that you have **not** copied and pasted from an external source, even if you consult another source to help craft your response. While the short answer questions are not necessarily being assessed on grammatical correctness or sentence structure, you should do your best to communicate yourself clearly.

## Part 1: Clustering [Suggested Time: 25 min]

This first part of the code challenge is meant to test your k-means and hierarchical agglomerative clustering knowledge.

### 1.1) Using the gif below for reference, describe the steps of the k-means clustering algorithm.
* If the gif doesn't run, you may access it via [this link](images/centroid.gif).

<img src='images/centroid.gif'>


```python
"""

Written answer here

"""
```

### 1.2) In a similar way, describe the process behind Hierarchical Agglomerative Clustering.


```python
"""

Written answer here

"""
```

Next, you will apply k-means clustering to your now friend, the wine dataset. 

You will use scikit-learn to fit k-means clustering models, and you will determine the optimal number of clusters to use by looking at silhouette scores. 

We load the wine dataset for you in the cell below. 


```python
from sklearn.datasets import load_wine

X, y = load_wine(return_X_y=True)
wine = load_wine()
X = pd.DataFrame(X, columns = wine.feature_names)
```

### 1.3) Write a function called `get_labels` that will find `k` clusters in a dataset of features `X`, and return the labels for each row of `X`

_Hint: Within the function, you'll need to:_
* instantiate a k-means clustering model (use `random_state = 1` for reproducibility),
* fit the model to the data, and
* return the labels for each point.


```python
# Replace None and pass with appropriate code
def get_labels(k, X):
    
    # Instantiate a k-means clustering model with random_state=1 and n_clusters=k
    kmeans = None
    
    # Fit the model to the data
    None
    
    # return the predicted labels for each row in the data
    pass 
```

**1.4) Fit the k-means algorithm to the wine data for k values in the range 2 to 9 using the function you've written above. Obtain the silhouette scores for each trained k-means clustering model, and place the values in a list called `silhouette_scores`.** 

We have provided you with some starter code in the cell below.

_Hints: What imports do you need? Do you need to pre-process the data in any way before fitting the k-means clustering algorithm?_ 


```python
# Code here

silhouette_scores= []

for k in range(2, 10):
    labels = None 
    
    score = silhouette_score(None, None, metric='euclidean')
    
    silhouette_scores.append(score)
```

Run the cell below to plot the silhouette scores obtained for each different value of k against k, the number of clusters we asked the algorithm to find. 


```python
plt.plot(range(2, 10), silhouette_scores, marker='o')
plt.title('Silhouette scores vs number of clusters')
plt.xlabel('k (number of clusters)')
plt.ylabel('silhouette score')
```

**1.5) Which value of k would you choose based on the plot of silhouette scores? How does this number compare to the number of classes in the wine dataset?**

Hint: this number should be <= 5.  If it's not, check your answer in the previous section.


```python
"""

Written answer here

"""
```

## Part 2: Time Series [Suggested Time: 15 minutes]

<!---Create stock_df and save as .pkl
stocks_df = pd.read_csv("raw_data/all_stocks_5yr.csv")
stocks_df["clean_date"] = pd.to_datetime(stocks_df["date"], format="%Y-%m-%d")
stocks_df.drop(["date", "clean_date", "volume", "Name"], axis=1, inplace=True)
stocks_df.rename(columns={"string_date": "date"}, inplace=True)
pickle.dump(stocks_df, open("write_data/all_stocks_5yr.pkl", "wb"))
--->

In the second part of the assessment, you'll be looking at OHLC (Open, High, Low, Close) daily stock data.


```python
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
```


```python
stocks_df = pickle.load(open("write_data/all_stocks_5yr.pkl", "rb"))
stocks_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15.07</td>
      <td>15.12</td>
      <td>14.63</td>
      <td>14.75</td>
      <td>February 08, 2013</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14.89</td>
      <td>15.01</td>
      <td>14.26</td>
      <td>14.46</td>
      <td>February 11, 2013</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14.45</td>
      <td>14.51</td>
      <td>14.10</td>
      <td>14.27</td>
      <td>February 12, 2013</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14.30</td>
      <td>14.94</td>
      <td>14.25</td>
      <td>14.66</td>
      <td>February 13, 2013</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14.94</td>
      <td>14.96</td>
      <td>13.16</td>
      <td>13.99</td>
      <td>February 14, 2013</td>
    </tr>
  </tbody>
</table>
</div>



### 2.1) Transform the `date` feature so that it becomes a `datetime` object that contains the following format: YYYY-MM-DD and set `date` to be the index of `stocks_df`.


```python
# Code here
```

### 2.2) Perform downsampling `stocks_df` that takes the mean of the `open`, `high`, `low`, and `close` features on a monthly basis. Store the results in `stocks_monthly_df`.

> Hint: `stocks_monthly_df` should have 61 rows and 4 columns after you perform downsampling.


```python
# Code here
```


```python
stocks_monthly_df.shape
```

### 2.3) Create a line graph that visualizes the monthly open stock prices from `stocks_monthly_df` for the purposes of identifying if average monthly open stock price is stationary or not using the rolling mean and rolling standard deviation.

> Hint: 
> * store your sliced version of `stocks_monthly_df` in a new DataFrame called `open_monthly_df`;
> * use a window size of 3 to represent one quarter of time in a year


```python
# Code here

open_monthly_df = None

rolmean = None
rolstd = None

# note: do not rename the variables otherwise the plot code will not work
```


```python
fig, ax = plt.subplots(figsize=(13, 10))
ax.plot(open_monthly_df, color="blue",label="Average monthly opening stock price")
ax.plot(rolmean, color="red", label="Rolling quarterly mean")
ax.plot(rolstd, color="black", label="Rolling quarterly std. deviation")
ax.set_ylim(0, 120)
ax.legend()
fig.suptitle("Average monthly open stock prices, Feb. 2013 to Feb. 2018")
fig.tight_layout()
```

Based on your visual inspection of the graph, is the monthly open stock price stationary?


```python
"""

Written answer here

"""
```

### 2.4) Use the Dickey-Fuller Test to identify if `open_monthly_df` is stationary


```python
# Code here
```

Does this confirm your answer from Question 3? Explain why the time series is stationary or not based on the output from the Dickey-Fuller Test.


```python
"""

Written answer here

"""
```

### 2.5) Looking at the decomposition of the time series in `open_monthly_df`, it looks like the peaks are the same value. To confirm or deny this, create a function that returns a dictionary where each key is year and each value is the maximum value from the `seasonal` object for each year.


```python
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(np.log(open_monthly_df))

# Gather the trend, seasonality and noise of decomposed object
seasonal = decomposition.seasonal

# Plot gathered statistics
plt.figure(figsize=(13, 10))
plt.plot(seasonal,label='Seasonality', color="blue")
plt.title("Seasonality of average monthly open stock prices, Feb. 2013 to Feb. 2018")
plt.ylabel("Average monthly open stock prices")
plt.tight_layout()
plt.show()
```


```python
# Replace "pass" with appropriate code

def calc_yearly_max(seasonal_series):
    """Returns the max seasonal value for each year"""
    pass
```


```python
calc_yearly_max(seasonal)
```

## Part 3: Natural Language Processing [Suggested Time: 20 minutes]

In this exercise we will attempt to classify text messages as "SPAM" or "HAM" using TF-IDF Vectorization. Once we successfully classify our texts we will examine our results to see which words are most important to each class of text messages. 

Complete the functions below and answer the question(s) at the end. 


```python
#import necessary libraries 
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import string
from nltk.corpus import stopwords
from nltk import word_tokenize
```


```python
# read in data
df_messages = pd.read_csv('data/spam.csv', usecols=[0,1])
```


```python
# convert string labels to 1 or 0 
le = LabelEncoder()
df_messages['target'] = le.fit_transform(df_messages['v1'])
```


```python
# examine or data
df_messages.head()
```


```python
# separate features and labels 
X = df_messages['v2']
y = df_messages['target']

```


```python
# generate a list of stopwords 
stopwords_list = stopwords.words('english') + list(string.punctuation)

```

### 3.1) Create a function that takes in our various texts along with their respective labels and uses TF-IDF to vectorize the texts.  Recall that TF-IDF helps us "vectorize" text (turn text into numbers) so we can do "math" with it.  It is used to reflect how relevant a term is in a given document in a numerical way. 


```python
#generate tf-idf vectorization (use sklearn's TfidfVectorizer) for our data, replace "pass" below
def tfidf(X, y,  stopwords_list): 
    '''
    Generate train and test TF-IDF vectorization for our data set
    
    Parameters
    ----------
    X: pandas.Series object
        Pandas series of text documents to classify 
    y : pandas.Series object
        Pandas series containing label for each document
    stopwords_list: list ojbect
        List containing words and punctuation to remove. 
    Returns
    --------
    tf_idf_train :  sparse matrix, [n_train_samples, n_features]
        Vector representation of train data
    tf_idf_test :  sparse matrix, [n_test_samples, n_features]
        Vector representation of test data
    y_train : array-like object
        labels for training data
    y_test : array-like object
        labels for testing data
    vectorizer : vectorizer object
        fit TF-IDF vecotrizer object

    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    
    pass
```


```python
tf_idf_train, tf_idf_test, y_train, y_test, vecotorizer = tfidf(X, y, stopwords_list)
```

### 3.2) Now that we have a set of vectorized training data we can use this data to train a _classifier_ to learn how to classify a specific text based on the vectorized version of the text. Below we have initialized a simple Naive Bayes Classifier and Random Forest Classifier. Complete the function below which will accept a classifier object, a vectorized training set, vectorized test set, and list of training labels and return a list of predictions for our training set and a separate list of predictions for our test set.


```python
nb_classifier = MultinomialNB()
rf_classifier = RandomForestClassifier(n_estimators=100)
```


```python
# create a function that takes in a classifier and trains it on our tf-idf vectors 
# and generates test and train predictions. Replace "pass" in the code below.
def classify_text(classifier, tf_idf_train, tf_idf_test, y_train):
    '''
    Train a classifier to identify whether a message is spam or ham
    
    Parameters
    ----------
    classifier: sklearn classifier
       initialized sklearn classifier (MultinomialNB, RandomForestClassifier, etc.)
    tf_idf_train : sparse matrix, [n_train_samples, n_features]
        TF-IDF vectorization of train data
    tf_idf_test : sparse matrix, [n_test_samples, n_features]
        TF-IDF vectorization of test data
    y_train : pandas.Series object
        Pandas series containing label for each document in the train set
    Returns
    --------
    train_preds :  list object
        Predictions for train data
    test_preds :  list object
        Predictions for test data
    '''
    #fit the classifier with our training data
    
    #predict the labels of our train data and store them in train_preds
    
    #predict the labels of our test data and store them in test_preds
    pass
```


```python
# generate predictions for Naive Bayes Classifier
nb_train_preds, nb_test_preds = classify_text(nb_classifier,tf_idf_train, tf_idf_test, y_train)
```


```python
print(confusion_matrix(y_test, nb_test_preds))
print(accuracy_score(y_test, nb_test_preds))
```


```python
#generate predictions for Random Forest Classifier
rf_train_preds, rf_test_preds = classify_text(rf_classifier,tf_idf_train, tf_idf_test, y_train)
```


```python
print(confusion_matrix(y_test, rf_test_preds))
print(accuracy_score(y_test, rf_test_preds))
```

You can see both classifiers do a pretty good job classifying texts as either "SPAM" or "HAM". Let's figure out which words are the most important to each class of texts! Recall that Inverse Document Frequency can help us determine which words are most important in an entire corpus or group of documents. 

### 3.3) Create a function that calculates the IDF of each word in our collection of texts.


```python
# Replace "pass" with the appropriate code

def get_idf(class_, df, stopwords_list):
    '''
    Get ten words with lowest IDF values representing 10 most important
    words for a defined class (spam or ham)
    
    Parameters
    ----------
    class_ : str object
        string defining class 'spam' or 'ham'
    df : pandas DataFrame object
        data frame containing texts and labels
    stopwords_list: list object
        List containing words and punctuation to remove. 
    --------
    important_10 : pandas dataframe object
        Dataframe containing 10 words and respective IDF values
        representing the 10 most important words found in the texts
        associated with the defined class
    '''
    #generate series containing all texts associated with the defined class
    docs = 'code here'
    
    #initialize dictionary to count document frequency 
    # (number of documents that contain a certain word)
    class_dict = {}
    
    #loop over each text and split each text into a list of its unique words 
    for doc in docs:
        words = set(doc.split())
        
        #loop over each word and if it is not in the stopwords_list add the word 
        #to class_dict with a value of 1. if it is already in the dictionary
        #increment it by 1
        pass
        
    #take our dictionary and calculate the 
    #IDF (number of docs / number of docs containing each word) 
    #for each word and return the 10 words with the lowest IDF 
    pass
```


```python
get_idf('spam', df_messages, stopwords_list)
```


```python
get_idf('ham', df_messages, stopwords_list)
```

### 3.4) Imagine that the word "school" has the highest TF-IDF value in the second document of our test data. What does that tell us about the word school?


```python
"""

Written answer here

"""
```

## Part 4: Regularization and Optimization of Neural Networks [Suggested Time: 20 minutes]

Now you're going to train full neural networks on a small set of data. It is a binary classification problem in which you need to identify whether or not a dot will belong to the teal or orange class. Let's first import some of the necessary libraries.


```python
import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.datasets import make_gaussian_quantiles, make_circles
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras import Sequential, regularizers
from keras.layers import Dense
```


```python
np.random.seed(0)
# generate 2d classification dataset
X, y = make_circles(n_samples=450, noise=0.12)
df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'teal', 1:'orange'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    if key != 2:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
pyplot.show()
```


![png](index_files/index_65_0.png)


In the two cells below, the set of data has been split into a training and testing set and then fit to a neural network with two hidden layers. Run the two cells below to see how well the model performs.


```python
# train/test/split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
```


```python
np.random.seed(0)

#Instantiate Classifier
classifier = Sequential()

#Hidden Layer
classifier.add(Dense(
    32, 
    activation='relu', 
    input_dim=2,
    kernel_initializer='random_normal',

))

#Hidden Layer
classifier.add(Dense(
    32,
    activation='relu', 
    input_dim=2,
    kernel_initializer='random_normal',

))

#Output Layer
classifier.add(Dense(
    1, 
    activation='sigmoid',
    kernel_initializer='random_uniform',
))

classifier.compile(optimizer ='adam',loss="binary_crossentropy",metrics =['accuracy'])

classifier.fit(X_train, y_train, epochs=25, verbose=0, batch_size=10, shuffle=False)

# TRAIN

#predict classes
predicted_vals_train = classifier.predict_classes(X_train)
#show accuracy score
print(accuracy_score(y_train,predicted_vals_train))


# TEST

#predict classess
predicted_vals_test = classifier.predict_classes(X_test)
#show accuracy score
print(accuracy_score(y_test,predicted_vals_test))
```

    0.8126984126984127
    0.6666666666666666


### 4.1) Modify the code below to use L2 regularization


The model appears to be overfitting. To deal with this overfitting, modify the code below to include L2 regularization in the model. 


```python
np.random.seed(0)

#Instantiate Classifier
classifier2 = Sequential()

#Hidden Layer
classifier2.add(Dense(
    32, 
    activation='relu', 
    input_dim=2,
    kernel_initializer='random_normal'

))

#Hidden Layer
classifier2.add(Dense(
    32,
    activation='relu', 
    input_dim=2,
    kernel_initializer='random_normal'

))

#Output Layer
classifier2.add(Dense(
    1, 
    activation='sigmoid',
    kernel_initializer='random_uniform',
))

classifier2.compile(optimizer ='adam',loss="binary_crossentropy",metrics =['accuracy'])

classifier2.fit(X_train, y_train, epochs=25, verbose=0, batch_size=10, shuffle=False)

# TRAIN

#predict classes
predicted_vals_train = classifier2.predict_classes(X_train)
#show accuracy score
print(accuracy_score(y_train,predicted_vals_train))

# TEST

#predict classess
predicted_vals_test = classifier2.predict_classes(X_test)
#show accuracy score
print(accuracy_score(y_test,predicted_vals_test))

```

Did the regularization you performed prevent overfitting?


```python
"""

Written answer here

"""
```

### 4.2) Explain how regularization is related to the bias/variance tradeoff within Neural Networks and how it's related to the results you just achieved in the training and test accuracies of the previous models. What does regularization change in the training process (be specific to what is being regularized and how it is regularizing)?



```python
"""

Written answer here

"""
```

### 4.2) How might L1  and dropout regularization change a neural network's architecture?


```python
"""

Written answer here

"""
```
