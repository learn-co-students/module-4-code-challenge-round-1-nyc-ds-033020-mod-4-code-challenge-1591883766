
# Module 4 Code Challenge

This code challenge is designed to test your understanding of the Module 4 material. It covers:

* Principal Component Analysis
* Clustering
* Time Series
* Natural Language Processing

_Read the instructions carefully._ You will be asked both to write code and respond to a few short answer questions.

The goal here is to demonstrate your knowledge. Showing that you know things about certain concepts is more important than getting the best model. You can use any libraries you want to solve the problems in the assessment. 

### Note on the short answer questions

For the short answer questions, _please use your own words._ The expectation is that you have **not** copied and pasted from an external source, even if you consult another source to help craft your response. While the short answer questions are not necessarily being assessed on grammatical correctness or sentence structure, you should do your best to communicate yourself clearly.

---

## Part 1: Principal Component Analysis [Suggested Time: 15 minutes]

---

In the first part of the code challenge, you'll apply the unsupervised learning technique of Principal Component Analysis to the wine dataset. 

We load the wine dataset for you in the cell below. 


```python
# Run this cell without changes

# Relevant imports
import pandas as pd
import warnings
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Supress warnings
warnings.simplefilter("ignore")

# Load data
wine = load_wine()
X, y = load_wine(return_X_y=True)
X = pd.DataFrame(X, columns=wine.feature_names)
y = pd.Series(y)
y.name = 'class'

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling
scaler_1 = StandardScaler()
X_train_scaled = pd.DataFrame(scaler_1.fit_transform(X_train), columns=X_train.columns)

# Inspect the first five rows of the scaled dataset
X_train_scaled.head()
```

### 1.1) Fit PCA to the training data

Call the PCA instance you'll create `wine_pca`. Set `n_components=0.9` and make sure to use `random_state = 42`.

_Make sure you are using the **preprocessed data!**_


```python
# Your code here
```

### 1.2) How many principal components are there in the fitted PCA object?

_Hint: Look at the list of attributes of trained `PCA` objects in the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)_


```python
# Your code here
```

### 1.3) Is PCA more useful or less useful when you have high multicollinearity among your features? Explain why.


```python
"""

Your written answer here

"""
```

--- 

## Part 2: Clustering [Suggested Time: 20 minutes]

---

This second part of the code challenge is meant to test your clustering knowledge.

* If the gif doesn't run, you may access it via [this link](images/centroid.gif).

<img src='images/centroid.gif'>

### 2.1) Using the gif above for reference, describe the steps of the k-means clustering algorithm.


```python
"""

Your written answer here

"""
```

Now let's use the wine dataset again, this time for clustering.

You will use scikit-learn to fit k-means clustering models, and you will determine the optimal number of clusters to use by looking at silhouette scores. 

### 2.2) Write a function called `get_labels()` that will find `k` clusters in a dataset of features `X`, and return the labels for each row of `X`. 

Review the doc-string in the function below to understand the requirements of this function.

_Hint: Within the function, you'll need to:_
* instantiate a k-means clustering model (use `random_state = 1` for reproducibility),
* fit the model to the data, and
* return the labels for each point 


```python
# Replace None with appropriate code

# Relevant import(s) here
None

def get_labels(k, X):
    """ 
    Finds the labels from a k-means clustering model 
    
    Parameters: 
    -----------
    k: float object
        number of clusters to use in the k-means clustering model
    X: Pandas DataFrame or array-like object
        Data to cluster
    
    Returns: 
    --------
    labels: array-like object
        Labels attribute from the k-means model
    
    """
    
    # Instantiate a k-means clustering model with random_state=1 and n_clusters=k
    kmeans = None
    
    # Fit the model to the data
    None
    
    # Return the predicted labels for each row in the data produced by the model
    return None
```

In the cell below we fit the k-means algorithm to the wine data for $k$ values in the range 2 to 9 using the function you've written above. Then we obtain the silhouette scores for each trained k-means clustering model, and place the values in a list called `silhouette_scores`.


```python
# Run this cell without changes

from sklearn.metrics import silhouette_score

# Preprocessing is needed. Scale the data
scaler_2 = StandardScaler()
X_scaled = scaler_2.fit_transform(X)

# Create empty list for silhouette scores
silhouette_scores = []

# Range of k values to try
k_values = range(2, 10)

for k in k_values:
    labels = get_labels(k, X_scaled)
    score = silhouette_score(X_scaled, labels, metric='euclidean')
    silhouette_scores.append(score)
```

Next, we plot the silhouette scores obtained for each different value of $k$, against $k$, the number of clusters we asked the algorithm to find. 


```python
# Run this cell without changes

import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(k_values, silhouette_scores, marker='o')
plt.title('Silhouette scores vs number of clusters')
plt.xlabel('k (number of clusters)')
plt.ylabel('silhouette score');
```

### 2.3) Which value of $k$ would you choose based on the above plot of silhouette scores? How does this number compare to the number of classes in the [wine dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html)?

Hint: this number should be <= 5. If it's not, check the function written for Question 2.2.


```python
"""

Your written answer here

"""
```

---

## Part 3: Natural Language Processing [Suggested Time: 20 minutes]

---

In this third section we will attempt to classify text messages as "SPAM" or "HAM" using TF-IDF Vectorization. Once we successfully classify our texts we will consider how to interpret the vectorization.

Complete the functions below and answer the question at the end. 


```python
# Run this cell without changes

# Import necessary libraries 
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
# nltk.download('stopwords') # un-comment this if you get an error from nltk
```


```python
# Run this cell without changes

# Read in data
df_messages = pd.read_csv('data/spam.csv', usecols=[0,1])

# Convert string labels to 1 or 0 
le = LabelEncoder()
df_messages['target'] = le.fit_transform(df_messages['v1'])

# Examine our data
df_messages.head()
```


```python
# Run this cell without changes

# Separate features and labels 
X = df_messages['v2']
y = df_messages['target']

# Generate a list of stopwords 
stopwords_list = stopwords.words('english') + list(string.punctuation)
```

### 3.1) Create a function that takes in our various texts along with their respective labels and uses TF-IDF to vectorize the texts.

- Review the doc-string in the function below to understand the requirements of this function.
- Recall that TF-IDF helps us "vectorize" text (turn text into numbers) so we can do "math" with it.  It is used to quantify how relevant a term is in a given document.
- **DO NOT** perform tokenization, removal of stop words, or TF-IDF vectorization "by hand".  Use `sklearn`'s `TfidfVectorizer`.


```python
# Replace "pass" with appropriate code

def tfidf(X, y, stopwords_list): 
    """
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
        fit TF-IDF vectorizer object

    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    
    pass
```


```python
# Run this cell without changes
tf_idf_train, tf_idf_test, y_train, y_test, vectorizer = tfidf(X, y, stopwords_list)
```

### 3.2) Complete the function below to return a list of predictions for our training set and a separate list of predictions for our test set.

Now that we have a set of vectorized training data we can use this data to train a _classifier_ to learn how to classify a specific text based on the vectorized version of the text. Below we have initialized a simple Naive Bayes Classifier and Random Forest Classifier. 

Review the doc-string in the function below to understand the requirements of this function. The function should accept a classifier object, a vectorized training set, vectorized test set, and a list of training labels to return separate lists of predictions for the training and the test sets.


```python
# Run this cell without changes
nb_classifier = MultinomialNB()
rf_classifier = RandomForestClassifier(n_estimators=100)
```


```python
# Replace None with appropriate code

def classify_text(classifier, tf_idf_train, tf_idf_test, y_train):
    """
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
    """
    # Fit the classifier with our training data
    None
    
    # Predict the labels of our train data and store them in train_preds
    None
    
    # Predict the labels of our test data and store them in test_preds
    None
    
    return train_preds, test_preds
```

Generate and evaluate predictions for Naive Bayes Classifier


```python
# Run this cell without changes

nb_train_preds, nb_test_preds = classify_text(nb_classifier, tf_idf_train, tf_idf_test, y_train)

print(confusion_matrix(y_test, nb_test_preds))
print(accuracy_score(y_test, nb_test_preds))
```

Generate and evaluate predictions for Random Forest Classifier


```python
# Run this cell without changes

rf_train_preds, rf_test_preds = classify_text(rf_classifier, tf_idf_train, tf_idf_test, y_train)

print(confusion_matrix(y_test, rf_test_preds))
print(accuracy_score(y_test, rf_test_preds))
```

You can see both classifiers do a pretty good job classifying texts as either "SPAM" or "HAM". 

### 3.3) Based on the code below, the word "genuine" has the highest TF-IDF value in the second document of our test data. What does that tell us about the word "genuine"?


```python
# Run this cell without changes

tf_idf_test_df = pd.DataFrame(tf_idf_test.toarray(), columns=vectorizer.vocabulary_.keys())
second_doc = tf_idf_test_df.loc[1]
second_doc.idxmax(axis=1)
```


```python
# Run this cell without changes
second_doc['genuine']
```


```python
"""

Your written answer here

"""
```

---

## Part 4: Time Series [Suggested Time: 20 minutes]

---

<!---Create stock_df and save as .pkl
stocks_df = pd.read_csv("raw_data/all_stocks_5yr.csv")
stocks_df["clean_date"] = pd.to_datetime(stocks_df["date"], format="%Y-%m-%d")
stocks_df.drop(["date", "clean_date", "volume", "Name"], axis=1, inplace=True)
stocks_df.rename(columns={"string_date": "date"}, inplace=True)
pickle.dump(stocks_df, open("write_data/all_stocks_5yr.pkl", "wb"))
--->

Here you'll be looking at OHLC (Open, High, Low, Close) daily stock data.


```python
# Run this cell without changes

import pickle
import numpy as np

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

stocks_df = pickle.load(open('write_data/all_stocks_5yr.pkl', 'rb'))
stocks_df.head()
```

### 4.1) Transform the `date` feature so that it becomes a `datetime` object, and set `date` to be the index of `stocks_df`.

The format of the `date` feature is `'%B %d, %Y'` . Use this when converting the `date` feature to a `datetime` object in order for the code to run faster.

Be sure that the `date` index of `stocks_df` is in the format: YYYY-MM-DD (should do so automatically).


```python
# Your code here
```

### 4.2) Downsample `stocks_df` using the mean of the `open`, `high`, `low`, and `close` features on a monthly basis. Store the results in `stocks_monthly_df`.

Hint: `stocks_monthly_df` should have 61 rows and 4 columns after you perform downsampling.


```python
# Your code here
```


```python
# Run this cell without changes
stocks_monthly_df.shape
```

### 4.3) Create a line graph that visualizes the monthly open stock prices from `stocks_monthly_df`.

This is for the purposes of identifying if average monthly open stock price is stationary or not, using the rolling mean and rolling standard deviation.

Store a sliced version of `stocks_monthly_df` which grabs the `open` column in a new object called `open_monthly_series`.

Hint: use a window size of 3 to represent one quarter of a year


```python
# Replace None with appropriate code

open_monthly_series = None

roll_mean = None
roll_std = None

# Note: do not rename the variables otherwise the plot code will not work
```


```python
# Run this cell without changes
fig, ax = plt.subplots(figsize=(13, 10))
ax.plot(open_monthly_series, color='blue',label='Average monthly opening stock price')
ax.plot(roll_mean, color='red', label='Rolling quarterly mean')
ax.plot(roll_std, color='black', label='Rolling quarterly std. deviation')
ax.set_ylim(0, 120)
ax.legend()
fig.suptitle('Average monthly open stock prices, Feb. 2013 to Feb. 2018')
fig.tight_layout()
```

Based on your visual inspection of the above graph, is the monthly open stock price stationary? Explain your answer


```python
"""

Your written answer here

"""
```
