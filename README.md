# **Sentimental-Analysis-Project**

# Demo



https://user-images.githubusercontent.com/108968831/207074084-77b887f7-6615-47c8-b2ba-75e8897946db.mp4




# INTRODUCTION


Introduction
Sentiment analysis refers to identifying as well as classifying the sentiments that are expressed in the text source. Tweets are often useful in generating a vast amount of sentiment data upon analysis. These data are useful in understanding the opinion of the people about a variety of topics.

Therefore we need to develop an Automated Machine Learning Sentiment Analysis Model in order to compute the customer perception. Due to the presence of non-useful characters (collectively termed as the noise) along with useful data, it becomes difficult to implement models on them.

# Problem Statement
In this project, we try to implement a Twitter sentiment analysis model that helps to overcome the challenges of identifying the sentiments of the tweets. The necessary details regarding the dataset are:

The dataset provided is the Sentiment140 Dataset which consists of 1,600,000 tweets that have been extracted using the Twitter API. The various columns present in the dataset are:

- **sentiment:** the polarity of the tweet (positive or negative)
- **ids:** Unique id of the tweet
- **date:** the date of the tweet
- **query:** It refers to the query. If no such query exists then it is NO QUERY.
- **user_id:** It refers to the name of the user that tweeted
- **tweets:** It refers to the text of the tweet
## Steps to follow

The various steps involved in the Machine Learning Pipeline are :

- Import Necessary Dependencies :
- Read and Load the Dataset :
- Exploratory Data Analysis
- Data Visualization of Target Variables :
  
- Data Preprocessing : 

- Splitting our data into Train and Test Subset:
- Transforming Dataset using TF-IDF Vectorizer
- Function for Model Evaluation
- Model Building
- Conclusion
## Let’s get started, 

## Step-1: Import Necessary Dependencies
     Basic Python Libraries

  1. Pandas – library for data analysis and data manipulation
  2. Matplotlib – library used for data visualization
  3. Seaborn – a library based on matplotlib and it provides a high-level interface for data visualization
  4. WordCloud – library to visualize text data
  5. re – provides functions to pre-process the strings as per the given regular expression
    Natural Language Processing

  1. nltk – Natural Language Toolkit is a collection of libraries for natural language processing
  2. stopwords – a collection of words that don’t provide any meaning to a sentence
  3. WordNetLemmatizer – used to convert different forms of words into a single item but still keeping the context intact.
  
    Scikit-Learn (Machine Learning Library for Python)

    TF-IDF Vectorizer – transform text to vectors

  
  Evaluation Metrics

  1. Accuracy Score – no. of correctly classified instances/total no. of instances
  2. Precision Score – the ratio of correctly predicted instances over total positive instances
  3. Recall Score – the ratio of correctly predicted instances over total instances in that class
  4. Roc Curve – a plot of true positive rate against false positive rate
  5. Classification Report – report of precision, recall and f1 score
  6. Confusion Matrix – a table used to describe the classification models

## Step-2: Read and Load the Dataset

We now know that we are working with a typical CSV file (i.e., the delimiter is ,, etc.). We proceed to loading the data into memory.

## Step-3: Exploratory Data Analysis 
- Five top records of data
- Columns/features in data
- Length of the dataset
- Shape of data
- Data information
- Checking for Null values
- Rows and columns in the dataset
- Check unique Target Values

## Step-4: Data Visualization of Target Variables
- plot a count plot for positive and negative records in dataset
- Plot a cloud of words for negative tweets
- Plot a cloud of words for positive tweets

## Step-5: Data Preprocessing 
The preprocessing of the text data is an essential step as it makes the raw text ready for mining, i.e., it becomes easier to extract information from the text and apply machine learning algorithms to it. If we skip this step then there is a higher chance that you are working with noisy and inconsistent data. The objective of this step is to clean noise those are less relevant to find the sentiment of tweets such as punctuation, special characters, numbers, and terms which don’t carry much weightage in context to the text.

In the above-given problem statement before training the model, we have performed various pre-processing steps on the dataset that mainly dealt with removing stopwords, removing emojis. The text document is then converted into the lowercase for better generalization.

Subsequently, the punctuations were cleaned and removed thereby reducing the unnecessary noise from the dataset. After that, we have also removed the repeating characters from the words along with removing the URLs as they do not have any significant importance.

At last, we then performed Stemming(reducing the words to their derived stems) 

## Step-7: Transforming Dataset using TF-IDF Vectorizer
- Fit the Vectorizer 
- Transform the data using Vectorizer

This is another method which is based on the frequency method but it is different to the bag-of-words approach in the sense that it takes into account, not just the occurrence of a word in a single document (or tweet) but in the entire corpus.
TF-IDF works by penalizing the common words by assigning them lower weights while giving importance to words which are rare in the entire corpus but appear in good numbers in few documents.

## Step-8: Model Evaluation
After training the model we then apply the evaluation measures to check how the model is performing. Accordingly, we use the following evaluation parameters to check the performance of the models respectively

## Conclusion:
- Accuracy Score:
We will evaluate our model using various metrics such as Accuracy Score, Precision Score, Recall Score, Confusion Matrix and create a roc curve to visualize how our model performed.

we can conclude Logistic Regression model gives better acuuarcy among all models we preoceeded with Logistic Regression.

Accuracy is the ratio of the total number of correct predictions and the total number of predictions.Our data is balanced so we can use accuarcy to predict model ability. Logistic Regression gives 79% accuracy for the data.Hence, our model learnt 79 percent to predict or distinguished between happy or sad sentiments accurately.





