# Sentiment_analysis_twitter
it will do sentiment analysis on the twitter dataset
It is the natural Language Processing  problem(NLP) it is done by classifing the tweets by negative,pasitive,nutral
by the mechine learning models for classification,text mining,text analysis,data mining tasks
# About the problem
Natural language processing is the very important reserch in datascience.Now a daysSentiment analysis is the very common first level
task in the NLP.We need to classify the tweets into negavite,pasitive and nutral. In the real time it is very time consuming to do this by humans along with
time it need thousands of people. So to avoid that we build a model that classifyies the tweets for us
We will do so by following a sequence of steps needed to solve the general sentiment analysis problem.
We start with preprocessing the raw text of the tweets. Then we will explore the preprocessed text and try to get some intuition about the tweets.
After thet,we will extract the numerical features from the data and finally use the numerical features to train the model.
# Understanding the problem statement 
Lets go through the problem statement.
The goal of the problem is to detect hate speech in tweets. we say the tweets containing hate speech if it has a racist or sexist sentiment associates with it. SO the tasks
is to classify racist or sexist tweets from other tweets.
Steps inside the problem <br />
* first we start with the data preprocessing and cleaning <br />
The preprocessing is the essential step in the data mining if generate the raw data ready for mining.If we skip this step we are likly to have a very noise 
data to build the model and it will lead to the less accuracy.
The objective of the step is to clean the noise those are less respectfull to find the statement of tweets such as punctuation, special charecters,numbers and charecters
like smilyes. <br />
* Stemming <br />
Let me explain something about stemming, we have chance to come across some words having the same meaning but in different parts of speech,  this step converts that type 
of words to thier root word <br />
EX. loves and loved both have the same meaning but in different parts of speech so this step convert these words to "love" a root single word
* CountVectorizer <br />
In this step we exctract the numerical feature of the cleaned data using the bag of words model.
converting the data into dataframe that having the columns with the words in the data and the samples filled with frequency of the particular word in the dataset. <br />
* Traing <br />
with this numerical features we train the model with train data, and evaluate its accuracy with evaluation dataset adn finally test it by test data <br />
* Conclution <br />
in this note we got some information about the approach sentiment analysis.
Did you feel this was helpful? Do you have any other useful ideas? Feel free to discuss your experience..happy pythoning <br />
# Author
Pavithra Devi M <br /> 
pavipd495@gmail.com

