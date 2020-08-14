import GetOldTweets3 as got
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline
import plotly
import cufflinks as cf
import plotly
cf.go_offline()
import datetime
sid = SentimentIntensityAnalyzer()
Previous_Date = (datetime.datetime.today() - datetime.timedelta(days=1)).date()
NextDay_Date = (datetime.datetime.today() + datetime.timedelta(days=1)).date()
prev = Previous_Date.strftime("%Y-%m-%d")
nxt = NextDay_Date.strftime("%Y-%m-%d")
def Get_tweets(prev_date,nxt_date,limit):
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch('Biden')\
                                           .setSince(prev)\
                                           .setUntil(nxt)\
                                           .setMaxTweets(limit)
    tweetCriteria2 = got.manager.TweetCriteria().setQuerySearch('Trump')\
                                           .setSince(prev)\
                                           .setUntil(nxt)\
                                           .setMaxTweets(limit)
    l1 = got.manager.TweetManager.getTweets(tweetCriteria)
    l2 = got.manager.TweetManager.getTweets(tweetCriteria2)
    Biden_tweets  = [t.text for t in l1]
    Trump_tweets  = [t.text for t in l2]
    Trump_df = pd.DataFrame(Trump_tweets,columns=['Text'])
    Biden_df = pd.DataFrame(Biden_tweets,columns=['Text'])
    Trump_df['Date'] = [t.date.date() for t in l2]
    Biden_df['Date'] = [t.date.date() for t in l1]
    
    return Trump_df,Biden_df
    
Trump_df,Biden_df = Get_tweets(prev,nxt,100)
def Get_Sentiment_Score(Trump_df,Biden_df):
    Trump_df['Score'] = Trump_df['Text'].apply(lambda x:sid.polarity_scores(x))
    Trump_df['Sentiment_Score'] = Trump_df['Score'].apply(lambda x:x['compound'])
    Biden_df['Score'] = Biden_df['Text'].apply(lambda x:sid.polarity_scores(x))
    Biden_df['Sentiment_Score'] = Biden_df['Score'].apply(lambda x:x['compound'])
    return Trump_df,Biden_df
Trump_df,Biden_df = Get_Sentiment_Score(Trump_df,Biden_df)
t1 = pd.DataFrame()
t1 = Trump_df.groupby(['Date']).mean()
t1['Date'] = t1.index.get_values()
t1.reset_index(drop=True,inplace = True)
b1 = pd.DataFrame()
b1 = Biden_df.groupby(['Date']).mean()
b1['Date'] = b1.index.get_values()
b1.reset_index(drop=True,inplace = True)
t1.sort_values(by=['Date'],inplace=True)
b1.sort_values(by=['Date'],inplace=True)
labels = ['Sentiment','Date']
Trump_sent = pd.concat((t1['Sentiment_Score'],t1['Date']),axis=1,keys = labels)
Biden_sent = pd.concat((b1['Sentiment_Score'],b1['Date']),axis=1,keys = labels)
Biden_sent.set_index('Date',inplace=True)
Trump_sent.set_index('Date',inplace=True)
Names = ['Trump','Biden']
sentiments = pd.concat([Trump_sent,Biden_sent],axis = 1,keys = Names)
sentiments.iplot(xTitle = 'Date',yTitle = 'Sentiment Score')
