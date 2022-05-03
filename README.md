# Name: Chau Nguyen

# Table of Contents

* [Introduction](#intro)
    * [Overview of dataset](#overview)
    * [Classifier for <code>is_submitter</code>](#classifier)
        * [Imbalanced classes](#imb)
        * [Evaluation metrics](#metrics)

* [Code](#code)

* [Methods](#methods)
    * [Data Pre-Processing](#pp)
        * [Part 1: Exploratory data analysis](#eda)
        * [Part 2: Data cleaning](#clean)
        * [Part 3: Feature engineering](#fe)
    * [Modeling](#modeling)
        * [Part 4: Building toy model on small data partition](#toy_model)
        * [Part 5: Big data & SparkML](#big_ml)
        * [Part 6: Model evaluation](#eval)

* [Conclusion](#conclusion)
    * [Lessons learned](#lessons)
        * [Basic ML: Goals, metrics, domain expertise](#lessons_basic)
        * [Big data ML: Patience, cloud computing, pipeline](lessons_big)
    * [Things I would do differently](#diff)

# Introduction <a name = 'intro'>

## Overview of dataset <a name = 'overview'>

Size, count, min date, max date, etc

## Classifier for <code>is_submitter</code> <a name='classifier'>

* In reddit threads, we can usually tell who the submitter (or OP - "Original Poster") is if they use the same account to make comments on posts they've made.

* Reddit users have the ability to switch between multiple accounts and comment on their own thread, creating fabricated interaction.

* In practice, Reddit the company may have records such as IP address and browser signature that could help them identify whether 2 accounts are indeed the same user. However, this information is not publicly available in the API.

* Therefore, I wanted to see if I could build a classifier model with SparkML to classifier whether comments in a thread were submitted by the Original Poster or not.

### Imbalanced classes <a name = 'imb'>

In the small data partition for June 2021, 34,585 out of 226,085 training samples (around 15%) have <code> abel is_submitter == 1</code>. Out of 7,141,851 training records of the full data set, 1,014,861 are <code>is_submitter == 1</code>, or about 14%.  

I did not look at the test samples in either the small nor big samples to prevent personal biases and data leakage. However, both these ratios tell me that I have an **imbalanced classifier** to model.

### Evaluation metrics <a name = 'metrics'>


Only 14-15% of comments in my reddit dataset were written by the original poster of the thread, I needed an evaluation metric that is NOT accuracy of the overall classifier. This is because a model can classifier that ALL of all the comments in my data were not written by the original poster (<code>is_submitter==0</code>) and be accurate 85% of the time. 

To evaluate a classifier's performance in an imbalanced dataset, I should focus on:

* Precision (TP = TP + FP or what % of all the comments the classifier said were written by the OP were actually written by the OP)

* Recall (TP = TP + FN or what % of all the comments written by OPs did the classifier get correctly)

Which of these 2 prioritize depends on the ideal outcome - or what I want the classifier to accomplish. 

*  For example, if my use case is to catch (and ban) submitters who use multiple accounts to comment on the same thread, then I want to prioritize recall.

*  However, I don't want to completely abandon precision either, because an auto-ban-classifier with low precision will cause legitimate, single-account users undue annoyance of wrongly getting banned, which is not good.

* This is why he F1-score, or the harmonic mean of precision & recall is another popular evaluation metric for imbalanced classes.

What if I want to take both precision and recall into consideration, but prioritize recall a little more than precision so that I can catch more multi-account commenters while minimizing false bans on single-account users? I can calculate the F-2 score by setting the <code>beta</code> parameter of SparkML's <code>MulticlassClassificationEvaluator</code> equal to 2, which tells the classifier to prioritize recall over precision but not ignore it altogether.

## Methods: <a name = 'methods'>

### Part 1: Exploratory data analysis <a name = 'eda'>

##### Step 1: Making sense of features from the reddit comments dataset.

* I am a frequent reddit user, therefore I had some familiarity with how a reddit comment threads are structured, what the upvote/ downvote system is, what comment collapsing means, how flairs generally work, etc.

##### Step 2: Gaining *domain expertise* on /r/pcmasterrace

* However, prior to this project, I was not a follower of the /r/pcmasterrace subreddit, I did not have the *domain knowledge* to understand the above features of this particular subreddit for the following reasons:

    * Different subreddits have their own sets of rules enforced by their own moderators. For example, in some subreddits, only subscribers are allowed to comment, upvote or downvote certain posts. In other subreddits, new accounts or accounts with fewer than a pre-definded threshold of karma score cannot create post of comments. Thus, I cannot apply my previous "domain knowledge" as a user of other subreddits to /r/pcmasterrace without learning more about it. 

    * Thus, aside from looking at just the columns from the dataset in Spark, I browsed the subreddit to get a sense for contents posted on it and the type of interaction between users.

##### Step 3: Exploratory data analysis: SparkSQL + pySpark + pandas + seaborn
    
Because I was not familiar with the subreddit itself, I did not want to come up with a hypothesis to test or a what kind of supervised model to build until I saw what the data looks like. I spent a lot of time working exploring the June 2021 parquet of the data, which has about 300k rows.

SparkSQL was extremely helpful for summary statistics, count and aggregates of the June dataset. For example, if I wanted to see how many posts (<code>id</code>) a unique user (<code>author</code>) have made in a thread (<code>link_id</code>), I would run the follow command:
```
spark.sql(
    '''
    SELECT DISTINCT link_id, author, COUNT(id) as post_count
    FROM df
    GROUP BY link_id, author
    ORDER BY post_count desc
    LIMIT 10;
    '''
).show(truncate=False)
```
SparkSQL was quite fast and sufficient if I wanted to understand how certain rows and columns could potentitally be grouped together or to get an idea of what's inside a long text string.

Although SQL aggegate tables are really helpful, but as a visual learner, I also needed to "see" the data. For quick visualization, my pipeline is:
* Use pySpark <code>withColumn()</code> method to quickly create a new variable in my Spark DataFrame
* Use <code>createOrReplaceTempView</code> method to create a temp table that has this new variable
* Use sparkSQL to <code>SELECT</code> this new variable, along with other variables and aggregates I want to visualize with
* Use <code>toPandas()</code> method to convert the aggregated sparkSQL table (now much smaller than the original) into a much smaller pandas dataframe.

* As I have more experience with pandas, matplotlib and seaborn than I do SparkSQL and pySpark, I could make quick data visualizations.

* I chose to use pySpark <code>withColumn()</code> to create new features, then selecting them from SparkSQL and converting the table back into pandas for visualizations instead of writing queries inside SparkSQL directly for **reproduction** purposes, because if the feature looks interesting, it could be part of my Data Cleaning & Feature Engineering pipelines right away.


### Part 2: Data Cleaning  <a name='clean'>

As explained above I did some data cleaning in parallel as I was exploring the data. I  decided early on in the process that there were complicated variables that I had to sacrifice, such as those related to <code>awards</code> and <code>author_flair</code>. I also dropped redundant columns such as <code>subreddit_id</code> (because my dataset only consisted of comments from /r/pcmasterrace)

I did some standard cleaning steps such as converting unix time in seconds into UTC timestamps, date, and time. 

### Part 3: Feature Engineering <a name ='fe'>

I also calculated new features, such as the length of a user's flair, the length of a comment, account age by the time a comment was posted, the total number of posts a unique user made in a certain thread.

## Modeling

### Part 4: Building toy model on small data partition <a name = 'toy_model' >

* Cross-validation: I set both small_ML and big_ML to do 5 folds of cross validation 

* F-2 score: by setting the <code>beta</code> parameter of SparkML's <code>MulticlassClassificationEvaluator</code> equal to 2, I could use the F2-score as the evaluator's metric in the cross validation process. 


## Part 5: Big data & SparkML <a name = 'big_ml'>

## Part 6: Model evaluation <a name = 'eval'>

# Conclusion <a name = 'conclusion'>

## Lessons learned <a name = 'lessons'>

### Basic ML: Goals, metrics, domain expertise <a name = 'lessons_basic'>

* Knowing that I needed to an imbalanced classifier was child's play

* Actually building it was hard

### Big data ML: Patience, cloud computing, pipeline <a name ='lessons_big'>
* Cloud computing is time consuming - **plan accordingly**

* Very difficult to do adhoc adjustments and calculations with a big dataset even with cloud computing - **plan accordingly**

* Extremely important to have a working script on a smaller dataset before trying anything "big data" **plan accordingly**

* Better yet, have a working pipeline **plan accordingly** (more on this later)

## Things I would do differently <a name = 'diff'>

Looking back, here are things I would have done differently:

* Picked a subreddit that I am more familiar with, so my "domain expertise" was not lost.

    * With a subreddit that I had more background knowledge of, I could have spent more time on feature engineering.

    * For example: in /r/cfb (College Football subreddit), users can have up to 2 flairs, usually 2 different college football teams. Because of historical rilvaries in the sport, user interaction based on school flairs are quite interesting. 

* Created a better pipeline. Because this was my first time working with pySpark, all my data cleaning and modeling were done in separate notebooks. I had to copy & paste cells around when I worked on different tasks on the different datasets, which was not great for version controlling, admittedly.

    * If I had more time, I would have writen 3 .py scripts, each with custom, callable functions:
            1) clean.py
            2) EDA.py
            3) ML.py
   
    * Then, I would read these functions into 2  separate Jupyter Notebooks: small.ipynb to run & display results using the smaller partition of the data & refine the 3 py scripts; big.ipynb for one final run.

* Done additional data pre-processing, such as scaling and/ or normalizing the continuous variables. This was not necessarily for the particular model I chose because Gradient Boosted Trees are tree-based and therefor does not require feature scaling, but if I wanted to trained more than 1 type of classifier, I should keep this in the back of my head.

* Trained more than 1 classifers and done more hyper-parameter tuning. 

* Tried different under/ oversampling methods to see if I could improve the classifier. Gradient Boosted Tree performed exceptionally in my unbalanced data set, but I would have liked to see how others did as well.  







