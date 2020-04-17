import re
import findspark
findspark.init()
from pyspark import SparkConf
from pyspark import SparkContext as sc
from numpy.core.defchararray import lower
from pyspark.ml.feature import Word2Vec
from pyspark.sql import SQLContext
from pyspark.ml.feature import CountVectorizer, IDF
import numpy as np
import seaborn as sns
import os
import pandas as pd
import ast
import matplotlib as mat
from  matplotlib import pyplot as plt
from wordcloud import WordCloud
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import *
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import Tokenizer, StopWordsRemover,StringIndexer
import tweepy as tw
from tweepy.auth import OAuthHandler
from cassandra.cluster import Cluster
from cassandra import ConsistencyLevel
#import pyspark_cassandra

cluster=Cluster()
session=cluster.connect('Movies')

def sentiment_model():
    print("Predicting Sentiments")
    imdb_df_pd  = session.execute('SELECT *  FROM movie_sent')

    #imdb_df_pd = pd.read_csv("IMDB_Dataset.csv")

    for col in imdb_df_pd.columns:
        if imdb_df_pd[col].dtypes == 'object':
            imdb_df_pd[col] = imdb_df_pd[col].astype('str')

        #imdb_df_pd.head(10)
    imdb_df = sqlContext.createDataFrame(imdb_df_pd)

    # Print the schema in a tree format
    #imdb_df.printSchema()
   #Categorize  sentiment to 0 or 1
    indexer = StringIndexer(inputCol="sentiment", outputCol="score")
    imdb_df = indexer.fit(imdb_df).transform(imdb_df)
    imdb_df = imdb_df.drop("sentiment")
    #imdb_df.show()

    imdb_df = imdb_df.select(regexp_replace('review', '[!?.;:#-/<>]+', ' ').alias('review'), 'score')
    imdb_df = imdb_df.select(regexp_replace('review', '\"', ' ').alias('review'), 'score')
    imdb_df = imdb_df.select(regexp_replace('review', ',', ' ').alias('review'), 'score')
    imdb_df = imdb_df.select(regexp_replace('review', '(\'s\s+)', ' ').alias('review'), 'score')
    imdb_df = imdb_df.select(regexp_replace('review', '(\'\s+)', ' ').alias('review'), 'score')
    #imdb_df.show()

    # remove all single characters
    imdb_df = imdb_df.select(regexp_replace('review', '\s+[a-zA-Z]\s+', ' ').alias('review'), 'score')


    # remove single characters from the start
    imdb_df = imdb_df.select(regexp_replace('review', '^[a-zA-Z]\s+', '').alias('review'), 'score')


    # remove single digit
    imdb_df = imdb_df.select(regexp_replace('review', '[0-9]+', ' ').alias('review'), 'score')

    #Substituting multiple spaces with single space
    imdb_df = imdb_df.select(regexp_replace('review', '\s+', ' ').alias('review'), 'score')

    # Converting to Lowercase
    imdb_df = imdb_df.select(lower(imdb_df.review).alias('review'), 'score')

    #imdb_df.show()


    # Tokenize text

    tokenizer = Tokenizer(inputCol='review', outputCol='words_token')

    df = tokenizer.transform(imdb_df)
    # Remove stop words
    remover = StopWordsRemover(inputCol='words_token', outputCol='words_clean', caseSensitive=False)

    # df = remover.transform(df)

    # df.show(10)
    cv = CountVectorizer(inputCol="words_clean", outputCol="tf", vocabSize=2 ** 17, minDF=5.0)

    # we now create a pipelined transformer
    cv_pipeline = Pipeline(stages=[tokenizer, remover, cv]).fit(imdb_df)

    #cv_pipeline.transform(imdb_df).show(5)

    idf = IDF(inputCol="tf", outputCol="idf")

    idf_pipeline = Pipeline(stages=[cv_pipeline, idf]).fit(imdb_df)
    #idf_pipeline.transform(imdb_df).show(5)

    training_df, validation_df, testing_df = imdb_df.randomSplit([0.6, 0.3, 0.1], seed=0)

    #print(training_df.count(), validation_df.count(), testing_df.count())

    lr = LogisticRegression(maxIter=50, regParam=0.0, elasticNetParam=0.0, featuresCol="idf", labelCol="score")
    lr_pipeline = Pipeline(stages=[idf_pipeline, lr]).fit(training_df)
    print("Prediction Accuracy before Tuning")
    lr_pipeline.transform(validation_df). \
        select(expr('float(prediction = score)').alias('correct')). \
        select(avg('correct').alias('Accuracy')).show()

    # identify noise in the model
    vocabulary = cv_pipeline.stages[2].vocabulary
    # vocabulary = idf_pipeline.stages[0].stages[2].vocabulary
    #print(vocabulary)
    weights = lr_pipeline.stages[1].coefficients.toArray()
    #print(weights)
    coeffs_df = pd.DataFrame({'word': vocabulary, 'weight': weights})
    #print(coeffs_df.sort_values('weight').head(5))
    #print(coeffs_df.sort_values('weight', ascending=False).head(5))
    # Fit the model

    # data is overfitted
    # modify the loss function and penalize weight values that are too large.
    # use either L! (Lasso) or Ridge(L2)
    from pyspark.ml.tuning import ParamGridBuilder

    #evlue the mode and find best fit model. done seperately

    # best parameters are regParam = 0.01 and  name='elasticNetParam' = 0.2

    lr = LogisticRegression(maxIter=50, regParam=0.01, elasticNetParam=0.2, featuresCol="idf", labelCol="score")
    lr_pipeline_fitted = Pipeline(stages=[idf_pipeline, lr]).fit(training_df)

    print("Prediction Accuracy - After Tuning")
    lr_pipeline_fitted.transform(validation_df). \
        select(expr('float(prediction = score)').alias('correct')). \
        select(avg('correct').alias('accuracy')).show()

    # identify noise in the model

    vocabulary = cv_pipeline.stages[2].vocabulary
    #print(vocabulary)
    weights = lr_pipeline_fitted.stages[1].coefficients.toArray()
    #print(weights)
    coeffs_df = pd.DataFrame({'word': vocabulary, 'weight': weights})
    #print(coeffs_df.sort_values('weight').head(5))
    #print(coeffs_df.sort_values('weight', ascending=False).head(5))
    return lr_pipeline_fitted
    print("end of sentiment model")

def getMovieData():
    pandas_df = session.execute('SELECT *  FROM movie_data')
    #pandas_df = pd.read_csv("movies_data.csv")
    for col in pandas_df.columns:
        if pandas_df[col].dtypes.name == 'object':
            pandas_df[col] = pandas_df[col].astype('str')
    return pandas_df

def movie_dataset():
    print("getting  data")
    pandas_df = pd.read_csv("movies_metadata.csv")

    for col in pandas_df.columns:
        if pandas_df[col].dtypes == 'object':
            pandas_df[col] = pandas_df[col].astype('str')
    pandas_df['budget'] = pd.to_numeric(pandas_df['budget'], errors='coerce')
    pandas_df['budget'] = pandas_df['budget'].replace(0, np.nan)
    pandas_df[pandas_df['budget'].isnull()].shape
    credits_pd = pd.read_csv('credits.csv')
    keywords_pd = pd.read_csv('keywords.csv')
    for col in credits_pd.columns:
        if credits_pd[col].dtypes == 'object':
            credits_pd[col] = credits_pd[col].astype('str')
    for col in keywords_pd.columns:
        if keywords_pd[col].dtypes == 'object':
            keywords_pd[col] = keywords_pd[col].astype('str')
    pandas_df = pandas_df.merge(credits_pd, on='id')
    pandas_df  = pandas_df.merge(keywords_pd, on='id')

    def clean_numeric(x):
        try:
            return float(x)
        except:
            return np.nan

    pandas_df['popularity'] = pandas_df['popularity'].apply(clean_numeric).astype('float')
    pandas_df['vote_count'] = pandas_df['vote_count'].apply(clean_numeric).astype('float')
    pandas_df['vote_average'] = pandas_df['vote_average'].apply(clean_numeric).astype('float')
    pandas_df['year'] = pd.to_datetime(pandas_df['release_date'], errors='coerce'). \
        apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
    # pandas_df['year']  = pandas_df['year'].astype('float')
    base_poster_url = 'http://image.tmdb.org/t/p/w185/'
    pandas_df['poster_path'] = "<img src='" + base_poster_url + pandas_df['poster_path'] + "' style='height:100px;'>"

    pandas_df['production_countries'] = pandas_df['production_countries'].fillna('[]').apply(ast.literal_eval). \
        apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    pandas_df['production_companies'] = pandas_df['production_companies'].fillna('[]').apply(ast.literal_eval). \
        apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

    pandas_df['spoken_languages'] = pandas_df['spoken_languages'].fillna('[]').apply(ast.literal_eval). \
        apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

    pandas_df['genres'] = pandas_df['genres'].fillna('[]').apply(ast.literal_eval). \
        apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

    pandas_df['belongs_to_collection'] = pandas_df['belongs_to_collection'].replace('nan', '[]'). \
        apply(ast.literal_eval). \
        apply(lambda x: x['name'] if isinstance(x, dict) else np.nan)
    pandas_df['keywords'] = pandas_df['keywords'].fillna('[]').apply(ast.literal_eval). \
        apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

    def get_director(x):
        for i in x:
            if i['job'] == 'Director':
                return i['name']
        return np.nan
    pandas_df['cast'] = pandas_df['cast'].fillna('[]').apply(ast.literal_eval). \
        apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    pandas_df['cast'] = pandas_df['cast'].apply(lambda x: x[:3] if len(x) >= 3 else x)
    pandas_df['cast'] = pandas_df['cast'].apply(lambda x: [str.lower(i.replace("  ", " ")) for i in x])
    pandas_df['crew'] = pandas_df['crew'].fillna('[]').apply(ast.literal_eval)
    pandas_df['director'] = pandas_df['crew'].apply(get_director)
    pandas_df['director'] = pandas_df['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", " ")))
    pandas_df['keywords'] = pandas_df['keywords'].apply(lambda x: [str.lower(i.replace("  ", " ")) for i in x])


    for col in pandas_df.columns:
        if pandas_df[col].dtypes.name == 'object':
            pandas_df[col] = pandas_df[col].astype('str')

    #pandas_df.info()
    print("getting  data ended")
    pandas_df.to_csv("movies_data.csv")
    return pandas_df

def movie_wordcloud(df):
    title_df = df.select("id", "title")
    # Clean text
    df_clean = title_df.select("id", lower(regexp_replace('title', "[^a-zA-Z\\s]", "")).alias('title'))

    # Tokenize text
    tokenizer = Tokenizer(inputCol='title', outputCol='words_token')
    df_words_token = tokenizer.transform(df_clean).select('id', 'words_token')

    # Remove stop words
    remover = StopWordsRemover(inputCol='words_token', outputCol='words_clean')
    df_words_no_stopw = remover.transform(df_words_token).select('id', 'words_clean')

    #df_words_no_stopw.show(10)

    wordsDF = df_words_no_stopw.select(explode("words_clean").alias("words"))

    wordsDF = wordsDF.select(trim(wordsDF.words).alias("words"))
    #wordsDF.show()

    wordCountDF = wordsDF.groupBy("words").count().orderBy(desc("count")).limit(16)
    #wordCountDF.show()
    pandD = wordCountDF.toPandas()
    pandD.drop(0, inplace=True)

    sns.barplot(y='words', x='count', data=pandD)
    plt.title("Movie Title  Analysis")
    plt.xlabel('Words Frequency')
    plt.ylabel('Words')
    #plt.show()

    wordCountDF = wordsDF.groupBy("words").count().orderBy(desc("count")).limit(101)
    pandD = wordCountDF.toPandas()
    pandD.drop(0, inplace=True)  # drop first row

    wordcloudConvertDF = pandD.set_index('words').T.to_dict('records')
    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=100, relative_scaling=0.5,
                          colormap='Dark2') \
        .generate_from_frequencies(dict(*wordcloudConvertDF))
    plt.figure(figsize=(14, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title("Words Cloud - Movie Titles")
    plt.axis('off')
    plt.show()

    """# Overview Cloud

    overview_df = df.select("id", "overview")
    # Clean text
    df_clean = df.select("id", lower(regexp_replace('overview', "[^a-zA-Z\\s]", "")).alias('overview'))

    # Tokenize text
    tokenizer = Tokenizer(inputCol='overview', outputCol='words_token')
    df_words_token = tokenizer.transform(df_clean).select('id', 'words_token')

    # Remove stop words
    remover = StopWordsRemover(inputCol='words_token', outputCol='words_clean')
    df_words_no_stopw = remover.transform(df_words_token).select('id', 'words_clean')

    df_words_no_stopw.show(10)

    wordsDF = df_words_no_stopw.select(explode("words_clean").alias("words"))

    wordsDF = wordsDF.select(trim(wordsDF.words).alias("words"))
    wordsDF.show()

    wordCountDF = wordsDF.groupBy("words").count().orderBy(desc("count")).limit(16)
    wordCountDF.show()
    pandD = wordCountDF.toPandas()
    pandD.drop(0, inplace=True)

    sns.barplot(y='words', x='count', data=pandD)
    plt.title("Movie Overview  Analysis")
    plt.xlabel('Words Frequency')
    plt.ylabel('Words')
    #plt.show()

    wordCountDF = wordsDF.groupBy("words").count().orderBy(desc("count")).limit(101)
    pandD = wordCountDF.toPandas()
    pandD.drop(0, inplace=True)

    wordcloudConvertDF = pandD.set_index('words').T.to_dict('records')
    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=100, relative_scaling=0.5,
                          colormap='Dark2') \
        .generate_from_frequencies(dict(*wordcloudConvertDF))
    plt.figure(figsize=(14, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title("Words Cloud  - Movie Overview")
    plt.axis('off')
    plt.show()"""

def getStory(df):
    # Graph1
    df['year'] = pd.to_datetime(df['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
    year_count = df.groupby('year')['title'].count()
    #plt.figure(figsize=(18, 5))
    year_count.plot()
    plt.title('Number of movies releases in a particular year')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.subplots_adjust(left=0.2, right=0.7, top=0.9, bottom=0.3)
    #plt.show()

    df[df['year'] != 'NaT'][['title', 'year']].sort_values('year').head(10)
    # Graph2
    df['runtime'] = df['runtime'].astype('float')
    #plt.figure(figsize=(12, 6))
    sns.distplot(df[(df['runtime'] < 300) & (df['runtime'] > 0)]['runtime'])
    plt.title('Overall Runtime of the Movies')
    plt.xlabel('Runtime')
    plt.ylabel('Density')
    plt.subplots_adjust(left=0.2, right=0.7, top=0.9, bottom=0.3)
    #plt.show()

    # # Graph3  - giving problem
    def clean_numeric(x):
        try:
             return float(x)
        except:
             return np.nan

    df['year'] = df['year'].replace('NaT', np.nan)
    df['year'] = df['year'].apply(clean_numeric)
    df['genres'] = df['genres'].fillna('[]').apply(ast.literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    s = df.apply(lambda x: pd.Series(x['genres']), axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'genre'
    gen_df = df.drop('genres', axis=1).join(s)
    pop_gen = pd.DataFrame(gen_df['genre'].value_counts()).reset_index()
    pop_gen.columns = ['genre', 'movies']
    print(pop_gen.head(10))
    genres = ['Drama', 'Comedy', 'Thriller', 'Romance', 'Action', 'Horror', 'Crime', 'Adventure', 'Science Fiction','Mystery', 'Fantasy', 'Mystery', 'Animation']
    pop_gen_movies = gen_df[(gen_df['genre'].isin(genres)) & (gen_df['year'] >= 2000) & (gen_df['year'] <= 2016)]
    ctab = pd.crosstab([pop_gen_movies['year']], pop_gen_movies['genre']).apply(lambda x: x / x.sum(), axis=1)
    ctab[genres].plot(kind='line', stacked=False, colormap='jet', figsize=(12, 8)).legend(loc='center left',bbox_to_anchor=(1, 0.5))
    plt.title('Trend of Genre in Particular Year')
    plt.xlabel('Year')
    plt.ylabel('Density')
    plt.subplots_adjust(left=0.2, right=0.7, top=0.9, bottom=0.3)
    #plt.show()

    # Graph4
    df['return'] = df['revenue'] / df['budget']
    df[df['return'].isnull()].shape
    df_mat = df[(df['return'].notnull()) & (df['runtime'] > 0) & (df['return'] < 10)]
    sns.jointplot('return', 'runtime', data=df_mat)
    plt.title('Relativity among Runtime and Return Gained by the Movie')
    plt.subplots_adjust(left=0.2, right=0.7, top=0.9, bottom=0.3)
    #plt.show()

    # Graph5
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    def get_month(x):
        try:
            return month_order[int(str(x).split('-')[1]) - 1]
        except:
            return np.nan

    df['month'] = df['release_date'].apply(get_month)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
    sns.boxplot(x='month', y='return', data=df[df['return'].notnull()], palette="muted", ax=ax, order=month_order)
    ax.set_ylim([0, 12])
    plt.title('Monthly Return gained by Movies')
    plt.xlabel('Month')
    plt.ylabel('Returns Gained')
    plt.subplots_adjust(left=0.2, right=0.7, top=0.9, bottom=0.3)
    #plt.show()

    # Graph6
    df['runtime'] = df['runtime'].astype('float')
    #plt.figure(figsize=(18, 5))
    year_runtime = df[df['year'] != 'NaT'].groupby('year')['runtime'].mean()
    plt.plot(year_runtime.index, year_runtime)
    plt.xticks(np.arange(1874, 2024, 10.0))
    plt.title('Overall Runtime of movies in particular Year')
    plt.xlabel('Year')
    plt.ylabel('Runtime')
    plt.subplots_adjust(left=0.2, right=0.7, top=0.9, bottom=0.3)
    #plt.show()


    # Graph7
    #plt.figure(figsize=(18, 8))
    sns.barplot(x='genre', y='movies', data=pop_gen.head(15))
    plt.title("Different Genres with their relative usage")
    plt.xlabel('Genre')
    plt.ylabel('Count of movies')
    plt.xticks(rotation=45)
    plt.subplots_adjust(left=0.2, right=0.7, top=0.9, bottom=0.3)
    plt.show()


def top_25(df):
    # Top 25 chart- General

    # v is the number of votes for the movie
    # m is the minimum votes required to be listed in the chart
    # R is the average rating of the movie
    # C is the mean vote across the whole report
    #print(df.count(), len(df.columns))
    # remove null values for vore counts and vote averages

    movie_df = df.select('title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres'). \
                    filter(df.vote_average.isNotNull() & df.vote_count.isNotNull())
    C_df = movie_df.select(mean(movie_df.vote_average).alias("C")).collect()[0]
    C = C_df[0]
    #print(C)
    # Greenwald-Khanna algorithm: approx quantile uses this
    m_array = movie_df.approxQuantile("vote_count", [0.94], 0.05)
    m = m_array[0]
    #print(m)

    # filter values
    top_df = movie_df.filter(movie_df.vote_count > m)
    #print(top_df.count(), len(top_df.columns))

    # addcolumn for m and C
    top_df = top_df.withColumn("m", top_df.vote_count - top_df.vote_count + m)
    top_df = top_df.withColumn("C", top_df.vote_count - top_df.vote_count + C)

    top_df = top_df.withColumn('Ranking',
                           (top_df.vote_count * top_df.vote_average + top_df.m * top_df.C) / (
                                   top_df.m + top_df.vote_count))
    print("Top 10  movies Chart")

    top_df.orderBy(desc("Ranking")).limit(10).show()

# top chart for genre wise

def build_chart(df, genre, percentile=0.85):

    movie_df = df.select('title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres'). \
        filter(df.vote_average.isNotNull() & df.vote_count.isNotNull())
    gen_df = movie_df.withColumn("genres1", explode(split(
        regexp_extract(
            regexp_replace(df.genres, "\s", ""), "^\[(.*)\]$", 1), ","))) \
        .drop("genres") \
        .withColumnRenamed("genres1", "genres")

    #gen_df.show(10)
    genre = "'" + genre + "'"
    movie_df = gen_df.filter(gen_df.genres == genre)
    #movie_df.show()
    C_df = movie_df.select(mean(movie_df.vote_average).alias("C")).collect()[0]
    C = C_df[0]
    #print(C)
    # Greenwald-Khanna algorithm: approx quantile uses this
    m_array = movie_df.approxQuantile("vote_count", [percentile], 0.05)
    m = m_array[0]
    #print(m)

    # filter values
    top_df = movie_df.filter(movie_df.vote_count > m)
    #(top_df.count(), len(top_df.columns))

    # addcolumn for m and C
    top_df = top_df.withColumn("m", top_df.vote_count - top_df.vote_count + m)
    top_df = top_df.withColumn("C", top_df.vote_count - top_df.vote_count + C)

    top_df = top_df.withColumn('Ranking',
                               (top_df.vote_count * top_df.vote_average + top_df.m * top_df.C) / (
                                           top_df.m + top_df.vote_count))

    top_df = top_df.orderBy(desc("Ranking"))

    return top_df

def als_model(userid, df):

    als_df_pd = session.execute('SELECT *  FROM movie_rating')
    #als_df_pd = pd.read_csv("ratings_small.csv")
    movie_list_df = df.select('id', 'title')
    movie_list_df = movie_list_df.withColumn('userId', lit(userid))

    for col in als_df_pd.columns:
        if als_df_pd[col].dtypes == 'object':
            als_df_pd[col] = als_df_pd[col].astype('str')
    ratings = sqlContext.createDataFrame(als_df_pd)

    #ratings.printSchema()
    #ratings.show()
    #print((ratings.count(), len(ratings.columns)))

    mv_notwatched_df = ratings.filter(ratings.userId == userid)\
        .select('movieId')\
        .join(movie_list_df, ratings.movieId == movie_list_df.id, 'right_outer')\
        .drop("movieId")\
        .withColumnRenamed("id", "movieId")

    #ratings.groupBy("userID").count().show()

    usercount = ratings.agg(countDistinct(ratings.userId).alias("Users_Count")).head()[0]

    #print('The number of distinct values of  Users is: ', str(usercount))

    (training, test) = ratings.randomSplit([0.8, 0.2])

    # # Build the recommendation model using ALS on the training data
    # # Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics

    als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating",
              coldStartStrategy="drop")
    model = als.fit(training)

    # # Evaluate the model by computing the RMSE on the test data
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print("ALS- Model Root-mean-square error before Tuning= " + str(rmse))
    # # Generate top 10 movie recommendations for each user

    print("Top 10 movies recommended for each user")
    userRecs = model.recommendForAllUsers(10)
    userRecs.show(10)

    # Generate top 10 user recommendations for each movie
    print("Top 10 movies recommended for each movie")
    movieRecs = model.recommendForAllItems(10)
    movieRecs.show(10)

    # Tune the model

    pipeline = Pipeline(stages=[als])

    paramGrid = ParamGridBuilder() \
        .addGrid(als.regParam, [0.1, 0.01]) \
        .build()

    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator,
                              numFolds=2)  # use 3+ folds in practice

    # Run cross-validation, and choose the best set of parameters.
    cvModel_fitted = crossval.fit(training)

    #print("best model")
    bestModel = cvModel_fitted.bestModel


    print("ALS model - Root-mean-square error after Tuning= " + str(rmse))
    predictions = cvModel_fitted.transform(test)

    print("Best  Prediction Model")
    predictions.show(10)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)

    print("Root-mean-square error after cross validation = " + str(rmse))



    param_dict = cvModel_fitted.bestModel.stages[0].extractParamMap()

    print("List of Movies not watched by the User")
    mv_notwatched_df.show(10)

    print("Top 10 movies Recommended")
    top_10  = cvModel_fitted.transform(mv_notwatched_df).orderBy(desc('prediction')).limit(10)
    top_10.show(10)
    return  top_10
#create a record of 94% movies

def getTweetFullDataset(df):
    print("Getting Twitter data")
    consumer_key = '0oFsUmrYI59vugFr6XyqJ0C5U'
    consumer_secret = 'XGeaakfoudApfwIgN6gX0Tcm83KLWlM98CwcZ8mQv5xwqTgKRY'
    access_token = '2612720178-qj2HtZy96iX6zEoaUFjmpNktpOIJeOXfCGsGLEt'
    access_secret = 'hTy0uckpgI9nzvB9UeSs7uGQ3VNP99p5MYyWztnJYqJJl'
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    api = tw.API(auth)
    # Cassandra Connection Local
    session = cluster.connect('wrd_cass')
    stmt = session.prepare("insert into movie_tweets(id, title, tweetText) VALUES (?,?,?)")
    row_number = 0

    for movie in df['title']:
    # initialize a list to hold all the Tweets
        search_query = movie
        print("search_query :", search_query)

        for tweet in tw.Cursor(api.search, q=search_query + " -filter:retweets", lang='en', result_type='recent').items(10):
            # Remove all the special characters
            text=tweet.text
            #print(text)
            # process tweet
            processed_tweet = ' '.join(re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", str(text)).split())

            #processed_tweet = re.sub(r'\W', ' ', str(tweet.text))

            # remove all single characters
            #processed_tweet = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_tweet)

            # Remove single characters from the start
            #processed_tweet = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_tweet)

            # Substituting multiple spaces with single space
            #processed_tweet = re.sub(r'\s+', ' ', processed_tweet, flags=re.I)

            # Removing prefixed 'b'
            #processed_tweet = re.sub(r'^b\s+', '', processed_tweet)

            # Converting to Lowercase
            # processed_tweet = processed_tweet.lower()
            print(processed_tweet)
            # insert tweets for movie
            qry = stmt.bind([row_number,  search_query, processed_tweet])
            qry.consistency_level = ConsistencyLevel.LOCAL_ONE
            #session.execute(qry)
            row_number = row_number + 1

    print("insert successful")

def getSparkTweet( df, title):
    tweets_df = df.filter(df.tweets.contains(title))
    result=1
    return  result,tweets_df

def getTitleTweets(title):
    consumer_key = 'xx'
    consumer_secret = 'xx'
    access_token = 'xx'
    access_secret = 'xx'
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    api = tw.API(auth)
    # Cassandra Connection Local
    #session = cluster.connect('wrd_cass')
    #stmt = session.prepare("insert into movie_tweets(id, title, tweetText) VALUES (?,?,?)")
    #row_number = 0
    search_query = title
    #print("search_query :", search_query)
    tweets = pd.DataFrame(columns=["review"])
    """processed_tweet = 'Test'
    df = pd.DataFrame([[processed_tweet]], columns=['text'])
    print(df.head())
    tweets = tweets.append(df)"""
    result=0
    for tweet in tw.Cursor(api.search, q=search_query + " -filter:retweets", lang='en', result_type='recent').items(50):
        try:
            # Remove all the special characters
            text = tweet.text
            # print(text)
            # process tweet
            processed_tweet = ' '.join(re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", str(text)).split())
            #print(processed_tweet)
            df = pd.DataFrame([[processed_tweet]], columns=['review'])
            tweets= tweets.append(df)
            result=1
        except:
            continue
    # print(tweets.head(10))
    if result==0:
        tweet_df = tweets
    else:
        tweet_df = sqlContext.createDataFrame(tweets)

    return  result, tweet_df

def predict_sentiment_df(top10, ls_sentiment_model):
    top10_sentiment = pd.DataFrame(columns=["Title", "Sentiment", "Count"])
    tweet_pd = session.execute('SELECT *  FROM  tweets')
    for col in tweet_pd.columns:
        if tweet_pd[col].dtypes == 'object':
            tweet_pd[col] = als_df_pd[col].astype('str')
    tweet_df = sqlContext.createDataFrame(tweet_pd)
    for i in top10.collect():
        try:
            #print(i.title)
            #result, tweet_df = getTitleTweets(i.title)
            result, tweet_df = getSparkTweet(tweet_df,i.title)
            if result != 0:
                ls_pred = ls_sentiment_model.transform(tweet_df)
                #ls_pred.show()
                ls_pred_cnt = ls_pred.groupBy(ls_pred.prediction).count()
                #ls_pred_cnt.show()
                neg_cnt= ls_pred_cnt.collect()[0][1]
                pos_cnt = ls_pred_cnt.collect()[1][1]
                df_row = pd.DataFrame([[i.title,"Positive", pos_cnt], [i.title,"Negative", neg_cnt]],
                              columns=["Title", "Sentiment", "Count"])
                top10_sentiment=top10_sentiment.append(df_row)
        except:
            continue

    #print(top10_sentiment.head())
    graph = sns.barplot(x="Count", y="Title", hue="Sentiment",dodge=True, data=top10_sentiment)
    plt.title('Sentiment Analysis')
    plt.xlabel('Count')
    plt.ylabel('Movie Title')
    plt.subplots_adjust(left=0.2, right=0.7, top=0.9, bottom=0.3)
    plt.show()

def word2vec_model():
    word2vec_df = df.select("id", "title","cast","director","genres", "keywords")
    word2vec_df = word2vec_df.filter(word2vec_df.keywords != "[]")
    word2vec_df = word2vec_df.filter(word2vec_df.cast != "[]")
    word2vec_df = word2vec_df.filter(word2vec_df.genres != "[]")


    word2vec_df = word2vec_df.withColumn("cast_str", regexp_extract(word2vec_df.cast, "^\[(.*)\]$", 1))
    word2vec_df = word2vec_df.withColumn("genres_str", regexp_extract(word2vec_df.genres, "^\[(.*)\]$",1))
    word2vec_df = word2vec_df.withColumn("keywords_str", regexp_extract(word2vec_df.keywords, "^\[(.*)\]$", 1))
    word2vec_df = word2vec_df.dropna()

    word2vec_df.printSchema()
    #for i in word2vec_df.collect():
    #        token = i.keywords_str
    #        print(token)

    word2vec_df = word2vec_df.select("id", "title",
                                     concat(word2vec_df.cast_str, lit(","),
                                             word2vec_df.director, lit(","),
                                             word2vec_df.genres_str, lit(","),
                                             word2vec_df.keywords_str).alias('token_string')
                                      )

    # for i in word2vec_df.collect():
    #     token = i.token_string
    #     print(token)

    word2vec_df = word2vec_df.withColumn("token", split(word2vec_df.token_string, ","))

    #word2vec_df = word2vec_df.withColumn("token", split(
    #                                     regexp_extract(df.word2vec_df, "^\[(.*)\]$", 1), ","))

    word2vec_df.printSchema()
    word2vec_df.show()
    #
    # # Learn a mapping from words to Vectors.
    word2Vec = Word2Vec(vectorSize=5, minCount=2, inputCol="token", outputCol="result")
    model = word2Vec.fit(word2vec_df)
    print(model)
    result = model.transform(word2vec_df)
    result.show()
    title_df = word2vec_df.filter (word2vec_df.title == 'Batman Forever').collect()[0]
    print(title_df.token)
    vec = model.transform(title_df.token)
    print (vec)
    synonyms = model.findSynonyms( vec, 5)
    #for word, cosine_distance in synonyms:
    #    print("{}: {}".format(word, cosine_distance))
    #

    #

    # for row in result.collect():
    #     print(row)
    #     #t, vector = row
    #     #print("Text: [%s] => \nVector: %s\n" % (", ".join(text), str(vector)))
    #
    #vec = model.transform("batman")
    #model

if __name__ == "__main__":
    os.chdir("D:\\trentsemester2\\bigData\\the-movies-dataset")
    sparkconf = SparkConf().setAppName("movie").setMaster("local[*]")
    sparkcont = sc(conf=sparkconf)
    sparkcont.setLogLevel("ERROR")
    sqlContext = SQLContext(sparkcont)
    pandas_df = getMovieData()
    df = sqlContext.createDataFrame(pandas_df)
    movie_wordcloud(df)
    #story_df = pd.read_csv("movies_metadata.csv")
    getStory(pandas_df)
    top_25(df)
    print("Top 15 romance movies")
    build_chart(df=df, genre='Romance').limit(15).show()
    #word2vec_model()
    top10 = als_model(8,df)
    ls_sentiment_model = sentiment_model()
    predict_sentiment_df(top10, ls_sentiment_model)
    print("End of Project")




