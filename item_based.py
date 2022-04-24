# import libraries
# ---------------------------

from audioop import reverse
from sklearn.neighbors import NearestNeighbors
from locale import normalize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

# import dataset
# ---------------------------
ratings = pd.read_csv('ratings.csv')
# print(ratings.head())

# hyper parameters
# ---------------------------

# min number of reviews
min_reviews = 5

# number of similar students
k = 50

# similarity threshold
threshold = 0.4

# number of suggested lectures
N = 15

# data preprocessing
# ---------------------------

# remove ETH data
ratings = ratings.drop(ratings[ratings['university'] == 'ETH'].index)

# count ratings given by each student
num_ratings = ratings.groupby('student_id')[
    'rating'].count().sort_values(ascending=False)

# remove students with too many ratings or too little
ratings = ratings.drop(ratings[ratings['student_id'].isin(
    num_ratings[(num_ratings > 70) | (num_ratings < min_reviews)].index)].index)

# drop duplicates ratings
ratings = ratings.drop_duplicates(
    subset=['student_id', 'lecture_id'], keep='last')


# options
# ---------------------------

# selected student to analyze
student_id = ratings.iloc[1921]['student_id']

# print(ratings[ratings['student_id'] == student_id]['lecture_name'])

# similarity computation
# ---------------------------

# remove bias
ratings_means = ratings.groupby(by='student_id')['rating'].mean()
ratings = pd.merge(ratings, ratings_means,
                   on='student_id', suffixes=('', '_mean'))
ratings['rating_normalized'] = ratings['rating'] - ratings['rating_mean']

# rating matrix
rating_matrix = ratings.pivot(
    index='lecture_id', columns='student_id', values='rating').fillna(0).dropna()

# similarity matrix of lectures
similarity_matrix = pd.DataFrame(cosine_similarity(
    rating_matrix), index=rating_matrix.index, columns=rating_matrix.index)

# rating prediction
# ---------------------------


# lectures that the selected student already did
# lectures_done = ratings[ratings['student_id']
#                         == student_id]['lecture_id'].to_list()

# # lecture that the student can still do
# candidate_lectures = possible_lectures.drop(
#     possible_lectures[possible_lectures.isin(lectures_done)].index)

# print('Candidate lectures\n', candidate_lectures)

# top N recommendation
# ---------------------------


def predict(lecture_id):

    # get similar lectures
    similar_lectures = similarity_matrix[lecture_id]

    # remove selected lecture
    similar_lectures = similar_lectures.drop(lecture_id)

    print(similar_lectures)
    # for neighbor in neighbors:
    #     print(neighbor)

    # rating_mean = ratings_means.loc[student_id]

    r_hat = rating_mean + np.dot(r_normalized['rating_normalized'], neighbors.loc[r_normalized['student_id']]
                                 ) / np.sum(np.abs(neighbors.loc[r_normalized['student_id']]))

    # return r_hat[0]


# get the stuff the student did
lectures_done = ratings[ratings['student_id']
                        == student_id][['lecture_id', 'rating']][:5]
print(lectures_done)

lecture_ids = ratings['lecture_id'].dropna().unique()

predictions = []
for lecture_id in lecture_ids[:1000]:
    # print(lecture_id)
    similar_lectures = similarity_matrix.loc[lecture_id,
                                             lectures_done['lecture_id']]

    r_hat = np.dot(similar_lectures,
                   lectures_done['rating']) / np.sum(np.abs(similar_lectures))

    # print(similar_lectures)
    # print(r_hat)
    predictions.append([lecture_id, r_hat])

# predictions = pd.DataFrame(predictions, columns=[
#     'lecture_id', 'rating_predicted']).sort_values(by='rating_predicted', ascending=False).dropna()

# topN = predictions[:N]

# print('Suggested lectures')
# print(ratings[ratings['lecture_id'].isin(
#     topN['lecture_id'])].merge(topN, on='lecture_id')[['lecture_name', 'rating_predicted']].drop_duplicates().sort_values(by='rating_predicted', ascending=False).reset_index(drop=True))

# # lectures of a student
# lectures = ratings[ratings['student_id'] == student_id]

# test = []
# for lecture_id in lectures['lecture_id']:
#     test.append([lecture_id, predict(student_id, lecture_id)])

# test = pd.DataFrame(test, columns=[
#     'lecture_id', 'rating_predicted']).fillna(0)

# print(pd.concat([lectures[['lecture_name', 'rating']].reset_index(drop=True),
#       test['rating_predicted'].reset_index(drop=True)], axis=1))

# print('RMSE', mean_squared_error(
#     lectures['rating'], test['rating_predicted'], squared=False))
