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

print(ratings[ratings['student_id'] == student_id]['lecture_name'])

# similarity computation
# ---------------------------

# remove bias
ratings_means = ratings.groupby(by='student_id')['rating'].mean()
ratings = pd.merge(ratings, ratings_means,
                   on='student_id', suffixes=('', '_mean'))
ratings['rating_normalized'] = ratings['rating'] - ratings['rating_mean']

# rating matrix
rating_matrix = ratings.pivot(
    index='student_id', columns='lecture_id', values='rating').fillna(0)

# fit neighbors
knn = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')
knn.fit(rating_matrix.to_numpy())

# store k nearest neighbors
([similarities], [neighbors_index]) = knn.kneighbors(
    [rating_matrix.loc[student_id]])
neighbors = pd.DataFrame(
    {'similarity': 1 - similarities}, index=rating_matrix.iloc[neighbors_index].index)

# remove the selected student if in the original data
neighbors = neighbors.drop(student_id)

print('Neighbors\n', neighbors)

# rating prediction
# ---------------------------

# lectures that other students did
possible_lectures = ratings[ratings['student_id'].isin(
    neighbors.index)]['lecture_id'].drop_duplicates()

# lectures that the selected student already did
lectures_done = ratings[ratings['student_id']
                        == student_id]['lecture_id'].to_list()

# lecture that the student can still do
candidate_lectures = possible_lectures.drop(
    possible_lectures[possible_lectures.isin(lectures_done)].index)

# print('Candidate lectures\n', candidate_lectures)

# top N recommendation
# ---------------------------


def predict(student_id, lecture_id):

    r_normalized = ratings[(ratings['lecture_id'] == lecture_id) & (
        ratings['student_id'].isin(neighbors.index))]

    # print(r_normalized[['student_id', 'rating_normalized']])

    # print(neighbors.loc[r_normalized['student_id']])
    rating_mean = ratings_means.loc[student_id]

    r_hat = rating_mean + np.dot(r_normalized['rating_normalized'], neighbors.loc[r_normalized['student_id']]
                                 ) / np.sum(np.abs(neighbors.loc[r_normalized['student_id']]))

    return r_hat[0]


predictions = []
for lecture_id in candidate_lectures:
    predictions.append([lecture_id, predict(student_id, lecture_id)])

predictions = pd.DataFrame(predictions, columns=[
    'lecture_id', 'rating_predicted']).sort_values(by='rating_predicted', ascending=False)


topN = predictions[:N]

print('Suggested lectures')
print(ratings[ratings['lecture_id'].isin(
    topN['lecture_id'])].merge(topN, on='lecture_id')[['lecture_name', 'rating_predicted']].drop_duplicates().sort_values(by='rating_predicted', ascending=False).reset_index(drop=True))

# lectures of a student
lectures = ratings[ratings['student_id'] == student_id]

test = []
for lecture_id in lectures['lecture_id']:
    test.append([lecture_id, predict(student_id, lecture_id)])

test = pd.DataFrame(test, columns=[
    'lecture_id', 'rating_predicted']).fillna(0)

print(pd.concat([lectures[['lecture_name', 'rating']].reset_index(drop=True),
      test['rating_predicted'].reset_index(drop=True)], axis=1))

print('RMSE', mean_squared_error(
    lectures['rating'], test['rating_predicted'], squared=False))


# print(predictions)
# predict('3', '50942157')

# # extract unique user ids
# student_ids = ratings['student_id'].unique()
# num_users = len(student_ids)
# print(num_users)

# # extract unique lecture ids
# lecture_ids = ratings['lecture_id'].unique()
# num_lectures = len(lecture_ids)
# print(num_lectures)

# # compute user similarity matrix
# similarities = np.zeros()

# train_ratings, test_ratings = train_test_split(ratings)


# cosine_similarity(x_train, y_train)

# # evaluate results
# # ---------------------------
# # mae
# mean_absolute_error(y_true, y_pred)

# # rmse
# mean_squared_error(y_true, y_pred, squared=False)
