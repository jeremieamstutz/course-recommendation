# import libraries
# ---------------------------

from locale import normalize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity


# parameters
# ---------------------------

# max number of similar students
k = 10


# options
# ---------------------------

# selected student to analyze
student_id = 3


# import dataset
# ---------------------------
ratings = pd.read_csv('ratings.csv')
print(ratings.head())


# data preprocessing
# ---------------------------

# remove ETH data
ratings = ratings.drop(ratings[ratings['university'] == 'ETH'].index)

# count ratings given by each student
num_ratings = ratings.groupby('student_id')[
    'rating'].count().sort_values(ascending=False)

# remove students with too many ratings or too little
ratings = ratings.drop(ratings[ratings['student_id'].isin(
    num_ratings[(num_ratings > 70) | (num_ratings < 4)].index)].index)

# drop duplicates ratings
ratings = ratings.drop_duplicates(
    subset=['student_id', 'lecture_id'], keep='last')


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

# similarity matrix of students
similarity_matrix = pd.DataFrame(cosine_similarity(
    rating_matrix), index=rating_matrix.index, columns=rating_matrix.index)


# rating prediction
# ---------------------------

# get similar students
similar_students = similarity_matrix[student_id].nlargest(k+1)
# remove the selected student
similar_students = similar_students.drop(student_id)
print('Similar students', similar_students)

# lectures that other students did
possible_lectures = ratings[ratings['student_id'].isin(
    similar_students.index)]['lecture_id'].drop_duplicates()
# lectures that the selected student already did
lectures_done = ratings[ratings['student_id']
                        == student_id]['lecture_id'].to_list()

candidate_lectures = possible_lectures.drop(
    possible_lectures[possible_lectures.isin(lectures_done)].index)

print('Candidate lectures', candidate_lectures)

# top N recommendation
# ---------------------------


def predict(student_id, lecture_id):

    # user who rated lecture_id


    r_normalized = ratings[(ratings['lecture_id'] == lecture_id) & (
        ratings['student_id'].isin(similar_students.index))]['student_id', 'rating_normalized']

    print(r_normalized)

    r_normalized.join(similar_students, on='')

    r_hat = np.dot(r_normalized, similar_students)

    # r_hat = np.dot(r_normalized, neighbors) / np.sum(np.abs(neighbors))


predict('3', '50942131')

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
