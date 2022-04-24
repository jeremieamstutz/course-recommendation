# import libraries
# ---------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk

def mean_absolute_error(targets, predictions):
    return np.abs(predictions - targets).mean()

def root_mean_squared_error(targets, predictions):
    return np.sqrt(((predictions - targets) ** 2).mean())

def hit_rate():
    return

def average_reciprocal_hit_rate():
    return

def cumulative_hit_rate():
    return

def rating_hit_rate():
    return

def coverage():
    return

def diversity():
    return

def novelty():
    return

def churn():
    return

def responsiveness():
    return
