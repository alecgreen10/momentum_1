import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error


TRAIN_TEST_CUTOFF = '06-04-2022'
TRAIN_VALID_RATIO = 0.75



dow_historical_data = pd.read_csv('train_1.csv', index_col=[0])


#############################################################################################

new_df = dow_historical_data[['AAPL','AMGN']].copy()
cols = new_df.columns
new_df["Target"] = (new_df["AAPL"].pct_change(4).shift(-4) > 0).astype(int)
new_df.dropna(inplace=True)
new_df.index = pd.to_datetime(new_df.index)
index = new_df.index[new_df.index < TRAIN_TEST_CUTOFF]
index = index[:int(len(index) * TRAIN_VALID_RATIO)]
scaler = StandardScaler().fit(new_df.loc[index, cols])
new_df[cols] = scaler.transform(new_df[cols])


#############################################################################################


























