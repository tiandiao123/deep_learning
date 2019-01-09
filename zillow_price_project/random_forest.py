import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

color = sns.color_palette()

train_data = pd.read_csv("./train_2017.csv", parse_dates=["transactiondate"])
print(train_data.shape)
print(train_data.head())


train_df = train_data


prop_df = pd.read_csv("./properties_2017.csv")
print(prop_df.head())
print(prop_df.shape)



