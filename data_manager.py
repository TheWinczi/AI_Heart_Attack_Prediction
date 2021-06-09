from pandas import read_csv, DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np


def load_data():
    heart_data = read_csv("data/heart.csv")
    saturation_data = read_csv("data/o2Saturation.csv")

    heart_data.insert(loc=2, column="saturation", value=saturation_data, allow_duplicates=True)
    return heart_data


def process_data(data: DataFrame):
    x = data.iloc[:, range(len(data.keys())-1)]
    y = data["output"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1, stratify=y)

    std = StandardScaler()
    std.fit(x_train)
    x_train_std = std.transform(x_train)
    x_test_std = std.transform(x_test)

    return x_train_std, x_test_std, y_train, y_test

