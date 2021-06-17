from pandas import read_csv, DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt




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


def draw_importances_diagram(importances: list[float], labels: list[str]):
    indices = np.argsort(importances)[::-1]

    plt.figure()
    plt.title("Features importances")
    plt.bar(range(len(importances)),
            importances[indices],
            align='center')

    plt.xticks(range(len(labels)),
               labels[indices],
               rotation=90)

    plt.tight_layout()
    plt.show()


def process_data_using_importances(data: DataFrame, gamma: float, forest, diagram: bool = False):
    importances = forest.feature_importances_
    labels = data.columns.to_numpy()[:-1]
    indices = np.argsort(importances)[::-1]

    if diagram:
        draw_importances_diagram(importances, labels)

    new_labels = []
    for i, index in enumerate(indices):
        if i > gamma:
            break
        else:
            new_labels.append(labels[index])

    new_labels.append('output')

    return process_data(data[new_labels])
