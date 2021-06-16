from data_manager import *
from classifiers import *


def lunch_classifiers(x_train, x_test, y_train, y_test):
    tree = decision_tree(x_train, x_test, y_train, y_test)
    forest = random_forest(x_train, x_test, y_train, y_test)
    knn = knn_classifier(x_train, x_test, y_train, y_test)
    svc = svc_classifier(x_train, x_test, y_train, y_test)
    return tree, forest, knn, svc


def main():
    data = load_data()
    x_train, x_test, y_train, y_test = process_data(data)

    print("Before significance filter:")
    tree, forest, knn, svc = lunch_classifiers(x_train, x_test, y_train, y_test)

    gamma = 0.75
    x_train, x_test, y_train, y_test = process_data_using_importances(data, gamma, forest)

    print("\nAfter significance filter:")
    lunch_classifiers(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()
