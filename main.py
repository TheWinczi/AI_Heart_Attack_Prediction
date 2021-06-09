from data_manager import *
from classifiers import *



def main():
    data = load_data()
    x_train, x_test, y_train, y_test = process_data(data)

    decision_tree(x_train, x_test, y_train, y_test)



if __name__ == "__main__":
    main()