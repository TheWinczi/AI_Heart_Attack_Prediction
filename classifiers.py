from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score



def decision_tree(x_train, x_test, y_train, y_test):
    tree = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=1)
    tree.fit(x_train, y_train)

    y_pred = tree.predict(x_test)
    print(f"Decision Tree: {accuracy_score(y_test, y_pred)}")
