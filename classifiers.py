from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def decision_tree(x_train, x_test, y_train, y_test):
    tree = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=1)
    tree.fit(x_train, y_train)

    y_pred = tree.predict(x_test)
    print(f"Decision Tree: {accuracy_score(y_test, y_pred)}")
    return tree


def random_forest(x_train, x_test, y_train, y_test):
    forest = RandomForestClassifier(n_estimators=55, criterion='entropy', max_depth=5, n_jobs=4, random_state=1)
    forest.fit(x_train, y_train)

    y_pred = forest.predict(x_test)
    print(f"Random forest: {accuracy_score(y_test, y_pred)}")
    return forest


def knn_classifier(x_train, x_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=14, metric='minkowski')
    knn.fit(x_train, y_train)

    y_pred = knn.predict(x_test)
    print(f"KNN: {accuracy_score(y_test, y_pred)}")
    return knn


def svc_classifier(x_train, x_test, y_train, y_test):
    svc = SVC(C=1.0, kernel='rbf', random_state=1)
    svc.fit(x_train, y_train)

    y_pred = svc.predict(x_test)
    print(f"SVN: {accuracy_score(y_test, y_pred)}")
    return svc
