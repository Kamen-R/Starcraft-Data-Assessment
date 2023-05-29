from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def get_data(data, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=["LeagueIndex", "LeagueName", "GameID", "Age", "TotalHours", "HoursPerWeek"]),
                                                        data["LeagueIndex"], test_size=test_size, random_state=random_state)
    
    X_valid = X_test[len(X_test) // 3:]
    X_test = X_test[:len(X_test)// 3]
    y_valid = y_test[len(y_test) // 3:]
    y_test = y_test[:len(y_test)// 3]

    return X_train, X_test, y_train, y_test, X_valid, y_valid

def LogisticCV(X_train, y_train, C):
    clf = LogisticRegressionCV(Cs = C, cv=10, multi_class="multinomial").fit(X_train, y_train)
    return clf

def KNeigbors(X_train, y_train, neighbors):
    neigh = KNeighborsClassifier(n_neighbors=neighbors)
    neigh.fit(X_train, y_train)
    return neigh

def RandomForest(X_train, y_train, depth):
    decision_clf = RandomForestClassifier(max_depth=depth, random_state=0)
    decision_clf.fit(X_train, y_train)
    return decision_clf

def clf_results(clf, X_test, y_test):
    pred = (clf.predict(X_test) == y_test).sum()
    pred_above = (clf.predict(X_test) + 1 == y_test).sum()
    pred_under = (clf.predict(X_test) - 1 == y_test).sum()
    print("Classifier name:", clf)
    print("Proportion of correct predictions:",  round(pred / len(X_test), 4))
    print("Proportion of predictions within 1 rank:", round((pred + pred_above + pred_under) / len(X_test), 4))
    return None
