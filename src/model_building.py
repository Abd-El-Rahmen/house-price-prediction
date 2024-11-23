from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge

def build_random_forest(X_train , Y_train):
    model = RandomForestRegressor(n_estimators=100 , random_state=42)
    model.fit(X_train, Y_train)
    return model

def build_linear_regression(X_train , Y_train):
    # Ridge regression is used here instead of simple linear regression to prevent overfitting
    model = Ridge(alpha=1.0)  
    model.fit(X_train, Y_train)
    return model

def build_decision_tree(X_train , Y_train):
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, Y_train)
    return model
