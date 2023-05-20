import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from dtreeviz import decision_boundaries
import dtreeviz
import matplotlib.pyplot as plt

if True:
    dataset_url = "https://raw.githubusercontent.com/parrt/dtreeviz/master/data/cars.csv"
    df_cars = pd.read_csv(dataset_url)

    X = df_cars.drop('MPG', axis=1)
    features = list(X.columns)
    # features = ['WGT','CYL']

    X = X[features]
    y = df_cars['MPG']

    dt = DecisionTreeRegressor(max_depth=3, criterion="absolute_error")
    dt.fit(X.values, y.values)

    viz_rmodel = dtreeviz.model(dt, X, y,
                               feature_names=features,
                               target_name='MPG')
    viz_rmodel.view().show()
    # viz_rmodel.rtree_feature_space(features=['WGT','CYL'], show=['splits', 'legend'])
    # viz_rmodel.rtree_feature_space(features=['WGT'], show=['splits', 'legend'])
    # viz_rmodel.rtree_feature_space()

    viz_rmodel = dtreeviz.model(dt, X, y,
                                feature_names=features,
                                target_name='MPG')
    viz_rmodel.rtree_feature_space3D(
        # features=['WGT','ENG'],
        features=['WGT','CYL'],
        # features=['CYL', 'ENG'],
                                     fontsize=10,
                                     elev=10, azim=20,
                                     show={'splits', 'title'},
        dist=20,
                                     colors={'tessellation_alpha': .5})
    plt.show()

if False:
    from sklearn.datasets import load_iris

    iris = load_iris()
    features = list(iris.feature_names)
    class_names = iris.target_names
    X = iris.data
    y = iris.target

    dtc_univar = DecisionTreeClassifier(max_depth=3, min_samples_leaf=1, random_state=666)
    dtc_univar.fit(X, y)

    viz_model = dtreeviz.model(dtc_univar,
                               X_train=X, y_train=y,
                               feature_names=features,
                               target_name='iris',
                               class_names=class_names)

    # viz_model.view().show()
    viz_model.ctree_feature_space(#features=['petal length (cm)'],
                                  # gtype='barstacked',
                                  figsize=(5, 1.5))
    plt.show()

if False:
    dataset_url = "https://raw.githubusercontent.com/parrt/dtreeviz/master/data/titanic/titanic.csv"
    dataset = pd.read_csv(dataset_url)
    # Fill missing values for Age
    dataset.fillna({"Age":dataset.Age.mean()}, inplace=True)
    # Encode categorical variables
    dataset["Sex_label"] = dataset.Sex.astype("category").cat.codes.astype(int)
    dataset["Cabin_label"] = dataset.Cabin.astype("category").cat.codes
    dataset["Embarked_label"] = dataset.Embarked.astype("category").cat.codes

    features = ["Pclass", "Age", "Fare", "Sex_label", "Cabin_label", "Embarked_label"]
    # features = ['Age', 'Sex_label']
    # features = ['Sex_label']
    target = "Survived"

    tree_classifier = DecisionTreeClassifier(max_depth=3)
    tree_classifier.fit(dataset[features].values, dataset[target].values)

    viz_model = dtreeviz.model(tree_classifier,
                               X_train=dataset[features], y_train=dataset[target],
                               feature_names=features,
                               target_name=target, class_names=["survive", "perish"])

    # viz_model.view(fontname='Courier New', scale=.75).show()

    # viz_model.instance_feature_importance(x=dataset[features].iloc[0,:])
    # viz_model.ctree_feature_space(features=['Age','Sex_label'], show=['splits', 'legend'])
    viz_model.ctree_feature_space(features=['Fare'], show=['splits', 'legend'])
    plt.show()