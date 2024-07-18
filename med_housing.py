import sys

import sklearn.cluster
import sklearn.compose
import sklearn.impute
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
assert sys.version_info >= (3,7) # Python 3.7 or above
import sklearn
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from cluster_similarity import ClusterSimilarity

def load_housing_data() -> pd.DataFrame:
    """
    Load entire housing.csv data into a DataFrame
    """
    path = Path("datasets/housing.tgz")
    if not path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, path)
        with tarfile.open(path) as tar_path:
            tar_path.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

def shuffle_and_split_data(data:pd.DataFrame, test_ratio) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Shuffle data into test/train, with test_ratio defining the percent of test set

    Ex: test_ratio = 0.2, the test set makes up 20%, while training set gets the remaining 80%
    """
    indices = np.random.permutation(len(data))
    test_size = int(len(data) * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

def is_id_in_test_set(identifier, test_ratio:float):
    """
    Compute a hash for each instance identifier, if it is less than 20% of max
    hash value, then the instance belongs in the test set. If it is over 20%, the 
    instance belongs in the training set
    """
    return crc32(np.int64(identifier)) < test_ratio * 2**32

def split_data_with_id_hash(data: pd.DataFrame, test_ratio:float, id_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute a hash for every instance and determine if that instance belongs in test data set or training data set. An
    instance's placement (train/test) is preserved, meaning, if new instances are provided in the dataset, a test set will
    not include an instance that was previously in training set. 
    """
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

def graphics_display_value_population(data: pd.DataFrame, c_in:str="median_house_value") -> None:
    """
    Scatter plot of longitude and latitude.
    Size of dot represents population (bigger size = greater population)
    Heatmap represents median house value of the district
    """
    data.plot(kind="scatter", x="longitude", y="latitude", grid=True,
              s=data["population"]/100, label="population",
              c=c_in, colormap="jet", colorbar=True,
              legend=True, sharex=False, figsize=(10,7))
    plt.show()

def graphics_scatter_matrix(data: pd.DataFrame, attributes: list) -> None:
    """
    Display correlation of attributes using panda's scatter matrix function
    """
    pd.plotting.scatter_matrix(data[attributes], figsize=(12,8))
    plt.show()

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]

def ratio_pipeline():
    return sklearn.pipeline.make_pipeline(
        sklearn.impute.SimpleImputer(strategy="median"),
        sklearn.preprocessing.FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        sklearn.preprocessing.StandardScaler()
    )

if __name__ == "__main__":

    # Get data and split into train/test
    housing:pd.DataFrame

    housing = load_housing_data()
    housing_with_id = housing.reset_index()
    
    # Stratify sample based on median income
    housing["income_category"] = pd.cut(housing["median_income"],
                                    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                    labels=[1,2,3,4,5])
    
    strat_train_set, strat_test_set = sklearn.model_selection.train_test_split(
        housing, test_size=0.2, stratify=housing["income_category"], random_state=42
    )

    # Remove income category
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_category", axis = 1, inplace=True)

    housing = strat_train_set.copy()
    
    # Look for correlations against median house price
    housing.drop("ocean_proximity", axis=1, inplace=True)
    housing["rooms_per_house"] = housing["total_rooms"]/housing["households"]
    housing["bedrooms_ratio"]  = housing["total_bedrooms"]/housing["total_rooms"]
    housing["people_per_house"] = housing["population"]/housing["households"]
    corr_matrix = housing.corr()
    #print(corr_matrix["median_house_value"].sort_values(ascending=False))

    
    
    ######################################
    #        Clean data         ##########
    ######################################

    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()
    
    # housing.dropna(subset=["total_bedrooms"], inplace=True) 1#===> Remove all NaN instances in the dataset
    # housing.drop("total_bedrooms",axis=1)                   2#===> Drop total_bedrooms column
    # median = housing["total_bedrooms"].median()
    # housing.fillna({"total_bedrooms":median}, inplace=True) 3#===> Impute NaN values with the median

    # Impute data
    imputer = sklearn.impute.SimpleImputer(strategy="median")
    housing_num = housing.select_dtypes(include=[np.number]) # Create a copy of Dataframe with numerical attributes
    imputer.fit(housing_num) # Compute medians of each attribute and store in statistics_ variable
    X = imputer.transform(housing_num) # Impute missing values in training data

    # Wrap NumPy array back into a Dataframe
    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)

    # Label encoding to convert categorical values into numerical values
    housing_cat = housing[["ocean_proximity"]]
    # ordinal_encoder = sklearn.preprocessing.OrdinalEncoder()
    # housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

    # One hot encoding to convert categorical values into vectors
    cat_encoder = sklearn.preprocessing.OneHotEncoder()
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
    

    # Normalize data
    min_max_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1,1))
    housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)
    housing_scaled = pd.DataFrame(housing_num_min_max_scaled, columns=housing_num.columns, index=housing_num.index)

    # Standardize data
    std_scaler = sklearn.preprocessing.StandardScaler()
    std_scaler.fit_transform(housing_num)

    ######################################
    #        Pipelines          ##########
    ######################################
    
    # Pipeline for numerical attributes
    num_pipeline = sklearn.pipeline.Pipeline([
        ("impute", sklearn.impute.SimpleImputer(strategy="median")),
        ("standardize", sklearn.preprocessing.StandardScaler()),
    ])

    housing_num_prepared = num_pipeline.fit_transform(housing_num)
    df_housing_num_prepared = pd.DataFrame(
        housing_num_prepared, columns=num_pipeline.get_feature_names_out(),
        index=housing_num.index
    )

    num_attribs = np.array(housing_num.columns)
    cat_attribs = ["ocean_proximity"]

    # Pipeline for category attributes
    cat_pipeline = sklearn.pipeline.Pipeline([
        ("impute", sklearn.impute.SimpleImputer(strategy="most_frequent")),
        ("Encode", sklearn.preprocessing.OneHotEncoder(handle_unknown="ignore"))
    ])

    # Pipeline to replace features with their logarithm
    log_pipeline = sklearn.pipeline.make_pipeline(
        sklearn.impute.SimpleImputer(strategy="median"),
        sklearn.preprocessing.FunctionTransformer(np.log, feature_names_out="one-to-one"),
        sklearn.preprocessing.StandardScaler()
    )


    cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
    default_num_pipeline = sklearn.pipeline.make_pipeline(
        sklearn.impute.SimpleImputer(strategy="median"),
        sklearn.preprocessing.StandardScaler()
    )

    preprocessing = sklearn.compose.ColumnTransformer([
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),   
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]), 
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population", "households", "median_income"]),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", cat_pipeline, sklearn.compose.make_column_selector(dtype_include=object)),
    ],
        remainder=default_num_pipeline)
    
    housing_prepared = preprocessing.fit_transform(housing)
    

    ######################################
    #        Train models       ##########
    ######################################

    # Linear Regression
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LinearRegression

    lin_reg = make_pipeline(preprocessing, LinearRegression())
    lin_reg.fit(housing, housing_labels)

    housing_predictions = lin_reg.predict(housing)
    
    from sklearn.metrics import mean_squared_error
    lin_rmse = mean_squared_error(housing_labels, housing_predictions, squared=False)
    # Root mean square error is $68,972 not very good. Linear Regression is underfitting the training data
    

    # Decision Tree Regressor
    from sklearn.tree import DecisionTreeRegressor
    tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
    tree_reg.fit(housing, housing_labels)
    housing_predictions = tree_reg.predict(housing)

    tree_rmse = mean_squared_error(housing_labels, housing_predictions, squared=False)
    # Decision Tree Regressor provided a 0.0 rmse... most likely overfit the data

    # Using cross validation to evaluate decision tree
    from sklearn.model_selection import cross_val_score
    tree_rmses = -cross_val_score(tree_reg, housing, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
    # Cross validation tells us the Decision Tree has an RMSE of 67,013 with a std of 1,460. 
    # Decision Tree is severley overfitting since the training set yielded no errors while validation yielded a high error


    # Random Forrest Regressor
    from sklearn.ensemble import RandomForestRegressor
    forrest_reg = make_pipeline(preprocessing, RandomForestRegressor(random_state=42))
    forrest_reg.fit(housing, housing_labels)
    forrest_predictions = forrest_reg.predict(housing)
    
    forrest_rmse = mean_squared_error(housing_labels, forrest_predictions, squared=False)
    # Got 17,519

    forrest_rmses = -cross_val_score(forrest_reg, housing, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
    # Cross validation yielded a 47124, this tells us that this model, although performed well, is still overfitting data