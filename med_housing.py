import sys

import sklearn.model_selection
assert sys.version_info >= (3,7) # Python 3.7 or above
import sklearn
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32

def load_housing_data() -> pd.DataFrame:
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

if __name__ == "__main__":

    # Get data and split into train/test
    housing = load_housing_data()
    housing_with_id = housing.reset_index()
    train_set, test_set = split_data_with_id_hash(data=housing_with_id,test_ratio=0.2, id_column="index")
    
    # Graphics to display dataset
    
    housing["income_category"] = pd.cut(housing["median_income"],
                                        bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                        labels=[1,2,3,4,5])
    housing["income_category"].value_counts().sort_index().plot.bar(rot=0, grid=True)
    plt.xlabel("Income Category")
    plt.ylabel("Number of districts")
    plt.show()




