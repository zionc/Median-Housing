import sys
assert sys.version_info >= (3,7) # Python 3.7 or above
import sklearn
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np

def load_housing_data() -> pd.DataFrame:
    path = Path("datasets/housing.tgz")
    if not path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, path)
        with tarfile.open(path) as tar_path:
            tar_path.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

def shuffle_and_split_data(data:pd.DataFrame, test_ratio):
    """
    Shuffle data into test/train, with test_ratio defining the percent of test set

    Ex: test_ratio = 0.2, the test set makes up 20%, while training set gets the remaining 80%
    """
    indices = np.random.permutation(len(data))
    test_size = int(len(data) * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

if __name__ == "__main__":
    housing = load_housing_data()
    housing.hist(bins=50,figsize=(14,8))
    train_set, test_set = shuffle_and_split_data(housing,0.2)
