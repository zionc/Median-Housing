import sys
assert sys.version_info >= (3,7) # Python 3.7 or above
import sklearn
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request


def load_housing_data() -> pd.DataFrame:
    path = Path("datasets/housing.tgz")
    if not path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, path)
        with tarfile.open(path) as tar_path:
            tar_path.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

if __name__ == "__main__":
    housing = load_housing_data()
    print(housing.head())
