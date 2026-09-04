from pathlib import Path
import argparse

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_DATA = Path(__file__).resolve().parents[1] / "Aditya" / "Feature Extraction" / "features_combined.csv"
TARGET = "laser_power"
EXCLUDED_COLUMNS = {"dataset", "subject", "vertex_channel", "gamma_band_hz", TARGET}


def main(data_path: Path) -> None:
    data = pd.read_csv(data_path)
    feature_columns = [column for column in data.columns if column not in EXCLUDED_COLUMNS]
    features = data[feature_columns]
    target = data[TARGET]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_indices, test_indices = next(splitter.split(features, target, groups=data["subject"]))
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        KNeighborsRegressor(n_neighbors=15, weights="distance", n_jobs=-1),
    )
    model.fit(features.iloc[train_indices], target.iloc[train_indices])
    predictions = model.predict(features.iloc[test_indices])

    print("Model: K-Nearest Neighbors Regression")
    print(f"Rows: {len(data):,} | Features: {len(feature_columns)}")
    print(f"MAE: {mean_absolute_error(target.iloc[test_indices], predictions):.4f}")
    print(f"RMSE: {mean_squared_error(target.iloc[test_indices], predictions) ** 0.5:.4f}")
    print(f"R2: {r2_score(target.iloc[test_indices], predictions):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict laser power with k-nearest neighbors.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    main(parser.parse_args().data)