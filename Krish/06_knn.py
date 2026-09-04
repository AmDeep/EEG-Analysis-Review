"""k-nearest-neighbors baseline for laser-power classification."""
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eeg_model_utils import evaluate_classifier, load_data, parse_args


if __name__ == "__main__":
    args = parse_args(__doc__)
    X, y, groups = load_data(args.csv)
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=15, weights="distance", p=2, n_jobs=-1)),
    ])
    evaluate_classifier(model, "knn", X, y, groups, args.output_dir)
