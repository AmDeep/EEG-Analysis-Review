"""Linear support-vector-machine baseline for laser-power classification."""
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from eeg_model_utils import evaluate_classifier, load_data, parse_args


if __name__ == "__main__":
    args = parse_args(__doc__)
    X, y, groups = load_data(args.csv)
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearSVC(C=0.1, class_weight="balanced", dual=False,
                            max_iter=10000, random_state=42)),
    ])
    evaluate_classifier(model, "linear_svm", X, y, groups, args.output_dir)
