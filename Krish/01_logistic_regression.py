"""Multinomial logistic-regression baseline for laser-power classification."""
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eeg_model_utils import evaluate_classifier, load_data, parse_args


if __name__ == "__main__":
    args = parse_args(__doc__)
    X, y, groups = load_data(args.csv)
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", random_state=42)),
    ])
    evaluate_classifier(model, "logistic_regression", X, y, groups, args.output_dir)
