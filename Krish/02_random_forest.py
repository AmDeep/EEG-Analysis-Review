"""Random-forest baseline for laser-power classification."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from eeg_model_utils import evaluate_classifier, load_data, parse_args


if __name__ == "__main__":
    args = parse_args(__doc__)
    X, y, groups = load_data(args.csv)
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=3,
                                           class_weight="balanced_subsample", n_jobs=-1, random_state=42)),
    ])
    evaluate_classifier(model, "random_forest", X, y, groups, args.output_dir)
