"""Histogram-gradient-boosting baseline for laser-power classification."""
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from eeg_model_utils import evaluate_classifier, load_data, parse_args


if __name__ == "__main__":
    args = parse_args(__doc__)
    X, y, groups = load_data(args.csv)
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(learning_rate=0.08, max_iter=300, max_leaf_nodes=15,
                                                   l2_regularization=1.0, random_state=42)),
    ])
    evaluate_classifier(model, "hist_gradient_boosting", X, y, groups, args.output_dir,
                        balanced_sample_weights=True)
