# Held-out-subject baseline results

All scores come from the same five-fold, dataset-specific-subject-held-out split. Higher is better.

| Model | Accuracy | Balanced accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| Logistic regression | 12.9% | 14.5% | 10.6% |
| Random forest | 24.8% | 13.1% | 12.4% |
| Extra trees | 22.2% | 14.2% | 12.8% |
| Histogram gradient boosting | 18.0% | **16.0%** | **13.0%** |
| Linear SVM | 14.3% | 13.9% | 10.1% |
| k-nearest neighbors | 24.4% | 9.5% | 9.3% |

For context, always predicting the most common level (3.50) would achieve about 26.0% ordinary accuracy but only 6.7% balanced accuracy. Therefore, ordinary accuracy alone is misleading in this data: it rewards predicting the common power level. Histogram gradient boosting is the most balanced baseline because it has the best balanced accuracy and macro-F1, even though random forest has the highest raw accuracy among the six models.

The held-out-subject scores are low but meaningful as a conservative starting point. They indicate that these pooled, epoch-level EEG features contain limited generalizable information about the exact one-of-15 laser-power levels under this validation scheme. The next useful check is leave-one-dataset-out validation; it will measure transfer to a wholly unseen study and should be interpreted separately from the results here.
