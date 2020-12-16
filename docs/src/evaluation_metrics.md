# Evaluation metrics

Lighthouse automatically generates a suite of evaluation metrics.
Here, we briefly describe these. This page uses terms defined in [Terminology](@ref),
so see that page for any unfamiliar words.

## Confusion matrices

Lighthouse plots confusion matrices, which are simple tables
showing the empirical distribution of predicted class (the rows)
versus the elected class (the columns). These come in two variants: 

* row-normalized: this means each row has been normalized to sum to 1. Thus, the row-normalized confusion matrix shows the empirical distribution of elected classes for a given predicted class. E.g. the first row of the row-normalized confusion matrix shows the empirical probabilities of the elected classes for a sample which was predicted to be in the first class.
* column-normalized: this means each column has been normalized to sum to 1. Thus, the column-normalized confusion matrix shows the empirical distribution of predicted classes for a given elected class. E.g. the first column of the column-normalized confusion matrix shows the empirical probabilities of the predicted classes for a sample which was elected to be in the first class.

[insert example plot]

## Inter-rater reliability

## ROC curves

## PR curves (precision-recall curves)

## PR-gain curves (precision-recall-gain curves)

## Prediction-reliability calibration

##
