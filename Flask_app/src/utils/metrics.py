
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from snorkel.slicing import PandasSFApplier, slicing_function
import itertools



def decorate(func):
    """
    Function to decorate the output

    :param func: the function to wrap the output of
    :return: nested func
    """
    def inner(self, *args, **kwargs):
        print("~" * 60)
        func(self, *args, **kwargs)
        print("~" * 60)
    return inner


def splitStatistics(splitlist):
    """
    Function to print split statistics

    :param splitlist: _X_train, _X_dev, _X_test, _y_train, _y_dev, _y_test
    :return: print split stats
    """
    cat_list = ['Data Retention', 'Data Security', 'Do Not Track', 'First Party Collection/Use', 'International and Specific Audiences', 'Introductory/Generic', 'Policy Change', 'Practice not covered', 'Privacy contact information', 'Third Party Sharing/Collection', 'User Access, Edit and Deletion', 'User Choice/Control']
    _X_train, _X_val, _X_test, _y_train, _y_val, _y_test = splitlist
    for label, (_X, _y) in {"TRAIN SET":[_X_train, _y_train], "DEV SET":[_X_val, _y_val], "TEST SET":[_X_test, _y_test]}.items():
        print(label)
        print("Number of unique segments: {}".format(_X.shape[0]))
        print("Percentage of segments containing each of the following categories:")
        df = pd.Series(np.sum(_y, axis = 0), index= cat_list)
        print(pd.DataFrame({
            "Counts": df,
            "Percentage": (round(df/sum(df)*100, 2)).astype('str') + "%"}))
        print("-" * 60)


def getOptimalTreshold(y_true, y_prob):
    """
    :param y_true: ground truth
    :param y_prob: Prediction probabilities
    :return optimal threshold
    """
    prec, rec, thr = precision_recall_curve(y_true.ravel(), y_prob.ravel())
    f1_list = (2 * prec * rec) / (prec + rec)
    optim_threshold_all = thr[np.argmax(f1_list)]

    optim_threshold_list = []
    for i in range(y_true.shape[1]):
        prec, rec, thr = precision_recall_curve(y_true[:,i].ravel(), y_prob[:,i].ravel())
        f1_list = (2 * prec * rec) / (prec + rec)
        optim_threshold = thr[np.argmax(f1_list)]
        optim_threshold_list.append(optim_threshold)

    return optim_threshold_all, optim_threshold_list

@slicing_function()
def selected_cats(x):
    """Segments with the `First Party Collection/Use` and `User Access, Edit and Deletion` categories."""
    return all(cat in x.category for cat in ["User Choice/Control", "User Access, Edit and Deletion"])


@slicing_function()
def short_text(x):
    """Projects with short titles and descriptions."""
    return len(x.segment_text.split()) < 7  # less than 7 words



def get_metrics(y_true, y_pred, classes, df=None):
    """
    Function to obtain performance evaluation metrics from y_true and y_pred
    :param y_true: ground truth
    :param y_pred: Prediction probabilities
    :param classes: categories
    :param df: dataset
    :return return metrics dict (overall and per-class)
    Attribution: Code adapted from https://madewithml.com/
    """
    # Performance
    metrics = {"overall": {}, "class": {}, "report":{}}

    # Overall metrics
    overall_metrics = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    metrics["overall"]["precision"] = overall_metrics[0]
    metrics["overall"]["recall"] = overall_metrics[1]
    metrics["overall"]["f1"] = overall_metrics[2]
    metrics["overall"]["num_samples"] = np.float64(len(y_true))
    metrics["report"] = classification_report(y_true, y_pred, target_names=classes, output_dict = True)

    # Per-class metrics
    class_metrics = precision_recall_fscore_support(y_true, y_pred, average=None)
    for i in range(len(classes)):
        metrics["class"][classes[i]] = {
            "precision": class_metrics[0][i],
            "recall": class_metrics[1][i],
            "f1": class_metrics[2][i],
            "num_samples": np.float64(class_metrics[3][i]),
        }

    # Slicing metrics
    if df is not None:
        # Slices
        slicing_functions = [selected_cats, short_text]
        applier = PandasSFApplier(slicing_functions)
        slices = applier.apply(df)

        # Score slices
        # Use snorkel.analysis.Scorer for multiclass tasks
        # Naive implementation for our multilabel task
        # based on snorkel.analysis.Scorer
        metrics["slices"] = {}
        metrics["slices"]["class"] = {}
        for slice_name in slices.dtype.names:
            mask = slices[slice_name].astype(bool)
            if sum(mask):  # pragma: no cover, test set may not have enough samples for slicing
                slice_metrics = precision_recall_fscore_support(
                    y_true[mask], y_pred[mask], average="micro"
                )
                metrics["slices"]["class"][slice_name] = {}
                metrics["slices"]["class"][slice_name]["precision"] = slice_metrics[0]
                metrics["slices"]["class"][slice_name]["recall"] = slice_metrics[1]
                metrics["slices"]["class"][slice_name]["f1"] = slice_metrics[2]
                metrics["slices"]["class"][slice_name]["num_samples"] = len(y_true[mask])

        # Weighted overall slice metrics
        metrics["slices"]["overall"] = {}
        for metric in ["precision", "recall", "f1"]:
            metrics["slices"]["overall"][metric] = np.mean(
                list(
                    itertools.chain.from_iterable(
                        [
                            [metrics["slices"]["class"][slice_name][metric]]
                            * metrics["slices"]["class"][slice_name]["num_samples"]
                            for slice_name in metrics["slices"]["class"]
                        ]
                    )
                )
            )

    return metrics