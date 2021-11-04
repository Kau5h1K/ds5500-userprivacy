import json
import tempfile
import warnings
from argparse import Namespace
from pathlib import Path

import mlflow
import optuna
import torch
from numpyencoder import NumpyEncoder
from optuna.integration.mlflow import MLflowCallback
import os

from src.config import cfg
from src.data import preprocess
from src.utils import gen
from src.models import CNN

# Ignore warning
warnings.filterwarnings("ignore")


def performTuning(param_dict, study_name="optimization", n_trials=100):
    """
    Perform hyper-param tuning using Optuna
    :param param_dict: global param dict
    :param study_name: Optuna study name
    :param n_trials: number of trials to run tuning for

    Attribution: boilerplate adapted from https://github.com/optuna/optuna
    """

    print("🟠 Performing Hyper-parameter tuning for {} trials...".format(n_trials))
    # Convert Dict to ArgeParse namespace
    params = Namespace(**param_dict)

    # Set up Optuna params with MLFlow callback
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(study_name=study_name, direction="maximize", pruner=pruner)
    mlflow_callback = MLflowCallback(tracking_uri=mlflow.get_tracking_uri(), metric_name="f1")
    study.optimize(lambda trial: CNN.objectiveCNN(params, trial), n_trials=n_trials, callbacks=[mlflow_callback])

    print("🟢 Hyper-parameter tuning completed successfully!")
    # Get stats of all trials
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values(["value"], ascending=False)
    print("Sample results of trials")
    print(trials_df.head())

    # Get params of best trial
    print(f"Best value F1: {study.best_trial.value}")
    params = {**params.__dict__, **study.best_trial.params}
    params["threshold"] = study.best_trial.user_attrs["threshold"]

    # Save params of best trial to local system
    os.makedirs(cfg.PARAM.BEST_PARAM_DPATH, exist_ok = True)
    gen.saveParams(params, os.path.join(cfg.PARAM.BEST_PARAM_DPATH, "best_param_dict.json"), cls=NumpyEncoder)
    print("Parameters of best trial")
    print(json.dumps(params, indent=2, cls=NumpyEncoder))



def trainwithBP(param_dict, experiment_name="best", run_name="model", save=False):
    """
    Train/Retrain the model with Custom/best params obtained from tuning
    :param param_dict: global param dict
    :param experiment_name: MLFlow experiment name
    :param run_name: MLFlow run name
    :param save_artifacts: save performance and params (bool)
    Attribution: boilerplate adapted from https://www.mlflow.org/docs/latest/tutorials-and-examples/tutorial.html
                                          https://towardsdatascience.com/manage-your-machine-learning-lifecycle-with-mlflow-part-1-a7252c859f72
                                          https://www.run.ai/guides/machine-learning-operations/mlflow/
    """
    # Convert Dict to ArgeParse namespace
    params = Namespace(**param_dict)

    # Start experiment
    mlflow.set_experiment(experiment_name=experiment_name)
    # Start run
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id

        # Perform training for one run
        artifacts = CNN.performRunCNN(params=params)

        # Set custom tags
        tags = {}
        mlflow.set_tags(tags)

        # Log performance evaluation metrics from run
        performance = artifacts["performance"]
        print(json.dumps(performance["overall"], indent=2))
        metrics = {"precision": performance["overall"]["precision"],
                   "recall": performance["overall"]["recall"],
                   "f1": performance["overall"]["f1"],
                   "best_val_loss": artifacts["loss"]}
        mlflow.log_metrics(metrics)

        # Log run artifacts to MLFlow registry
        with tempfile.TemporaryDirectory() as dp:
            gen.saveParams(vars(artifacts["params"]), Path(dp, "params.json"), cls=NumpyEncoder)
            gen.saveParams(performance, Path(dp, "performance.json"))
            artifacts["label_encoder"].save(Path(dp, "label_encoder.json"))
            artifacts["tokenizer"].save(Path(dp, "tokenizer.json"))
            torch.save(artifacts["model"].state_dict(), Path(dp, "model.pt"))
            mlflow.log_artifacts(dp)
        mlflow.log_params(vars(artifacts["params"]))

    # Save performance metrics and run ID to local system
    if not save:
        open(os.path.join(cfg.PARAM.BEST_PARAM_DPATH, "run_ID.txt"), "w").write(run_id)
        gen.saveParams(performance, os.path.join(cfg.PARAM.BEST_PARAM_DPATH, "metrics.json"))


def predictSegment(segment, run_id):
    """
    Utility func to classify a segment into cats
    :param segment: a sequence of text
    :param run_id: MLFlow run to use
    :return prediction
    """
    # Load artifacts from the run and predict
    artifacts = loadRunArtifacts(run_id=run_id)
    prediction = CNN.predictCNN(segments=[segment], artifacts=artifacts)
    print(json.dumps(prediction, indent=2))

    return prediction


def getRunParams(run_id):
    """
    Utility func to get params of MLFlow run
    :param run_id: MLFlow run to use
    :return params
    """
    artifact_uri = mlflow.get_run(run_id=run_id).info.artifact_uri.split("file://")[-1]
    params = gen.loadParams(filepath=Path(artifact_uri, "params.json"))
    print(json.dumps(params, indent=2))
    return params


def getRunMetrics(run_id):
    """
    Utility func to get performance metrics of MLFlow run
    :param run_id: MLFlow run to use
    :return metrics
    """
    artifact_uri = mlflow.get_run(run_id=run_id).info.artifact_uri.split("file://")[-1]
    metrics = gen.loadParams(filepath=Path(artifact_uri, "metrics.json"))
    print(json.dumps(metrics, indent=2))
    return metrics


def loadRunArtifacts(run_id, device=torch.device("cpu")):
    """
    Utility func to load run artifacts from MLFLow registry
    :param run_id: MLFlow run to use
    :param device: torch device
    :return artifacts

    Attribution: boilerplate adapted from https://www.mlflow.org/docs/latest/tutorials-and-examples/tutorial.html
    """

    # Load artifacts from MLFLow registry
    experiment_id = mlflow.get_run(run_id=run_id).info.experiment_id
    artifacts_dir = os.path.join(cfg.MLFLOW.MODEL_REGISTRY, experiment_id, run_id, "artifacts")
    params = Namespace(**gen.loadParams(filepath=os.path.join(artifacts_dir, "params.json")))
    label_encoder = preprocess.LabelEncoder.load(fp=Path(artifacts_dir, "label_encoder.json"))
    tokenizer = preprocess.Tokenizer.load(fp=Path(artifacts_dir, "tokenizer.json"))
    model_state = torch.load(Path(artifacts_dir, "model.pt"), map_location=device)
    performance = gen.loadParams(filepath=Path(artifacts_dir, "performance.json"))

    # Load model state
    model = CNN.buildCNN(params=params, vocab_size=len(tokenizer), num_classes=len(label_encoder))
    model.load_state_dict(model_state)

    artifacts = {"params": params, "label_encoder": label_encoder, "tokenizer": tokenizer, "model": model, "performance": performance}

    return artifacts

def deleteMLFlowExperiment(experiment_name):
    """
    Utility func to load run artifacts from MLFLow registry
    :param experiment_name: MLFlow experiment to delete

    Attribution: boilerplate adapted from https://www.mlflow.org/docs/latest/tutorials-and-examples/tutorial.html
    """
    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    client.delete_experiment(experiment_id=experiment_id)
    print(f"🔴 Deleted MLFlow experiment {experiment_name}!")