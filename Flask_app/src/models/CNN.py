import itertools
import json
import numpy as np
import pandas as pd
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpyencoder import NumpyEncoder
import optuna

from src.config import cfg
from src.data import preprocess
from src.utils import gen
from src.utils import metrics
from src.utils import embeddings
from src.models import Trainer

class CNN(nn.Module):
    """
    Class to define CNN architecture
    Attribution: Boilerplate code adapted from
        https://github.com/yunjey/pytorch-tutorial
        https://github.com/MorvanZhou/PyTorch-Tutorial
        https://github.com/kyungyunlee/sampleCNN-pytorch
    """

    def __init__(self, embedding_dim, vocab_size, num_filters, filter_sizes, hidden_dim, dropout_p, num_classes, padding_idx=0, pretrained_embeddings=None, freeze_embeddings=False):
        super().__init__()

        # Initialize embeddings
        if pretrained_embeddings is None:
            self.embeddings = nn.Embedding(embedding_dim=embedding_dim, num_embeddings=vocab_size, padding_idx=padding_idx)
        else:
            pretrained_embeddings = torch.from_numpy(pretrained_embeddings).float()
            self.embeddings = nn.Embedding(
                embedding_dim=embedding_dim, num_embeddings=vocab_size,
                padding_idx=padding_idx, _weight=pretrained_embeddings)

        if freeze_embeddings:
            self.embeddings.weight.requires_grad = False

        # Conv params
        self.filter_sizes = filter_sizes
        self.conv = nn.ModuleList([nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=f) for f in filter_sizes])

        # FC params
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(num_filters * len(filter_sizes), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs, channel_first=False):
        # Embed input
        (x_in,) = inputs
        x_in = self.embeddings(x_in)
        if not channel_first:
            x_in = x_in.transpose(1, 2)

        z = []
        max_seq_len = x_in.shape[2]
        for i, f in enumerate(self.filter_sizes):

            # `SAME` padding
            padding_left = int((self.conv[i].stride[0] * (max_seq_len - 1) - max_seq_len + self.filter_sizes[i]) / 2)
            padding_right = int(math.ceil((self.conv[i].stride[0] * (max_seq_len - 1) - max_seq_len + self.filter_sizes[i])/ 2))

            # Conv
            _z = self.conv[i](F.pad(x_in, (padding_left, padding_right)))

            # Pooling
            _z = F.max_pool1d(_z, _z.size(2)).squeeze(2)
            z.append(_z)

        # Concat outputs
        z = torch.cat(z, 1)

        # FC
        z = self.fc1(z)
        z = self.dropout(z)
        z = self.fc2(z)

        return z

class CNN_V2(nn.Module):
    """
    Class to define CNN architecture
    Attribution: Boilerplate code adapted from
        https://github.com/yunjey/pytorch-tutorial
        https://github.com/MorvanZhou/PyTorch-Tutorial
        https://github.com/kyungyunlee/sampleCNN-pytorch
    """

    def __init__(self, embedding_dim, vocab_size, num_filters, filter_sizes, hidden_dim, dropout_p, num_classes, padding_idx=0, pretrained_embeddings=None, freeze_embeddings=False):
        super().__init__()

        # Initialize embeddings
        if pretrained_embeddings is None:
            self.embeddings = nn.Embedding(embedding_dim=embedding_dim, num_embeddings=vocab_size, padding_idx=padding_idx)
        else:
            pretrained_embeddings = torch.from_numpy(pretrained_embeddings).float()
            self.embeddings = nn.Embedding(
                embedding_dim=embedding_dim, num_embeddings=vocab_size,
                padding_idx=padding_idx, _weight=pretrained_embeddings)

        if freeze_embeddings:
            self.embeddings.weight.requires_grad = False

        self.convolutions = nn.ModuleList([nn.Conv2d(1, num_filters, (k, embedding_dim)) for k in filter_sizes])

        self.relu = nn.ReLU()

        self.drop_out = nn.Dropout(p=dropout_p)

        units = [num_filters* len(filter_sizes)] + [hidden_dim]

        self.linear_layers = nn.ModuleList([nn.Linear(units[k], units[k+1]) for k in range(len(units) - 1)])

        self.linear_last = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs, channel_first=False):
        #size(N,1,length) to size(N,1,length,dims)
        (x,) = inputs
        x = x.unsqueeze(1)
        x = self.embeddings(x)

        #size(N,1,length,dims) to size(N,1,length)

        x = [self.relu(conv(x)).squeeze(3) for conv in self.convolutions]

        #size(N,1,length) to (N, Co * len(Ks))

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        # x = [F.avg_pool1d(i, i.size(2)).squeeze(2) for i in x]

        x = torch.cat(x, 1)

        #size(N, Co * len(Ks)) to size(N, Hu_last)

        for linear in self.linear_layers:

            x = linear(x)

            x = self.relu(x)

        #size(N, Hu_last) to size(N, C)

        x = self.drop_out(x)
        x = self.linear_last(x)

        return x


def buildCNN(params, vocab_size, num_classes, tokenizer, device=torch.device("cpu")):
    """
    Call CNN class to instantiate the model
    :param params: global param dict
    :param vocab_size: total vocal size (num unique tokens)
    :param num_classes: total num of classes
    :param tokenizer: tokenizer object fit on data
    :param device: torch device
    :return CNN model object
    """

    FREEZE_EMBEDDINGS = params.freeze_embed
    PRETRAINED_EMBEDDINGS = embeddings.processEmbeddings(params, tokenizer)

    filter_sizes = list(range(1, int(params.max_filter_size) + 1))
    model = CNN(embedding_dim=int(params.embedding_dim), vocab_size=int(vocab_size), num_filters=int(params.num_filters),
                filter_sizes= filter_sizes, hidden_dim=int(params.hidden_dim),
                dropout_p=float(params.dropout_p), num_classes=int(num_classes),
                pretrained_embeddings=PRETRAINED_EMBEDDINGS, freeze_embeddings=FREEZE_EMBEDDINGS)
    model = model.to(device)
    return model



class CNNDataset(torch.utils.data.Dataset):
    """
    Prepare dataset into CNN model readable format
    Attribution:
     boilerplate code adapted from https://pytorch.org/tutorials/
    """
    def __init__(self, X, y, max_filter_size):
        self.X = X
        self.y = y
        self.max_filter_size = max_filter_size

    def __len__(self):
        return len(self.y)

    def __str__(self):
        return f"<Dataset(N={len(self)})>"

    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        return [X, y]

    def collate_fn(self, batch):
        # Get inputs
        batch = np.array(batch, dtype=object)
        X = batch[:, 0]
        y = np.stack(batch[:, 1], axis=0)

        # Pad inputs
        X = preprocess.Addpadding(lists=X, max_list_len=self.max_filter_size)

        X = torch.LongTensor(X.astype(np.int32))
        y = torch.FloatTensor(y.astype(np.int32))

        return X, y

    def create_dataloader(self, batch_size, shuffle=False, drop_last=False):
        return torch.utils.data.DataLoader(dataset=self, batch_size=batch_size, collate_fn=self.collate_fn, shuffle=shuffle, drop_last=drop_last, pin_memory=True)



def performRunCNN(dataset, params, trial = None):
    """
    :param params: global param dict
    :param trial: Optune trial indicator
    :return Run artifacts that include params, model artifacts, and performance metrics
    """
    # Set seeds for reproducibility
    gen.setSeeds(seed=params.seed)

    # Set torch device (CPU or CUDA)
    device = gen.setDevice(cuda=params.cuda)

    # Load Dataset
    df = dataset

    # Get length of dataset
    params.num_samples = len(df)

    # Clean segments
    df.segment_text = df.segment_text.apply(preprocess.cleanText, lower=params.lower, stem=params.stem)

    # Encode categories
    cats = df.category
    label_encoder = preprocess.LabelEncoder()
    label_encoder.fit(cats)
    y = label_encoder.encode(cats)

    # Get category weight distribution for loss function
    cats_list = list(itertools.chain.from_iterable(cats.values))
    counts = np.bincount([label_encoder.class_to_index[cat_] for cat_ in cats_list])
    cat_weights = {i: 1.0 / count for i, count in enumerate(counts)}

    # Set seeds for splitting
    gen.setSeeds(seed=params.seed)

    # Split Multilabel data
    X = df.segment_text.to_numpy()
    X_train, X_, y_train, y_ = preprocess.train_test_split_multilabel(
        X=X, y=y, train_size=params.train_size
    )
    X_val, X_test, y_val, y_test = preprocess.train_test_split_multilabel(X=X_, y=y_, train_size=0.5)
    test_df = pd.DataFrame({"segment_text": X_test, "category": label_encoder.decode(y_test)})

    # Print stats if a single run
    if trial is None:
        print("-" * 60)
        print("Successfully split the dataset into {:g}% train, {:g}% val and {:g}% test!".format((params.train_size)*100, (1-params.train_size)/2*100, (1-params.train_size)/2*100))
        print("Number of unique segments in total: {}".format(X.shape[0]))
        metrics.splitStatistics(splitlist=[X_train, X_val, X_test, y_train, y_val, y_test])

    # Initialize Tokenizer
    tokenizer = preprocess.Tokenizer(char_level=params.char_level)

    # Fit Tokenizer
    tokenizer.fit_on_texts(texts=X_train)

    # Transform Train, val and test using Tokenizer
    X_train = np.array(tokenizer.texts_to_sequences(X_train), dtype=object)
    X_val = np.array(tokenizer.texts_to_sequences(X_val), dtype=object)
    X_test = np.array(tokenizer.texts_to_sequences(X_test), dtype=object)

    # Create  Train and val dataloaders
    train_dataset = CNNDataset(X=X_train, y=y_train, max_filter_size=params.max_filter_size)
    val_dataset = CNNDataset(X=X_val, y=y_val, max_filter_size=params.max_filter_size)
    train_dataloader = train_dataset.create_dataloader(batch_size=params.batch_size)
    val_dataloader = val_dataset.create_dataloader(batch_size=params.batch_size)

    # Build and initialize CNN
    model = buildCNN(params=params, vocab_size=len(tokenizer), num_classes=len(label_encoder), tokenizer=tokenizer, device=device)

    # Set CNN params for training
    print(f"Parameters: {json.dumps(params.__dict__, indent=2, cls=NumpyEncoder)}")
    cat_weights_tensor = torch.Tensor(np.array(list(cat_weights.values())))
    loss_fn = nn.BCEWithLogitsLoss(weight=cat_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.05, patience=5)

    # Call Trainer module
    trainer = Trainer(model=model, device=device, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, trial=trial)

    # Train CNN
    best_val_loss, best_model, train_losses, val_losses = trainer.train(params.num_epochs, params.patience, train_dataloader, val_dataloader)

    # Find optimal threshold
    _, y_true, y_prob = trainer.eval_step(dataloader=train_dataloader)
    params.threshold, perlabel_thresholds = metrics.getOptimalTreshold(y_true=y_true, y_prob=y_prob)

    # Evaluate model
    artifacts = {"params": params, "label_encoder": label_encoder, "tokenizer": tokenizer, "model": best_model, "loss": best_val_loss, 'train_losses':train_losses, 'val_losses': val_losses, 'perlabel_thresholds': perlabel_thresholds}
    device = torch.device("cpu")
    y_true, y_pred, performance = evaluateCNN(df=test_df, artifacts=artifacts)
    artifacts["metrics"] = performance

    # Return run artifacts
    return artifacts


def objectiveCNN(dataset, params, trial):
    """
    Objective function for Optuna hyperparameter tuning
    :param params: global param dict
    :param trial: Optuna trial
    :returns performance metric to tune

    Attribution: boilerplate code adapted from https://github.com/optuna/optuna
    """
    # Set tuning params
    if params.embedding_dim is None and params.embed is None:
        params.embedding_dim = trial.suggest_int("embedding_dim", 128, 512)
    params.num_filters = trial.suggest_int("num_filters", 128, 512)
    params.hidden_dim = trial.suggest_int("hidden_dim", 128, 512)
    params.dropout_p = trial.suggest_uniform("dropout_p", 0.3, 0.8)
    params.lr = trial.suggest_loguniform("lr", 5e-5, 5e-4)

    # Start trial
    print(f"\nTrial {trial.number}:")
    print(json.dumps(trial.params, indent=2))
    artifacts = performRunCNN(dataset=dataset, params=params, trial=trial)

    # Set tags and attributes
    params = artifacts["params"]
    performance = artifacts["metrics"]
    print(pd.DataFrame(performance["report"]).T)
    #print(json.dumps(performance["overall"], indent=2))
    trial.set_user_attr("threshold", params.threshold)
    trial.set_user_attr("precision", performance["overall"]["precision"])
    trial.set_user_attr("recall", performance["overall"]["recall"])
    trial.set_user_attr("f1", performance["overall"]["f1"])

    return performance["overall"]["f1"]



def evaluateCNN(df, artifacts, device=torch.device("cpu")):
    """
    Perform model evaluation on unseen data
    :param df: dataset
    :param artifacts: run artifacts to evaluate
    :param device: torch device
    :return y_true, y_pred, performance
    """
    # Get artifacts (load model, encoder and tokenizer)
    params = artifacts["params"]
    model = artifacts["model"]
    tokenizer = artifacts["tokenizer"]
    label_encoder = artifacts["label_encoder"]
    model = model.to(device)
    classes = label_encoder.classes

    # Prepare dataset into model readable format
    X = np.array(tokenizer.texts_to_sequences(df.segment_text.to_numpy()), dtype="object")
    y = label_encoder.encode(df.category)
    dataset = CNNDataset(X=X, y=y, max_filter_size=int(params.max_filter_size))
    dataloader = dataset.create_dataloader(batch_size=int(params.batch_size))

    # Get predictions based on optimal threshold
    trainer = Trainer(model=model, device=device)
    y_true, y_prob = trainer.predict_step(dataloader=dataloader)
    y_pred = np.array([np.where(prob >= float(params.threshold), 1, 0) for prob in y_prob])

    # Get performance evaluation metrics
    performance = metrics.get_metrics(df=df, y_true=y_true, y_pred=y_pred, classes=classes)

    return y_true, y_pred, performance


def predictCNN(segments, artifacts, device:torch.device = torch.device("cpu")):
    """
    Perform model predictions on unseen data
    :param segments: list of segments (paragraphs)
    :param artifacts: run artifacts to evaluate
    :param device: torch device
    :return category predictions
    """
    # Retrieve artifacts
    params = artifacts["params"]
    label_encoder = artifacts["label_encoder"]
    tokenizer = artifacts["tokenizer"]
    model = artifacts["model"]

    # Prepare dataset into model readable format
    preprocessed_segments = [preprocess.cleanText(segment, lower=params.lower, stem=params.stem) for segment in segments]
    X = np.array(tokenizer.texts_to_sequences(preprocessed_segments), dtype="object")
    y_blank = np.zeros((len(X), len(label_encoder)))
    dataset = CNNDataset(X=X, y=y_blank, max_filter_size=int(params.max_filter_size))
    dataloader = dataset.create_dataloader(batch_size=int(params.batch_size))

    # Get model predictions
    trainer = Trainer(model=model, device=device)
    _, y_prob = trainer.predict_step(dataloader)
    y_pred = [np.where(prob >= float(params.threshold), 1, 0) for prob in y_prob]
    categories = label_encoder.decode(y_pred)
    predictions = [{"input_text": segments[i], "preprocessed_text": preprocessed_segments[i], "predicted_tags": categories[i]} for i in range(len(categories))]

    return predictions