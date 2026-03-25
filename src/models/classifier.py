import torch
import torch.nn as nn
import random
import pandas as pd
from sklearn.base import BaseEstimator


def _get_discriminator_out_size(discriminator, img_channels, img_size):
    """Run a dummy forward pass to determine the flattened feature size."""
    dummy = torch.zeros(1, img_channels, img_size, img_size)
    with torch.no_grad():
        out = discriminator(dummy)
    return out.view(1, -1).shape[1]


class CNNTransferLearning(BaseEstimator, nn.Module):
    def __init__(self, dcgan_discriminator, num_classes, img_channels=3, img_size=64, learning_rate=0.001):
        super(CNNTransferLearning, self).__init__()

        self.dcgan_discriminator = dcgan_discriminator
        self.num_classes = num_classes
        self.img_channels = img_channels
        self.img_size = img_size
        self.learning_rate = learning_rate

        # Freeze discriminator weights
        for param in self.dcgan_discriminator.parameters():
            param.requires_grad = False

        flat_size = _get_discriminator_out_size(dcgan_discriminator, img_channels, img_size)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            # No Softmax here — CrossEntropyLoss includes log-softmax internally
        )

    def set_trainable_layers(self, n_blocks_from_top: int = 0):
        """Unfreeze the last n conv blocks of the discriminator (0 = fully frozen)."""
        for p in self.dcgan_discriminator.parameters():
            p.requires_grad = False
        if n_blocks_from_top == 0:
            return
        layers = list(self.dcgan_discriminator.main.children())
        conv_indices = [i for i, l in enumerate(layers) if isinstance(l, nn.Conv2d)]
        if n_blocks_from_top >= len(conv_indices):
            for p in self.dcgan_discriminator.parameters():
                p.requires_grad = True
            return
        for layer in layers[conv_indices[-n_blocks_from_top]:]:
            for p in layer.parameters():
                p.requires_grad = True

    def forward(self, x):
        if isinstance(x, list):
            x = torch.stack(x)
        x = x.to(dtype=torch.float32)
        x = self.dcgan_discriminator(x)
        x = self.classifier(x)
        return x

    def fit(self, X, y, learning_rate=None):
        if isinstance(y, list) and y and isinstance(y[0], float):
            y = torch.tensor(y).long()
        elif isinstance(y, pd.Series):
            y = torch.tensor(y.values).long()
        elif isinstance(y, list):
            y = torch.stack([yi.to(dtype=torch.long) for yi in y])

        self.train()
        criterion = nn.CrossEntropyLoss()
        lr = learning_rate if learning_rate is not None else self.learning_rate
        trainable = list(self.classifier.parameters()) + [
            p for p in self.dcgan_discriminator.parameters() if p.requires_grad
        ]
        optimizer = torch.optim.Adam(trainable, lr=lr)

        X_tensor = torch.stack([x.to(dtype=torch.float32) for x in X])

        for _ in range(10):
            optimizer.zero_grad()
            outputs = self.forward(X_tensor)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        return self

    def predict(self, X):
        X_tensor = torch.stack([x.to(dtype=torch.float32) for x in X])
        self.eval()
        with torch.no_grad():
            outputs = self.forward(X_tensor)
        _, predicted = torch.max(outputs, 1)
        return predicted.numpy()

    def score(self, X, y=None):
        """Return softmax probabilities. Signature accepts y to satisfy sklearn interface."""
        X_tensor = torch.stack([x.to(dtype=torch.float32) for x in X])
        self.eval()
        with torch.no_grad():
            outputs = self.forward(X_tensor)
        return nn.functional.softmax(outputs, dim=1).numpy()

    def batched_train(self, X, y, num_epochs=10, batch_size=32, learning_rate=None):
        """Batched training path (distinct from sklearn fit())."""
        if isinstance(y, pd.Series):
            y = torch.tensor(y.values).long()
        elif not isinstance(y, torch.Tensor):
            y = torch.tensor(y).long()

        self.train()
        criterion = nn.CrossEntropyLoss()
        lr = learning_rate if learning_rate is not None else self.learning_rate
        trainable = list(self.classifier.parameters()) + [
            p for p in self.dcgan_discriminator.parameters() if p.requires_grad
        ]
        optimizer = torch.optim.Adam(trainable, lr=lr)

        indices = list(range(len(X)))
        for _ in range(num_epochs):
            random.shuffle(indices)
            num_batches = len(indices) // batch_size
            for b in range(num_batches):
                batch_idx = indices[b * batch_size:(b + 1) * batch_size]
                batch_X = torch.stack([X[i].to(dtype=torch.float32) for i in batch_idx])
                batch_y = y[batch_idx]

                optimizer.zero_grad()
                outputs = self.forward(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        return self
