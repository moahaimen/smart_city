from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.metrics.evaluation import classification_metrics, regression_metrics
from src.simulation.severity import map_pm25_to_severity


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return inputs
        return inputs[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs if self.downsample is None else self.downsample(inputs)
        return self.activation(self.net(inputs) + residual)


class PollutionTCN(nn.Module):
    def __init__(self, input_size: int, channel_size: int, num_blocks: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        blocks = []
        in_channels = input_size
        for block_index in range(num_blocks):
            dilation = 2 ** block_index
            blocks.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=channel_size,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_channels = channel_size
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Linear(channel_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        temporal = inputs.transpose(1, 2)
        features = self.tcn(temporal)
        return self.head(features[:, :, -1]).squeeze(-1)


@dataclass
class PredictorBundle:
    model: PollutionTCN
    feature_columns: list[str]
    window_size: int
    feature_mean: np.ndarray
    feature_std: np.ndarray
    device: str
    checkpoint_path: Path
    model_config: dict[str, int | float]

    def transform(self, windows: np.ndarray) -> np.ndarray:
        return (windows - self.feature_mean.reshape(1, 1, -1)) / self.feature_std.reshape(1, 1, -1)

    def predict(self, windows: np.ndarray, batch_size: int = 256) -> np.ndarray:
        if len(windows) == 0:
            return np.empty((0,), dtype=np.float32)
        transformed = self.transform(windows).astype(np.float32)
        self.model.eval()
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(transformed), batch_size):
                batch = torch.from_numpy(transformed[start : start + batch_size]).to(self.device)
                outputs = self.model(batch).detach().cpu().numpy()
                predictions.append(outputs)
        return np.concatenate(predictions, axis=0)

    def save(self) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_type": "tcn",
                "state_dict": self.model.state_dict(),
                "feature_columns": self.feature_columns,
                "window_size": self.window_size,
                "feature_mean": self.feature_mean,
                "feature_std": self.feature_std,
                "model_config": self.model_config,
            },
            self.checkpoint_path,
        )

    @classmethod
    def load(cls, checkpoint_path: str | Path, device: str = "cpu") -> "PredictorBundle":
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_config = dict(checkpoint["model_config"])
        model = PollutionTCN(
            input_size=len(checkpoint["feature_columns"]),
            channel_size=int(model_config["channel_size"]),
            num_blocks=int(model_config["num_blocks"]),
            kernel_size=int(model_config["kernel_size"]),
            dropout=float(model_config["dropout"]),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        return cls(
            model=model,
            feature_columns=list(checkpoint["feature_columns"]),
            window_size=int(checkpoint["window_size"]),
            feature_mean=np.asarray(checkpoint["feature_mean"], dtype=np.float32),
            feature_std=np.asarray(checkpoint["feature_std"], dtype=np.float32),
            device=device,
            checkpoint_path=Path(checkpoint_path),
            model_config=model_config,
        )


def _make_loader(inputs: np.ndarray, targets: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator if shuffle else None)


def train_tcn_regressor(
    sequence_splits: dict[str, dict[str, np.ndarray | pd.DataFrame]],
    config: dict,
    results_dirs: dict[str, Path],
    seed: int,
) -> dict[str, object]:
    model_config = config["model"]
    data_config = config["data"]
    severity_thresholds = config["severity"]["pm25_thresholds"]
    feature_columns = list(data_config["feature_columns"])
    device = "cpu"

    train_x = np.asarray(sequence_splits["train"]["x"], dtype=np.float32)
    train_y = np.asarray(sequence_splits["train"]["y"], dtype=np.float32)
    val_x = np.asarray(sequence_splits["val"]["x"], dtype=np.float32)
    val_y = np.asarray(sequence_splits["val"]["y"], dtype=np.float32)
    test_x = np.asarray(sequence_splits["test"]["x"], dtype=np.float32)
    test_y = np.asarray(sequence_splits["test"]["y"], dtype=np.float32)

    feature_mean = train_x.mean(axis=(0, 1))
    feature_std = train_x.std(axis=(0, 1))
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)

    def normalize(values: np.ndarray) -> np.ndarray:
        return ((values - feature_mean.reshape(1, 1, -1)) / feature_std.reshape(1, 1, -1)).astype(np.float32)

    train_x_norm = normalize(train_x)
    val_x_norm = normalize(val_x)

    architecture = {
        "channel_size": int(model_config["channel_size"]),
        "num_blocks": int(model_config["num_blocks"]),
        "kernel_size": int(model_config["kernel_size"]),
        "dropout": float(model_config["dropout"]),
    }
    model = PollutionTCN(
        input_size=len(feature_columns),
        channel_size=architecture["channel_size"],
        num_blocks=architecture["num_blocks"],
        kernel_size=architecture["kernel_size"],
        dropout=architecture["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    loss_fn = nn.MSELoss()
    train_loader = _make_loader(train_x_norm, train_y, batch_size=int(model_config["batch_size"]), shuffle=True, seed=seed)
    val_loader = _make_loader(val_x_norm, val_y, batch_size=int(model_config["batch_size"]), shuffle=False, seed=seed)

    best_state = None
    best_val_loss = float("inf")
    history_rows: list[dict[str, float | int]] = []

    for epoch in range(1, int(model_config["epochs"]) + 1):
        model.train()
        train_losses = []
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                predictions = model(inputs)
                val_losses.append(float(loss_fn(predictions, targets).detach().cpu().item()))

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else train_loss
        history_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_path = results_dirs["models"] / "tcn_regressor.pt"
    bundle = PredictorBundle(
        model=model,
        feature_columns=feature_columns,
        window_size=int(data_config["window_size"]),
        feature_mean=feature_mean.astype(np.float32),
        feature_std=feature_std.astype(np.float32),
        device=device,
        checkpoint_path=checkpoint_path,
        model_config=architecture,
    )
    bundle.save()

    test_predictions = bundle.predict(test_x)
    true_severity = map_pm25_to_severity(test_y, severity_thresholds)
    pred_severity = map_pm25_to_severity(test_predictions, severity_thresholds)

    regression = regression_metrics(test_y, test_predictions)
    classification = classification_metrics(true_severity, pred_severity)
    history_df = pd.DataFrame(history_rows)
    predictions_df = sequence_splits["test"]["meta"].copy()
    predictions_df["y_true"] = test_y
    predictions_df["y_pred"] = test_predictions
    predictions_df["true_severity"] = true_severity
    predictions_df["predicted_severity"] = pred_severity

    history_df.to_csv(results_dirs["tables"] / "tcn_training_history.csv", index=False)
    predictions_df.to_csv(results_dirs["tables"] / "tcn_test_predictions.csv", index=False)
    pd.DataFrame([regression]).to_csv(results_dirs["tables"] / "tcn_regression_metrics.csv", index=False)
    pd.DataFrame([classification]).to_csv(results_dirs["tables"] / "tcn_classification_metrics.csv", index=False)

    return {
        "bundle": bundle,
        "history": history_df,
        "predictions": predictions_df,
        "regression_metrics": regression,
        "classification_metrics": classification,
        "model_name": "tcn",
    }
