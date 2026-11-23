"""PBMC68k Reduced matched v1/v3 experiment; optional historical two-stage protocol."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / ".local-deps").is_dir():
    sys.path.insert(0, str(ROOT / ".local-deps"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".cache" / "numba"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

from joblib import dump
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.sparse import issparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common import SEED, save_figure

LABEL_COLUMN = "bulk_labels"
MAX_EPOCHS = 60
PATIENCE = 5
MIN_DELTA = 1e-4
VALIDATION_FRACTION = .20
HIDDEN = [128, 64]
BATCH_SIZE = 32
LEARNING_RATE = .001
WEIGHT_DECAY = .0001


def write_json(path, record):
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def ids_hash(ids):
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def make_mlp(n_features, n_classes, hidden=HIDDEN):
    return nn.Sequential(
        nn.Linear(n_features, hidden[0]), nn.ReLU(),
        nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
        nn.Linear(hidden[1], n_classes),
    )


def initialize_training(x, y, n_classes):
    """Fresh model, optimizer, scaler and shuffle generator for every trajectory."""
    torch.manual_seed(SEED)
    scaler = StandardScaler()
    train_x = torch.from_numpy(scaler.fit_transform(x))
    train_y = torch.from_numpy(y)
    dataset = TensorDataset(train_x, train_y, torch.arange(len(y)))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))
    model = make_mlp(x.shape[1], n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    return model, optimizer, scaler, train_x, train_y, loader


def train_epoch(model, optimizer, loader):
    model.train()
    order = []
    for x, y, indices in loader:
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        order.extend(indices.tolist())
    return hashlib.sha256(np.asarray(order, dtype="<i8").tobytes()).hexdigest()


def fit_selection(x_train, y_train, x_validation, y_validation, n_classes,
                  early_stopping=True):
    """Runs on 420/105 only; fixed60 control uses the identical training routine."""
    model, optimizer, scaler, train_x, train_y, loader = initialize_training(
        x_train, y_train, n_classes)
    initial_hash = state_hash(model)
    validation_x = torch.from_numpy(scaler.transform(x_validation))
    validation_y = torch.from_numpy(y_validation)
    history = []
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stopping_reference = float("inf")
    stale = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        order_hash = train_epoch(model, optimizer, loader)
        model.eval()
        with torch.inference_mode():
            train_loss = nn.functional.cross_entropy(model(train_x), train_y).item()
            validation_logits = model(validation_x)
            validation_loss = nn.functional.cross_entropy(
                validation_logits, validation_y).item()
            predicted = validation_logits.argmax(1).numpy()
        if not np.isfinite([train_loss, validation_loss]).all():
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
        _, _, f1, _ = precision_recall_fscore_support(
            y_validation, predicted, labels=np.arange(n_classes),
            average="macro", zero_division=0)
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_accuracy": accuracy_score(y_validation, predicted),
            "validation_macro_f1": f1,
            "batch_order_sha256": order_hash, "state_sha256": state_hash(model),
        })
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
        # Patience counts failures to beat the last meaningful reference by min_delta.
        if validation_loss < stopping_reference - MIN_DELTA:
            stopping_reference, stale = validation_loss, 0
        else:
            stale += 1
        if early_stopping and stale >= PATIENCE:
            break
    if early_stopping:
        model.load_state_dict(best_state)
    model.eval()
    saved_epoch = best_epoch if early_stopping else MAX_EPOCHS
    record = {
        "stage": "selection" if early_stopping else "control_fixed60",
        "n_train": len(y_train), "n_validation": len(y_validation),
        "best_epoch": best_epoch, "best_validation_loss": best_loss,
        "saved_epoch": saved_epoch, "epochs_run": len(history),
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "min_delta": MIN_DELTA,
        "early_stopped": early_stopping and len(history) < MAX_EPOCHS,
        "checkpoint_rule": "absolute minimum validation_loss" if early_stopping
                           else "fixed epoch 60",
        "patience_rule": "validation_loss < stopping_reference - min_delta",
        "initial_state_sha256": initial_hash, "saved_state_sha256": state_hash(model),
        "seed": SEED,
    }
    saved_metrics = history[saved_epoch - 1]
    for key in ("validation_loss", "validation_accuracy", "validation_macro_f1"):
        record[key + "_at_saved_epoch"] = saved_metrics[key]
    return model, scaler, pd.DataFrame(history), record


def fit_final(x_train, y_train, n_classes, epochs):
    """Fresh initialization and fresh scaler fitted on the full 525 training cells."""
    model, optimizer, scaler, train_x, train_y, loader = initialize_training(
        x_train, y_train, n_classes)
    initial_hash = state_hash(model)
    initial_optimizer_entries = len(optimizer.state)
    history = []
    for epoch in range(1, epochs + 1):
        order_hash = train_epoch(model, optimizer, loader)
        model.eval()
        with torch.inference_mode():
            train_loss = nn.functional.cross_entropy(model(train_x), train_y).item()
        if not np.isfinite(train_loss):
            raise FloatingPointError(f"Non-finite final training loss at epoch {epoch}")
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "batch_order_sha256": order_hash, "state_sha256": state_hash(model)})
    model.eval()
    record = {
        "stage": "final_refit", "n_train": len(y_train), "n_validation": 0,
        "epochs_run": epochs, "initial_state_sha256": initial_hash,
        "saved_state_sha256": state_hash(model),
        "initialization": "fresh random initialization with seed 42",
        "optimizer_initial_state_entries": initial_optimizer_entries,
        "seed": SEED,
    }
    return model, scaler, pd.DataFrame(history), record


def save_mlp(model, n_features, n_classes, record, path):
    torch.save({"state_dict": model.state_dict(), "n_features": n_features,
                "n_classes": n_classes, "hidden": HIDDEN, **record}, path)


def evaluate_mlp(model, scaler, x_test, y_test, n_classes):
    """One test forward pass after all training/epoch decisions are fixed."""
    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(scaler.transform(x_test)))
        test_loss = nn.functional.cross_entropy(logits, torch.from_numpy(y_test)).item()
        predicted = logits.argmax(1).numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predicted, labels=np.arange(n_classes), average="macro", zero_division=0)
    return predicted, {
        "test_loss": test_loss, "test_accuracy": accuracy_score(y_test, predicted),
        "test_macro_precision": precision, "test_macro_recall": recall,
        "test_macro_f1": f1, "n_test": len(y_test), "test_forward_passes": 1,
    }


def plot_training(selection, final, best_epoch, output):
    fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
    ax.plot(selection.epoch, selection.train_loss, label="Training loss", color="#207785")
    ax.plot(selection.epoch, selection.validation_loss, label="Validation loss", color="#B75B3E")
    ax.axvline(best_epoch, color="#66518E", ls="--", label=f"Best epoch = {best_epoch}")
    ax.scatter([best_epoch], [selection.validation_loss.min()], color="#66518E", zorder=3)
    ax.set(xlabel="Epoch", ylabel="Mean cross-entropy", title="Stage 1 · 420/105 epoch selection")
    ax.legend(frameon=False)
    save_figure(output, "mlp_train_validation_loss", fig)
    fig, ax = plt.subplots(figsize=(6.5, 4), layout="constrained")
    ax.plot(final.epoch, final.train_loss, marker="o", color="#207785")
    ax.set(xlabel="Epoch", ylabel="Mean training cross-entropy",
           title=f"Stage 2 · Fresh refit on 525 cells for {best_epoch} epochs")
    ax.set_xticks(final.epoch)
    save_figure(output, "mlp_final_training_loss", fig)


def save_evaluation(predictions, y_test, encoder, test_ids, output, train_cells=525):
    table = pd.DataFrame({"cell_id": test_ids, "true_label": encoder.inverse_transform(y_test)})
    classes = np.arange(len(encoder.classes_))
    metrics, errors = [], []
    for name, pred in predictions.items():
        slug = name.lower().replace(" ", "_")
        table[name] = encoder.inverse_transform(pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, pred, labels=classes, average="macro", zero_division=0)
        metrics.append((name, accuracy_score(y_test, pred), precision, recall, f1))
        report = classification_report(y_test, pred, labels=classes,
                                       target_names=encoder.classes_, output_dict=True,
                                       zero_division=0)
        pd.DataFrame(report).T.to_csv(output / "tables" / f"per_class_{slug}.csv",
                                      index_label="label")
        cm = confusion_matrix(y_test, pred, labels=classes)
        pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_).to_csv(
            output / "tables" / f"confusion_{slug}.csv", index_label="true_label")
        for i, j in zip(*np.nonzero(cm)):
            if i != j:
                errors.append((name, encoder.classes_[i], encoder.classes_[j], int(cm[i, j])))
        fig, ax = plt.subplots(figsize=(8.9, 5.7), layout="constrained")
        sns.heatmap(cm, annot=True, fmt="d", mask=cm == 0, cmap="Blues",
                    xticklabels=classes + 1, yticklabels=encoder.classes_,
                    cbar_kws={"label": "Cells"}, ax=ax)
        ax.tick_params(axis="y", labelrotation=0)
        ax.set(xlabel="Predicted class (1–10, same order as rows)", ylabel="True class",
               title=f"PBMC68k Reduced · {name} · {train_cells} training cells")
        save_figure(output, f"confusion_{slug}", fig)
    table.to_csv(output / "tables" / "classification_predictions.csv", index=False)
    frame = pd.DataFrame(metrics, columns=[
        "model", "accuracy", "macro_precision", "macro_recall", "macro_f1"])
    frame.to_csv(output / "tables" / "classification_metrics.csv", index=False)
    pd.DataFrame(errors, columns=["model", "true_label", "prediction", "cells"]).sort_values(
        ["model", "cells"], ascending=[True, False]).to_csv(
            output / "tables" / "classification_errors.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.4), layout="constrained")
    frame.set_index("model").T.rename(index={
        "accuracy": "Accuracy", "macro_precision": "Macro precision",
        "macro_recall": "Macro recall", "macro_f1": "Macro F1",
    }).plot.bar(ax=ax, color=["#237A92", "#D2863F", "#6C6699"], rot=0)
    ax.set(ylim=(0, 1), ylabel="Score", title=f"Final models · Shared {train_cells}/175 split")
    ax.legend(title=None, frameon=False, loc="upper center", ncols=3)
    save_figure(output, "model_comparison", fig)
    return frame


def read_splits(data, root, y, encoder):
    """Original CSV is authoritative; there is no new outer train/test random split."""
    original = pd.read_csv(root / "results" / "tables" / "classification_split.csv")
    if original.cell_id.duplicated().any() or set(original.cell_id) != set(data.obs_names):
        raise ValueError("Original split must contain each dataset cell exactly once.")
    if set(original.split) != {"train", "test"}:
        raise ValueError("Unexpected original split labels.")
    positions = data.obs_names.get_indexer(original.cell_id)
    np.testing.assert_array_equal(original.true_label, encoder.inverse_transform(y[positions]))
    original_train = data.obs_names.get_indexer(original.loc[original.split == "train", "cell_id"])
    # The original predictions persist the original test evaluation order.
    test_ids = pd.read_csv(root / "results" / "tables" / "classification_predictions.csv",
                           usecols=["cell_id"]).cell_id
    if test_ids.duplicated().any() or set(test_ids) != set(original.loc[original.split == "test", "cell_id"]):
        raise ValueError("Original test prediction order does not match split membership.")
    test = data.obs_names.get_indexer(test_ids)
    if len(original_train) != 525 or len(test) != 175:
        raise ValueError("Expected the original 525/175 split.")
    if np.intersect1d(original_train, test).size:
        raise ValueError("Training and test sets overlap.")
    class_counts = np.bincount(y[original_train], minlength=len(encoder.classes_))
    if class_counts.min() < 2:
        raise ValueError("Each original training class needs at least two cells.")
    train, validation = train_test_split(
        original_train, test_size=VALIDATION_FRACTION,
        stratify=y[original_train], random_state=SEED)
    if len(train) != 420 or len(validation) != 105:
        raise ValueError("Expected a 420/105 selection split.")
    for name, indices in (("train", train), ("validation", validation), ("test", test)):
        if np.setdiff1d(np.arange(len(encoder.classes_)), y[indices]).size:
            raise ValueError(f"The {name} subset is missing classes.")
    if np.intersect1d(train, validation).size:
        raise ValueError("Selection subsets overlap.")
    return original, original_train, train, validation, test


def markdown_table(frame):
    headers = list(frame.columns)
    rows = [headers, ["---"] * len(headers)]
    for record in frame.itertuples(index=False, name=None):
        rows.append([f"{v:.6f}" if isinstance(v, float) else str(v) for v in record])
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def run_two_stage(adata, root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    for protected in (root / "results", root / "results_v2"):
        if output == protected or protected in output.parents:
            raise ValueError("Preserve results and results_v2. Choose a new output directory.")
    artifacts = [
        "tables/classification_metrics.csv", "tables/mlp_selection_training.csv",
        "tables/mlp_final_training.csv", "models/mlp.pt", "early_stopping_control/metrics.csv",
    ]
    if any((output / name).exists() for name in artifacts):
        raise FileExistsError(f"Results already exist in {output}. Choose a fresh output directory.")
    for folder in ("tables", "models", "figures", "early_stopping_control"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    # Packaged dataset is local: this function does not download the PBMC3k dataset.
    data = sc.datasets.pbmc68k_reduced()
    if LABEL_COLUMN not in data.obs or data.obs[LABEL_COLUMN].isna().any():
        raise ValueError("Expected complete bulk_labels.")
    encoder = LabelEncoder()
    y = encoder.fit_transform(data.obs[LABEL_COLUMN])
    x = np.asarray(data.X.toarray() if issparse(data.X) else data.X, dtype=np.float32)
    original, full, train, validation, test = read_splits(data, root, y, encoder)
    n_classes, n_features = len(encoder.classes_), data.n_vars
    assignments = original.copy()
    assignments["selection_split"] = assignments.split.map({"train": "train", "test": "test"})
    assignments.loc[assignments.cell_id.isin(data.obs_names[validation]), "selection_split"] = "validation"
    assignments["selection_order"] = -1
    assignments["final_train_order"] = -1
    assignments["test_order"] = -1
    for indices, column in ((train, "selection_order"), (validation, "selection_order"),
                            (full, "final_train_order"), (test, "test_order")):
        order = {cell: i for i, cell in enumerate(data.obs_names[indices])}
        mask = assignments.cell_id.isin(order)
        assignments.loc[mask, column] = assignments.loc[mask, "cell_id"].map(order)
    assignments.to_csv(output / "tables" / "classification_split.csv", index=False)
    counts = pd.crosstab(assignments.true_label, assignments.selection_split).reindex(
        columns=["train", "validation", "test"])
    counts.columns = ["selection_train", "selection_validation", "test"]
    counts["final_train"] = counts.selection_train + counts.selection_validation
    counts.to_csv(output / "tables" / "class_counts.csv", index_label="label")
    mapping = {
        "feature_names_in_order": data.var_names.tolist(),
        "classes_in_order": encoder.classes_.tolist(), "label_column": LABEL_COLUMN,
        "seed": SEED, "n_train": len(full), "n_validation_final": 0, "n_test": len(test),
        "n_selection_train": len(train), "n_selection_validation": len(validation),
        "data_shape": list(data.shape), "training_order": "original split CSV row order",
        "final_train_ids_sha256": ids_hash(data.obs_names[full]),
        "selection_train_ids_sha256": ids_hash(data.obs_names[train]),
        "validation_ids_sha256": ids_hash(data.obs_names[validation]),
        "test_ids_sha256": ids_hash(data.obs_names[test]),
    }
    write_json(output / "models" / "feature_label_mapping.json", mapping)
    print(f"Selection {len(train)}/{len(validation)}; final {len(full)}/{len(test)}", flush=True)

    selected, selection_scaler, selection_history, selection_record = fit_selection(
        x[train], y[train], x[validation], y[validation], n_classes)
    best_epoch = selection_record["best_epoch"]
    selection_history.to_csv(output / "tables" / "mlp_selection_training.csv", index=False)
    write_json(output / "tables" / "mlp_best_epoch.json", selection_record)
    save_mlp(selected, n_features, n_classes, selection_record, output / "models" / "mlp_selection.pt")
    dump(selection_scaler, output / "models" / "mlp_selection_scaler.joblib")

    # Strict control A: reset RNG/model/optimizer/loader and run the same 420-cell path to 60.
    fixed, fixed_scaler, fixed_history, fixed_record = fit_selection(
        x[train], y[train], x[validation], y[validation], n_classes, early_stopping=False)
    pd.testing.assert_frame_equal(selection_history, fixed_history.iloc[:len(selection_history)])
    if selection_record["initial_state_sha256"] != fixed_record["initial_state_sha256"]:
        raise AssertionError("Controlled trajectories used different initializations.")
    control = output / "early_stopping_control"
    fixed_history.to_csv(control / "training_fixed60.csv", index=False)
    selection_history.to_csv(control / "training_early_stopping.csv", index=False)
    save_mlp(fixed, n_features, n_classes, fixed_record, control / "mlp_fixed60.pt")
    dump(fixed_scaler, control / "scaler.joblib")

    # Final model is newly initialized, with a newly fitted scaler and empty Adam state.
    final, final_scaler, final_history, final_record = fit_final(
        x[full], y[full], n_classes, best_epoch)
    final_history.to_csv(output / "tables" / "mlp_final_training.csv", index=False)
    write_json(output / "tables" / "mlp_final_training.json", final_record)
    save_mlp(final, n_features, n_classes, final_record, output / "models" / "mlp.pt")
    dump(final_scaler, output / "models" / "mlp_scaler.joblib")
    sklearn_models = {
        "Logistic Regression": ("logistic_regression", make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))),
        "Random Forest": ("random_forest", RandomForestClassifier(
            n_estimators=300, random_state=SEED, n_jobs=2)),
    }
    for name, (slug, estimator) in sklearn_models.items():
        estimator.fit(x[full], y[full])
        dump(estimator, output / "models" / f"{slug}.joblib")

    # All models and training decisions are now fixed. Each test model is evaluated once.
    predictions = {name: estimator.predict(x[test])
                   for name, (_, estimator) in sklearn_models.items()}
    predictions["MLP"], test_record = evaluate_mlp(final, final_scaler, x[test], y[test], n_classes)
    write_json(output / "tables" / "mlp_test_metrics.json", test_record)
    metrics = save_evaluation(predictions, y[test], encoder, data.obs_names[test], output)
    plot_training(selection_history, final_history, best_epoch, output)

    control_rows = []
    control_predictions = pd.DataFrame({
        "cell_id": data.obs_names[test], "true_label": encoder.inverse_transform(y[test])})
    for name, model, scaler, record in (
        ("A_fixed60", fixed, fixed_scaler, fixed_record),
        ("B_early_stopping", selected, selection_scaler, selection_record),
    ):
        predicted, evaluation = evaluate_mlp(model, scaler, x[test], y[test], n_classes)
        control_predictions[name] = encoder.inverse_transform(predicted)
        control_rows.append({
            "strategy": name, "n_train": len(train), "n_validation": len(validation),
            "epochs_run": record["epochs_run"], "model_epoch": record["saved_epoch"],
            "validation_loss": record["validation_loss_at_saved_epoch"], **evaluation})
    control_metrics = pd.DataFrame(control_rows)
    control_metrics.to_csv(control / "metrics.csv", index=False)
    control_predictions.to_csv(control / "predictions.csv", index=False)
    write_json(control / "protocol.json", {
        "fixed60": fixed_record, "early_stopping": selection_record,
        "identical_initialization": True, "identical_prefix_all_epoch_metrics_and_hashes": True,
        "prefix_epochs_verified": len(selection_history),
        "train_ids_sha256": mapping["selection_train_ids_sha256"],
        "validation_ids_sha256": mapping["validation_ids_sha256"],
        "test_ids_sha256": mapping["test_ids_sha256"],
        "B_checkpoint": "../models/mlp_selection.pt",
        "scope": "420-cell epoch-selection control; independent of 525-cell final refit",
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), layout="constrained")
    axes[0].plot(fixed_history.epoch, fixed_history.train_loss, label="Training loss")
    axes[0].plot(fixed_history.epoch, fixed_history.validation_loss, label="Validation loss")
    axes[0].axvline(best_epoch, ls="--", color="#66518E", label=f"B selected epoch {best_epoch}")
    axes[0].set(xlabel="Epoch", ylabel="Mean cross-entropy", title="Identical training path · 420 cells")
    axes[0].legend(frameon=False)
    control_metrics.set_index("strategy")[["test_accuracy", "test_macro_f1"]].plot.bar(
        ax=axes[1], rot=0, color=["#237A92", "#D2863F"])
    axes[1].set(ylim=(0, 1), ylabel="Score", title="Controlled test comparison")
    axes[1].legend(["Test Accuracy", "Test Macro-F1"], frameon=False)
    fig.savefig(control / "comparison.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    comparisons = []
    for version, directory, n_train, strategy in (
        ("v1", root / "results", 525, "fixed 60 epochs"),
        ("v2", root / "results_v2", 420, "420/105 early stopping"),
        ("v3", output, 525, f"420/105 selects {best_epoch}; fresh 525-cell refit"),
    ):
        table = pd.read_csv(directory / "tables" / "classification_metrics.csv")
        table.insert(0, "version", version)
        table["final_training_cells"] = n_train
        table["mlp_training_protocol"] = strategy
        comparisons.append(table)
    comparison = pd.concat(comparisons, ignore_index=True)
    comparison.to_csv(output / "tables" / "experiment_comparison.csv", index=False)
    write_report(output, metrics, comparison, control_metrics, counts,
                 selection_record, final_record, test_record)
    print(metrics.to_string(index=False), flush=True)
    print(control_metrics.to_string(index=False), flush=True)
    print(f"Selected epoch {best_epoch}; refit epochs {len(final_history)}", flush=True)
    return adata


def write_report(output, metrics, comparison, controls, counts, selection, final, test):
    best = selection["best_epoch"]
    control_table = controls[[
        "strategy", "epochs_run", "model_epoch", "validation_loss",
        "test_accuracy", "test_macro_f1"]]
    display_counts = counts.reset_index().rename(columns={
        "true_label": "类别", "selection_train": "选轮训练",
        "selection_validation": "选轮验证", "test": "测试", "final_train": "最终训练"})
    text = f"""# 第三版单细胞分类实验结果说明

## 1 科学问题与本次完成内容

本节开始前先明确比较对象。只改变早停策略的实验在第 9 节及 early_stopping_control 目录中：
A 和 B 都有相同的 420/105/175 划分，A 也计算验证指标，但固定采用第 60 轮，
B 用验证 Loss 停止并恢复最佳轮。历史 v1 没有验证集，v3 最终模型采用两阶段重训，
这两个历史版本不能用于只改变早停的结论。固定轮数训练同样可以设置验证集。


本实验使用本地 Scanpy PBMC68k Reduced 的 700 个细胞、765 个表达特征和 10 类 bulk_labels，
比较基因表达预测细胞类型的结果。第 06 部分在 CPU 上完成了验证集选轮、全训练集重训、
三模型最终评价，以及固定 60 轮和 Early Stopping 的严格对照。

本次读取原 results 的划分表，固定原来的 525 个训练细胞和 175 个测试细胞。
没有调用新的外层随机划分。测试顺序取自原 classification_predictions.csv 的 cell_id 列。
原划分表没有保存训练时的行顺序，本次按原划分表的行顺序固定最终训练输入，
把每个阶段的行顺序写入新划分表。模型超参数在测试评分前固定。

## 2 为什么需要验证集

做了什么。把 525 个训练细胞按标签分层，以种子 42 分成 420 个选轮训练细胞和 105 个验证细胞。
只对 420 个细胞更新参数，验证细胞用于观察对未参与梯度更新样本的预测损失。

为什么这么做。训练 Loss 持续下降可能只是增强了对训练样本的拟合。
验证 Loss 提供选择训练轮数的依据，从而让测试集承担最终评价的职责。

得到了什么。各类都覆盖三个子集，数量如下。验证集沿用本次选轮阶段的划分，
严格对照 A/B 也使用完全相同的这些细胞。

{markdown_table(display_counts)}

最小的初始 CD4 T 类在验证集中只有 1 个细胞，测试中只有 2 个细胞。
单个细胞的错分就可能明显改变逐类指标和 Macro-F1。

## 3 第一阶段如何选择 Epoch

做了什么。用 StandardScaler 在 420 个选轮训练细胞上 fit，然后对验证集 transform。
MLP 为 765 → 128 → ReLU → 64 → ReLU → 10。输出 logits，交叉熵直接接收 logits 和整数标签。
Adam 的学习率为 {LEARNING_RATE}，weight_decay 为 {WEIGHT_DECAY}，batch_size 为 {BATCH_SIZE}。
每轮结束后用该轮参数计算平均 Train Loss、Validation Loss、Validation Accuracy 和 Validation Macro-F1。

为什么这么做。max_epochs={MAX_EPOCHS} 设置训练上限。验证 Loss 相对最近一次明显改善的参考值
下降超过 min_delta={MIN_DELTA} 时重置计数，连续 {PATIENCE} 轮未达到该幅度就停止。
参数快照按验证 Loss 的绝对最低值保存，避免遗漏幅度较小但仍最低的一轮。

得到了什么。第一阶段运行 {selection['epochs_run']} 轮，最低验证 Loss 出现在第 {best} 轮，
为 {selection['best_validation_loss']:.6f}。该轮验证 Accuracy 为
{selection['validation_accuracy_at_saved_epoch']:.2%}，验证 Macro-F1 为
{selection['validation_macro_f1_at_saved_epoch']:.6f}。
该阶段的权重保存在 models/mlp_selection.pt，缩放器保存在 models/mlp_selection_scaler.joblib。

## 4 第二阶段为什么重新训练

做了什么。新建 MLP、Adam 和独立的 shuffle 生成器，以种子 42 从头初始化，
在全部 525 个原始训练细胞上重新 fit 一个 StandardScaler，训练恰好 {best} 轮。
没有将第一阶段的参数或优化器状态载入最终模型，也没有继续用第一阶段的缩放器。

为什么这么做。第一阶段需要留出细胞来估计训练时长。训练轮数确定后，这 105 个细胞的表达与标签
可以加入最终拟合，增加可用训练数据。重新初始化使最终模型从第 1 轮起就在全部训练数据和新缩放尺度上学习。
若接着第一阶段权重训练，会混合两个阶段的学习过程，训练轮数也不再对应选出的轮数。

得到了什么。最终模型训练了 {final['epochs_run']} 轮，模型保存在 models/mlp.pt，
缩放器保存在 models/mlp_scaler.joblib。最终模型是全 525 个细胞重训所得，
第一阶段模型用于选轮和 420 个训练细胞的严格对照。
两阶段使用同一个固定随机种子，初始参数可以相同，但每阶段都重新创建网络和空状态优化器。
新模型的初始化哈希、逐轮参数哈希和数据顺序哈希可用于核对这些过程。

## 5 三个最终模型如何工作

输入均为同一批 525 个细胞的 765 维已发布表达特征，目标均为 10 类参考标签。
预测时输入同一顺序的 175 个细胞，输出每个细胞的一个类别。
分类没有使用包内已有 PCA、UMAP 或聚类标签作为特征。

逻辑回归学习每个类别的线性分数，Pipeline 中的 StandardScaler 仅在 525 个训练细胞上拟合，
max_iter 为 2000，随机种子为 42。随机森林用 300 棵树集成，种子为 42，CPU 使用两个工作线程。
MLP 用 ReLU 隐层学习非线性特征组合，最终训练轮数由第一阶段确定。
三个模型的最终训练样本完全相同，MLP 额外使用训练池内部的验证过程来确定轮数。

模型训练全部结束后执行测试。最终 MLP 只进行一次测试前向计算，
Test Loss 为 {test['test_loss']:.6f}，未用测试分数修改 Epoch 或超参数。

{markdown_table(metrics)}

本次全量重训的 Accuracy 最高模型为
{metrics.loc[metrics.accuracy.idxmax(), 'model']}，其 Accuracy 为
{metrics.accuracy.max():.2%}。Macro-F1 最高模型为
{metrics.loc[metrics.macro_f1.idxmax(), 'model']}，其 Macro-F1 为
{metrics.macro_f1.max():.6f}。这里的排序只描述当前单次划分。

Accuracy 是全部细胞中预测正确的比例。Macro Precision、Macro Recall 和 Macro-F1
分别先逐类别计算，再按类别等权平均。Macro-F1 能反映小类表现，
其数值也会受小类样本量和标签差异影响。

## 6 每张图怎样阅读

- figures/mlp_train_validation_loss.png 展示第一阶段。横轴为 Epoch，纵轴为平均交叉熵，
两条线分别是训练和验证损失，虚线及圆点标记最佳 Epoch。
观察验证损失是否随着训练损失下降而下降，可判断训练是否继续改善验证目标。
- figures/mlp_final_training_loss.png 展示第二阶段。横轴为最终训练轮数，纵轴是 525 个细胞的训练损失。
曲线记录新初始化模型的拟合过程。该阶段没有独立验证曲线，评价依据是事先固定的测试结果。
- figures/model_comparison.png 横轴为四种指标，纵轴为 0 到 1 的得分，颜色区分三个全量重训模型。
比较时应同时关注总体 Accuracy 与各类别等权的 Macro-F1。
- figures/confusion_logistic_regression.png、confusion_random_forest.png 和 confusion_mlp.png
的行是参考标签，列是预测标签。列编号 1–10 与行标签顺序一致。
对角线是正确预测，非对角单元是错分。数字为测试细胞数，空白表示 0，颜色越深人数越多。
- early_stopping_control/comparison.png 左图给出同一 420 个训练细胞上固定 60 轮的训练和验证轨迹，
虚线标记 B 所选轮数。右图比较 A/B 的测试 Accuracy 和 Macro-F1。
该图研究 420 个训练细胞上的轮数策略。

## 7 每个表格和模型文件的作用

- tables/classification_split.csv 保存原始 split、第一阶段 selection_split，以及
selection_order、final_train_order、test_order。顺序从 0 开始，-1 表示该细胞不属于对应阶段。
selection_order 在选轮训练和验证两组内分别计数，应结合 selection_split 读取。
- tables/class_counts.csv 保存逐类的选轮训练、验证、测试和最终训练人数。
最终训练为选轮训练与验证的合并。
- tables/mlp_selection_training.csv 每行对应第一阶段一轮，保存四项训练验证指标、
批次顺序哈希和轮末权重哈希。
- tables/mlp_final_training.csv 每行对应第二阶段一轮，保存训练损失、批次顺序和轮末权重哈希。
- tables/mlp_best_epoch.json 保存第一阶段最佳 Epoch、验证指标、停止信息、参数和初始化哈希。
tables/mlp_final_training.json 保存从头初始化及最终训练轮数等记录。
- tables/mlp_test_metrics.json 保存最终 MLP 一次测试的损失和分类指标。
- tables/classification_metrics.csv 保存三种全 525 个训练细胞模型的统一测试指标。
- tables/per_class_logistic_regression.csv、per_class_random_forest.csv、per_class_mlp.csv
保存每类 Precision、Recall、F1 和 Support 及整体汇总。
Support 是该类测试细胞数，查看小类表现时应同时读取它。
- tables/confusion_*.csv 是三张混淆矩阵的数值版，行列是相同的类别序列。
tables/classification_errors.csv 只列非对角错分，按模型及错分人数排序。
- tables/classification_predictions.csv 保存同一顺序的测试 cell_id、参考标签和三列最终预测。
- tables/experiment_comparison.csv 汇总从 v1、v2 和 v3 读取的三模型结果及训练条件。
- models/feature_label_mapping.json 保存基因列顺序、类别编码顺序、各阶段规模和样本顺序哈希。
模型加载时需维持输入基因顺序。
- models/logistic_regression.joblib 包含缩放和分类 Pipeline，
models/random_forest.joblib 保存树集成模型，
models/mlp.pt 和 mlp_scaler.joblib 是全 525 个细胞训练的最终 MLP 及缩放器。
models/mlp_selection.pt 与 mlp_selection_scaler.joblib 属于第一阶段。
- early_stopping_control/training_fixed60.csv、training_early_stopping.csv 保存严格对照 A/B 的训练轨迹。
metrics.csv 保存二者的验证和测试指标，predictions.csv 保存测试预测，protocol.json 保存协议及前缀一致性检查。
mlp_fixed60.pt 为 A 的第 60 轮权重，scaler.joblib 在共同的 420 个训练细胞上拟合。
B 使用 ../models/mlp_selection.pt。
- run_metadata.json 保存本次解释器、依赖版本、设备和实际运行时间。
verification.json 保存本次已执行的核验结果，包括从头重训复核、A/B 轨迹一致性和旧结果哈希检查。

## 8 新旧实验如何比较

v1 用 525 个训练细胞，MLP 固定训练 60 轮。
v2 用 420 个训练细胞和 105 个验证细胞，MLP 用 Early Stopping 保存最佳验证轮。
v3 用 420/105 选轮，然后以新初始化模型在全部 525 个训练细胞上训练所选轮数。

{markdown_table(comparison)}

这些实验改变了训练细胞数、训练时长和选择协议。
原实验未保存训练行顺序，本次明确固定并记录顺序，数据顺序也可能影响随机优化和随机森林的结果。
分层拆分依赖输入行顺序，因此 v3 的 420/105 子集与 v2 的成员也存在差异。
A/B 严格对照使用的是 v3 中完全相同的 420/105 子集。
不能把版本间的性能差异直接解释为 Early Stopping 带来的提升。
v3 的三种最终模型都拟合相同的 525 个细胞，适用于当前设定下的模型对照。

## 9 严格的轮数选择策略对照

做了什么。A 与 B 均用同一 420 个训练细胞和同一 105 个验证细胞，
网络、Adam 参数、随机初始化、Scaler 和 shuffle 生成器种子全部相同。
A 固定训练 60 轮并使用第 60 轮参数。B 触发 Early Stopping 后恢复最低验证 Loss 的参数。
两者训练分别执行，程序逐轮核对 B 停止之前的损失、验证指标、
批次顺序哈希和权重哈希，确保共同训练前缀完全一致。

为什么这么做。这样控制了训练样本、初始化和训练顺序，
可以在这个固定数据划分上比较两种训练轮数选择策略。
B 复用第一阶段得到的模型，其选轮完成在任何新测试评估之前。

得到了什么。

{markdown_table(control_table)}

A 的 Validation Loss 为第 60 轮的值，B 的 Validation Loss 为其保存轮的值。
本次 A 的测试 Accuracy 为 {controls.iloc[0].test_accuracy:.2%}，
B 为 {controls.iloc[1].test_accuracy:.2%}。
A 的测试 Macro-F1 为 {controls.iloc[0].test_macro_f1:.6f}，
B 为 {controls.iloc[1].test_macro_f1:.6f}。
Accuracy 和 Macro-F1 衡量的侧重点不同，应结合逐类表现理解。
本对照的测试集仍是同一 175 个细胞。两种模型都只在权重确定后各测试一次。
这些对照模型使用 420 个训练细胞，须独立于第二阶段 525 个细胞重训的最终模型解读。

## 10 局限与后续改进

数据总量只有 700 个细胞，类别明显不平衡。最小测试类的两个样本不足以稳定描述泛化。
PBMC68k Reduced 是发布前已预处理的数据，上游特征选择和缩放无法按当前划分重新拟合。
本次 Scaler 的拟合遵守阶段边界，独立性仍受公开子集和标签来源限制。

固定细胞级留出只能评价本次子集，跨供体、批次或平台表现需要新的独立数据。
反复查看同一测试集的多个实验也会限制后续将其当作全新证据的力度。
本次没有根据这些新测试指标重选参数。后续开发宜在训练池内部完成选择，
再由新的独立供体数据进行最后确认。

可先用训练池内部的分层交叉验证研究最佳轮数是否稳定，再考虑正则化和类别加权。
420 个细胞上选出的 Epoch 在 525 个细胞上对应更多梯度更新，最佳时长的迁移也需要验证。
后续还可比较传统表达特征和预训练单细胞向量，并同时报告计算成本及每类表现。
这些是后续研究方向，本次结果表来自实际执行的三模型与 A/B 对照。

## 11 复现

首次在项目根目录运行以下命令，默认写入 results_v3

```bash
conda run -n ai4sci python src/06_cell_classification.py
```

结果已存在时，以新的输出目录复现

```bash
conda run -n ai4sci python src/06_cell_classification.py --output results_v3_recheck
```

该入口只执行第 06 部分，读取本地随 Scanpy 提供的数据和已有划分文件。
"""
    (output / "实验结果说明.md").write_text(text)


def run(adata, root, output, protocol="matched"):
    """Default experiment compares v1/v3 with identical 420/105/175 data."""
    if protocol == "two-stage":
        return run_two_stage(adata, root, output)
    root, output = Path(root).resolve(), Path(output).resolve()
    for name in ("results", "results_v2", "results_v3"):
        protected = root / name
        if output == protected or protected in output.parents:
            raise ValueError("Use a new results_matched output directory.")
    if (output / "comparison.csv").exists() or (output / "v1" / "tables" / "mlp_training.csv").exists():
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    data = sc.datasets.pbmc68k_reduced()
    encoder = LabelEncoder()
    if data.obs[LABEL_COLUMN].isna().any():
        raise ValueError("Missing reference labels.")
    y = encoder.fit_transform(data.obs[LABEL_COLUMN])
    x = np.asarray(data.X.toarray() if issparse(data.X) else data.X, dtype=np.float32)
    _, full, train, validation, test = read_splits(data, root, y, encoder)
    n_classes = len(encoder.classes_)
    split_rows = []
    for label, indices in (("train", train), ("validation", validation), ("test", test)):
        for order, index in enumerate(indices):
            split_rows.append((data.obs_names[index], encoder.classes_[y[index]], label, order))
    split = pd.DataFrame(split_rows, columns=["cell_id", "true_label", "split", "order"])
    split.to_csv(output / "classification_split.csv", index=False)
    pd.crosstab(split.true_label, split.split).reindex(
        columns=["train", "validation", "test"]).to_csv(output / "class_counts.csv")

    # Independent executions, each with fresh model/Adam/scaler and reset RNG/loader.
    fixed, fixed_scaler, fixed_log, fixed_record = fit_selection(
        x[train], y[train], x[validation], y[validation], n_classes, early_stopping=False)
    stopped, stopped_scaler, stopped_log, stopped_record = fit_selection(
        x[train], y[train], x[validation], y[validation], n_classes, early_stopping=True)
    pd.testing.assert_frame_equal(fixed_log.iloc[:len(stopped_log)], stopped_log)
    assert fixed_record["initial_state_sha256"] == stopped_record["initial_state_sha256"]
    np.testing.assert_array_equal(fixed_scaler.mean_, stopped_scaler.mean_)
    np.testing.assert_array_equal(fixed_scaler.scale_, stopped_scaler.scale_)
    assert fixed_scaler.n_samples_seen_ == stopped_scaler.n_samples_seen_ == 420
    assert not np.intersect1d(train, validation).size
    assert not np.intersect1d(full, test).size

    # These two fixed baselines are shared by v1 and v3, with exactly the same fit data.
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=2)
    lr.fit(x[train], y[train])
    rf.fit(x[train], y[train])
    # All training and selection decisions are fixed before the first test prediction.
    common_predictions = {"Logistic Regression": lr.predict(x[test]), "Random Forest": rf.predict(x[test])}
    metric_frames, mlp_rows = [], []
    versions = (
        ("v1", fixed, fixed_scaler, fixed_log, fixed_record),
        ("v3", stopped, stopped_scaler, stopped_log, stopped_record),
    )
    for version, model, scaler, log, record in versions:
        directory = output / version
        for folder in ("tables", "models", "figures"):
            (directory / folder).mkdir(parents=True, exist_ok=True)
        log.to_csv(directory / "tables" / "mlp_training.csv", index=False)
        write_json(directory / "tables" / "mlp_checkpoint.json", record)
        save_mlp(model, data.n_vars, n_classes, record, directory / "models" / "mlp.pt")
        dump(scaler, directory / "models" / "mlp_scaler.joblib")
        dump(lr, directory / "models" / "logistic_regression.joblib")
        dump(rf, directory / "models" / "random_forest.joblib")
        write_json(directory / "models" / "feature_label_mapping.json", {
            "feature_names_in_order": data.var_names.tolist(),
            "classes_in_order": encoder.classes_.tolist(),
            "n_train": len(train), "n_validation": len(validation), "n_test": len(test),
            "seed": SEED,
        })
        predictions = dict(common_predictions)
        predictions["MLP"], test_metrics = evaluate_mlp(model, scaler, x[test], y[test], n_classes)
        write_json(directory / "tables" / "mlp_test_metrics.json", test_metrics)
        metrics = save_evaluation(predictions, y[test], encoder, data.obs_names[test], directory, train_cells=420)
        metrics.insert(0, "version", version)
        metric_frames.append(metrics)
        mlp_rows.append({
            "version": version, "n_train": 420, "n_validation": 105, "n_test": 175,
            "epochs_run": record["epochs_run"], "model_epoch": record["saved_epoch"],
            "validation_loss": record["validation_loss_at_saved_epoch"],
            "validation_accuracy": record["validation_accuracy_at_saved_epoch"],
            "validation_macro_f1": record["validation_macro_f1_at_saved_epoch"], **test_metrics,
        })
        fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
        ax.plot(log.epoch, log.train_loss, label="Training loss", color="#207785")
        ax.plot(log.epoch, log.validation_loss, label="Validation loss", color="#B75B3E")
        epoch = record["saved_epoch"]
        ax.axvline(epoch, color="#66518E", ls="--", label=f"Saved model epoch {epoch}")
        ax.scatter([epoch], [record["validation_loss_at_saved_epoch"]], color="#66518E")
        ax.set(xlabel="Epoch", ylabel="Mean cross-entropy", title=f"{version} · Matched 420/105 split")
        ax.legend(frameon=False)
        save_figure(directory, "mlp_train_validation_loss", fig)

    comparison = pd.concat(metric_frames, ignore_index=True)
    comparison.to_csv(output / "comparison.csv", index=False)
    mlp_comparison = pd.DataFrame(mlp_rows)
    mlp_comparison.to_csv(output / "mlp_comparison.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for ax, (version, _, _, log, record) in zip(axes, versions):
        ax.plot(log.epoch, log.train_loss, label="Training loss")
        ax.plot(log.epoch, log.validation_loss, label="Validation loss")
        ax.axvline(record["saved_epoch"], ls="--", color="#66518E",
                   label=f"Saved epoch {record['saved_epoch']}")
        ax.set(xlabel="Epoch", ylabel="Mean cross-entropy", title=version)
        ax.legend(frameon=False)
    fig.savefig(output / "training_loss_comparison.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    protocol_record = {
        "experiment": "matched v1 fixed60 vs v3 early stopping",
        "only_changed_factor": "MLP stopping and checkpoint selection strategy",
        "train_cells": 420, "validation_cells": 105, "test_cells": 175,
        "train_ids_sha256": ids_hash(data.obs_names[train]),
        "validation_ids_sha256": ids_hash(data.obs_names[validation]),
        "test_ids_sha256": ids_hash(data.obs_names[test]),
        "initial_state_sha256": fixed_record["initial_state_sha256"],
        "identical_prefix_epochs_verified": len(stopped_log),
        "prefix_metrics_weights_and_batch_order_equal": True,
        "scaler_means_scales_equal": True, "scaler_fit_cells": 420,
        "hidden": HIDDEN, "input_features": data.n_vars, "classes": n_classes,
        "optimizer": "Adam", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE, "seed": SEED, "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE, "min_delta": MIN_DELTA,
        "test_forward_passes_per_mlp": 1,
        "v1": fixed_record, "v3": stopped_record,
        "historical_archive": ["results", "results_v2", "results_v3"],
    }
    write_json(output / "protocol.json", protocol_record)
    report = f"""# v1 与 v3 相同条件重新实验

## 本次问题与比较范围

本次重新执行两个独立训练过程，比较固定训练 60 轮与验证集早停策略。
本目录内 v1 和 v3 都有相同的 420 个训练细胞、105 个验证细胞和 175 个测试细胞。
两组共享网络结构、随机初始化、Adam 参数、Scaler、训练批次顺序和测试顺序。
唯一改变的实验因素是 MLP 的停止及保存轮次策略。

本次 v1 是重新实验的固定轮数基线，和原 results 中的历史 v1 分开保存。
历史 results_v3 中的全 525 个细胞两阶段重训记录属于此前的另一个协议。
本报告中的结果全部来自本次实际训练。

## 数据与训练

使用本地 Scanpy PBMC68k Reduced，700 × 765，10 类。
从原 results/tables/classification_split.csv 读取训练池与测试集，
按原测试预测表保留 175 个测试细胞的顺序。
在原 525 个训练池中按类别分层、种子 42 划分出 420/105。
训练、验证、测试互不重叠，所有类别均有样本。

Scaler 只在 420 个训练细胞拟合，两组验证与测试数据只 transform。
MLP 为 765→128→ReLU→64→ReLU→10，直接输出 logits，
使用 CrossEntropyLoss、Adam、学习率 0.001、weight_decay 0.0001、batch_size 32。

v1 和 v3 每轮都计算训练 Loss、验证 Loss、验证 Accuracy 和验证 Macro-F1。
v1 固定采用第 60 轮，不使用验证指标改变训练时长。
v3 在连续 5 轮未达到 min_delta=0.0001 的改善幅度时停止，恢复最低验证 Loss 的快照。

两组分别从头训练，初始化哈希一致。
前 {len(stopped_log)} 轮的批次顺序、权重哈希和全部训练验证指标逐轮一致。
所有训练和模型选择完成后，再对两组 MLP 分别测试一次。

Logistic Regression 与 Random Forest 同样只使用 420 个训练细胞，
它们的参数与预测在两个版本中完全相同，方便把差异限定在 MLP 训练轮数选择。

## 本次 MLP 实测结果

{markdown_table(mlp_comparison)}

## 三模型结果

{markdown_table(comparison)}

## 怎样看图和文件

training_loss_comparison.png 左侧是 v1 的 60 轮记录，右侧是 v3 的提前停止记录。
横轴是 Epoch，纵轴是平均交叉熵，虚线标出实际采用的模型轮次。
两个版本的 figures/mlp_train_validation_loss.png 单独绘制各自训练与验证曲线。
验证 Loss 从最低点上升时，说明继续拟合训练数据没有同步改善验证目标。

每个版本的 model_comparison.png 比较三模型的 Accuracy 与宏平均指标。
confusion_*.png 和对应 CSV 的行是真实标签，列是预测标签，对角线为正确数量。

classification_split.csv 保存共同划分及组内顺序，class_counts.csv 保存各类规模。
v1/tables 和 v3/tables 分别保存逐轮指标、模型轮次记录、最终测试指标、
逐类指标、逐细胞预测、混淆矩阵及错分。
models 下的 mlp.pt 是对应版本实际采用的权重，mlp_scaler.joblib 是对应缩放器。
protocol.json 保存固定条件、两组轮次和训练过程一致性检查。
comparison.csv 是三模型对比，mlp_comparison.csv 是本次 MLP 早停对照的完整数值。

## 如何解释这次结果

本次固定 60 轮的测试 Accuracy 为 {mlp_rows[0]['test_accuracy']:.2%}，
早停版本为 {mlp_rows[1]['test_accuracy']:.2%}。
对应 Macro-F1 为 {mlp_rows[0]['test_macro_f1']:.6f} 和 {mlp_rows[1]['test_macro_f1']:.6f}。
应按同一划分下的实际结果描述各指标，不能预设早停一定更好。

这是一组单随机种子、小样本的细胞级留出。最小验证类仅 1 个细胞、最小测试类仅 2 个细胞。
同一个测试集已被用于多个历史实验，本次没有根据新测试分数调整策略，
但后续更充分的确认需要训练池内部多次分层评估及新的独立供体数据。
公开 Reduced 子集的上游预处理仍不能在当前切分内重做。

## 复现

在项目根目录执行

```bash
conda run -n ai4sci python src/06_cell_classification.py --output results_matched_recheck
```

默认协议现在是本次相同条件的 v1/v3 实验。
此前两阶段协议可用 --protocol two-stage 和新的 --output 目录运行。
"""
    (output / "实验结果说明.md").write_text(report)
    print(comparison.to_string(index=False), flush=True)
    print(mlp_comparison.to_string(index=False), flush=True)
    return adata



if __name__ == "__main__":
    import argparse
    from importlib.metadata import version
    from time import perf_counter

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--protocol", choices=["matched", "two-stage"], default="matched")

    args = parser.parse_args()
    if args.output is None:
        args.output = ROOT / ("results_matched" if args.protocol == "matched" else "results_v3")
    started = perf_counter()
    run(None, ROOT, args.output, protocol=args.protocol)
    metadata = {
        "python_path": sys.executable, "seed": SEED, "device": "cpu", "torch_threads": 2,
        "protocol": args.protocol,
        "elapsed_seconds": round(perf_counter() - started, 3),
        "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
        "packages": {name: version(name) for name in (
            "scanpy", "torch", "numpy", "scikit-learn", "pandas", "joblib")},
    }
    write_json(args.output / "run_metadata.json", metadata)
    print(f"Results saved to {args.output.resolve()}", flush=True)
