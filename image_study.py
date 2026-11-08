from __future__ import annotations

import io
import zipfile
from pathlib import PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_ARCHIVE_BYTES = 200 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 500 * 1024 * 1024
MAX_IMAGES = 2000
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MANIFEST_COLUMNS = ["image_path", "label", "background_group"]

SCENARIO_EXAMPLES = {
    "Dogs vs. wolves": "Intended signal: animal appearance. Suspected shortcut: snow or ice.",
    "Waterbirds": "Intended signal: bird species. Suspected shortcut: water versus land background.",
    "Cows vs. camels": "Intended signal: animal appearance. Suspected shortcut: grassland versus desert.",
}


def _safe_archive_path(value: str) -> str:
    normalized = value.replace("\\", "/").strip()
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe or empty image path in manifest: {value!r}")
    return str(path)


def read_image_archive(
    archive_bytes: bytes, manifest: pd.DataFrame
) -> tuple[pd.DataFrame, list[bytes]]:
    missing_columns = [column for column in MANIFEST_COLUMNS if column not in manifest]
    if missing_columns:
        raise ValueError(
            "The annotation CSV must include: image_path, label, background_group."
        )
    if len(archive_bytes) > MAX_ARCHIVE_BYTES:
        raise ValueError("The image ZIP must be no larger than 200 MB.")
    if manifest.empty or len(manifest) > MAX_IMAGES:
        raise ValueError(f"Provide between 1 and {MAX_IMAGES} annotated images.")

    annotations = manifest[MANIFEST_COLUMNS].copy()
    if annotations.isna().any().any():
        raise ValueError("The annotation CSV cannot have blank paths, labels, or backgrounds.")
    annotations["image_path"] = annotations["image_path"].map(_safe_archive_path)
    annotations["label"] = annotations["label"].astype(str).str.strip()
    annotations["background_group"] = (
        annotations["background_group"].astype(str).str.strip()
    )
    if (annotations[["label", "background_group"]] == "").any().any():
        raise ValueError("Labels and background groups must not be blank.")
    if annotations["image_path"].duplicated().any():
        raise ValueError("Each image_path must occur only once in the annotation CSV.")
    if annotations["label"].nunique() < 2:
        raise ValueError("The dataset must contain at least two image classes.")
    if annotations["background_group"].nunique() < 2:
        raise ValueError("Provide at least two background groups to compare.")
    for path in annotations["image_path"]:
        if PurePosixPath(path).suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image type for {path!r}; use JPG, PNG, or WEBP.")

    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            image_members: dict[str, zipfile.ZipInfo] = {}
            uncompressed_size = 0
            for info in archive.infolist():
                if info.is_dir():
                    continue
                member_path = _safe_archive_path(info.filename)
                uncompressed_size += info.file_size
                if uncompressed_size > MAX_UNCOMPRESSED_BYTES:
                    raise ValueError("The uncompressed image ZIP must be no larger than 500 MB.")
                if info.file_size > MAX_IMAGE_BYTES:
                    raise ValueError(f"Image {member_path!r} is larger than 20 MB.")
                if PurePosixPath(member_path).suffix.lower() in IMAGE_EXTENSIONS:
                    if member_path in image_members:
                        raise ValueError(f"The ZIP contains a duplicate path: {member_path}")
                    image_members[member_path] = info
            payloads: list[bytes] = []
            for path in annotations["image_path"]:
                info = image_members.get(path)
                if info is None:
                    raise ValueError(f"Image {path!r} is not present in the ZIP archive.")
                payload = archive.read(info)
                try:
                    with Image.open(io.BytesIO(payload)) as image:
                        if image.width * image.height > 40_000_000:
                            raise ValueError(
                                f"Image {path!r} is larger than the 40-megapixel safety limit."
                            )
                        image.verify()
                except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
                    raise ValueError(f"Image {path!r} cannot be safely decoded: {error}") from error
                payloads.append(payload)
    except zipfile.BadZipFile as error:
        raise ValueError("Upload a valid ZIP archive containing the annotated images.") from error

    return annotations, payloads


@st.cache_resource(show_spinner="Loading the pretrained ResNet-18 image feature extractor...")
def _load_feature_extractor():
    import torch
    from torch import nn
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval()
    return torch, model, weights.transforms()


def extract_image_embeddings(image_payloads: list[bytes]) -> np.ndarray:
    torch, model, transform = _load_feature_extractor()
    embeddings: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(image_payloads), 16):
            batch = []
            for payload in image_payloads[start : start + 16]:
                try:
                    with Image.open(io.BytesIO(payload)) as image:
                        batch.append(transform(image.convert("RGB")))
                except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
                    raise ValueError(f"An image could not be decoded: {error}") from error
            output = model(torch.stack(batch))
            embeddings.append(output.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def _metric_summary(
    labels: pd.Series, predictions: np.ndarray
) -> dict[str, float]:
    return {
        "Accuracy": float(accuracy_score(labels, predictions)),
        "Precision": float(
            precision_score(labels, predictions, average="weighted", zero_division=0)
        ),
        "Recall": float(
            recall_score(labels, predictions, average="weighted", zero_division=0)
        ),
        "F1-score": float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        ),
    }


def run_background_study(
    embeddings: np.ndarray,
    annotations: pd.DataFrame,
    test_fraction: float,
    random_seed: int,
) -> dict[str, Any]:
    if len(embeddings) != len(annotations):
        raise ValueError("The number of image embeddings does not match the annotation rows.")
    if annotations["label"].nunique() < 2 or annotations["background_group"].nunique() < 2:
        raise ValueError("The study needs at least two labels and two background groups.")
    joint_groups = annotations["label"] + "\u241f" + annotations["background_group"]
    group_counts = joint_groups.value_counts()
    if group_counts.min() < 2:
        raise ValueError(
            "Each label/background combination needs at least two images so "
            "both development and held-out examples can be created."
        )

    indices = np.arange(len(annotations))
    try:
        development_indices, heldout_indices = train_test_split(
            indices,
            test_size=test_fraction,
            random_state=random_seed,
            stratify=joint_groups,
        )
    except ValueError as error:
        raise ValueError(
            "There are too few images per label/background group for this test "
            "fraction. Add examples to each group or choose a larger test fraction."
        ) from error

    development = annotations.iloc[development_indices]
    heldout = annotations.iloc[heldout_indices]
    common_background_by_label: dict[str, str] = {}
    for label in sorted(development["label"].unique()):
        label_background_counts = (
            development.loc[development["label"] == label, "background_group"]
            .value_counts()
        )
        if len(label_background_counts) < 2:
            raise ValueError(
                f"Class {label!r} needs development images from at least two "
                "background groups."
            )
        top_count = label_background_counts.max()
        if int((label_background_counts == top_count).sum()) != 1:
            raise ValueError(
                f"Class {label!r} has no unique most-common training background. "
                "Provide a clear background correlation or use more examples."
            )
        common_background_by_label[str(label)] = str(label_background_counts.idxmax())
    if len(set(common_background_by_label.values())) < 2:
        raise ValueError(
            "The most common development background is the same for every class, "
            "so this split cannot test a class-associated background shortcut. "
            "Choose a dataset where class and background are differently associated."
        )

    aligned_train_mask = np.array(
        [
            common_background_by_label[str(row.label)] == row.background_group
            for row in development.itertuples()
        ]
    )
    aligned_heldout_mask = np.array(
        [
            common_background_by_label[str(row.label)] == row.background_group
            for row in heldout.itertuples()
        ]
    )
    train_indices = development_indices[aligned_train_mask]
    common_indices = heldout_indices[aligned_heldout_mask]
    shifted_indices = heldout_indices[~aligned_heldout_mask]
    if len(train_indices) < 4:
        raise ValueError("Not enough aligned development images to train a classifier.")
    if len(common_indices) < 2 or len(shifted_indices) < 2:
        raise ValueError(
            "The held-out split needs both common-background and alternate-background "
            "examples. Add more images in each label/background group."
        )

    train_labels = annotations.iloc[train_indices]["label"].reset_index(drop=True)
    if set(train_labels) != set(annotations["label"]):
        raise ValueError(
            "The aligned training examples must contain every class. Add more "
            "images or adjust the test fraction."
        )
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_seed),
    )
    classifier.fit(embeddings[train_indices], train_labels)

    predictions: dict[str, np.ndarray] = {}
    scenarios = {
        "Common background (in-distribution)": common_indices,
        "Alternate background (shifted)": shifted_indices,
    }
    results: dict[str, Any] = {}
    class_labels = sorted(annotations["label"].unique().tolist())
    for scenario_name, scenario_indices in scenarios.items():
        scenario_labels = annotations.iloc[scenario_indices]["label"].reset_index(drop=True)
        scenario_predictions = classifier.predict(embeddings[scenario_indices])
        predictions[scenario_name] = scenario_predictions
        results[scenario_name] = {
            "metrics": _metric_summary(scenario_labels, scenario_predictions),
            "rows": int(len(scenario_indices)),
            "confusion_matrix": confusion_matrix(
                scenario_labels, scenario_predictions, labels=class_labels
            ).tolist(),
            "class_labels": class_labels,
        }

    group_rows = []
    for (label, background), group in heldout.groupby(["label", "background_group"]):
        aligned = common_background_by_label[str(label)] == str(background)
        scenario_name = (
            "Common background (in-distribution)"
            if aligned
            else "Alternate background (shifted)"
        )
        scenario_indices = common_indices if aligned else shifted_indices
        scenario_frame = annotations.iloc[scenario_indices].reset_index(drop=True)
        scenario_rows = scenario_frame[
            (scenario_frame["label"] == label)
            & (scenario_frame["background_group"] == background)
        ]
        if scenario_rows.empty:
            continue
        scenario_pos = scenario_rows.index.to_numpy()
        group_predictions = predictions[scenario_name][scenario_pos]
        group_rows.append(
            {
                "Class": str(label),
                "Background": str(background),
                "Examples": int(len(scenario_rows)),
                "Accuracy": float(accuracy_score(scenario_rows["label"], group_predictions)),
                "Role": "Common" if aligned else "Alternate",
            }
        )
    group_table = pd.DataFrame(group_rows)
    background_association = (
        development.groupby(["label", "background_group"])
        .size()
        .rename("count")
        .reset_index()
    )
    return {
        "kind": "image_background_study",
        "completed_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "image_count": int(len(annotations)),
        "train_count": int(len(train_indices)),
        "heldout_count": int(len(heldout_indices)),
        "test_fraction": float(test_fraction),
        "random_seed": int(random_seed),
        "class_labels": class_labels,
        "background_labels": sorted(annotations["background_group"].unique().tolist()),
        "common_background_by_label": common_background_by_label,
        "development_group_counts": [
            {
                "label": str(row.label),
                "background_group": str(row.background_group),
                "count": int(row.count),
            }
            for row in background_association.itertuples(index=False)
        ],
        "scenarios": results,
        "subgroup_metrics": group_table.to_dict(orient="records"),
        "feature_extractor": "ImageNet-pretrained ResNet-18, frozen feature extractor",
        "classifier": "Standardized logistic-regression linear probe",
    }
