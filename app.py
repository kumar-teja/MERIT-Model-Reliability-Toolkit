from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from image_study import (
    SCENARIO_EXAMPLES,
    extract_image_embeddings,
    read_image_archive,
    run_background_study,
)


st.set_page_config(
    page_title="MERIT | Model Reliability Investigation Toolkit",
    page_icon="🔬",
    layout="wide",
)

MODEL_NAMES = (
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "K-Nearest Neighbors",
)
METRIC_NAMES = ("Accuracy", "Precision", "Recall", "F1-score")


def make_synthetic_data(
    rows: int, shortcut_strength: float, random_seed: int
) -> pd.DataFrame:
    """Create binary data with a genuine signal and a controllable shortcut."""
    rng = np.random.default_rng(random_seed)
    target = rng.integers(0, 2, size=rows)
    direction = target * 2 - 1
    genuine_signal = direction + rng.normal(0, 1.0, size=rows)
    shortcut_agrees = rng.random(rows) < shortcut_strength
    shortcut_signal = np.where(shortcut_agrees, target, 1 - target)
    return pd.DataFrame(
        {
            "genuine_signal": genuine_signal,
            "shortcut_signal": shortcut_signal.astype(int),
            "target": target,
        }
    )


def prepare_features(
    data: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    if target_column not in data.columns:
        raise ValueError("The selected target column is not present in the dataset.")

    working = data.copy()
    rows_before = len(working)
    working = working.loc[working[target_column].notna()].copy()
    dropped_target_rows = rows_before - len(working)
    if working.empty:
        raise ValueError("The selected target has no non-missing values.")

    y = working.pop(target_column)
    if y.nunique(dropna=True) < 2:
        raise ValueError("The target must contain at least two distinct classes.")

    X = working
    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    if numeric_columns:
        X[numeric_columns] = X[numeric_columns].replace([np.inf, -np.inf], np.nan)

    empty_columns = X.columns[X.isna().all()].tolist()
    X = X.drop(columns=empty_columns)
    if X.shape[1] == 0:
        raise ValueError(
            "No usable feature columns remain after excluding the target and "
            "columns that contain no values."
        )

    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]
    audit = {
        "rows_before_target_check": rows_before,
        "rows_dropped_for_missing_target": dropped_target_rows,
        "rows_used": len(X),
        "feature_columns": X.columns.tolist(),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "empty_columns_excluded": empty_columns,
        "missing_values_by_column": X.isna().sum().to_dict(),
    }
    return X, y, audit


def make_preprocessor(
    numeric_columns: list[str], categorical_columns: list[str]
) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )
    return ColumnTransformer(transformers, remainder="drop")


def make_classifier(name: str, training_rows: int, random_seed: int):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, random_state=random_seed)
    if name == "Decision Tree":
        return DecisionTreeClassifier(random_state=random_seed)
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=150, random_state=random_seed, n_jobs=-1
        )
    if name == "K-Nearest Neighbors":
        return KNeighborsClassifier(n_neighbors=min(5, training_rows))
    raise ValueError(f"Unknown model: {name}")


def calculate_metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    return {
        "Accuracy": float(accuracy_score(y_true, predictions)),
        "Precision": float(
            precision_score(y_true, predictions, average="weighted", zero_division=0)
        ),
        "Recall": float(
            recall_score(y_true, predictions, average="weighted", zero_division=0)
        ),
        "F1-score": float(
            f1_score(y_true, predictions, average="weighted", zero_division=0)
        ),
    }


def run_experiment(
    data: pd.DataFrame,
    target_column: str,
    model_names: list[str],
    test_fraction: float,
    random_seed: int,
    intervention_column: str,
    intervention: str,
) -> dict[str, Any]:
    X, y, audit = prepare_features(data, target_column)
    class_counts = y.value_counts()
    stratify = y if class_counts.min() >= 2 else None
    if stratify is None:
        raise ValueError(
            "Each target class needs at least two rows to create a stratified "
            "train/test split. Add data for the smallest class and try again."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_fraction,
        random_state=random_seed,
        stratify=stratify,
    )

    if intervention_column not in X_test.columns:
        raise ValueError("Choose an intervention feature from the available columns.")

    changed_test = X_test.copy()
    rng = np.random.default_rng(random_seed)
    if intervention == "Shuffle feature values":
        changed_test[intervention_column] = rng.permutation(
            changed_test[intervention_column].to_numpy()
        )
    elif intervention == "Reverse numeric feature":
        if not pd.api.types.is_numeric_dtype(changed_test[intervention_column]):
            raise ValueError("Reversing is available only for numeric features.")
        changed_test[intervention_column] = (
            changed_test[intervention_column].max()
            + changed_test[intervention_column].min()
            - changed_test[intervention_column]
        )
    else:
        raise ValueError("Select a supported feature intervention.")

    numeric_columns = audit["numeric_columns"]
    categorical_columns = audit["categorical_columns"]
    results: dict[str, Any] = {
        "audit": audit,
        "class_counts": {str(key): int(value) for key, value in y.value_counts().items()},
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "target_column": target_column,
        "intervention_column": intervention_column,
        "intervention": intervention,
        "models": {},
    }

    for model_name in model_names:
        pipeline = Pipeline(
            [
                (
                    "preprocessor",
                    make_preprocessor(numeric_columns, categorical_columns),
                ),
                (
                    "classifier",
                    make_classifier(model_name, len(X_train), random_seed),
                ),
            ]
        )
        pipeline.fit(X_train, y_train)
        baseline_predictions = pipeline.predict(X_test)
        intervention_predictions = pipeline.predict(changed_test)
        baseline_metrics = calculate_metrics(y_test, baseline_predictions)
        intervention_metrics = calculate_metrics(y_test, intervention_predictions)

        without_feature_metrics = None
        remaining_features = [column for column in X.columns if column != intervention_column]
        if remaining_features:
            reduced_numeric = [column for column in numeric_columns if column in remaining_features]
            reduced_categorical = [
                column for column in categorical_columns if column in remaining_features
            ]
            reduced_pipeline = Pipeline(
                [
                    (
                        "preprocessor",
                        make_preprocessor(reduced_numeric, reduced_categorical),
                    ),
                    (
                        "classifier",
                        make_classifier(model_name, len(X_train), random_seed),
                    ),
                ]
            )
            reduced_pipeline.fit(X_train[remaining_features], y_train)
            without_feature_metrics = calculate_metrics(
                y_test, reduced_pipeline.predict(X_test[remaining_features])
            )

        results["models"][model_name] = {
            "baseline": baseline_metrics,
            "intervention": intervention_metrics,
            "without_intervention_feature": without_feature_metrics,
            "confusion_matrix": confusion_matrix(
                y_test, baseline_predictions, labels=pipeline.classes_
            ).tolist(),
            "class_labels": [str(label) for label in pipeline.classes_],
        }
    results["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    return results


def render_analysis(result: dict[str, Any] | None) -> None:
    st.title("Analysis")
    st.caption(
        "Research context, experimental design, and the latest measured results."
    )

    st.subheader("Research question")
    st.write(
        "How do spurious-correlation strength and classifier choice affect "
        "generalization, and which practical interventions improve robustness "
        "when those relationships change?"
    )
    st.write(
        "A model can score well on a familiar test set by using a shortcut that "
        "is easier to learn than the intended signal. MERIT makes one feature "
        "change explicit and compares predictions on the same held-out examples."
    )

    with st.expander("Research design and interpretation", expanded=True):
        st.markdown(
            """
            - **Controlled synthetic case:** a noisy genuine signal and a binary
              shortcut are created with a user-selected correlation strength.
            - **Uploaded-data case:** one selected feature is perturbed on the
              test split. This is a sensitivity probe, not proof that the feature
              is spurious or causal.
            - **Fair comparison:** models share one stratified split and fixed
              random seed; every preprocessing step is fitted on training rows.
            - **Robustness readout:** compare baseline, perturbed-test, and
              retrained-without-feature scores. A decrease is evidence for
              sensitivity under this experiment, not a deployment guarantee.
            """
        )

    st.subheader("What the metrics measure")
    st.dataframe(
        pd.DataFrame(
            [
                ("Accuracy", "Fraction of all predictions that are correct."),
                (
                    "Precision",
                    "For each class, how often predictions of that class are correct; "
                    "reported as a class-frequency-weighted average.",
                ),
                (
                    "Recall",
                    "For each class, how many actual examples are found; "
                    "reported as a class-frequency-weighted average.",
                ),
                (
                    "F1-score",
                    "Harmonic mean of precision and recall, weighted by class frequency.",
                ),
                ("Confusion matrix", "Counts of actual versus predicted classes."),
            ],
            columns=["Measure", "Interpretation"],
        ),
        hide_index=True,
        width="stretch",
    )

    if result is None:
        st.info("Run an experiment in the Project Tool tab to populate this analysis.")
        return

    if result.get("kind") == "image_background_study":
        st.subheader("Latest image background study")
        first, second, third = st.columns(3)
        first.metric("Annotated images", result["image_count"])
        second.metric("Aligned training images", result["train_count"])
        third.metric("Held-out images", result["heldout_count"])
        st.caption(
            f"Completed: {result['completed_at_utc']} · "
            f"Feature extractor: {result['feature_extractor']} · "
            f"Classifier: {result['classifier']}"
        )
        st.write(
            "The app identifies each class's most common background using only "
            "the development split, trains on those aligned examples, then "
            "compares held-out images with common versus alternate backgrounds."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Scenario": name,
                        "Examples": values["rows"],
                        **values["metrics"],
                    }
                    for name, values in result["scenarios"].items()
                ]
            ).style.format({metric: "{:.3f}" for metric in METRIC_NAMES}),
            hide_index=True,
            width="stretch",
        )
        subgroup_frame = pd.DataFrame(result["subgroup_metrics"])
        st.subheader("Held-out accuracy by class and background")
        st.dataframe(
            subgroup_frame.style.format({"Accuracy": "{:.3f}"}),
            hide_index=True,
            width="stretch",
        )
        figure, axis = plt.subplots(figsize=(9, 4))
        subgroup_frame["Group"] = (
            subgroup_frame["Class"] + " · " + subgroup_frame["Background"]
        )
        axis.bar(subgroup_frame["Group"], subgroup_frame["Accuracy"])
        axis.set_ylim(0, 1.05)
        axis.set_ylabel("Accuracy")
        axis.set_title("Held-out performance across background groups")
        axis.tick_params(axis="x", rotation=25)
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)
        for scenario_name, scenario in result["scenarios"].items():
            with st.expander(f"{scenario_name} confusion matrix"):
                st.dataframe(
                    pd.DataFrame(
                        scenario["confusion_matrix"],
                        index=[f"Actual {label}" for label in scenario["class_labels"]],
                        columns=[
                            f"Predicted {label}" for label in scenario["class_labels"]
                        ],
                    ),
                    width="stretch",
                )
        shifted_accuracy = result["scenarios"]["Alternate background (shifted)"][
            "metrics"
        ]["Accuracy"]
        common_accuracy = result["scenarios"]["Common background (in-distribution)"][
            "metrics"
        ]["Accuracy"]
        st.info(
            f"Accuracy was {common_accuracy:.3f} on held-out common-background "
            f"images and {shifted_accuracy:.3f} on held-out alternate-background "
            "images. This controlled split tests sensitivity to the annotated "
            "background association; it does not prove that the background caused "
            "any individual prediction."
        )
        return

    st.subheader("Latest experiment")
    audit = result["audit"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows used", audit["rows_used"])
    c2.metric("Training rows", result["train_rows"])
    c3.metric("Held-out rows", result["test_rows"])
    st.caption(
        f"Target: {result['target_column']} · Intervention: "
        f"{result['intervention']} on `{result['intervention_column']}` · "
        f"Completed: {result['completed_at_utc']}"
    )

    metric_rows = []
    for model_name, model_result in result["models"].items():
        for scenario in ("baseline", "intervention", "without_intervention_feature"):
            metrics = model_result[scenario]
            if metrics is not None:
                metric_rows.append(
                    {
                        "Model": model_name,
                        "Scenario": {
                            "baseline": "Baseline",
                            "intervention": "Perturbed test feature",
                            "without_intervention_feature": "Retrained without feature",
                        }[scenario],
                        **metrics,
                    }
                )
    metrics_frame = pd.DataFrame(metric_rows)
    st.dataframe(
        metrics_frame.style.format({metric: "{:.3f}" for metric in METRIC_NAMES}),
        hide_index=True,
        width="stretch",
    )

    chart_data = metrics_frame.melt(
        id_vars=["Model", "Scenario"], value_vars=list(METRIC_NAMES),
        var_name="Metric", value_name="Score",
    )
    figure, axis = plt.subplots(figsize=(10, 4))
    for (model_name, scenario), group in chart_data.groupby(["Model", "Scenario"]):
        axis.plot(
            group["Metric"],
            group["Score"],
            marker="o",
            label=f"{model_name} — {scenario}",
        )
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title("Baseline, intervention, and retraining comparison")
    axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)

    st.subheader("Baseline confusion matrices")
    matrix_columns = st.columns(min(len(result["models"]), 2))
    for index, (model_name, model_result) in enumerate(result["models"].items()):
        with matrix_columns[index % len(matrix_columns)]:
            st.markdown(f"**{model_name}**")
            st.dataframe(
                pd.DataFrame(
                    model_result["confusion_matrix"],
                    index=[f"Actual {label}" for label in model_result["class_labels"]],
                    columns=[f"Predicted {label}" for label in model_result["class_labels"]],
                ),
                width="stretch",
            )

    drops = []
    for model_name, model_result in result["models"].items():
        baseline_accuracy = model_result["baseline"]["Accuracy"]
        changed_accuracy = model_result["intervention"]["Accuracy"]
        drops.append(
            (baseline_accuracy - changed_accuracy, model_name, baseline_accuracy, changed_accuracy)
        )
    largest_drop = max(drops)
    st.info(
        f"{largest_drop[1]} had the largest accuracy decrease in this run "
        f"({largest_drop[2]:.3f} baseline to {largest_drop[3]:.3f} after the "
        "test-feature intervention). Interpret this as experiment-specific "
        "sensitivity; it does not establish causality or predict all deployment shifts."
    )


def render_image_tool() -> None:
    st.subheader("Image background study")
    st.write(
        "Start immediately with one of three bundled generated image scenarios, "
        "or upload your own labelled image collection for a background study."
    )
    image_source = st.radio(
        "Image dataset",
        ("Bundled generated image demo", "Upload your own image ZIP and annotations"),
        horizontal=True,
    )
    scenario = st.selectbox("Example research scenario", list(SCENARIO_EXAMPLES))
    st.info(SCENARIO_EXAMPLES[scenario])
    st.markdown(
        """
        **For your own data, include both classes in at least two background
        groups.** The annotations are supplied by you; MERIT does not detect or
        segment backgrounds automatically. Use at least four images per
        class/background combination at the default test fraction; ten or more
        per combination is a better starting point.
        """
    )
    template = pd.DataFrame(columns=["image_path", "label", "background_group"])
    st.download_button(
        "Download annotation CSV template",
        data=template.to_csv(index=False),
        file_name="merit_image_annotations_template.csv",
        mime="text/csv",
    )
    demo_directory = Path(__file__).resolve().parent / "data" / "image_demo"
    if image_source == "Bundled generated image demo":
        demo_names = {
            "Dogs vs. wolves": "synthetic_dogs_wolves",
            "Waterbirds": "synthetic_waterbirds",
            "Cows vs. camels": "synthetic_cows_camels",
        }
        demo_name = demo_names[scenario]
        archive_path = demo_directory / f"{demo_name}.zip"
        manifest_path = demo_directory / f"{demo_name}.csv"
        if not archive_path.exists() or not manifest_path.exists():
            st.error(
                "The bundled image demo is missing. From the project folder run "
                "`python scripts\\create_demo_datasets.py`."
            )
            return
        archive_bytes = archive_path.read_bytes()
        try:
            manifest = pd.read_csv(manifest_path)
        except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as error:
            st.error(f"The bundled annotations could not be read: {error}")
            return
        st.caption(
            "These are generated illustrations, not real animal or bird "
            "photographs. Download the selected demo to inspect its images and "
            "annotations."
        )
        st.download_button(
            "Download selected demo images (ZIP)",
            data=archive_bytes,
            file_name=archive_path.name,
            mime="application/zip",
        )
        st.download_button(
            "Download selected demo annotations (CSV)",
            data=manifest_path.read_bytes(),
            file_name=manifest_path.name,
            mime="text/csv",
        )
    else:
        st.caption(
            "The ZIP contains image files; the CSV maps each image_path inside "
            "that ZIP to its label and background_group. Paths are relative to "
            "the ZIP root."
        )
        archive_file = st.file_uploader("Upload images as a ZIP file", type=["zip"])
        annotation_file = st.file_uploader(
            "Upload image annotations as CSV", type=["csv"]
        )
        if archive_file is None or annotation_file is None:
            st.info("Upload both files to inspect and run an image study.")
            return
        archive_bytes = archive_file.getvalue()
        try:
            manifest = pd.read_csv(annotation_file)
        except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as error:
            st.error(f"The annotation CSV could not be read: {error}")
            return
    try:
        annotations, image_payloads = read_image_archive(
            archive_bytes, manifest
        )
    except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as error:
        st.error(f"Image dataset could not be prepared: {error}")
        return

    st.write(
        f"Ready to analyze **{len(annotations):,} images**, "
        f"{annotations['label'].nunique()} classes, and "
        f"{annotations['background_group'].nunique()} background groups."
    )
    st.dataframe(
        annotations.groupby(["label", "background_group"])
        .size()
        .rename("Images")
        .reset_index(),
        hide_index=True,
        width="stretch",
    )
    preview_indices = (
        annotations.groupby(["label", "background_group"], sort=True)
        .head(1)
        .index.tolist()
    )
    if preview_indices:
        st.image(
            [image_payloads[index] for index in preview_indices],
            caption=annotations.loc[preview_indices, "image_path"].tolist(),
            width=160,
        )
    test_fraction = st.slider(
        "Held-out image fraction", 0.2, 0.4, 0.25, step=0.05, key="image_test_fraction"
    )
    random_seed = st.number_input(
        "Image study random seed", min_value=0, max_value=99999, value=42,
        key="image_random_seed",
    )
    st.caption(
        "The first image run downloads ImageNet-pretrained ResNet-18 weights. "
        "MERIT freezes this feature extractor and fits a logistic-regression "
        "classifier on CPU."
    )
    if st.button("Run image background study", type="primary"):
        try:
            with st.spinner("Extracting image features and evaluating background groups..."):
                embeddings = extract_image_embeddings(image_payloads)
                result = run_background_study(
                    embeddings,
                    annotations,
                    test_fraction=test_fraction,
                    random_seed=int(random_seed),
                )
            st.session_state["merit_result"] = result
            st.success("Image study complete. Review the Analysis tab for results.")
        except (ValueError, OSError, RuntimeError, ImportError) as error:
            st.error(
                "The image study could not run. Check the image files and network "
                f"access for the first pretrained-weight download. Details: {error}"
            )

    result = st.session_state.get("merit_result")
    if result and result.get("kind") == "image_background_study":
        st.download_button(
            "Download image study results (JSON)",
            data=json.dumps(result, indent=2),
            file_name="merit_image_background_results.json",
            mime="application/json",
        )


def render_project_tool() -> None:
    st.title("Project Tool")
    st.caption(
        "Prepare data, run reliability experiments, and compare outcomes."
    )
    experiment_type = st.radio(
        "Experiment type",
        ("Tabular reliability study", "Image background study"),
        horizontal=True,
    )
    if experiment_type == "Image background study":
        render_image_tool()
        return
    source = st.radio(
        "Tabular data source",
        ("Synthetic controlled dataset", "Upload labelled CSV"),
        horizontal=True,
    )
    if source == "Upload labelled CSV":
        st.markdown("**Bundled CSV examples** — download one, then upload it below.")
        tabular_directory = Path(__file__).resolve().parent / "data" / "tabular"
        examples = [
            ("01_intended_signal_only.csv", "Intended signal, no shortcut feature"),
            ("02_balanced_no_shortcut.csv", "Shortcut cue is independent of the label"),
            ("03_strong_shortcut.csv", "Shortcut cue agrees with label 95% of the time"),
            ("04_reversed_shortcut.csv", "Shortcut cue agrees with label 5% of the time"),
        ]
        download_columns = st.columns(2)
        for index, (filename, description) in enumerate(examples):
            dataset_path = tabular_directory / filename
            if dataset_path.exists():
                with download_columns[index % 2]:
                    st.download_button(
                        f"Download: {description}",
                        data=dataset_path.read_bytes(),
                        file_name=filename,
                        mime="text/csv",
                    )
        st.caption(
            "These are generated teaching datasets, not collected measurements. "
            "They all share a noisy genuine signal so the shortcut can be compared fairly."
        )

    uploaded_data: pd.DataFrame | None = None
    if source == "Upload labelled CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded_file is None:
            st.info("Upload a labelled CSV to configure an experiment.")
            return
        try:
            uploaded_data = pd.read_csv(uploaded_file)
        except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as error:
            st.error(f"Could not read this CSV: {error}")
            return
        if uploaded_data.empty or len(uploaded_data.columns) < 2:
            st.error("The CSV must contain at least one row and two columns.")
            return
        st.write(f"Loaded {len(uploaded_data):,} rows and {len(uploaded_data.columns)} columns.")
        target_default = (
            next(
                (
                    list(uploaded_data.columns).index(column)
                    for column in ("target", "animal")
                    if column in uploaded_data.columns
                ),
                len(uploaded_data.columns) - 1,
            )
        )
        target_column = st.selectbox(
            "Target (label) column",
            list(uploaded_data.columns),
            index=target_default,
        )
        candidate_features = [column for column in uploaded_data.columns if column != target_column]
        if not candidate_features:
            st.error("Select a target while leaving at least one feature column.")
            return
        default_intervention = (
            next(
                (
                    candidate_features.index(column)
                    for column in ("shortcut_signal", "background_shortcut")
                    if column in candidate_features
                ),
                0,
            )
        )
        intervention_column = st.selectbox(
            "Feature to investigate",
            candidate_features,
            index=default_intervention,
        )
        intervention_options = ["Shuffle feature values"]
        if pd.api.types.is_numeric_dtype(uploaded_data[intervention_column]):
            intervention_options.append("Reverse numeric feature")
        intervention = st.selectbox("Test intervention", intervention_options)
        synthetic_options = None
    else:
        target_column = "target"
        intervention_column = "shortcut_signal"
        intervention = st.selectbox(
            "Test intervention",
            ("Shuffle feature values", "Reverse numeric feature"),
            help="The intervention changes only the held-out feature values.",
        )
        with st.expander("Synthetic data controls", expanded=True):
            synthetic_rows = st.slider("Number of rows", 200, 5000, 1200, step=100)
            shortcut_strength = st.slider(
                "Shortcut correlation strength",
                0.0,
                1.0,
                0.9,
                step=0.05,
                help="0.5 is random; values closer to 1 make the shortcut agree more often with the target.",
            )
            synthetic_options = (synthetic_rows, shortcut_strength)

    test_fraction = st.slider("Held-out test fraction", 0.2, 0.4, 0.25, step=0.05)
    random_seed = st.number_input("Random seed", min_value=0, max_value=99999, value=42)
    selected_models = st.multiselect(
        "Classifiers to compare",
        MODEL_NAMES,
        default=["Logistic Regression", "Random Forest"],
    )

    if st.button("Run reliability experiment", type="primary"):
        if not selected_models:
            st.error("Select at least one classifier.")
            return
        if source == "Synthetic controlled dataset":
            assert synthetic_options is not None
            data = make_synthetic_data(
                synthetic_options[0], synthetic_options[1], int(random_seed)
            )
        else:
            assert uploaded_data is not None
            data = uploaded_data
        try:
            result = run_experiment(
                data=data,
                target_column=target_column,
                model_names=selected_models,
                test_fraction=test_fraction,
                random_seed=int(random_seed),
                intervention_column=intervention_column,
                intervention=intervention,
            )
        except ValueError as error:
            st.error(f"Experiment could not run: {error}")
            return
        st.session_state["merit_result"] = result
        st.success("Experiment complete. Open the Analysis tab to review results.")

    st.divider()
    st.subheader("Preprocessing and data-quality audit")
    st.write(
        "MERIT does not require pre-cleaning before an experiment. It reports the "
        "applied steps so data preparation is visible and repeatable."
    )
    if source == "Upload labelled CSV" and uploaded_data is not None:
        st.dataframe(
            pd.DataFrame(
                {
                    "Column": uploaded_data.columns,
                    "Data type": [str(dtype) for dtype in uploaded_data.dtypes],
                    "Missing values": [int(count) for count in uploaded_data.isna().sum()],
                    "Missing (%)": [
                        round(float(count) * 100 / len(uploaded_data), 2)
                        for count in uploaded_data.isna().sum()
                    ],
                    "Distinct values": [
                        int(uploaded_data[column].nunique(dropna=True))
                        for column in uploaded_data.columns
                    ],
                }
            ),
            hide_index=True,
            width="stretch",
        )
    else:
        st.write(
            "Synthetic data contains `genuine_signal` (continuous), "
            "`shortcut_signal` (binary), and `target` (binary). Its ground-truth "
            "shortcut strength is controlled above."
        )

    st.markdown(
        """
        **Applied when an experiment runs**
        1. Rows with a missing target are excluded; feature columns that are
           entirely empty are reported and excluded. Infinite numeric values are
           treated as missing.
        2. Numeric features receive median imputation followed by standardization.
           Categorical features receive most-frequent imputation and one-hot encoding;
           unseen test categories are ignored by the encoder.
        3. The target is never included as a feature. A stratified train/test split
           is used, and preprocessing is fitted on training rows only.
        4. The selected feature is shuffled or (for numeric features) reversed in
           the held-out test data. A second model is retrained without that feature
           as a simple ablation comparison.
        """
    )

    result = st.session_state.get("merit_result")
    if result:
        st.subheader("Export experiment")
        st.download_button(
            "Download results (JSON)",
            data=json.dumps(result, indent=2),
            file_name="merit_experiment_results.json",
            mime="application/json",
        )


def main() -> None:
    st.title("MERIT")
    st.caption("Model Evaluation and Reliability Investigation Toolkit")
    analysis_tab, tool_tab = st.tabs(["Analysis", "Project Tool"])
    with analysis_tab:
        render_analysis(st.session_state.get("merit_result"))
    with tool_tab:
        render_project_tool()


if __name__ == "__main__":
    main()
