# MERIT

**Model Evaluation and Reliability Investigation Toolkit**  
*Testing when machine-learning models fail*

MERIT is a Streamlit research prototype for investigating whether a classifier's
performance changes when a feature relationship is disrupted. It pairs familiar
classification metrics with controlled tabular interventions and image
background-shift experiments so a strong score on one test distribution is not
mistaken for evidence of robustness.

The interface has two tabs:

- **Analysis** explains the research question, experiment design, metrics,
  limitations, and the latest run's comparisons, plots, and confusion matrices.
- **Project Tool** generates controlled data or accepts a labelled CSV, exposes
  the data-quality and preprocessing steps, or accepts an annotated image
  archive for a background study. Both workflows export experiment results as
  JSON.

## Research question and motivation

**How do the strength of spurious correlations and the choice of classifier
affect generalization, and which practical interventions improve robustness when
those correlations change?**

A classifier may learn a feature that happens to correlate with the label in
training data rather than a stable, meaningful relationship. Standard accuracy,
precision, recall, F1-score, and a confusion matrix describe predictions on the
evaluated sample; by themselves, they do not establish which relationships the
model relied on or whether performance will persist under distribution shift.

MERIT operationalizes a small, repeatable sensitivity experiment: measure a
baseline on held-out examples, alter one selected feature in those same examples,
and compare. It also retrains a model without the selected feature as a simple
ablation. The results are evidence about the tested setup—not a causal
attribution, proof of shortcut learning, or guarantee of real-world behavior.

## Quick start

Use Python 3.10 or later.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit. In **Project Tool**, choose the
synthetic tabular dataset or **Image background study**. The image workflow
includes a small, generated dog/wolf illustration dataset; it is synthetic,
not wildlife photography. You can also upload your own images and annotations.

## Project layout

```text
DSF_PROJECT/
├── app.py
├── image_study.py
├── scripts/
│   └── create_demo_datasets.py
├── data/
│   ├── image_demo/
│   └── tabular/
├── README.md
├── run_merit.bat
└── requirements.txt
```

The included demo assets can be regenerated with:

```powershell
python scripts\create_demo_datasets.py
```

On Windows, double-click `run_merit.bat` to start the local app, or run the
command below from the project folder:

```powershell
python -m streamlit run app.py
```

## Workflow

1. **Data input:** Generate synthetic binary-classification data or upload a
   labelled CSV. For an upload, select the target column and the feature to
   investigate.
2. **Data inspection:** Review types, missing counts, missing percentages, and
   distinct-value counts. Missing target rows and empty feature columns are
   explicitly accounted for.
3. **Preprocessing:** Split into training and held-out data, then fit a
   preprocessing pipeline on the training data only. Numeric fields are
   median-imputed and standardized. Categorical fields are most-frequent
   imputed and one-hot encoded. Infinite numeric values are treated as missing.
   The target is kept separate from input features. Unknown test categories are
   handled by ignoring their one-hot columns.
4. **Baseline:** Fit the selected classifiers on the same split and evaluate on
   the original held-out features.
5. **Intervention:** On the held-out set only, either shuffle the selected
   feature values between examples or reverse a numeric feature around the
   midpoint of its observed minimum and maximum. The fitted baseline model is
   evaluated again on these altered examples.
6. **Ablation:** For comparison, retrain another instance of each classifier
   using the training split without the selected feature, then evaluate on the
   original held-out data.
7. **Analysis and export:** Compare baseline, perturbed-test, and ablation
   metrics; review confusion matrices and the accuracy comparison; download a
   JSON record of the experiment settings, audit, and results.

## Image background study

The Project Tool includes three 160-image **generated illustration** datasets:
dogs versus wolves on snow/green-field backgrounds, waterbirds versus landbirds
on lake/forest backgrounds, and cows versus camels on grassland/desert
backgrounds. In each dataset, the animal silhouettes are an intended cue, and
the class-associated background is an intentionally strong shortcut. Both
backgrounds occur with each class, but each class has one common background.
These make the experiment immediately runnable without a separate download.
They are simple programmatic illustrations—not animal photographs, and not
scientific evidence about real animal recognition.

### Input files

Upload an image ZIP archive and an annotation CSV with these required columns:

```csv
image_path,label,background_group
images/dog_snow_001.jpg,dog,snow
images/wolf_snow_001.jpg,wolf,snow
images/dog_grass_001.jpg,dog,grass
images/wolf_grass_001.jpg,wolf,grass
```

`image_path` is relative to the ZIP root and must match a file in the archive.
Each image needs a class label and a manually assigned background group.
Include both classes in each background group. With the default 25% holdout, at
least four images per class/background combination are generally needed for
the stratified split; ten or more per combination is a better starting point.
The app validates image paths and archive size limits without extracting
untrusted ZIP paths to disk. A blank annotation CSV template can be downloaded
from the app.

### Image experiment steps

1. The app creates one stratified development/held-out split per
   class/background combination.
2. Using only development annotations, it identifies the most common
   background group for each class. It trains on those aligned development
   images, making the intended background shortcut explicit.
3. A frozen ImageNet-pretrained ResNet-18 converts images to features. A
   standardized logistic-regression linear probe is fitted on aligned training
   examples.
4. The same model is evaluated on held-out examples with the class's common
   background (**in-distribution**) and with alternate backgrounds (**shifted**).
5. MERIT reports weighted accuracy, precision, recall and F1, confusion
   matrices, and per-class/background accuracy. Results can be downloaded as
   JSON.

The first image run downloads pretrained ResNet-18 weights (about 45 MB) and
therefore needs network access. The model is CPU-only. The app limits the
archive to 200 MB, total uncompressed contents to 500 MB, each image to 20 MB,
each image to 40 megapixels, and the dataset to 2,000 images.

### Important design limitation

This is a **background-group generalization test**, not automatic
counterfactual image editing. Researchers label backgrounds, and MERIT evaluates
held-out images collected in alternate backgrounds. It does not segment an
animal, replace a background, or guarantee that the animal itself stays
identical across backgrounds. To make a stronger causal claim, create matched
counterfactual pairs (the same animal foreground on different backgrounds),
control duplicate/near-duplicate leakage, and document how images were sampled.

### Research image sources

For a research-based extension using real photographs, obtain and cite the
original dataset and follow its current terms:

- [WILDS benchmark datasets](https://wilds.stanford.edu/datasets/) describe
  real distribution-shift research data. **iWildCam** contains camera-trap
  animal images and explicitly studies changes across camera locations, which
  can include background and acquisition changes. It is not a pre-made
  dog-versus-wolf snow dataset; follow WILDS' download instructions and the
  dataset license.
- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
  provides real cat and dog images for a species/background study, not wolves.
  The dataset is associated with CC BY-SA 4.0; check the original terms and
  attribution requirements before redistributing any images.

No third-party research photos are bundled here. Real image corpora can be
large, have dataset-specific access or attribution rules, and may not provide
the background labels needed by this experiment. Keep source URLs, licenses,
and annotation provenance with any research dataset you use.

## Synthetic experiment

The generated dataset contains:

- `target`: a balanced binary class label.
- `genuine_signal`: a noisy numeric signal whose distribution depends on the
  target.
- `shortcut_signal`: a binary feature that agrees with the target according to
  the configured shortcut strength.

At strength `0.5`, the shortcut is uninformative; at `1.0`, it always agrees
with the target. Shuffling or reversing the shortcut in held-out data disrupts
that learned relationship while leaving the genuine signal unchanged. This
known construction helps explain the experiment; it does not mean every
uploaded-data feature is known to be spurious.

### Downloadable tabular teaching CSVs

`data/tabular/` contains four reproducible, 1,000-row examples with the same
balanced `animal` label and noisy `genuine_signal`:

| CSV | Shortcut column | Intended demonstration |
|---|---|---|
| `01_intended_signal_only.csv` | Absent | Establish a baseline with only the intended signal |
| `02_balanced_no_shortcut.csv` | 50% label agreement | Show an uninformative background cue |
| `03_strong_shortcut.csv` | 95% label agreement | Show a misleading cue that appears useful in-distribution |
| `04_reversed_shortcut.csv` | 5% label agreement | Show a cue whose relationship is mostly reversed |

Download them under **Project Tool → Tabular reliability study → Upload labelled
CSV**, then upload one CSV at a time. Choose `animal` as the target and
`background_shortcut` as the investigated feature when it exists. Shuffle the
held-out feature to compare the baseline and intervention scores. The CSVs are
generated examples, not real animal observations. Their recipe and seed are
documented in `data/tabular/README.txt` and `scripts/create_demo_datasets.py`.

## Models and evaluation

The prototype compares Logistic Regression, Decision Tree, Random Forest, and
K-Nearest Neighbors. Models share the same stratified split and random seed.
K-Nearest Neighbors uses at most five neighbors, capped by the number of
training examples.

Reported metrics are:

- **Accuracy:** fraction of all predictions that are correct.
- **Precision:** class-wise precision, averaged with class-frequency weights.
- **Recall:** class-wise recall, averaged with class-frequency weights.
- **F1-score:** class-wise harmonic mean of precision and recall, averaged with
  class-frequency weights.
- **Confusion matrix:** counts by actual and predicted class.

Weighted metrics provide a single multiclass-compatible summary and can mask
weak performance on rare classes. Inspect per-class data and confusion matrices
before drawing conclusions.

## Preprocessing and data handling details

MERIT deliberately surfaces preprocessing as part of the research workflow,
rather than treating it as invisible setup:

- The uploaded CSV is read as a table. The user chooses the label; all other
  columns are candidate features.
- Rows without a target are dropped and counted. A target with fewer than two
  classes is rejected.
- Numeric infinities become missing values. Entirely empty feature columns are
  excluded and listed in the experiment audit.
- Numeric columns are median-imputed and standardized. Categorical columns are
  imputed with the most frequent training value and one-hot encoded.
- The test data is never used to fit imputers, scalers, encoders, or classifiers.
- A stratified split is used; an experiment is rejected when there are too few
  examples in the smallest target class for stratification.

The data preview reports null counts before the experiment. The exported JSON
includes the effective feature groups, excluded empty columns, missing-value
counts, number of rows dropped for missing labels, split sizes, selected
intervention, class counts, and each model's metrics.

## Interpretation and limitations

- **Perturbation is not causal inference.** A performance change shows model
  sensitivity to the particular manipulation on the particular split; it does
  not show that the feature caused a prediction or is ethically/semantically
  spurious.
- **Feature permutation can create implausible examples.** Shuffling one column
  may break dependencies between features. Reversal is likewise a diagnostic
  stress test, not necessarily a realistic deployment scenario.
- **Ablation changes the learning problem.** A retrained model without a feature
  is a comparison, not a guaranteed mitigation. Other features may be
  correlated proxies.
- **A single split is uncertain.** A fixed seed makes a run repeatable, not
  statistically conclusive. Repeat with multiple seeds and representative data
  before making research or deployment claims.
- **Image background labels are researcher-provided.** Label errors or
  inconsistent background definitions affect conclusions. The automatic
  majority-background rule constructs a deliberately correlated training set;
  results do not directly measure a model trained on the full supplied data.
- **The image model is a linear probe.** ResNet-18 is pretrained on ImageNet and
  frozen; MERIT trains only a logistic-regression classifier on extracted
  features. This is a practical baseline, not a fully fine-tuned model.
- **Metrics have limits.** Weighted averages may hide minority-class failures;
  calibration, confidence intervals, subgroup evaluation, and operational
  decision costs are not yet modeled.
- **Data scope is limited.** MERIT does not infer a causal graph, discover
  spurious features automatically, handle time-series data, or validate the
  representativeness of uploaded datasets.

Use conclusions such as “the model was sensitive to shuffling feature X in this
held-out sample,” not “feature X caused the model to fail.”

## Technology

- Python
- Streamlit
- pandas and NumPy
- scikit-learn
- Matplotlib
- PyTorch and Torchvision (pretrained image feature extraction)

Install the pinned-compatible dependency ranges in `requirements.txt`.

## Deploying on Streamlit Community Cloud

The message **“The app’s code is not connected to a remote GitHub repository”**
is shown by the **Deploy** button when Streamlit Community Cloud cannot find a
GitHub repository for the current app. It is not a Python error, and changing
app code alone cannot connect it to GitHub.

For local use, close the Deploy dialog and run `run_merit.bat` (or
`python -m streamlit run app.py` from this folder). The Deploy button is not
needed for localhost.

To publish to Community Cloud:

1. Sign in to GitHub and create a repository. Community Cloud generally needs a
   repository it can access; a public repository is the simplest option.
2. Push the complete project, including `app.py`, `image_study.py`,
   `requirements.txt`, `data/`, and `scripts/`, to the repository's default
   branch.
3. In Streamlit Community Cloud, choose **Create app**, select that GitHub
   repository and branch, set the main file to `app.py`, and deploy.
4. Wait for dependency installation and app startup in the Cloud logs.

The image workflow adds large PyTorch dependencies and downloads pretrained
weights on its first image run, so check the deployment platform's current
memory and package-size limits. Never commit private datasets, credentials, or
API tokens to a public repository. If you have not created a GitHub repository
yet, the Deploy dialog is expected until you do.
