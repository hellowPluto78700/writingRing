# Acceleration Reconstruction Evaluation Helpers

*Developer Guide for the Reusable A/B/C/D Experiment Modules*

## Scope

This guide documents the reusable Python helper modules that support raw-acceleration and reconstructed-acceleration CNN experiments. It defines the module responsibilities, shared data contracts, principal public interfaces, experiment protocols, and artifact conventions.

> **Design principle:** Notebooks should define an experiment and present results. Dataset validation, model definition, training, embedding extraction, metrics, evaluation orchestration, configuration, and artifact I/O belong in reusable modules.

## Reusable modules

| **Module**        | **Primary role**                                        |
|-------------------|---------------------------------------------------------|
| **config.py**     | Experiment protocol and hyperparameters                 |
| **datasets.py**   | Loading, validation, splits, normalization, DataLoaders |
| **model.py**      | Mask-aware CNN and 256-D encoder                        |
| **training.py**   | Optimization, early stopping, split metrics             |
| **embedding.py**  | Embedding extraction and paired checks                  |
| **metrics.py**    | Retrieval, geometry, and clustering metrics             |
| **evaluation.py** | Full representation and paired evaluation               |
| **io.py**         | Checkpoint, result, and provenance serialization        |

# 1. Architecture and Shared Contracts

The package is organized as a directional pipeline. Lower layers are reusable and do not depend on notebook state; higher layers compose them into complete experiments.

```text
config.py
  |
  v
datasets.py --> model.py --> training.py
  |
  v
embedding.py
  |
  v
metrics.py
  |
  v
evaluation.py
  |
  v
io.py
```

> **Reference notebook:** experiment_B_reconstruction_frozen_cnn.ipynb demonstrates the strict frozen raw-CNN-on-reconstruction Experiment B workflow and can be used as an integration example.

## 1.1 Dataset source semantics

| **Source**         | **Meaning**                                               | **Typical experiments**  |
|--------------------|-----------------------------------------------------------|--------------------------|
| **raw**            | Acceleration channels 15:18 from paddedSpikeIMU           | A; B reference; D test   |
| **reconstruction** | Aligned \*\_padded_reconstructed_accel_m_s2.npy signal    | B query; C; D test       |
| **mixed**          | Each logical sample appears twice: raw and reconstruction | D2 training / validation |

## 1.2 DataLoader batch contract

All training and embedding code expects the following minimum batch interface:

```python
{
    "x": Tensor[B, 3, T],
    "label": Tensor[B],
    "valid_mask": Tensor[B, T],
    "valid_length": Tensor[B],
    "sample_id": list[str],
    "user": list[str],          # metadata; trainer ignores it
    "action": list[str],        # metadata; trainer ignores it
    "source": list[str],        # raw or reconstruction
}
```

- x is normalized acceleration with channel-first layout.
- `valid_mask` identifies the valid prefix of each right-padded segment.
- Padding is forced back to exact zero after normalization.
- `sample_id` is stable across raw/reconstruction single-domain datasets; mixed data appends ::raw or ::reconstruction.

## 1.3 Model and embedding contract

```python
logits, h = model(x, valid_mask=valid_mask)

logits.shape == (B, num_classes)
h.shape == (B, 256)
z = L2_normalize(h)
```

> **Compatibility invariant:** The CNN layer names and architecture_config() are intentionally stable so existing baseline `model_state_dict` checkpoints can be restored with `strict=True`.

## 1.4 Experiment protocol summary

| **Experiment** | **Train / reference**                   | **Validation** | **Test query**                              | **Normalization**             |
|----------------|-----------------------------------------|----------------|---------------------------------------------|-------------------------------|
| **A**          | Raw                                     | Raw            | Raw                                         | Fit on raw train              |
| **B**          | Raw reference; frozen raw-trained model | Raw reference  | Reconstruction                              | Load from baseline checkpoint |
| **C**          | Reconstruction                          | Reconstruction | Reconstruction                              | Fit on reconstruction train   |
| **D2**         | Mixed raw + reconstruction              | Mixed          | Raw and reconstruction evaluated separately | Fit on mixed train            |

# 2. config.py - Experiment Protocol Configuration

`config.py` is declarative. It describes what an experiment should do but does not load data, train models, or write artifacts. The preset constructors make A/B/C/D protocols explicit and reduce accidental hyperparameter or domain mismatches.

## 2.1 Core dataclasses

| **Type**              | **Responsibility**           | **Important fields**                                                                                                  |
|-----------------------|------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| **UserSplitConfig**   | User-disjoint split policy   | train_fraction, val_fraction, explicit user lists, label coverage requirements                                        |
| **DataLoaderConfig**  | Batch loading policy         | batch_size, num_workers, pin_memory, persistent_workers                                                               |
| **CNNTrainingConfig** | CNN optimization policy      | epochs, lr, weight decay, class weights, patience, optional batch/gradient limits                                     |
| **ExperimentConfig**  | Complete experiment protocol | sources, normalization source, training flag, seed, split/loader/training/evaluation configs, checkpoint/output paths |

## 2.2 Preset constructors

**`experiment_a_config(...)`**

> Creates the raw -> raw baseline protocol.
>
> **Returns:** ExperimentConfig(name="A_raw", training_enabled=True)

**`experiment_b_config(baseline_checkpoint=...)`**

> Creates the frozen raw-trained CNN -> reconstruction query protocol. Normalization source is "checkpoint" and training is disabled.
>
> **Returns:** ExperimentConfig

**`experiment_c_config(...)`**

> Creates reconstruction train/validation/test with reconstruction-train normalization.
>
> **Returns:** ExperimentConfig

**`experiment_d_config(test_source="raw" \| "reconstruction")`**

> Creates D2 mixed training/validation and selects the domain used for evaluation. The same trained model should be evaluated once per test domain.
>
> **Returns:** ExperimentConfig

## 2.3 Recommended usage

```python
from accel_reconstruction_eval.config import experiment_c_config

config = experiment_c_config(
    output_dir="outputs/experiments/C_reconstruction",
    random_seed=12345,
)
config.validate()
```

# 3. datasets.py - Dataset Loading, Validation, Splits, and Normalization

`datasets.py` is the boundary between repository artifacts and model-ready tensors. It validates canonical padded packages, optionally validates reconstruction alignment, constructs a global sample manifest, assigns user-disjoint splits, fits normalization on valid samples only, and creates loaders with a uniform batch contract.

## 3.1 Important constants and artifacts

| **Item**                           | **Definition**                                                            |
|------------------------------------|---------------------------------------------------------------------------|
| **ACCELERATION_SLICE**             | slice(15, 18) from paddedSpikeIMU                                         |
| **RECONSTRUCTION_SUFFIX**          | \*\_padded_reconstructed_accel_m_s2.npy                                   |
| **RECONSTRUCTION_METADATA_SUFFIX** | \*\_padded_reconstructed_accel_metadata.json                              |
| **NormalizationStats**             | 3-channel mean/std + valid_time_points + fitted_on provenance             |
| **LoadedAccelerationData**         | one or more padded roots + metadata + packages + logical sample manifest  |
| **SplitAssignment**                | manifest with split/label_idx plus users, class maps, and split summaries |

## 3.2 Main public interfaces

**`load_acceleration_data(root, repository_root=None, require_reconstruction=False)`**

> Resolves one padded root or a non-empty sequence of padded roots, validates
> producer metadata and every padded package, memory-maps arrays, and builds
> one aligned logical sample manifest. Multiple roots stay physically separate;
> incompatible metadata, duplicate roots, duplicate `(user, action)` packages,
> or duplicate sample IDs fail explicitly.
>
> **Returns:** LoadedAccelerationData

For a two-action A/B/C/D run, pass the Action 0 and Action 1 combination roots
as an ordered list. The loader requires identical fixed-length batching
contracts, including target length, sampling rate, feature schema, channel
count, and right-padding convention. Experiment A persists the selected action
set and a path-independent canonical sample-ID digest in its checkpoint. B/C/D
use that identity in addition to split/class metadata; changing roots or the
selected sample cohort requires a new A checkpoint. A legacy checkpoint can
only be used with one root.

**`prepare_user_disjoint_splits(sample_manifest, ...)`**

> Creates or validates train/validation/test user assignments, checks leakage and optional label coverage, adds split and label_idx columns, and builds class mappings.
>
> **Returns:** SplitAssignment

**`fit_acceleration_normalization(packages, sample_manifest, split="train", source=...)`**

> Fits per-channel mean/std using only valid time points. For source="mixed", both raw and reconstruction are accumulated.
>
> **Returns:** `NormalizationStats`

**`build_dataset(..., split, source, normalization)`**

> Builds a raw, reconstruction, or D2 mixed Dataset.
>
> **Returns:** torch.utils.data.Dataset

**`build_split_loaders(..., train_source, val_source, test_source, normalization, ...)`**

> Creates the exact loaders expected by `training.py` and `embedding.py`.
>
> **Returns:** dict with train, train_eval, val, and test loaders

## 3.3 Normalization rules

- Statistics are fitted from valid (non-padding) time points only.
- The normalization source is part of the experiment protocol and must be recorded.
- Experiment B must load the raw-training normalization from the baseline checkpoint; it must not refit statistics on reconstruction.
- After normalization, invalid right-padding positions are reset to exactly zero.

## 3.4 D2 mixed dataset behavior

```text
logical sample i
  -> sample_i::raw
  -> sample_i::reconstruction

len(MixedAccelerationDataset) == 2 * logical_sample_count
```

This protocol changes the training distribution without changing the character labels or user-level split.

# 4. model.py - Mask-Aware Acceleration CNN

`model.py` defines the reusable acceleration-only classifier and its mask-aware temporal pooling behavior. Padding is removed from feature maps after every convolutional block, and valid lengths are propagated through the Conv1d output-length formula.

## 4.1 Public interfaces

**`MaskAwareAccelerationCNN(num_classes)`**

> Creates the 3-channel Conv1d classifier with a 256-D pre-classifier representation.
>
> **Returns:** nn.Module

**`model.forward(x, valid_mask=...)`**

> Runs the full encoder and classifier.
>
> **Returns:** (logits, embedding)

**`model.encode(x, valid_mask=...)`**

> Runs the encoder only.
>
> **Returns:** embedding with shape (B, 256)

**`model.classify_embedding(embedding)`**

> Applies the existing linear classifier to a 256-D embedding.
>
> **Returns:** logits with shape (B, C)

**`model.architecture_config()`**

> Returns the architecture descriptor stored in checkpoints.
>
> **Returns:** dict

## 4.2 Architecture

| **Block**                         | **Channels** | **Kernel / stride / padding** |
|-----------------------------------|--------------|-------------------------------|
| **Conv1 + BN + ReLU**             | 3 -\> 64     | 7 / 1 / 3                     |
| **Conv2 + BN + ReLU**             | 64 -\> 128   | 5 / 2 / 2                     |
| **Conv3 + BN + ReLU**             | 128 -\> 256  | 5 / 2 / 2                     |
| **Conv4 + BN + ReLU**             | 256 -\> 256  | 3 / 1 / 1                     |
| **Masked global average pooling** | 256          | Valid propagated prefix only  |
| **Linear classifier**             | 256 -\> C    | Character logits              |

# 5. training.py - CNN Training and Split Evaluation

`training.py` contains domain-agnostic CNN optimization. It only depends on the batch contract, not on whether the samples are raw, reconstructed, or mixed.

## 5.1 Main public interfaces

**`build_cross_entropy(device=..., train_labels=None, num_classes=None, use_class_weights=False)`**

> Builds unweighted cross-entropy by default or inverse-frequency balanced class weights when requested.
>
> **Returns:** nn.CrossEntropyLoss

**`build_adam_optimizer(model, learning_rate=5e-4, weight_decay=0.0)`**

> Builds the baseline Adam optimizer.
>
> **Returns:** torch.optim.Adam

**`run_cnn_epoch(model, loader, criterion, device=..., optimizer=None, max_batches=None, grad_clip_norm=None)`**

> Runs a train epoch when optimizer is provided; otherwise runs evaluation.
>
> **Returns:** dict with loss, accuracy, balanced_accuracy, macro_f1

**`fit_cnn(model, train_loader=..., val_loader=..., ..., num_epochs, patience, ...)`**

> Trains with early stopping and selects the best state using validation balanced accuracy by default.
>
> **Returns:** (history DataFrame, best_state, best_epoch, best_metric)

**`evaluate_cnn_splits(model, loaders, criterion, device=...)`**

> Evaluates named loaders using the same classification metric implementation.
>
> **Returns:** DataFrame indexed by split

> **Moduleization change:** Former notebook globals for gradient clipping and training-batch limits are explicit optional parameters. Defaults preserve the previous full-training behavior.

# 6. embedding.py - Representation Extraction and Paired Embeddings

`embedding.py` converts model outputs into a typed representation bundle while preserving dictionary-style access used by the original notebooks.

## 6.1 EmbeddingBundle

| **Field**     | **Shape / type** | **Meaning**                                          |
|---------------|------------------|------------------------------------------------------|
| **h**         | (N, 256)         | Raw pre-classifier embedding                         |
| **z**         | (N, 256)         | Row-wise L2-normalized embedding for cosine geometry |
| **y**         | (N,)             | Integer labels                                       |
| **cnn_pred**  | (N,)             | CNN argmax predictions                               |
| **logits**    | (N, C)           | Classifier logits                                    |
| **sample_id** | (N,)             | Stable sample identity                               |

Both attribute and mapping access are supported: bundle.h and bundle\["h"\] are equivalent.

## 6.2 Main public interfaces

**`extract_embeddings(model, loader, device=...)`**

> Extracts h, logits, predictions, labels, sample IDs, and L2-normalized z.
>
> **Returns:** `EmbeddingBundle`

**`extract_embedding_splits(model, loaders, device=...)`**

> Runs extraction for named loaders.
>
> **Returns:** dict\[str, `EmbeddingBundle`\]

**`standardize_feature_splits(train_x, val_x, test_x)`**

> Fits linear-probe feature standardization on train only and applies it to all splits.
>
> **Returns:** standardized train/val/test + train mean/std

**`validate_paired_bundles(reference, query)`**

> Checks identical shape, labels, and sample ordering for raw/reconstruction paired analyses.
>
> **Returns:** None; raises on mismatch

**`paired_embedding_cosine(reference, query)`**

> Computes per-sample cosine similarity and distance between aligned z embeddings.
>
> **Returns:** (similarity, distance) arrays

# 7. metrics.py - Reusable Representation Metrics

`metrics.py` contains individual, mostly stateless metric implementations. `evaluation.py` composes these functions into a reproducible protocol.

| **Metric family**          | **Public functions**                                                 | **Purpose**                                    |
|----------------------------|----------------------------------------------------------------------|------------------------------------------------|
| **Classification helpers** | confusion_matrix_numpy, macro_average_by_class                       | Basic diagnostics and aggregation              |
| **Cosine retrieval / kNN** | cosine_topk, knn_predict_from_topk, select_knn_k                     | Raw-train gallery neighborhood readout         |
| **Neighborhood purity**    | same_label_at_k                                                      | Local label purity at K                        |
| **Retrieval**              | retrieval_map                                                        | Micro and macro mean average precision         |
| **Prototype geometry**     | class_centroids, prototype_classification                            | Nearest class-centroid readout                 |
| **Class separation**       | macro_intra_cosine_distance, centroid_inter_cosine_distance          | D_intra, D_inter, and derived separation ratio |
| **KMeans / clustering**    | kmeans_best_of_n, normalized_mutual_information, adjusted_rand_index | Test-only global label/cluster alignment       |
| **Silhouette**             | cosine_silhouette                                                    | Per-sample, micro, and macro cosine silhouette |
| **Visualization support**  | balanced_subsample_indices                                           | Class-balanced sampling for PCA/plots          |

> **Interpretation rule:** D_inter / D_intra should never be interpreted alone. Inspect D_intra and D_inter individually to rule out embedding collapse or global contraction.

# 8. evaluation.py - Full Representation Evaluation

`evaluation.py` is the orchestration layer. It freezes the protocol around the train/reference, validation, and test/query bundles so A, B, C, and D use the same metric implementation.

## 8.1 Configuration and result objects

| **Type**                           | **Role**                                                                                                         |
|------------------------------------|------------------------------------------------------------------------------------------------------------------|
| **LinearProbeConfig**              | Linear-probe epochs, optimizer settings, batch size, patience, seed offset                                       |
| **RepresentationEvaluationConfig** | K values, kNN candidates, retrieval batch size, KMeans/silhouette settings, seed                                 |
| **RepresentationEvaluationResult** | Summary table plus predictions, retrieval tables, geometry tables, probe state/history, KMeans/silhouette arrays |
| **PairedPreservationResult**       | Raw/reconstruction paired similarity, prediction transitions, per-class preservation                             |

## 8.2 Primary evaluation entry point

```python
result = evaluate_representation(
    train_bundle=train_embeddings,
    val_bundle=val_embeddings,
    test_bundle=test_embeddings,
    num_classes=len(class_to_idx),
    device=DEVICE,
    idx_to_class=idx_to_class,
)
```

The returned summary includes:

- CNN accuracy, balanced accuracy, and macro F1.
- Validation-selected kNN K and test kNN metrics.
- Prototype classifier metrics.
- Raw-train-fitted linear-probe metrics.
- Micro/macro mAP and SameLabel@1.
- D_intra, D_inter, and D_inter/D_intra.
- KMeans NMI, ARI, and cosine silhouette.

## 8.3 Experiment B cross-domain behavior

```python
result_B = evaluate_representation(
    train_bundle=raw_train,
    val_bundle=raw_val,
    test_bundle=recon_test,
    num_classes=len(class_to_idx),
    device=DEVICE,
    idx_to_class=idx_to_class,
)
```

- kNN gallery remains raw train.
- kNN K is selected using raw validation.
- Prototype centroids come from raw train.
- Linear probe is trained on raw train and selected on raw validation.
- Only the final query/test representation is reconstructed.

## 8.4 Paired preservation analysis

```python
paired = evaluate_paired_preservation(
    raw_test,
    recon_test,
    idx_to_class=idx_to_class,
)
```

Outputs include mean/median embedding cosine similarity and distance, prediction agreement, per-class preservation, and four correctness transitions: correct->correct, correct->wrong, wrong->correct, and wrong->wrong.

# 9. io.py - Checkpoints, Evaluation Artifacts, and Provenance

`io.py` provides atomic writes and a common artifact schema. New experiment checkpoints preserve the keys expected by the existing raw baseline while adding experiment source and provenance information.

## 9.1 Checkpoint interfaces

**`build_experiment_checkpoint(...)`**

> Builds schema-versioned checkpoint metadata including model state/config, class mapping, normalization provenance, user splits, best epoch/validation score, experiment config, and optional producer metadata.
>
> **Returns:** dict

**`save_checkpoint(path, checkpoint)`**

> Atomically writes a torch checkpoint.
>
> **Returns:** Path

**`load_checkpoint(path, map_location="cpu")`**

> Loads and validates both new experiment checkpoints and the legacy acceleration CNN representation checkpoint type.
>
> **Returns:** dict

**`restore_model_from_checkpoint(model, checkpoint, strict=True, device=None, freeze=False)`**

> Restores model weights and optionally freezes the network for Experiment B.
>
> **Returns:** model

**`normalization_from_checkpoint(checkpoint)`**

> Reconstructs a `NormalizationStats` object from checkpoint fields.
>
> **Returns:** `NormalizationStats`

## 9.2 Result serialization

| **Interface**                        | **Artifacts**                                                                                                                                     |
|--------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| **save_embedding_bundle()**          | Compressed NPZ containing h, z, y, cnn_pred, logits, sample_id                                                                                    |
| **save_representation_evaluation()** | Summary, kNN selection, SameLabel@K, retrieval, geometry, per-class intra distance, probe history, unsupervised summary, arrays, probe checkpoint |
| **save_paired_preservation()**       | Paired summary, transitions, per-class table, cosine similarity/distance arrays                                                                   |
| **save_provenance()**                | JSON-safe provenance payload                                                                                                                      |
| **load_summary_csv()**               | Loads a saved summary for final experiment comparison                                                                                             |

## 9.3 Core checkpoint fields

```python
{
    "model_state_dict": ...,
    "model_config": ...,
    "class_to_idx": ...,
    "normalization_mean": [x, y, z],
    "normalization_std": [x, y, z],
    "normalization_fitted_on": "raw:train" | "reconstruction:train" | "mixed:train" | ...,
    "train_users": [...],
    "val_users": [...],
    "test_users": [...],
    "best_epoch": ...,
    "best_val_balanced_accuracy": ...,
    "experiment_config": {...},
}
```

# 10. Recommended Experiment Workflows

## 10.1 Common setup

1. Create or load an ExperimentConfig.
2. Load and validate acceleration packages with load_acceleration_data().
3. Prepare the user-disjoint split and class mapping.
4. Fit normalization from the protocol-defined training source, or load it from a checkpoint for B.
5. Build train/train_eval/validation/test loaders.
6. Train or restore the CNN.
7. Extract train/validation/test embedding bundles.
8. Run evaluate_representation().
9. Save checkpoint, embeddings, evaluation tables/arrays, and provenance.

## 10.2 Experiment A - Raw baseline

```python
config = experiment_a_config(...)
# raw train / raw val / raw test
# normalization fitted on raw train
# train model from scratch
result_A = evaluate_representation(
    train_bundle=raw_train,
    val_bundle=raw_val,
    test_bundle=raw_test,
    ...,
)
```

## 10.3 Experiment B - Frozen raw model on reconstruction

```python
config = experiment_b_config(baseline_checkpoint=...)
checkpoint = load_checkpoint(config.baseline_checkpoint)
normalization = normalization_from_checkpoint(checkpoint)
restore_model_from_checkpoint(model, checkpoint, freeze=True, device=DEVICE)

result_B = evaluate_representation(
    train_bundle=raw_train,
    val_bundle=raw_val,
    test_bundle=recon_test,
    ...,
)
paired_B = evaluate_paired_preservation(raw_test, recon_test, ...)
```

> **Interpretation:** A drop in B alone cannot distinguish task-information loss from raw-to-reconstruction domain shift. Compare B with C.

## 10.4 Experiment C - Train and test on reconstruction

```python
config = experiment_c_config(...)
# reconstruction train / val / test
# normalization fitted on reconstruction train
# same CNN architecture, trained from scratch
result_C = evaluate_representation(
    train_bundle=recon_train,
    val_bundle=recon_val,
    test_bundle=recon_test,
    ...,
)
```

## 10.5 Experiment D2 - Mixed-domain training

```python
config = experiment_d_config(test_source="raw")
# mixed raw+reconstruction train/val
# mixed-train normalization
# train one model

result_D_raw = evaluate_representation(..., test_bundle=raw_test, ...)
result_D_recon = evaluate_representation(..., test_bundle=recon_test, ...)
```

The same trained checkpoint should be used for both D test domains to measure domain invariance rather than training two different models.

# 11. Result Interpretation Matrix

| **Observed pattern**                                          | **Primary interpretation**                                                                         |
|---------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| **A high; B low; C approximately A**                          | Reconstruction retains recoverable task information, but the raw-trained CNN suffers domain shift. |
| **A high; B low; C also low**                                 | Reconstruction likely removes or distorts information needed for character classification.         |
| **C \> A with improved geometry**                             | Reconstruction may act as a useful denoising or information-bottleneck transform.                  |
| **D \> A on held-out users**                                  | Mixed-domain exposure may improve cross-user/domain invariance.                                    |
| **D strong on raw but weak on reconstruction, or vice versa** | Mixed training has not achieved balanced domain invariance.                                        |

## 11.1 Geometry indicators

| **Metric**                            | **Preferred direction** | **What it indicates**                                             |
|---------------------------------------|-------------------------|-------------------------------------------------------------------|
| **D_intra**                           | Down                    | Tighter same-label embeddings                                     |
| **D_inter**                           | Maintain or up          | Greater centroid separation                                       |
| **D_inter / D_intra**                 | Up                      | Improved relative class separation; inspect components separately |
| **Silhouette**                        | Up                      | Samples align better with their own labels than competing labels  |
| **SameLabel@K**                       | Up                      | Cleaner local neighborhoods                                       |
| **mAP**                               | Up                      | Better same-label retrieval ranking                               |
| **kNN / prototype / linear probe BA** | Up                      | Improved recoverable label structure under different readouts     |
| **NMI / ARI**                         | Up                      | Better global cluster-label alignment                             |

# 12. Package Placement and Integration

Recommended package layout:

```text
snn/
└── accel_reconstruction_eval/
    ├── __init__.py
    ├── config.py
    ├── datasets.py
    ├── model.py
    ├── training.py
    ├── embedding.py
    ├── metrics.py
    ├── evaluation.py
    └── io.py
```

## 12.1 Dependency expectations

- Core runtime: Python, NumPy, pandas, PyTorch.
- Repository integration: snn.action0_dataset, snn.action0_engine, and writingring.segment_padding are used when available.
- Several modules include limited direct-module fallbacks for notebook/testing portability; repository execution remains the intended environment.

## 12.2 Non-negotiable experiment invariants

- Use identical user splits and `class_to_idx` mappings across A/B/C/D.
- Experiment B: freeze the raw-trained checkpoint and reuse its normalization.
- Use the same metric implementation and evaluation configuration across conditions.
- Preserve raw/reconstruction `sample_id`, label, `valid_length`, and `valid_mask` alignment.
- For causal reconstruction-geometry comparisons, use the same fixed feature extractor.
- Record normalization provenance, source domains, seed, user split, and checkpoint identity.

## 12.3 Minimal notebook responsibility

After modularization, a notebook should primarily:

- Select a dataset root and experiment configuration.
- Call the reusable load/train/extract/evaluate/save interfaces.
- Display tables and figures.
- Add experiment-specific interpretation.

> **Target state:** Experiment notebooks should be thin and auditable; the reusable package should contain the implementation details that must stay identical across A/B/C/D.
