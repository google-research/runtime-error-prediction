# Static Prediction of Runtime Errors by Learning to Execute Programs with External Resource Descriptions

This is the repository for the paper [_Static Prediction of Runtime Errors by Learning to Execute Programs with External Resource Descriptions_](https://arxiv.org/abs/2203.03771),
and for the _Python Runtime Errors (PRE)_ dataset. Please cite this work as:

```text
@misc{bieber2022static,
    title={Static Prediction of Runtime Errors by Learning to Execute Programs with External Resource Descriptions},
    author={David Bieber and Rishab Goel and Daniel Zheng and Hugo Larochelle and Daniel Tarlow},
    year={2022},
    eprint={2203.03771},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## Overview

This repository contains the source code used by the paper _Static Prediction of Runtime Errors by Learning to Execute Programs with External Resource Descriptions_. This includes the [dataset construction](core/data), [models](core/models), [sweeps](core/distributed/sweeps.py), [training loop](core/lib/trainer.py), and [evaluation logic](core/lib/trainer.py).

We detail below how to perform common tasks using this repository, including dataset loading, replicating dataset construction, and replicating training and eval.

## Python Runtime Errors Dataset

The dataset is derived from the [Project CodeNet dataset](https://github.com/IBM/Project_CodeNet). We filter the CodeNet dataset to 2.4 million Python submissions, and augment each with a label indicating any runtime error the program encounters when it is run on a sample input. We detail the contents of our augmented version of the dataset below.

The datasets can be found at `gs://python-runtime-errors/datasets/project-codenet/2021-12-29` and `gs://python-runtime-errors/datasets/project-codenet/2021-12-29-nodoc`.

<details>
  <summary>Direct links to download the dataset</summary>

  #### 2021-12-29 (gs://python-runtime-errors/datasets/project-codenet/2021-12-29)

  In this version of the dataset, the resource descriptions are present in the source of each submission as a docstring at the beginning of the file.

  * [train-ids.json](https://storage.googleapis.com/python-runtime-errors/datasets/project-codenet/2021-12-29/train-ids.json) (76.4 MB)
  * [valid-ids.json](https://storage.googleapis.com/python-runtime-errors/datasets/project-codenet/2021-12-29/valid-ids.json) (8.4 MB)
  * [test-ids.json](https://storage.googleapis.com/python-runtime-errors/datasets/project-codenet/2021-12-29/test-ids.json) (8.4 MB)
  * [train.tfrecord](https://storage.googleapis.com/python-runtime-errors/datasets/project-codenet/2021-12-29/train.tfrecord) (3.5 GB)
  * [valid.tfrecord](https://storage.googleapis.com/python-runtime-errors/datasets/project-codenet/2021-12-29/valid.tfrecord) (371.4 MB)
  * [test.tfrecord](https://storage.googleapis.com/python-runtime-errors/datasets/project-codenet/2021-12-29/test.tfrecord) (365.6 MB)
  * [LICENSE](https://storage.googleapis.com/python-runtime-errors/datasets/project-codenet/LICENSE) (Community Data License Agreement - Permissive - Version 2.0)

  #### 2021-12-29-nodoc (gs://python-runtime-errors/datasets/project-codenet/2021-12-29-nodoc)

  In this version of the dataset, the source of each submission is tokenized without modification.

  * [train-ids.json](https://storage.googleapis.com/python-runtime-errors/datasets/project-codenet/2021-12-29-nodoc/train-ids.json) (76.4 MB)
  * [valid-ids.json](https://storage.googleapis.com/python-runtime-errors/datasets/project-codenet/2021-12-29-nodoc/valid-ids.json) (8.4 MB)
  * [test-ids.json](https://storage.googleapis.com/python-runtime-errors/datasets/project-codenet/2021-12-29-nodoc/test-ids.json) (8.4 MB)
  * [train.tfrecord](https://storage.googleapis.com/python-runtime-errors/datasets/project-codenet/2021-12-29-nodoc/train.tfrecord) (3.1 GB)
  * [valid.tfrecord](https://storage.googleapis.com/python-runtime-errors/datasets/project-codenet/2021-12-29-nodoc/valid.tfrecord) (331.9 MB)
  * [test.tfrecord](https://storage.googleapis.com/python-runtime-errors/datasets/project-codenet/2021-12-29-nodoc/test.tfrecord) (327.6 MB)
  * [LICENSE](https://storage.googleapis.com/python-runtime-errors/datasets/project-codenet/LICENSE) (Community Data License Agreement - Permissive - Version 2.0)

Both versions of the dataset use the [tokenizer vocabulary found here](out/tokenizers/train-docstrings-1000000.json).

The problem ids and submission ids match the submissions in the original [Project CodeNet dataset](https://developer.ibm.com/exchanges/data/all/project-codenet/).

</details>

### Loading the Dataset

The [data_io](core/data/data_io.py) library provides functionality `load_dataset` for loading dataset iterators.

### Dataset Description

We make the Python Runtime Errors dataset available as a TFRecord of TFExamples.
Each example represents a single Python 3 submission to a competitive programming problem. Each example is labelled with a target class indicating the error (if any) encountered by the submission when run on a sample input.

<details>
  <summary>Here are the fields in each Example:</summary>

- **tokens**: A list of integer indexes that map to the tokens in the submission. Taken together, these tokens comprise a submission from the original Project CodeNet dataset.
- **docstring_tokens**: A list of integer indexes that map to tokens in the "resource description" corresponding to the submission's problem id. The resource description is parsed from the problem statement from the original Project CodeNet dataset.
- **edge_sources**: A list of integers. Each integer is the index of a node in the control flow graph of the submission. Taken together with the corresponding index in the edge_dests feature, each integer represents a single edge in the control flow graph.
- **edge_dests**: A list of integers. Each integer is the index of a node in the control flow graph of the submission. Taken together with the corresponding index in the edge_sources feature, each integer represents a single edge in the control flow graph.
- **edge_types**: A list of integers. Each integer is one of 0..5 representing the edge type of the corresponding edge in the edge_sources and edge_dests lists. The edge types are "unconditional branch" (0 = forward, 1 = reverse), "true branch" (2 = forward, 3 = reverse), and "false branch" (4 = forward, 5 = reverse).
- **node_token_span_starts**: A list of integers. The nth item in the list is the index of the first token in tokens corresponding to the nth node in the control flow graph.
- **node_token_span_ends**: A list of integers. The nth item in the list is the index of the last token in tokens corresponding to the nth node in the control flow graph.
- **token_node_indexes**: A list of integers, one integer per token in tokens. The nth item in the list is the index of the last node in the control flow graph that contains token n in its token span.
- **true_branch_nodes**: A list of integers, one integer per node in the control flow graph. The nth item in the list is the index of the node in the control flow graph that control would go to, from node n, if the true branch (or unconditional branch, if applicable) were followed.
- **false_branch_nodes**: A list of integers, one integer per node in the control flow graph. The nth item in the list is the index of the node in the control flow graph that control would go to, from node n, if the false branch (or unconditional branch, if applicable) were followed.
- **raise_nodes**: A list of integers, one integer per node in the control flow graph. The nth item in the list is the index of the node in the control flow graph that control would go to, from node n, if the an exception were raised at that point.
- **start_index**: An integer. The index in the list of control flow nodes of the start node for the program.
- **exit_index**: An integer. The index in the list of control flow nodes of the exit node for the program. The raise node is assumed to be immediately after the exit node, in models that use a raise node.
- **step_limit**: An integer. The number of model steps permitted for the submission.
- **target**: An integer indicating the target error class for the submission. This is the error collected from running the submission on the provided input, or else no error (1) if no error occurred. A timeout of 1 second of execution was used. The error classes are:
  - 1: No Error
  - 2: Other
  - 3: Timeout
  - 4: AssertionError
  - 5: AttributeError
  - 6: decimal
  - 7: EOFError
  - 8: FileNotFoundError
  - 9: ImportError
  - 10: IndentationError
  - 11: IndexError
  - 12: KeyError
  - 13: MathDomainError
  - 14: MemoryError
  - 15: ModuleNotFoundError
  - 16: NameError
  - 17: OSError
  - 18: OverflowError
  - 19: re.error
  - 20: RecursionError
  - 21: RuntimeError
  - 22: StopIteration
  - 23: SyntaxError
  - 24: TabError
  - 25: TypeError
  - 26: UnboundLocalError
  - 27: ValueError
  - 28: ZeroDivisionError
  - 29: numpy.AxisError

- **target_lineno**: The line number at which the error occurred, or else 0 if no error occurred during the execution of the submission.
- **target_node_indexes**: A list of indices of all control flow nodes that are consistent with the target line number target_lineno.
- **num_target_nodes**: An integer. The number of elements in 'target_node_indexes'.
- **post_domination_matrix**: An n x n 0/1 matrix. A 1 at element i,j indicates that i is post-dominated by j. This means that any path from i to the exit necessarily passes through node j.
- **post_domination_matrix_shape**: A 2-tuple of integers representing the shape of the post domination matrix.
- **problem_id**: A string e.g. "p00001" indicating the problem id corresponding to the submission's problem in the original Project CodeNet dataset.
- **submission_id**: A string e.g. "s149981901" indicating the submission id of the submission in the original Project CodeNet dataset.
- **in_dataset**: A 0/1 value indicating whether the example is to be included in the dataset for training and evaluation purposes.
- **num_tokens**: An integer. The number of tokens in the submission. This is the length of the tokens list.
- **num_nodes**: An integer. The number of nodes in the control flow graph of the submission.
- **num_edges**: An integer. The number of edges in the control flow graph of the submission.

</details>

<details>
  <summary>The error classes in the dataset are:</summary>

- 1: No Error
- 2: Other
- 3: Timeout
- 4: AssertionError
- 5: AttributeError
- 6: decimal
- 7: EOFError
- 8: FileNotFoundError
- 9: ImportError
- 10: IndentationError
- 11: IndexError
- 12: KeyError
- 13: MathDomainError
- 14: MemoryError
- 15: ModuleNotFoundError
- 16: NameError
- 17: OSError
- 18: OverflowError
- 19: re.error
- 20: RecursionError
- 21: RuntimeError
- 22: StopIteration
- 23: SyntaxError
- 24: TabError
- 25: TypeError
- 26: UnboundLocalError
- 27: ValueError
- 28: ZeroDivisionError
- 29: numpy.AxisError
</details>

### Generating the dataset

First, ensure that [codenet_paths.py](core/data/codenet_paths.py) reflects the location of the raw Project CodeNet data.

<details>
  <summary>1. Divide the problems into splits</summary>

Our pre-generated splits are available in [out/splits/default.json](out/splits/default.json).

We generated these splits using the following script:

```bash
python -m core.data.splits make_and_save_splits --path=out/splits/example-splits.json
```
</details>

<details>
  <summary>2. Run all submissions</summary>

Run the following on a GCP virtual machine.
```bash
python -m core.data.process_codenet run_codenet_submissions
```

This will collect data to the `codenet_paths.EVALS_ROOT` directory.
</details>

<details>
  <summary>3. Generate vocabulary for tokenizer</summary>

The following script will use the raw Project CodeNet data to generate a vocabulary file for the HuggingFace BPE tokenizer.
You can also skip this step and use our [pre-generated vocabulary file here](out/tokenizers/train-docstrings-1000000.json).

```bash
# First parse the problem descriptions.
python -m core.data.process_codenet generate_docstrings
# Then generate a vocab suitable for the resource descriptions and the source code.
python -m core.data.process_codenet generate_tokenizer --path=out/tokenizers/example-tokenizer.json --splits_path=out/splits/example-splits.json
```

The vocabulary used in the paper is available at [out/tokenizers/train-docstrings-1000000.json](out/tokenizers/train-docstrings-1000000.json).
</details>

<details>
  <summary>4. Generate the dataset</summary>

The following command generates the complete dataset (`fraction=1.0`), with the resource descriptions included in the programs as docstrings (`include_docstring=True`).

```bash
python -m core.data.process_codenet generate_codenet_dataset --tokenizer_path=out/tokenizers/example-tokenizer.json --dataset_path=out/datasets/example-dataset --splits_path=out/splits/example-splits.json --include_docstrings=True --fraction=1.0
```

We used this command to produce the dataset available at `gs://python-runtime-errors/datasets/project-codenet/2021-12-29`.
We also ran this same command with `include_docstring=False` to produce `gs://python-runtime-errors/datasets/project-codenet/2021-12-29-nodoc`.
</details>

The pre-generated datasets are available at `gs://python-runtime-errors/datasets/project-codenet/2021-12-29` and `gs://python-runtime-errors/datasets/project-codenet/2021-12-29-nodoc`.


### Dataset License

The Python Runtime Errors dataset is made available under the [Community Data License Agreement - Permissive - Version 2.0](https://storage.googleapis.com/python-runtime-errors/datasets/project-codenet/LICENSE).

## Training

### Setup

Python dependencies are listed in [setup.py](setup.py).

We provide a setup script to install all dependencies and mount the Cloud Storage bucket containing the dataset at [scripts/setup-tpu.sh](scripts/setup-tpu.sh). If running on CPU or GPU, replace the first line of setup-tpu.sh with [the appropriate jax installation command for your system](https://github.com/google/jax#installation).

<details>
  <summary>Additional setup details</summary>

  Training works fine on a CPU, GPU, or TPU, but is likely prohibitively slow on CPU, and is fastest on a TPU.

  The first line of setup-tpu.sh installs jax for use on TPU; if instead running on CPU or GPU, use [the appropriate jax installation command for your system](https://github.com/google/jax#installation).

  To start a Cloud TPU:

  1. Set DEFAULT_PROJECT in [gcp.py](core/distributed/gcp.py) to your GCP project name.
  2. You can then run `python -m core.distributed.gcp tpu_up_n --n=1 --offset=0` to start n Cloud TPUs, billed to your GCP project. (`--offset` is used to determine the names and zones of the tpus; `--n` indicates how many TPUs to start). This requires that gcloud is set up locally.

  To mount the Python Runtime Errors dataset bucket, run the following:

  ```bash
  # Connect to GCS Bucket for data
  if [ ! -f /mnt/python-runtime-errors/README.md ]; then
    sudo mkdir -p /mnt/python-runtime-errors
    sudo chown $(whoami) /mnt/python-runtime-errors
    gcsfuse python-runtime-errors /mnt/python-runtime-errors/
  fi
  ```

  [setup-tpu.sh](scripts/setup-tpu.sh) will do this mounting for you if you run it on your Cloud TPU or any other GCP VM.

  You can then use `/mnt/python-runtime-errors/datasets/project-codenet/2021-12-29-nodoc` as the `--dataset_path` below in place of `/path/to/dataset`.
  This matches the default dataset path set in [codenet_paths.py](core/data/codenet_paths.py).
</details>

### Exception IPA-GNN

The following command starts training an Exception IPA-GNN using the selected hyperparameters in the _Static Prediction of Runtime Errors by Learning to Execute Programs with External Resource Descriptions_ paper.

```bash
python3 -m scripts.runner \
  --dataset_path=/path/to/dataset \
  --config.model_class=IPAGNN \
  --config.raise_in_ipagnn=True \
  --config.optimizer=sgd \
  --config.batch_size=32 \
  --config.learning_rate=0.3 \
  --config.grad_clip_value=0.5 \
  --config.hidden_size=128 \
  --config.span_encoding_method=max \
  --config.transformer_dropout_rate=0.1 \
  --config.transformer_attention_dropout_rate=0 \
  --config.permissive_node_embeddings=False \
  --config.transformer_emb_dim=512 \
  --config.transformer_num_heads=8 \
  --config.transformer_num_layers=6 \
  --config.transformer_qkv_dim=512 \
  --config.transformer_mlp_dim=2048 \
  --config.eval_freq=15000 \
  --config.eval_subsample=1 \
  --config.eval_max_batches=500 \
  --config.save_freq=50000 \
  --config.train_steps=500000 \
  --config.study_id=example-study \
  --config.experiment_id=1 \
  --config.run_id=exception-example-run
```

Default configuration values are available in [config/default.py](config/default.py).


### IPA-GNN

The following command starts training a vanilla IPA-GNN using the selected hyperparameters in the _Static Prediction of Runtime Errors by Learning to Execute Programs with External Resource Descriptions_ paper.

```bash
python3 -m scripts.runner \
  --dataset_path=/path/to/dataset \
  --config.model_class=IPAGNN \
  --config.raise_in_ipagnn=False \
  --config.optimizer=sgd \
  --config.batch_size=32 \
  --config.learning_rate=0.3 \
  --config.grad_clip_value=1 \
  --config.hidden_size=128 \
  --config.span_encoding_method=first \
  --config.transformer_dropout_rate=0 \
  --config.transformer_attention_dropout_rate=0.1 \
  --config.permissive_node_embeddings=False \
  --config.transformer_emb_dim=128 \
  --config.transformer_num_heads=2 \
  --config.transformer_num_layers=128 \
  --config.transformer_qkv_dim=4 \
  --config.transformer_mlp_dim=512 \
  --config.eval_freq=15000 \
  --config.eval_subsample=1 \
  --config.eval_max_batches=500 \
  --config.save_freq=50000 \
  --config.train_steps=500000 \
  --config.study_id=example-study \
  --config.experiment_id=1 \
  --config.run_id=ipagnn-example-run
```

Default configuration values are available in [config/default.py](config/default.py).


## Evaluation

Add the following three flags to your train command to evaluate a trained model on the validation (`--split=valid`) or test set (`--split=test`).

`--config.restore_checkpoint_dir=study_id/experiment_id/run_id/top-checkpoints --mode=test --split=test`

---

Disclaimer: This is not an official Google project.
