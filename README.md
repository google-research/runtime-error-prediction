# Static Prediction of Runtime Errors by Learning to Execute Programs with External Resource Descriptions

This is the repository for the paper _Static Prediction of Runtime Errors by Learning to Execute Programs with External Resource Descriptions_,
and for the _Python Runtime Errors (PRE)_ dataset. Please cite this work as:

```text
@misc{bieber2022static,
      title={Static Prediction of Runtime Errors by Learning to Execute Programs with External Resource Descriptions},
      author={David Bieber and Rishab Goel and Daniel Zheng and Hugo Larochelle and Daniel Tarlow},
      year={2022},
}
```

## Overview

This repository contains the source code used by the paper _Static Prediction of Runtime Errors by Learning to Execute Programs with External Resource Descriptions_. This includes the [dataset construction](), [models](), [sweeps](), and [training loop](), and [evaluation logic]().

We detail below how to perform common tasks using this repository, including dataset loading, replicating dataset construction, and replicating training and eval.

## Python Runtime Errors Dataset

The dataset is derived from the [Project CodeNet dataset](https://github.com/IBM/Project_CodeNet). We filter the CodeNet dataset to 2.4 million Python submissions, and augment each with a label indicating any runtime error the program encounters when it is run on a sample input. We detail the contents of our augmented version of the dataset below.

### Loading the Dataset

The [data_io](core/data/data_io.py) library provides functionality `load_dataset` for loading dataset iterators.

### Dataset Description

We make the Python Runtime Errors dataset available as a TFRecord of TFExamples.

Here are the fields in each Example:

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

## Training

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



Disclaimer: This is not an official Google project.
