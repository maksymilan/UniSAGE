# UniSAGE：Unifying Static and Dynamic Attributes with Hyper-Structure

This project introduces a framework for training Graph Neural Network models on complex, heterogeneous graphs by orthogonally collecting dynamic and static information. Compared to conventional methods that mix different information types, our approach avoids potential information loss that can occur during the fusion process. By preserving the distinct characteristics of dynamic and static features, the model can learn more robust and informative representations for predictive tasks.

This framework utilizes datasets from the [relbench](https://github.com/snap-stanford/relbench.git) benchmark, processes them through a multi-stage pipeline, and trains a `UniSAGE` model for downstream classification and regression tasks.

## File Structure

A brief overview of the key files and directories in this repository:

```
.
├── requirements.txt
├── torch_geo/
├── model/
│   ├── UniSAGE.py
│   └── SSAgg.py
├── data/
│   ├── generate_data/
│   │   ├── raw_to_json.py
│   │   ├── build_dataset.sh
│   │   └── data_sampling.py
│   ├── preprocess.py
│   └── preprocess.sh
├── train_classification.py
├── train_regression.py
├── run_classification.sh
├── run_regression.sh
└── README.md
```

## Installation

1.  **Clone the Repository**

    ```bash
    git clone <repo-url>
    ```

2.  **Install Dependencies**
    It is recommended to use a virtual environment (e.g., conda or venv).

    ```bash
    pip install -r requirements.txt
    pip install torch_geo/*.whl
    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
    ```

3.  **Download RelBench Data**
    This project depends on the datasets provided by **RelBench**. You need to install `relbench` and download all required data to the default cache location (`~/.cache/relbench/`). You can do this by running the following Python script:

    ```python
    # save this as download_data.py and run it
    from relbench.datasets import get_dataset

    dataset_names = [
        "rel-amazon",
        "rel-avito",
        "rel-event",
        "rel-f1",
        "rel-hm",
        "rel-stack",
        "rel-trial",
    ]

    for name in dataset_names:
        print(f"Downloading dataset and tasks for: {name}...")
        try:
            # The get_db() method downloads both the database tables and all associated task files.
            get_dataset(name).get_db()
            print(f"Successfully downloaded {name}.")
        except Exception as e:
            print(f"Failed to download {name}. Error: {e}")

    print("\nAll dataset downloads attempted.")
    ```

    To run the script:

    ```bash
    python download_data.py
    ```

## Workflow and Usage

The data processing and training pipeline is divided into four main stages, automated by executable shell scripts. **All scripts should be run from the project's root directory.**

### Stage 1: Build Graph Data (`.jsonl`)

This stage converts the raw relational tables from RelBench into nested JSONL graph structures. Each line in the output file represents the full graph for a single root entity (e.g., a user, a product).

**Command:**

```bash
cd data/generate_data
./build_dataset.sh
```

  - **Backend Script:** `raw_to_json.py`
  - **Description:** This script is highly optimized to process large tables efficiently. The shell script will automatically build all supported datasets.
  - **Output:** JSONL files will be saved to `~/relbench-data/[dataset_name]/`. After running, return to the root directory: `cd ../..`

### Stage 2: Sample Tasks

To accelerate development and experimentation, this stage creates smaller, sampled versions of the official RelBench tasks and filters the large JSONL graph files accordingly.

**Command:**

```bash
cd data/generate_data
python data_sampling.py
```

  - **Backend Script:** `data_sampling.py`
  - **Description:** The script reads the full task tables and the JSONL files from Stage 1 to produce sampled `.parquet` files and a much smaller `.jsonl` file containing only the relevant graphs.
  - **Configuration:** All task configurations (ID keys, sample sizes) are managed within the `TASKS_CONFIG` dictionary in `data_sampling.py`.
  - **Output:** Sampled datasets will be saved to `~/relbench-data-test/sampling/`. After running, return to the root directory: `cd ../..`

### Stage 3: Preprocess & Encode Data (`.pt`)

This is the final data preparation step, where the sampled JSONL graphs are converted into PyTorch Geometric `Data` objects. Node features are generated using a pre-trained sentence embedding model.

**Command:**

```bash
cd data
./preprocess.sh
```

  - **Backend Script:** `preprocess.py`
  - **Description:** The shell script automates running `preprocess.py` for every sampled task. It reads the sampled JSONL files, builds graph structures (`edge_index`, timestamps), encodes node features into embeddings, and saves the final list of `PyG.Data` objects.
  - **Output:** The final `graph_dataset.pt` files will be saved in a structured manner inside `~/relbench-data-pt/`. After running, return to the root directory: `cd ..`

### Stage 4: Train Models

With the data fully processed, you can now train the models for all classification and regression tasks.

#### Classification Tasks

**Command:**

```bash
./run_classification.sh
```

  - **Backend Script:** `train_classification.py`
  - **Description:** This script iterates through all predefined classification tasks, launching a separate training process for each. Logs for each run are saved automatically.

#### Regression Tasks

**Command:**

```bash
./run_regression.sh
```

  - **Backend Script:** `train_regression.py`
  - **Description:** This script iterates through all predefined regression tasks and trains the model, evaluating using Mean Absolute Error (MAE).

#### Script Configuration and Parameters

You can easily configure the training process by modifying the variables at the top of the shell scripts (`run_classification.sh`, `run_regression.sh`):

| Variable            | Description                                          | Default (Class.) | Default (Reg.) |
| ------------------- | ---------------------------------------------------- | ---------------- | -------------- |
| `TRAIN_SCRIPT`      | The Python script to execute.                        | `train_...py`    | `train_...py`  |
| `DATA_ROOT`         | Path to the directory containing `.pt` files.          | `~/...-pt`       | `~/...-pt`     |
| `TASK_DATA_ROOT`    | Path to the directory with sampled `.parquet` files.   | `~/...-sampling` | `~/...-sampling`|
| `CPU_CORES_TO_USE`  | CPU core range to bind the process to (e.g., "10-19"). | `"0-9"`          | `"10-19"`      |

These shell script variables are used to generate the final command. The Python training scripts (`train_*.py`) accept the following command-line arguments:

| Argument              | Description                                        | Type    | Default Value    |
| --------------------- | -------------------------------------------------- | ------- | ---------------- |
| `--processed_path`    | Path to the input `graph_dataset.pt` file.         | `str`   | **Required** |
| `--task_path`         | Path to the directory with task `.parquet` files.  | `str`   | **Required** |
| `--dataset_id_key`    | Entity ID key in the `.pt` file.                   | `str`   | **Required** |
| `--task_id_key`       | Entity ID key in the `.parquet` files.             | `str`   | **Required** |
| `--label_key`         | Label column name in the `.parquet` files.         | `str`   | **Required** |
| `--timestamp_key`     | Timestamp column name in the `.parquet` files.     | `str`   | **Required** |
| `--gpu_ids`           | GPU device ID to use for training.                 | `str`   | `'0'`            |
| `--node_threshold`    | Max nodes for a graph to run on GPU.               | `int`   | `50000`          |
| `--seed`              | Random seed for reproducibility.                   | `int`   | `42`             |
| `--epochs`            | Number of training epochs.                         | `int`   | `5`              |
| `--batch_size`        | Batch size for DataLoaders.                        | `int`   | `16`             |
| `--lr`                | Optimizer learning rate.                           | `float` | `0.001`          |
| `--hidden_channels`   | Number of hidden dimensions in the model.          | `int`   | `64`             |
| `--num_heads`         | Number of attention heads in `UniSAGE`.            | `int`   | `8` (Class) / `2` (Reg) |
| `--ssagg_lambda`      | Lambda for the SSAgg layer.                        | `float` | `1.5`            |
| `--dropout`           | Dropout rate.                                      | `float` | `0.5` (Class) / `0.3` (Reg) |
| `--orthogonal_lambda` | Lambda for the orthogonal loss component.          | `float` | `0.1`            |
