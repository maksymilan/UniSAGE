# ==============================================================================
#           Automated Training Script for Classification Tasks
#
# Description:
#   This script automates the training for all predefined classification tasks.
#   It now binds the training process to specific CPU cores to manage resources.
#
# Usage:
#   ./run_training_classification.sh
# ==============================================================================

# --- Configuration ---

# Path to your Python script for classification training.
TRAIN_SCRIPT="train_classification.py"

# Root directory for all preprocessed and sampled data.
DATA_ROOT="${HOME}/relbench-data-pt"
TASK_DATA_ROOT="${HOME}/relbench-data-test/sampling"

CPU_CORES_TO_USE="0-9"

TASKS=(
    "rel-amazon/tasks/user-churn"
    "rel-amazon/tasks/item-churn" 
    "rel-avito/tasks/user-clicks"
    "rel-avito/tasks/user-visits"
    "rel-event/tasks/user-repeat"
    "rel-event/tasks/user-ignore"
    "rel-f1/tasks/driver-dnf"
    "rel-f1/tasks/driver-top3"
    "rel-hm/tasks/user-churn"
    "rel-stack/tasks/user-engagement"
    "rel-stack/tasks/user-badge"
    "rel-trial/tasks/study-outcome"
)

# --- Task Metadata Configuration ---

# DATASET_ID_KEY: The entity ID key used in the preprocessed .pt graph files.
declare -A DATASET_ID_KEYS=(
    ["rel-amazon/tasks/user-churn"]="customer_id"
    ["rel-amazon/tasks/item-churn"]="product_id" 
    ["rel-avito/tasks/user-clicks"]="UserID"
    ["rel-avito/tasks/user-visits"]="UserID"
    ["rel-event/tasks/user-repeat"]="user_id"
    ["rel-event/tasks/user-ignore"]="user_id"
    ["rel-f1/tasks/driver-dnf"]="driverId"
    ["rel-f1/tasks/driver-top3"]="driverId"
    ["rel-hm/tasks/user-churn"]="customer_id"
    ["rel-stack/tasks/user-engagement"]="UserId"
    ["rel-stack/tasks/user-badge"]="UserId"
    ["rel-trial/tasks/study-outcome"]="nct_id"
)

# TASK_ID_KEY: The entity ID key used in the task's .parquet files.
declare -A TASK_ID_KEYS=(
    ["rel-amazon/tasks/user-churn"]="customer_id"
    ["rel-amazon/tasks/item-churn"]="product_id" 
    ["rel-avito/tasks/user-clicks"]="UserID"
    ["rel-avito/tasks/user-visits"]="UserID"
    ["rel-event/tasks/user-repeat"]="user"
    ["rel-event/tasks/user-ignore"]="user"
    ["rel-f1/tasks/driver-dnf"]="driverId"
    ["rel-f1/tasks/driver-top3"]="driverId"
    ["rel-hm/tasks/user-churn"]="customer_id"
    ["rel-stack/tasks/user-engagement"]="OwnerUserId"
    ["rel-stack/tasks/user-badge"]="UserId"
    ["rel-trial/tasks/study-outcome"]="nct_id"
)

# LABEL_KEY: The column name of the target label in the task's .parquet files.
declare -A LABEL_KEYS=(
    ["rel-amazon/tasks/user-churn"]="churn"
    ["rel-amazon/tasks/item-churn"]="churn" 
    ["rel-avito/tasks/user-clicks"]="num_click"
    ["rel-avito/tasks/user-visits"]="num_click"
    ["rel-event/tasks/user-repeat"]="target"
    ["rel-event/tasks/user-ignore"]="target"
    ["rel-f1/tasks/driver-dnf"]="did_not_finish"
    ["rel-f1/tasks/driver-top3"]="qualifying"
    ["rel-hm/tasks/user-churn"]="churn"
    ["rel-stack/tasks/user-engagement"]="contribution"
    ["rel-stack/tasks/user-badge"]="WillGetBadge"
    ["rel-trial/tasks/study-outcome"]="outcome"
)

# TIMESTAMP_KEY: The column name of the timestamp in the task's .parquet files.
declare -A TIMESTAMP_KEYS=(
    ["rel-amazon/tasks/user-churn"]="timestamp"
    ["rel-amazon/tasks/item-churn"]="timestamp" 
    ["rel-avito/tasks/user-clicks"]="timestamp"
    ["rel-avito/tasks/user-visits"]="timestamp"
    ["rel-event/tasks/user-repeat"]="timestamp"
    ["rel-event/tasks/user-ignore"]="timestamp"
    ["rel-f1/tasks/driver-dnf"]="date"
    ["rel-f1/tasks/driver-top3"]="date"
    ["rel-hm/tasks/user-churn"]="timestamp"
    ["rel-stack/tasks/user-engagement"]="timestamp"
    ["rel-stack/tasks/user-badge"]="timestamp"
    ["rel-trial/tasks/study-outcome"]="timestamp"
)


# --- Main Execution Logic ---

echo "=== Starting Automated Training for Classification Tasks ==="
echo "=== All runs will be bound to CPU cores: [${CPU_CORES_TO_USE}] ==="

for task in "${TASKS[@]}"; do
    echo
    echo "======================================================================"
    echo "Preparing to train task: $task"
    echo "======================================================================"

    dataset_id_key=${DATASET_ID_KEYS[$task]}
    task_id_key=${TASK_ID_KEYS[$task]}
    label_key=${LABEL_KEYS[$task]}
    timestamp_key=${TIMESTAMP_KEYS[$task]}

    if [ -z "$dataset_id_key" ] || [ -z "$task_id_key" ] || [ -z "$label_key" ] || [ -z "$timestamp_key" ]; then
        echo "Error: Metadata for task '$task' is incomplete. Skipping."
        continue
    fi

    processed_path="${DATA_ROOT}/${task}/graph_dataset.pt"
    task_path="${TASK_DATA_ROOT}/${task}/"
    log_file_dir="${DATA_ROOT}/${task}"
    log_file="${log_file_dir}/train_classification.log"

    mkdir -p "$log_file_dir"

    echo "  - Processed data: ${processed_path}"
    echo "  - Task files dir: ${task_path}"
    echo "  - Log file:       ${log_file}"
    echo "  - Dataset ID key: ${dataset_id_key}"
    echo "  - Task ID key:    ${task_id_key}"
    echo "  - Label key:      ${label_key}"
    echo "  - Timestamp key:  ${timestamp_key}"

    if [ ! -f "$processed_path" ]; then
        echo "Error: Preprocessed file not found: '${processed_path}'. Please run preprocessing first. Skipping."
        continue
    fi

    COMMAND="taskset -c ${CPU_CORES_TO_USE} python3 ${TRAIN_SCRIPT} \
        --processed_path \"${processed_path}\" \
        --task_path \"${task_path}\" \
        --dataset_id_key \"${dataset_id_key}\" \
        --task_id_key \"${task_id_key}\" \
        --label_key \"${label_key}\" \
        --timestamp_key \"${timestamp_key}\" \
        --batch_size 1 \
        --gpu_ids 0 \
        --seed 42"

    echo
    echo "Executing command:"
    echo "$COMMAND"
    echo

    eval "$COMMAND" > "$log_file" 2>&1

    echo "Training for task '$task' complete. Log saved to ${log_file}"
done

echo
echo "======================================================================"
echo "All classification training tasks have been executed."
echo "======================================================================"