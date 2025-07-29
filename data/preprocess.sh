# ==============================================================================
#           Unified Automated Preprocessing Script for All Tasks
#
# Description:
#   This script automates the preprocessing of all sampled RelBench tasks.
#   It iterates through a predefined list of tasks, looks up the specific
#   parameters for each (like entity IDs and timestamp keys) from
#   configuration maps, and then calls the main Python preprocessing script
#   with the correct arguments.
#
# Usage:
#   1. Configure the PYTHON_SCRIPT, BASE_INPUT_DIR, and BASE_OUTPUT_DIR
#      variables below.
#   2. Ensure the Python preprocessing script is located at the specified path.
#   3. Ensure the sampled data exists in the input directory.
#   4. Run the script from your terminal: ./run_all_preprocessing.sh
# ==============================================================================

# --- Configuration ---

# 1. Path to your main Python preprocessing script.
PYTHON_SCRIPT="preprocess.py" # 确保此文件名与您的Python脚本匹配

# 2. Base directory where the 'sampling' folder is located.
BASE_INPUT_DIR="${HOME}/relbench-data-test/sampling"

# 3. Base directory where the final processed .pt files will be stored.
BASE_OUTPUT_DIR="${HOME}/relbench-data-pt"

# --- Master Task List (All Tasks) ---
TASKS=(
    # Regression Tasks
    "rel-amazon/tasks/user-ltv"
    "rel-amazon/tasks/item-ltv"
    "rel-avito/tasks/ad-ctr"
    "rel-event/tasks/user-attendance"
    "rel-f1/tasks/driver-position"
    "rel-hm/tasks/item-sales"
    "rel-stack/tasks/post-votes"
    "rel-trial/tasks/study-adverse"
    # Classification/AUROC Tasks
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

# --- Task-Specific Metadata Configuration ---

# TASK_CONFIG_ID_KEY: Maps full task name to its specific entity ID key.
declare -A TASK_CONFIG_ID_KEY=(
    ["rel-amazon/tasks/user-ltv"]="customer_id"
    ["rel-amazon/tasks/item-ltv"]="product_id"
    ["rel-amazon/tasks/user-churn"]="customer_id"
    ["rel-amazon/tasks/item-churn"]="product_id"
    ["rel-avito/tasks/ad-ctr"]="AdID"
    ["rel-avito/tasks/user-clicks"]="UserID"
    ["rel-avito/tasks/user-visits"]="UserID"
    ["rel-event/tasks/user-attendance"]="user_id"
    ["rel-event/tasks/user-repeat"]="user_id"
    ["rel-event/tasks/user-ignore"]="user_id"
    ["rel-f1/tasks/driver-position"]="driverId"
    ["rel-f1/tasks/driver-dnf"]="driverId"
    ["rel-f1/tasks/driver-top3"]="driverId"
    ["rel-hm/tasks/item-sales"]="article_id"
    ["rel-hm/tasks/user-churn"]="customer_id"
    ["rel-stack/tasks/post-votes"]="PostId"
    ["rel-stack/tasks/user-engagement"]="UserId"
    ["rel-stack/tasks/user-badge"]="UserId"
    ["rel-trial/tasks/study-adverse"]="nct_id"
    ["rel-trial/tasks/study-outcome"]="nct_id"
)

# TASK_CONFIG_JSONL_FILENAME: Maps full task name to its source JSONL filename.
declare -A TASK_CONFIG_JSONL_FILENAME=(
    ["rel-amazon/tasks/user-ltv"]="rel-amazon-customer.jsonl"
    ["rel-amazon/tasks/item-ltv"]="rel-amazon-product.jsonl"
    ["rel-amazon/tasks/user-churn"]="rel-amazon-customer.jsonl"
    ["rel-amazon/tasks/item-churn"]="rel-amazon-product.jsonl"
    ["rel-avito/tasks/ad-ctr"]="rel-avito-ad.jsonl"
    ["rel-avito/tasks/user-clicks"]="rel-avito-user.jsonl"
    ["rel-avito/tasks/user-visits"]="rel-avito-user.jsonl"
    ["rel-event/tasks/user-attendance"]="rel-event-user.jsonl"
    ["rel-event/tasks/user-repeat"]="rel-event-user.jsonl"
    ["rel-event/tasks/user-ignore"]="rel-event-user.jsonl"
    ["rel-f1/tasks/driver-position"]="rel-f1-driver.jsonl"
    ["rel-f1/tasks/driver-dnf"]="rel-f1-driver.jsonl"
    ["rel-f1/tasks/driver-top3"]="rel-f1-driver.jsonl"
    ["rel-hm/tasks/item-sales"]="rel-hm-article.jsonl"
    ["rel-hm/tasks/user-churn"]="rel-hm-customer.jsonl"
    ["rel-stack/tasks/post-votes"]="rel-stack-post.jsonl"
    ["rel-stack/tasks/user-engagement"]="rel-stack-user.jsonl"
    ["rel-stack/tasks/user-badge"]="rel-stack-user.jsonl"
    ["rel-trial/tasks/study-adverse"]="rel-trial-study.jsonl"
    ["rel-trial/tasks/study-outcome"]="rel-trial-study.jsonl"
)

# DATASET_CONFIG_TIME_KEYS: Maps dataset name to a space-separated list of time keys.
# THIS SECTION IS UPDATED BASED ON YOUR SCRIPTS.
declare -A DATASET_CONFIG_TIME_KEYS=(
    ["rel-amazon"]="review_time"
    ["rel-avito"]="SearchDate ViewDate PhoneRequestDate"
    ["rel-event"]="start_time timestamp"
    ["rel-f1"]="date"
    ["rel-hm"]="t_dat"
    ["rel-stack"]="CreationDate"
    ["rel-trial"]="date"
)

# --- Main Execution Logic ---

echo "=== Starting Automated Preprocessing for All Tasks ==="

# Check if the Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: The script '$PYTHON_SCRIPT' was not found."
    echo "Please update the PYTHON_SCRIPT variable at the top of this script."
    exit 1
fi

# Loop through the master list of tasks
for task in "${TASKS[@]}"; do
    echo
    echo "----------------------------------------------------------------------"
    echo "Processing Task: $task"
    echo "----------------------------------------------------------------------"

    # 1. Extract dataset name from the task path
    dataset_name=$(echo "$task" | cut -d'/' -f1)

    # 2. Look up parameters from the configuration maps
    id_key=${TASK_CONFIG_ID_KEY[$task]}
    jsonl_filename=${TASK_CONFIG_JSONL_FILENAME[$task]}
    time_keys_str=${DATASET_CONFIG_TIME_KEYS[$dataset_name]}
    read -r -a time_keys_arr <<< "$time_keys_str" # Convert space-separated string to a bash array

    # 3. Check if all parameters were found
    if [ -z "$id_key" ] || [ -z "$jsonl_filename" ] || [ -z "$time_keys_str" ]; then
        echo "Error: Missing configuration for task '$task' or dataset '$dataset_name'. Skipping."
        continue
    fi

    # 4. Construct input and output paths
    input_path="${BASE_INPUT_DIR}/${task}/${jsonl_filename}"
    output_path="${BASE_OUTPUT_DIR}/${task}"

    echo "  - Input JSONL: ${input_path}"
    echo "  - Output Dir:  ${output_path}"
    echo "  - ID Key:      ${id_key}"
    echo "  - Time Keys:   ${time_keys_arr[*]}"

    # 5. Check if the required input file exists
    if [ ! -f "$input_path" ]; then
        echo "Error: Input file not found: '${input_path}'."
        echo "Please ensure the sampling script has been run successfully for this task. Skipping."
        continue
    fi
    
    # 6. Create the output directory
    mkdir -p "$output_path"

    # 7. Execute the Python preprocessing script with the correct arguments
    echo "Executing Python script..."
    python3 "$PYTHON_SCRIPT" \
        --input_path "$input_path" \
        --output_path "$output_path" \
        --id_key "$id_key" \
        --time_keys "${time_keys_arr[@]}"

    echo "Task '$task' processed successfully."
done

echo
echo "======================================================================"
echo "All tasks have been preprocessed."
echo "======================================================================"