import pandas as pd
import orjson
from typing import Set, Dict, Any, Tuple
from pathlib import Path
from tqdm import tqdm
import argparse

TASKS_CONFIG = {
    "rel-amazon/tasks/user-ltv": {
        "parquet_id_column": "customer_id",
        "jsonl_id_key": "customer_id",
        "source_jsonl_file": "rel-amazon-customer.jsonl",
        "samples": (2000, 500, 500),
    },
    "rel-amazon/tasks/item-ltv": {
        "parquet_id_column": "product_id",
        "jsonl_id_key": "product_id",
        "source_jsonl_file": "rel-amazon-product.jsonl",
        "samples": (2000, 500, 500),
    },
    "rel-avito/tasks/ad-ctr": {
        "parquet_id_column": "AdID",
        "jsonl_id_key": "AdID",
        "source_jsonl_file": "rel-avito-ad.jsonl",
        "samples": (2000, 500, 500),
    },
    "rel-event/tasks/user-attendance": {
        "parquet_id_column": "user",
        "jsonl_id_key": "user_id",
        "source_jsonl_file": "rel-event-user.jsonl",
        "samples": (2000, 500, 500),
    },
    "rel-f1/tasks/driver-position": {
        "parquet_id_column": "driverId",
        "jsonl_id_key": "driverId",
        "source_jsonl_file": "rel-f1-driver.jsonl",
        "samples": (2000, 500, 500),
    },
    "rel-hm/tasks/item-sales": {
        "parquet_id_column": "article_id",
        "jsonl_id_key": "article_id",
        "source_jsonl_file": "rel-hm-article.jsonl",
        "samples": (2000, 500, 500),
    },
    "rel-stack/tasks/post-votes": {
        "parquet_id_column": "PostId",
        "jsonl_id_key": "PostId",
        "source_jsonl_file": "rel-stack-post.jsonl",
        "samples": (2000, 500, 500),
    },
    "rel-trial/tasks/study-adverse": {
        "parquet_id_column": "nct_id",
        "jsonl_id_key": "nct_id",
        "source_jsonl_file": "rel-trial-study.jsonl",
        "samples": (2000, 500, 500),
    },
    "rel-amazon/tasks/user-churn": {
        "parquet_id_column": "customer_id",
        "jsonl_id_key": "customer_id",
        "source_jsonl_file": "rel-amazon-customer.jsonl",
        "samples": (2000, 500, 500),
    },
    "rel-amazon/tasks/item-churn": {
        "parquet_id_column": "product_id",
        "jsonl_id_key": "product_id",
        "source_jsonl_file": "rel-amazon-product.jsonl",
        "samples": (2000, 500, 500),
    },
    "rel-avito/tasks/user-clicks": {
        "parquet_id_column": "UserID",
        "jsonl_id_key": "UserID",
        "source_jsonl_file": "rel-avito-user.jsonl",
        "samples": (2000, 500, 500),
    },
    "rel-avito/tasks/user-visits": {
        "parquet_id_column": "UserID",
        "jsonl_id_key": "UserID",
        "source_jsonl_file": "rel-avito-user.jsonl",
        "samples": (2000, 500, 500),
    },
    "rel-event/tasks/user-repeat": {
        "parquet_id_column": "user",
        "jsonl_id_key": "user_id",
        "source_jsonl_file": "rel-event-user.jsonl",
        "samples": (2000, 500, 500),
    },
    "rel-event/tasks/user-ignore": {
        "parquet_id_column": "user",
        "jsonl_id_key": "user_id",
        "source_jsonl_file": "rel-event-user.jsonl",
        "samples": (2000, 500, 500),
    },
    "rel-f1/tasks/driver-dnf": {
        "parquet_id_column": "driverId",
        "jsonl_id_key": "driverId",
        "source_jsonl_file": "rel-f1-driver.jsonl",
        "samples": (2000, 500, 500),
    },
    "rel-f1/tasks/driver-top3": {
        "parquet_id_column": "driverId",
        "jsonl_id_key": "driverId",
        "source_jsonl_file": "rel-f1-driver.jsonl",
        "samples": (2000, 500, 500),
    },
    "rel-f1/tasks/driver-position": {
        "parquet_id_column": "driverId",
        "jsonl_id_key": "driverId",
        "source_jsonl_file": "rel-f1-driver.jsonl",
        "samples": (2000, 500, 500),
    },
    "rel-hm/tasks/user-churn": {
        "parquet_id_column": "customer_id",
        "jsonl_id_key": "customer_id",
        "source_jsonl_file": "rel-hm-customer.jsonl",
        "samples": (2000, 500, 500),
    },
    "rel-stack/tasks/user-engagement": {
        "parquet_id_column": "OwnerUserId",
        "jsonl_id_key": "UserId",
        "source_jsonl_file": "rel-stack-user.jsonl",
        "samples": (2000, 500, 500),
    },
    "rel-stack/tasks/user-badge": {
        "parquet_id_column": "UserId",
        "jsonl_id_key": "UserId",
        "source_jsonl_file": "rel-stack-user.jsonl",
        "samples": (20000, 10000, 10000),
    },
    "rel-trial/tasks/study-outcome": {
        "parquet_id_column": "nct_id",
        "jsonl_id_key": "nct_id",
        "source_jsonl_file": "rel-trial-study.jsonl",
        "samples": (2000, 500, 500),
    },
}

def sample_single_task(task_name: str, config: Dict[str, Any], base_path_map: Dict[str, Path]):
    print(f"\n{'='*25} Processing Task: {task_name} {'='*25}")

    dataset_name = task_name.split('/')[0]
    parquet_id_column = config["parquet_id_column"]
    jsonl_id_key = config["jsonl_id_key"]
    source_jsonl_file = config["source_jsonl_file"]
    samples = config["samples"]

    task_cache_path = base_path_map["raw_tasks"] / task_name
    source_jsonl_path = base_path_map["built_jsonl"] / dataset_name / source_jsonl_file
    output_dir = base_path_map["output"] / task_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Source Task Path: {task_cache_path}")
    print(f"Source JSONL Path: {source_jsonl_path}")
    print(f"Sampled Output Path: {output_dir}")

    if not source_jsonl_path.exists():
        print(f"Error: Source JSONL file not found: {source_jsonl_path}")
        print("Please run 'build_datasets.sh' first.")
        return

    all_target_ids: Set = set()
    splits_to_process: list[Tuple[str, int]] = [
        ('train', samples[0]),
        ('val', samples[1]),
        ('test', samples[2])
    ]

    print(f"\n--- Step 1: Sampling task files and collecting entity IDs ---")
    for split_name, num_samples in splits_to_process:
        input_parquet_path = task_cache_path / f'{split_name}.parquet'
        output_parquet_path = output_dir / f'{split_name}.parquet'
        
        print(f"  Processing: {input_parquet_path}")

        try:
            task_df = pd.read_parquet(input_parquet_path)
        except Exception as e:
            print(f"    Error: Cannot read Parquet file '{input_parquet_path}': {e}")
            continue

        if parquet_id_column not in task_df.columns:
            print(f"    Error: ID column '{parquet_id_column}' not found in Parquet file.")
            continue

        if num_samples >= len(task_df):
            print(f"    Warning: Requested {num_samples} samples, but file only has {len(task_df)}. Using all rows.")
            selected_tasks_df = task_df
        else:
            selected_tasks_df = task_df.head(num_samples)
        
        selected_tasks_df.to_parquet(output_parquet_path, index=False)
        print(f"    Saved sampled task to: '{output_parquet_path}'")

        current_ids = set(selected_tasks_df[parquet_id_column].dropna())
        all_target_ids.update(current_ids)
        print(f"    Collected {len(current_ids)} IDs. Total unique IDs so far: {len(all_target_ids)}")

    print(f"\n--- Step 2: Indexing and filtering '{source_jsonl_path}' ---")
    objects_by_id: Dict[Any, str] = {}
    
    with open(source_jsonl_path, 'r', encoding='utf-8') as f_jsonl:
        for line in tqdm(f_jsonl, desc="    Indexing JSONL"):
            try:
                obj = orjson.loads(line)
                obj_id = obj.get(jsonl_id_key)
                if obj_id is not None and obj_id in all_target_ids and obj_id not in objects_by_id:
                    objects_by_id[obj_id] = line.strip()
            except orjson.JSONDecodeError:
                continue
    
    print(f"  Indexing complete. Found {len(objects_by_id)} / {len(all_target_ids)} matching objects.")

    final_output_jsonl_path = output_dir / source_jsonl_file
    print(f"\n--- Step 3: Writing filtered objects to '{final_output_jsonl_path}' ---")
    
    with open(final_output_jsonl_path, 'w', encoding='utf-8') as f_out:
        for json_line in objects_by_id.values():
            f_out.write(json_line + '\n')
    
    print(f"  Successfully wrote {len(objects_by_id)} unique JSON objects.")
    print(f"Task '{task_name}' sampling complete.")

def main():
    parser = argparse.ArgumentParser(
        description="Unified sampling script for RelBench tasks.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "task_name",
        type=str,
        nargs='?',
        default="all",
        help="The specific task to sample (e.g., 'rel-amazon/tasks/user-churn').\n"
             "If not provided, all tasks defined in TASKS_CONFIG will be processed."
    )
    args = parser.parse_args()

    base_path_map = {
        "raw_tasks": Path.home() / ".cache/relbench",
        "built_jsonl": Path.home() / "relbench-data-test",
        "output": Path.home() / "relbench-data-test" / "sampling",
    }
    
    if args.task_name == "all":
        print("Starting sampling process for ALL tasks...")
        for task, config in TASKS_CONFIG.items():
            sample_single_task(task, config, base_path_map)
        print(f"\n{'='*30} All tasks processed! {'='*30}")
    else:
        if args.task_name in TASKS_CONFIG:
            sample_single_task(args.task_name, TASKS_CONFIG[args.task_name], base_path_map)
        else:
            print(f"Error: Task '{args.task_name}' not found in TASKS_CONFIG.")
            print("Please choose from one of the following:")
            for task in TASKS_CONFIG:
                print(f"  - {task}")

if __name__ == '__main__':
    main()