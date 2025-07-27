import pandas as pd
import json
import torch
from torch_geometric.data import Data
import networkx as nx
import numpy as np
from transformers import AutoTokenizer, AutoModel
import os
import time
from sentence_transformers import SentenceTransformer
import argparse

def parse_json_limited_depth_keep_str(obj, max_depth, current_depth=1):
    """
    Recursively parse a JSON object up to max_depth levels.
    If max_depth == -1, parse all levels.
    Beyond max_depth, keep the remaining part as a JSON string.
    :param obj: The object to parse (dict, list, str, int, etc.)
    :param max_depth: Maximum recursion depth (-1 means no limit)
    :param current_depth: Current recursion depth
    :return: The parsed object
    """
    if max_depth != -1 and current_depth > max_depth:
        # If the maximum depth is reached, return the remaining part as a JSON string
        return json.dumps(obj, ensure_ascii=False)
    if isinstance(obj, dict):
        return {k: parse_json_limited_depth_keep_str(v, max_depth, current_depth + 1) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [parse_json_limited_depth_keep_str(item, max_depth, current_depth + 1) for item in obj]
    else:
        return obj  # Return basic types as is

def get_json_stats(obj, current_depth=1):
    """
    Recursively traverse the JSON object to get:
    - Maximum depth
    - Number of dicts
    - Number of lists
    :param obj: The JSON object (dict, list, etc.)
    :param current_depth: The current recursion depth
    :return: (max_depth, dict_count, list_count)
    """
    max_depth = current_depth
    dict_count = 0
    list_count = 0

    if isinstance(obj, dict):
        dict_count += 1
        for v in obj.values():
            d, dc, lc = get_json_stats(v, current_depth + 1)
            max_depth = max(max_depth, d)
            dict_count += dc
            list_count += lc
    elif isinstance(obj, list):
        list_count += 1
        for item in obj:
            d, dc, lc = get_json_stats(item, current_depth + 1)
            max_depth = max(max_depth, d)
            dict_count += dc
            list_count += lc

    return max_depth, dict_count, list_count

def json_to_graph(obj, time_keys, parent_node_id=None):
    """
    Convert JSON object to PyTorch Geometric graph structure, including node timestamps.
    """
    node_features = []
    node_timestamps = []  # <--- NEW: List to store timestamps for each node
    edge_index_by_depth = {}
    list_index_sequences_by_depth = {}
    node_id_counter = 0
    
    # ROOT node
    node_features.append("ROOT")
    node_timestamps.append(pd.NaT) # <--- NEW: ROOT node has no timestamp
    root_node_id = node_id_counter
    node_id_counter += 1
    
    # <--- NEW: Define common timestamp keys across all your datasets ---
    TIME_KEYS = time_keys if isinstance(time_keys, list) else [time_keys]

    def _recursive_build(obj, parent_id=None, node_counter=0, current_depth=1):
        nonlocal node_features, node_timestamps, edge_index_by_depth, list_index_sequences_by_depth
        
        if current_depth not in edge_index_by_depth:
            edge_index_by_depth[current_depth] = []
        if current_depth not in list_index_sequences_by_depth:
            list_index_sequences_by_depth[current_depth] = []
        
        if isinstance(obj, dict):
            # <--- NEW: Find the timestamp for the current dictionary entity ---
            entity_timestamp = pd.NaT
            for t_key in TIME_KEYS:
                if t_key in obj:
                    try:
                        # Use errors='coerce' to handle potential parsing errors gracefully
                        entity_timestamp = pd.to_datetime(obj[t_key], errors='coerce')
                        if pd.notna(entity_timestamp):
                            break
                    except (ValueError, TypeError):
                        pass

            for key, value in obj.items():
                # Skip the timestamp key itself from becoming a leaf node
                if key in TIME_KEYS:
                    continue
                
                if isinstance(value, str):
                    node_feature = f"{key}:{value}"
                    node_features.append(node_feature)
                    node_timestamps.append(entity_timestamp) # <--- NEW: Inherit timestamp
                    current_node_id = node_counter
                    node_counter += 1
                    if parent_id is not None:
                        edge_index_by_depth[current_depth].append([current_node_id, parent_id])
                        
                elif isinstance(value, dict):
                    node_features.append(key)
                    node_timestamps.append(entity_timestamp) # <--- NEW: Inherit timestamp
                    current_node_id = node_counter
                    node_counter += 1
                    if parent_id is not None:
                        edge_index_by_depth[current_depth].append([current_node_id, parent_id])
                    node_counter = _recursive_build(value, current_node_id, node_counter, current_depth + 1)
                        
                elif isinstance(value, list):
                    list_nodes = []
                    for i, item in enumerate(value):
                        # <--- NEW: Each item in a list is a new entity, find its own timestamp ---
                        item_timestamp = pd.NaT
                        if isinstance(item, dict):
                             for t_key in TIME_KEYS:
                                if t_key in item:
                                    try:
                                        item_timestamp = pd.to_datetime(item[t_key], errors='coerce')
                                        if pd.notna(item_timestamp):
                                            break
                                    except (ValueError, TypeError):
                                        pass
                        
                        if isinstance(item, (dict, list)):
                            node_features.append(key)
                            node_timestamps.append(item_timestamp) # <--- NEW: Use item's own timestamp
                            current_node_id = node_counter
                            list_nodes.append(current_node_id)
                            node_counter += 1
                            if parent_id is not None:
                                edge_index_by_depth[current_depth].append([current_node_id, parent_id])
                            node_counter = _recursive_build(item, current_node_id, node_counter, current_depth + 1)
                        else:
                            node_feature = f"{key}:{str(item)}"
                            node_features.append(node_feature)
                            node_timestamps.append(item_timestamp) # <--- NEW: Use item's own timestamp
                            current_node_id = node_counter
                            list_nodes.append(current_node_id)
                            node_counter += 1
                            if parent_id is not None:
                                edge_index_by_depth[current_depth].append([current_node_id, parent_id])
                    
                    if parent_id is not None:
                        list_nodes_with_parent = list_nodes + [parent_id]
                        list_index_sequences_by_depth[current_depth].append(list_nodes_with_parent)
                    else:
                        list_index_sequences_by_depth[current_depth].append(list_nodes)
                        
                else:
                    node_feature = f"{key}:{str(value)}"
                    node_features.append(node_feature)
                    node_timestamps.append(entity_timestamp) # <--- NEW: Inherit timestamp
                    current_node_id = node_counter
                    node_counter += 1
                    if parent_id is not None:
                        edge_index_by_depth[current_depth].append([current_node_id, parent_id])
                            
        elif isinstance(obj, list):
            list_nodes = []
            for i, item in enumerate(obj):
                node_features.append(f"list_item_{i}")
                node_timestamps.append(pd.NaT) # <--- NEW: No clear timestamp for root list items
                current_node_id = node_counter
                list_nodes.append(current_node_id)
                node_counter += 1
                if parent_id is not None:
                    edge_index_by_depth[current_depth].append([current_node_id, parent_id])
                if isinstance(item, (dict, list)):
                    node_counter = _recursive_build(item, current_node_id, node_counter, current_depth + 1)
            
            if parent_id is not None:
                list_nodes_with_parent = list_nodes + [parent_id]
                list_index_sequences_by_depth[current_depth].append(list_nodes_with_parent)
            else:
                list_index_sequences_by_depth[current_depth].append(list_nodes)
            
        return node_counter
    
    final_counter = _recursive_build(obj, root_node_id, node_id_counter, current_depth=1)
    
    # <--- MODIFIED: Return signature changed ---
    return node_features, node_timestamps, edge_index_by_depth, list_index_sequences_by_depth, final_counter

# <--- MODIFIED: Function signature changed to accept node_timestamps ---
def create_pytorch_geometric_data(node_features, node_timestamps, edge_index_by_depth, list_index_sequences_by_depth=None, node_embeddings=None):
    """
    Create PyTorch Geometric Data object from node features and edge index organized by depth.
    """
    all_edges = []
    for depth in sorted(edge_index_by_depth.keys()):
        all_edges.extend(edge_index_by_depth[depth])
    
    if all_edges:
        edge_index_tensor = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
    else:
        edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
    
    if node_embeddings is not None:
        x = node_embeddings
        # print(f"Using language model embeddings as node features: {x.shape}") # Commented out for cleaner output
    else:
        num_nodes = len(node_features)
        x = torch.arange(num_nodes, dtype=torch.float).unsqueeze(1)
        # print(f"Using simple index features: {x.shape}") # Commented out for cleaner output
    
    data = Data(x=x, edge_index=edge_index_tensor)
    data.node_features = node_features
    data.edge_index_by_depth = edge_index_by_depth
    data.list_index_sequences_by_depth = list_index_sequences_by_depth

    # <--- NEW: Convert and store node timestamps ---
    # Convert datetime objects to Unix timestamps (integer seconds). Use -1 for NaT.
    timestamps_unix = [int(ts.timestamp()) if pd.notna(ts) else -1 for ts in node_timestamps]
    data.node_timestamps = torch.tensor(timestamps_unix, dtype=torch.long)
    
    return data

def test_json_to_graph():
    """
    Test the JSON to graph conversion with a simple example.
    NOTE: This test function needs to be updated to match the new function signatures.
    """
    print("=== Testing JSON to Graph Conversion ===")
    
    # Simple test case
    test_json = {
        "name": "张三",
        "age": 25,
        "address": {
            "city": "北京",
            "district": "朝阳区"
        },
        "location": [{"信息更新日期": "2017-02-19", "居住地址": "绍兴市诸暨市中国浙江省绍兴市诸暨市漱云路", "居住状况": "其他"}, {"信息更新日期": "2017-01-23", "居住地址": "浙江绍兴诸暨次坞镇平阳地村ef号.", "居住状况": "未知"}, {"信息更新日期": "2017-01-20", "居住地址": "浙江省绍兴诸暨市次坞镇平阳地村ef号", "居住状况": "未知"}, {"信息更新日期": "2017-01-14", "居住地址": "陶朱街道漱云路欣泰小区A栋A单元bjb", "居住状况": "未知"}, {"信息更新日期": "2016-12-21", "居住地址": "鏆傜己", "居住状况": "未知"}],
        "hobby":["羽毛球", "网球"]
    }
    
    print("Test JSON:")
    print(json.dumps(test_json, ensure_ascii=False, indent=2))
    
    # <--- MODIFIED: Call signature updated ---
    node_features, node_timestamps, edge_index_by_depth, list_index_sequences_by_depth, total_nodes = json_to_graph(test_json,args.time_keys)
    
    print(f"\nGraph Conversion Results:")
    print(f"Total nodes: {total_nodes}")
    
    total_edges = sum(len(edges) for edges in edge_index_by_depth.values())
    print(f"Number of edges: {total_edges}")
    
    total_list_sequences = sum(len(sequences) for sequences in list_index_sequences_by_depth.values())
    print(f"Number of list sequences: {total_list_sequences}")
    
    print(f"\nNode Features:")
    for i, feature in enumerate(node_features):
        print(f"Node {i}: {feature}")

    # <--- NEW: Print timestamps for debugging ---
    print(f"\nNode Timestamps:")
    for i, ts in enumerate(node_timestamps):
        print(f"Node {i}: {ts}")
        
    print(f"\nEdges by Depth:")
    for depth in sorted(edge_index_by_depth.keys()):
        print(f"  Depth {depth}:")
        for i, edge in enumerate(edge_index_by_depth[depth]):
            print(f"    Edge {i}: {edge[0]} -> {edge[1]} ({node_features[edge[0]]} -> {node_features[edge[1]]})")
    
    print(f"\nList Index Sequences by Depth:")
    for depth in sorted(list_index_sequences_by_depth.keys()):
        print(f"  Depth {depth}:")
        for i, seq in enumerate(list_index_sequences_by_depth[depth]):
            print(f"    List {i}: nodes {seq} -> features: {[node_features[idx] for idx in seq]}")
    
    # <--- MODIFIED: Call signature updated ---
    graph_data = create_pytorch_geometric_data(node_features, node_timestamps, edge_index_by_depth, list_index_sequences_by_depth)
    print(f"\nPyTorch Geometric Data: {graph_data}")
    print(f"Edge index by depth keys: {sorted(graph_data.edge_index_by_depth.keys())}")
    print(f"List index sequences by depth keys: {sorted(graph_data.list_index_sequences_by_depth.keys())}")
    print(f"Node timestamps tensor: {graph_data.node_timestamps}")
    
    # This return is not used in the main script but is kept for consistency
    return node_features, edge_index_by_depth, list_index_sequences_by_depth, graph_data

# Run test
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSON data to create PyTorch Geometric graphs.")
    parser.add_argument('--input_path', type=str, required=True, help="Path to the input JSONL file containing the data.")
    parser.add_argument('--output_path', type=str,required=True, help="Path to save the processed graph dataset.")
    parser.add_argument('--id_key',type=str, required=True, help="Key to extract the unique identifier for each entry (e.g., 'driverId').")
    parser.add_argument('--time_keys', type=str, nargs='+', required=True,
                        help="A list of keys to be treated as timestamps (e.g., --time_keys date dob).")
    parser.add_argument('--model_name', type=str, default='sentence-transformers/average_word_embeddings_glove.6B.300d',
                        help='The name of the SentenceTransformer model to use for embeddings.')
    args = parser.parse_args()
    gpu_ids = [1]
    batch_size = 32
    print(f"Starting preprocessing with the following parameters:")
    print(f"  - Input JSONL: {args.input_path}")
    print(f"  - Output .pt File: {args.output_path}")
    print(f"  - Entity ID Key: {args.id_key}")
    print(f"  - Time Keys: {args.time_keys}")
    print(f"  - Embedding Model: {args.model_name}")
    print("\n" + "="*60 + "\n")

    # test_json_to_graph() # You can uncomment this to test the conversion logic
    print("\n" + "="*60 + "\n")
    # load data
    # NOTE: You might need to change the file path to your actual data file.
    # df = pd.read_excel('data_example.xlsx', header=None)
    # jsonl_file_path = 'relbench-reproduce/rel-data/rel-f1/all_f1_drivers.jsonl' # <--- TOBE EDIT: Update to your actual JSONL file path'
    jsonl_file_path = args.input_path
    id_key = args.id_key
    output_path = args.output_path
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        json_lines = [str(line) for line in f if line.strip()]  # Ensure no empty lines
    print(f"Loaded {len(json_lines)} rows of data")

    # Process all rows to create graphs
    all_graphs_data = []
    all_node_features = []
    node_feature_to_graph_mapping = []

    for row_idx in range(len(json_lines)):
        print(f"\nProcessing row {row_idx + 1}/{len(json_lines)}...")

        json_data = json.loads(json_lines[row_idx])

        # <--- MODIFIED: Extract customer_id. You might need to adapt the key 'customer_id' ---
        # customer_id = json_data.get('driverId') # <--- TOBE EDIT
        entity_id = json_data.get(id_key)  # <--- TOBE EDIT: Use the provided id_key
        if entity_id is None:
            print(f"Warning: '{id_key}' not found in row {row_idx}. Skipping this entry.")
            continue

        # <--- MODIFIED: Call the updated json_to_graph function ---
        node_features, node_timestamps, edge_index_by_depth, list_index_sequences_by_depth, total_nodes = json_to_graph(json_data, args.time_keys)

        print(f"  Graph {row_idx} (Entity {entity_id}): {total_nodes} nodes, {sum(len(edges) for edges in edge_index_by_depth.values())} edges")

        # Store graph structure info
        graph_info = {
            'row_idx': row_idx,
            'entity_id': entity_id, # <--- NEW: Store entity_id
            'node_features': node_features,
            'node_timestamps': node_timestamps, # <--- NEW: Store raw timestamps
            'edge_index_by_depth': edge_index_by_depth,
            'list_index_sequences_by_depth': list_index_sequences_by_depth,
            'total_nodes': total_nodes,
            'node_start_idx': len(all_node_features),
            'node_end_idx': len(all_node_features) + len(node_features)
        }
        all_graphs_data.append(graph_info)
        
        all_node_features.extend(node_features)
        
        for _ in range(len(node_features)):
            node_feature_to_graph_mapping.append(row_idx)
    
    print(f"\nTotal graphs created: {len(all_graphs_data)}")
    print(f"Total node features for embedding: {len(all_node_features)}")
    
    unique_node_features = list(set(all_node_features))
    feature_to_unique_idx = {feature: idx for idx, feature in enumerate(unique_node_features)}
    print(f"Unique node features after deduplication: {len(unique_node_features)}")
    
    # GPU and model loading configuration remains the same
    if gpu_ids and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_ids[0]}')
        print(f"Using GPU: {gpu_ids}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # NOTE: You need to provide the correct path to your model
    # tokenizer = AutoTokenizer.from_pretrained('/home/luckytiger/bge-reranker-large')
    # model = AutoModel.from_pretrained('/home/luckytiger/bge-reranker-large')

    model = SentenceTransformer("sentence-transformers/average_word_embeddings_glove.6B.300d", device=device)

    model = model.to(device)
    if len(gpu_ids) > 1 and torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        print(f"Using DataParallel on GPUs: {gpu_ids}")
    model.eval()

    # The embedding generation function remains the same
    # def generate_embeddings_batch(texts, batch_size):
        # embeddings = []
        # for i in range(0, len(texts), batch_size):
        #     batch_texts = texts[i:i + batch_size]
        #     encoded_input = tokenizer(
        #         batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=512
        #     )
        #     encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        #     with torch.no_grad():
        #         model_output = model(**encoded_input)
        #         batch_embeddings = model_output[0][:, 0]
        #         batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
        #         embeddings.append(batch_embeddings.cpu())
        #     if device.type == 'cuda':
        #         torch.cuda.empty_cache()
        # return torch.cat(embeddings, dim=0)

    def generate_embeddings_batch(texts, SBERT_model ,batch_size):
        embeddings_numpy = SBERT_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False
        )
        return torch.from_numpy(embeddings_numpy)

    print(f"\nGenerating embeddings for {len(unique_node_features)} unique node features...")
    unique_embeddings = generate_embeddings_batch(unique_node_features, model, batch_size)
    print(f"Unique embeddings shape: {unique_embeddings.shape}")
    
    # Create the final dataset
    dataset = []
    
    for graph_info in all_graphs_data:
        # Map node features to their corresponding embeddings
        node_embeddings = torch.stack([unique_embeddings[feature_to_unique_idx[f]] for f in graph_info['node_features']])
        
        # <--- MODIFIED: Pass timestamps to the creation function ---
        graph_data = create_pytorch_geometric_data(
            node_features=graph_info['node_features'],
            node_timestamps=graph_info['node_timestamps'],
            edge_index_by_depth=graph_info['edge_index_by_depth'],
            list_index_sequences_by_depth=graph_info['list_index_sequences_by_depth'],
            node_embeddings=node_embeddings
        )

        # <--- NEW: Attach the entity_id to the final Data object ---
        graph_data.entity_id = graph_info['entity_id']  # <--- TOBE EDIT: Use the entity_id extracted earlier

        # <--- REMOVED: The random label is no longer needed in the preprocessed data ---
        # graph_data.y = torch.tensor([torch.randint(0, 2, (1,)).item()], dtype=torch.long)
        
        dataset.append(graph_data)
        print(f"Graph {graph_info['row_idx']} (Entity {graph_data.entity_id}): {graph_data.num_nodes} nodes, {graph_data.num_edges} edges created.")

    print(f"\nDataset created with {len(dataset)} graphs")
    
    # Save the dataset
    print("Saving dataset...")
    # NOTE: Ensure the './data' directory exists
    # os.makedirs('./data', exist_ok=True)
    # torch.save(dataset, './data/graph_dataset.pt') # <--- TOBE EDIT
    os.makedirs(output_path, exist_ok=True)
    torch.save(dataset, os.path.join(output_path, 'graph_dataset.pt'))  # <--- TOBE EDIT: Save to the specified output path
    print(f"Dataset saved as '{output_path}/graph_dataset.pt'")

    # Final statistics printing remains the same
    total_nodes = sum([data.num_nodes for data in dataset])
    total_edges = sum([data.num_edges for data in dataset])
    embedding_dim = dataset[0].x.shape[1] if len(dataset) > 0 else 0
    
    print(f"\nDataset Statistics:")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Total nodes: {total_nodes}")
    print(f"Total edges: {total_edges}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Average nodes per graph: {total_nodes/len(dataset):.2f}")
    print(f"Average edges per graph: {total_edges/len(dataset):.2f}")