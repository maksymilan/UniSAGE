import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.seed import seed_everything
from model.UniSAGE import UniSAGE
from sklearn.metrics import roc_auc_score, mean_absolute_error
import numpy as np
from tqdm import tqdm
import argparse

def load_dataset(processed_path):
    """
    Load the saved dataset
    """
    dataset_path = processed_path
    print(f"Loading preprocessed graph data from: {dataset_path}")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Graph dataset not found at '{dataset_path}'. "
            "Please run the preprocessing script first."
        )
    
    try:
        dataset = torch.load(dataset_path, weights_only=False)
    except Exception as e:
        print(f"Method 1 failed: {e}")
        try:
            with torch.serialization.safe_globals([Data, Batch]):
                dataset = torch.load(dataset_path)
        except Exception as e2:
            print(f"Method 2 failed: {e2}")
            torch.serialization.add_safe_globals([Data, Batch])
            dataset = torch.load(dataset_path)
    
    print(f"Loaded dataset with {len(dataset)} graphs")
    
    return dataset


class TaskDataset(Dataset):
    """
    A PyTorch Dataset that combines a task file (e.g., from a CSV or Parquet)
    with a pre-processed graph map.
    """
    def __init__(self, task_df: pd.DataFrame, graph_map: dict, config: dict):
        self.tasks = task_df
        self.graph_map = graph_map
        self.entity_id_col = config['entity_id_col']
        self.label_col = config['label_col']
        self.timestamp_col = config['timestamp_col']

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task_row = self.tasks.iloc[idx]
        entity_id = task_row[self.entity_id_col]
        full_graph = self.graph_map.get(entity_id)
        
        if full_graph is None:
            return None

        if full_graph.num_nodes > 100000:
            print(f"Warning: Graph for entity {entity_id} has {full_graph.num_nodes} nodes, which is too large.")

        data = full_graph.clone()
        
        label_value = task_row[self.label_col]
        data.y = torch.tensor([float(label_value)], dtype=torch.float)

        task_timestamp = pd.to_datetime(task_row[self.timestamp_col])
        data.task_timestamp = torch.tensor([int(task_timestamp.timestamp())], dtype=torch.long)

        return data

def custom_collate(batch):
    """
    Custom collate function to properly handle edge_index_by_depth and list_index_sequences_by_depth
    when batching graphs together. Updates node indices correctly.
    This version avoids the default collate function's issues with dictionary attributes
    that have varying keys across data objects.
    """
    batch_for_default_collate = []
    for data in batch:
        new_data = Data()
        for key in data.keys():
            if key not in ['edge_index_by_depth', 'list_index_sequences_by_depth']:
                new_data[key] = data[key]
        batch_for_default_collate.append(new_data)
    
    batch_data = Batch.from_data_list(batch_for_default_collate)

    batch_edge_index_by_depth = {}
    batch_list_index_sequences_by_depth = {}
    
    node_offsets = [0]
    for data in batch:
        node_offsets.append(node_offsets[-1] + data.num_nodes)
    node_offsets = node_offsets[:-1]
    
    for graph_idx, data in enumerate(batch):
        offset = node_offsets[graph_idx]
        
        if hasattr(data, 'edge_index_by_depth') and data.edge_index_by_depth is not None:
            for depth, edges in data.edge_index_by_depth.items():
                if depth not in batch_edge_index_by_depth:
                    batch_edge_index_by_depth[depth] = []
                for edge in edges:
                    updated_edge = [edge[0] + offset, edge[1] + offset]
                    batch_edge_index_by_depth[depth].append(updated_edge)
        
        if hasattr(data, 'list_index_sequences_by_depth') and data.list_index_sequences_by_depth is not None:
            for depth, sequences in data.list_index_sequences_by_depth.items():
                if depth not in batch_list_index_sequences_by_depth:
                    batch_list_index_sequences_by_depth[depth] = []
                for seq in sequences:
                    updated_seq = [node_idx + offset for node_idx in seq]
                    batch_list_index_sequences_by_depth[depth].append(updated_seq)
    
    batch_data.edge_index_by_depth = batch_edge_index_by_depth
    batch_data.list_index_sequences_by_depth = batch_list_index_sequences_by_depth
    
    return batch_data


def custom_collate_safe(batch):
    """A safe wrapper for the collate function that filters out None values."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return custom_collate(batch)


def create_dataloaders(train_tasks, val_tasks, test_tasks, graph_map, task_config, batch_size=4):
    """Creates dataloaders from task DataFrames and a graph map."""
    train_dataset = TaskDataset(train_tasks, graph_map, task_config)
    val_dataset = TaskDataset(val_tasks, graph_map, task_config)
    test_dataset = TaskDataset(test_tasks, graph_map, task_config)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_safe)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_safe)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_safe)
    
    print(f"Train tasks: {len(train_dataset)}")
    print(f"Val tasks: {len(val_dataset)}")
    print(f"Test tasks: {len(test_dataset)}")
    print(f"Batch size: {batch_size}")
    
    return train_loader, val_loader, test_loader


class Model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_heads=4, ssagg_lambda=1.0, dropout=0.3, orthogonal_lambda=0.1):
        super(Model, self).__init__()
        self.orthogonal_lambda = orthogonal_lambda
        
        self.unisage = UniSAGE(
            in_channels=in_channels, 
            num_heads=num_heads,
            out_channels=1,
            hidden_channels=hidden_channels,
            ssagg_lambda=ssagg_lambda,
            dropout=dropout,
            orthogonal_method='loss'
        )
        
    def forward(self, x, edge_index_by_depth, list_index_sequences_by_depth, batch):
        x = self.unisage(x, edge_index_by_depth, list_index_sequences_by_depth)
        root_indices = self.get_root_indices(batch)
        graph_representations = x[root_indices]
        return graph_representations
    
    def get_orthogonal_loss(self):
        return self.unisage.get_orthogonal_loss()
    
    def use_orthogonal_loss(self):
        return self.unisage.use_orthogonal_loss()
    
    def get_root_indices(self, batch):
        root_indices = []
        current_offset = 0
        unique_graphs = torch.unique(batch, sorted=True)
        for graph_id in unique_graphs:
            root_indices.append(current_offset)
            nodes_in_graph = (batch == graph_id).sum().item()
            current_offset += nodes_in_graph
        return torch.tensor(root_indices, dtype=torch.long, device=batch.device)
    
    def reset_parameters(self):
        if hasattr(self.unisage, 'reset_parameters'):
            self.unisage.reset_parameters()

def apply_time_mask(batch, device, debug_first_batch=False):
    """
    Applies time masking to a batch of graphs.
    Filters edges and sequences based on the task timestamp of each graph.
    Returns two new dictionaries: masked_edge_index_by_depth and masked_list_sequences_by_depth.
    """
    task_ts_per_node = batch.task_timestamp[batch.batch]

    if debug_first_batch:
            print("\n" + "="*60)
            print("DEBUGGING TIME MASK ON THE FIRST BATCH")
            print(f"Batch contains {batch.num_graphs} graphs. Analyzing Graph 0...")
            graph_zero_task_ts = pd.to_datetime(batch.task_timestamp[0].item(), unit='s')
            print(f"Graph 0 Task Timestamp: {graph_zero_task_ts}")
            print("-" * 20)

    masked_edge_index_by_depth = {}
    for depth, edges in batch.edge_index_by_depth.items():
        if not edges: continue
        edge_index_tensor = torch.tensor(edges, dtype=torch.long, device=device).t()
        source_nodes = edge_index_tensor[0]
        
        task_timestamps_for_edges = task_ts_per_node[source_nodes]
        node_timestamps_for_edges = batch.node_timestamps[source_nodes]
        
        edge_mask = (node_timestamps_for_edges < task_timestamps_for_edges) | (node_timestamps_for_edges == -1)
        masked_edges_tensor = edge_index_tensor[:, edge_mask]
        
        if debug_first_batch:
            is_graph_zero_edge = (batch.batch[source_nodes] == 0)
            original_count = torch.sum(is_graph_zero_edge).item()
            masked_count = torch.sum(is_graph_zero_edge & edge_mask).item()
            if original_count > 0:
                print(f"Edge Masking at Depth {depth}:")
                print(f"  - Original Edges (Graph 0): {original_count}")
                print(f"  - Edges After Mask (Graph 0): {masked_count}")
                if masked_count < original_count:
                    print(f"  - SUCCESS: {original_count - masked_count} edge(s) were removed.")

        masked_edge_index_by_depth[depth] = masked_edges_tensor.t().tolist()
        
    if debug_first_batch:
        print("-" * 20)

    masked_list_sequences_by_depth = {}
    for depth, sequences in batch.list_index_sequences_by_depth.items():
        if not sequences: continue
        new_sequences_at_this_depth = []

        if debug_first_batch:
            original_seq_node_count = 0
            masked_seq_node_count = 0

        for seq_with_parent in sequences:
            if len(seq_with_parent) < 2: 
                continue

            seq_nodes = torch.tensor(seq_with_parent[:-1], dtype=torch.long, device=device)
            parent_node = seq_with_parent[-1]
            is_graph_zero_seq = (batch.batch[parent_node] == 0)
            
            graph_idx = batch.batch[parent_node]
            task_ts = batch.task_timestamp[graph_idx]
            
            node_timestamps = batch.node_timestamps[seq_nodes]
            node_mask = (node_timestamps < task_ts) | (node_timestamps == -1)
            historical_seq_nodes = seq_nodes[node_mask]

            if debug_first_batch and is_graph_zero_seq:
                original_seq_node_count += len(seq_nodes)
                masked_seq_node_count += len(historical_seq_nodes)

            if len(historical_seq_nodes) > 0:
                new_filtered_sequence = historical_seq_nodes.tolist() + [parent_node]
                new_sequences_at_this_depth.append(new_filtered_sequence)

        if debug_first_batch and original_seq_node_count > 0:
            print(f"Sequence Masking at Depth {depth}:")
            print(f"  - Original Sequence Nodes (Graph 0): {original_seq_node_count}")
            print(f"  - Sequence Nodes After Mask (Graph 0): {masked_seq_node_count}")
            if masked_seq_node_count < original_seq_node_count:
                print(f"  - SUCCESS: {original_seq_node_count - masked_seq_node_count} sequence node(s) were removed.")

        if new_sequences_at_this_depth:
            masked_list_sequences_by_depth[depth] = new_sequences_at_this_depth

    if debug_first_batch:
        print("="*60 + "\n")
    return masked_edge_index_by_depth, masked_list_sequences_by_depth

def train_model(model, train_loader, optimizer, criterion, device, epoch):
    """
    Train the model for one epoch
    """
    model.train()
    total_loss = 0
    total_primary_loss = 0
    total_orthogonal_loss = 0

    if not train_loader:
        return 0, 0, 0
    progress_bar = tqdm(train_loader, desc=f"Training on {device}", unit="batch")

    for batch_idx, batch in enumerate(progress_bar):
        if batch is None: continue
        batch = batch.to(device)
        optimizer.zero_grad()

        is_first_batch = (epoch == 0 and batch_idx == 0)
        masked_edge_index_by_depth, masked_list_sequences_by_depth = apply_time_mask(batch, device, debug_first_batch=is_first_batch)
        out = model(batch.x, masked_edge_index_by_depth, masked_list_sequences_by_depth, batch.batch)
        
        target = batch.y.view_as(out)
        primary_loss = criterion(out, target)

        orthogonal_loss = 0.0
        if model.use_orthogonal_loss():
            orthogonal_loss = model.get_orthogonal_loss()
            total_loss_tensor = primary_loss + model.orthogonal_lambda * orthogonal_loss
        else:
            total_loss_tensor = primary_loss

        total_loss_tensor.backward()
        optimizer.step()
        
        total_loss += total_loss_tensor.item()
        total_primary_loss += primary_loss.item()
        if isinstance(orthogonal_loss, torch.Tensor):
            total_orthogonal_loss += orthogonal_loss.item()
    
    avg_total_loss = total_loss / len(train_loader) if train_loader and len(train_loader) > 0 else 0
    avg_primary_loss = total_primary_loss / len(train_loader) if train_loader and len(train_loader) > 0 else 0
    avg_orthogonal_loss = total_orthogonal_loss / len(train_loader) if train_loader and len(train_loader) > 0 else 0
    
    return avg_total_loss, avg_primary_loss, avg_orthogonal_loss

def move_optimizer_to(optimizer, device):
    """
    Moves the state of an optimizer to a specified device.
    """
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

def val_model(model, val_loader, device, clamp_min: float, clamp_max: float):
    """
    Validate the model using MAE score
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(val_loader, desc="Validation", unit="batch")

    with torch.no_grad():
        for batch in progress_bar:
            if batch is None: continue
            batch = batch.to(device)
            
            masked_edge_index_by_depth, masked_list_sequences_by_depth = apply_time_mask(batch, device)
            
            out = model(batch.x, masked_edge_index_by_depth, masked_list_sequences_by_depth, batch.batch)
            
            pred = torch.clamp(out, min=clamp_min, max=clamp_max)
            all_preds.extend(np.atleast_1d(pred.squeeze().cpu().numpy()))
            all_labels.extend(np.atleast_1d(batch.y.squeeze().cpu().numpy()))

    if not all_labels: return np.array([]), np.array([])
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    return all_preds,all_labels

def test_model(model, test_loader, device, clamp_min: float, clamp_max: float):
    """
    Test the model using MAE score
    """
    model.eval()
    all_preds = []
    all_labels = []

    progress_bar = tqdm(test_loader, desc="Testing", unit="batch")

    with torch.no_grad():
        for batch in progress_bar:
            if batch is None: continue
            batch = batch.to(device)

            masked_edge_index_by_depth, masked_list_sequences_by_depth = apply_time_mask(batch, device)
            
            out = model(batch.x, masked_edge_index_by_depth, masked_list_sequences_by_depth, batch.batch)

            pred = torch.clamp(out, min=clamp_min, max=clamp_max)
            all_preds.extend(np.atleast_1d(pred.squeeze().cpu().numpy()))
            all_labels.extend(np.atleast_1d(batch.y.squeeze().cpu().numpy()))

    if not all_labels: return np.array([]), np.array([])
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    return all_preds, all_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on a preprocessed graph dataset.")
    parser.add_argument('--processed_path', type=str, required=True, help='Full path to the preprocessed graph dataset .pt file.')
    parser.add_argument('--task_path', type=str, required=True, help='Full path to the directory containing train/val/test.parquet task files.')
    parser.add_argument('--dataset_id_key', type=str, required=True, help="The entity ID key used when CREATING the graph dataset (e.g., 'driverId').")
    parser.add_argument('--task_id_key', type=str, required=True, help="The entity ID key in the TASK parquet files (e.g., 'UserId').")
    parser.add_argument('--label_key', type=str, required=True, help="The column name for the label in the task files (e.g., 'did_not_finish').")
    parser.add_argument('--timestamp_key', type=str, required=True, help="The column name for the timestamp in the task files.")
    parser.add_argument('--gpu_ids', type=str, default='0', help="Comma-separated list of GPU IDs to use (e.g., '0,1,2').")
    parser.add_argument("--node_threshold", type=int, default=50000, help="Threshold for the maximum number of nodes in a graph. Graphs exceeding this will be trained on cpu.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    # Hyperparameter Arguments
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training and evaluation.")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument('--hidden_channels', type=int, default=64, help="Number of hidden channels in the model.")
    parser.add_argument('--num_heads', type=int, default=2, help="Number of attention heads in the model.")
    parser.add_argument('--ssagg_lambda', type=float, default=1.5, help="Lambda for SSAGG layer.")
    parser.add_argument('--dropout', type=float, default=0.3, help="Dropout rate.")
    parser.add_argument('--orthogonal_lambda', type=float, default=0.1, help="Lambda for orthogonal loss.")
    
    args = parser.parse_args()

    seed_everything(args.seed)

    full_dataset = load_dataset(args.processed_path)
    gpu_device = torch.device(f'cuda:{args.gpu_ids}' if torch.cuda.is_available() else 'cpu')
    cpu_device = torch.device('cpu')

    if not full_dataset:
        raise ValueError("Graph dataset is empty.")
    
    first_data = full_dataset[0]
    print(first_data)
    entity_key = args.dataset_id_key
    if entity_key is None:
        raise AttributeError("Graphs in dataset must have a known entity ID attribute (e.g., 'driverId', 'customer_id').")
        
    small_graphs = [data for data in full_dataset if data.num_nodes <= args.node_threshold]
    large_graphs = [data for data in full_dataset if data.num_nodes > args.node_threshold]
    small_graph_map = {getattr(data, 'entity_id'): data for data in small_graphs}
    large_graph_map = {getattr(data, 'entity_id'): data for data in large_graphs}
    print(f"\nCreated a graph map with {len(small_graph_map)+len(large_graph_map)} entries using key '{entity_key}'.")
    
    base_task_path = os.path.expanduser(args.task_path)
    train_path = os.path.join(base_task_path, 'train.parquet')
    val_path = os.path.join(base_task_path, 'val.parquet')
    test_path = os.path.join(base_task_path, 'test.parquet')

    print(f"\nLoading tasks from: {base_task_path}")
    try:
        train_tasks = pd.read_parquet(train_path)
        val_tasks = pd.read_parquet(val_path)
        test_tasks = pd.read_parquet(test_path)
        print("Successfully loaded train, val, and test task files.")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Task file not found at {e.filename}. Please ensure you have downloaded the correct relbench task data.")

    task_config = {
        'entity_id_col': args.task_id_key,
        'label_col': args.label_key,
        'timestamp_col': args.timestamp_key
    }

    train_target_values = train_tasks[task_config['label_col']].to_numpy()
    clamp_min, clamp_max = np.percentile(train_target_values, [2, 98])

    print("\nCreating dataloaders for small graphs")
    s_train_loader, s_val_loader, s_test_loader = create_dataloaders(
        train_tasks, val_tasks, test_tasks, small_graph_map, task_config, batch_size=args.batch_size
    )
    print("\nCreating dataloaders for large graphs")
    l_train_loader, l_val_loader, l_test_loader = create_dataloaders(
        train_tasks, val_tasks, test_tasks, large_graph_map, task_config, batch_size=args.batch_size
    )

    in_channels = full_dataset[0].num_node_features
    
    model = Model(
        in_channels=in_channels, 
        hidden_channels=args.hidden_channels, 
        num_heads=args.num_heads,
        ssagg_lambda=args.ssagg_lambda,
        dropout=args.dropout,
        orthogonal_lambda=args.orthogonal_lambda
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.L1Loss() 
    
    print(f"\nModel: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    print(f"\nTraining for {args.epochs} epochs...")

    best_val_mae = float('inf')
    best_test_mae = float('inf')

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        model.to(gpu_device)
        move_optimizer_to(optimizer, gpu_device)
        train_loss_s, primary_loss_s, orthogonal_loss_s = train_model(model, s_train_loader, optimizer, criterion, gpu_device, epoch=epoch)

        model.to(cpu_device)
        move_optimizer_to(optimizer, cpu_device)
        train_loss_l, primary_loss_l, orthogonal_loss_l = train_model(model, l_train_loader, optimizer, criterion, cpu_device, epoch=epoch)

        train_loss = (train_loss_s + train_loss_l)
        primary_loss = (primary_loss_s + primary_loss_l)
        orthogonal_loss = (orthogonal_loss_s + orthogonal_loss_l)
        
        model.to(gpu_device)
        val_preds_s, val_labels_s = val_model(model, s_val_loader, gpu_device, clamp_min, clamp_max)
        model.to(cpu_device)
        val_preds_l, val_labels_l = val_model(model, l_val_loader, cpu_device, clamp_min, clamp_max)
        all_val_preds = np.concatenate([val_preds_s, val_preds_l])
        all_val_labels = np.concatenate([val_labels_s, val_labels_l])
        val_mae = mean_absolute_error(all_val_labels, all_val_preds)

        model.to(gpu_device)
        test_preds_s, test_labels_s = test_model(model, s_test_loader, gpu_device, clamp_min, clamp_max)
        model.to(cpu_device)
        test_preds_l, test_labels_l = test_model(model, l_test_loader, cpu_device, clamp_min, clamp_max)
        all_test_preds = np.concatenate([test_preds_s, test_preds_l])
        all_test_labels = np.concatenate([test_labels_s, test_labels_l])
        test_mae = mean_absolute_error(all_test_labels, all_test_preds)

        print(f"Epoch {epoch+1} Results -> Train Loss: {train_loss:.4f} (Primary: {primary_loss:.4f}, Orthogonal: {orthogonal_loss:.6f}), "
              f"Val MAE: {val_mae:.4f}, Test MAE: {test_mae:.4f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_test_mae = test_mae
    
    print(f"\nBest Validation MAE: {best_val_mae:.4f}")
    print(f"Corresponding Test MAE: {best_test_mae:.4f}")