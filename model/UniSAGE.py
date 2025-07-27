import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GATConv
from .SSAgg import SSAgg


class UniSAGE(MessagePassing):
    """
    UniSAGE: A unified graph neural network with hierarchical structure and orthogonal transformations.
    
    This model supports two orthogonalization methods for W1 and W2:
    1. 'loss' (default): Uses orthogonality loss during training to encourage W1 and W2 to be orthogonal
    2. 'direct': Directly orthogonalizes W2 with respect to W1 using Gram-Schmidt process
    
    When using 'loss' method, make sure to add get_orthogonal_loss() to your training loss.
    """
    def __init__(self, in_channels, out_channels, hidden_channels=None, num_heads=8, gru_hidden_size=None, ssagg_lambda=1.0, dropout=0.3, orthogonal_method='loss'):
        super(UniSAGE, self).__init__()
        
        # Set default values
        if hidden_channels is None:
            hidden_channels = in_channels
        if gru_hidden_size is None:
            gru_hidden_size = hidden_channels
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.gru_hidden_size = gru_hidden_size
        self.ssagg_lambda = ssagg_lambda
        self.dropout = dropout
        self.orthogonal_method = orthogonal_method  # 'loss' or 'direct'
        
        # Two learnable linear transformations W1 and W2 directly on input
        self.W1 = nn.Linear(in_channels, hidden_channels, bias=False)
        self.W2 = nn.Linear(in_channels, hidden_channels, bias=False)
        
        # Two GAT layers with different parameters for H1 and H2
        self.gat1 = GATConv(hidden_channels, hidden_channels, heads=num_heads, concat=False, dropout=dropout)
        self.gat2 = GATConv(hidden_channels, hidden_channels, heads=num_heads, concat=False, dropout=dropout)
        
        # Shared GRU for sequence processing
        self.gru = nn.GRU(hidden_channels, gru_hidden_size, batch_first=True)
        
        # SSAgg layer for final aggregation
        self.ssagg = SSAgg(hidden_channels * 2, hidden_channels * 2, heads=num_heads, concat=False, Lambda=ssagg_lambda, dropout=dropout)
        
        # Final linear transformation (input is now 4 * hidden_channels due to concat)
        self.final_linear = nn.Linear(hidden_channels * 4, out_channels)
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        # Initialize linear layers
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)
        
        # Initialize GRU
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # Initialize final linear layer
        nn.init.xavier_uniform_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)
    
    def get_orthogonal_W2(self):
        """
        Get orthogonalized W2 with respect to W1 using Gram-Schmidt process
        W2_ortho = W2 - proj_W1(W2)
        """
        W1_weight = self.W1.weight  # [hidden_channels, in_channels]
        W2_weight = self.W2.weight  # [hidden_channels, in_channels]
        
        # Compute projection of W2 onto W1
        # proj_W1(W2) = W1 * (W1^T * W2) / (W1^T * W1)
        
        # Compute W1^T * W1 (inner product matrix)
        W1T_W1 = torch.mm(W1_weight, W1_weight.T)  # [hidden_channels, hidden_channels]
        
        # Compute W1^T * W2
        W1T_W2 = torch.mm(W1_weight, W2_weight.T)  # [hidden_channels, hidden_channels]
        
        # Add small regularization for numerical stability
        reg_term = 1e-6 * torch.eye(self.hidden_channels, device=W1_weight.device)
        W1T_W1_reg = W1T_W1 + reg_term
        
        # Solve for coefficients: W1T_W1_reg @ coeffs = W1T_W2
        # coeffs = torch.linalg.solve(W1T_W1_reg, W1T_W2)  # [hidden_channels, hidden_channels]
        coeffs = torch.linalg.lstsq(W1T_W1_reg, W1T_W2).solution

        # Compute orthogonal W2: W2_ortho = W2 - W1 @ coeffs
        W2_ortho = W2_weight - torch.mm(coeffs, W1_weight)  # [hidden_channels, in_channels]
        
        return W2_ortho

    def get_orthogonal_loss(self):
        """
        Compute orthogonality loss between W1 and W2 weight matrices.
        The loss encourages W1 and W2 to be orthogonal to each other.
        
        Returns:
            orthogonal_loss: Scalar tensor representing the orthogonality loss
        """
        W1_weight = self.W1.weight  # [hidden_channels, in_channels]
        W2_weight = self.W2.weight  # [hidden_channels, in_channels]
        
        # Method 1: Frobenius inner product
        # Loss = ||W1^T @ W2||_F^2 (should be 0 for orthogonal matrices)
        inner_product = torch.mm(W1_weight, W2_weight.T)  # [hidden_channels, hidden_channels]
        frobenius_loss = torch.norm(inner_product, p='fro') ** 2
        
        # Method 2: Cosine similarity based loss (alternative approach)
        # Normalize the weight matrices
        W1_norm = F.normalize(W1_weight, p=2, dim=1)  # [hidden_channels, in_channels]
        W2_norm = F.normalize(W2_weight, p=2, dim=1)  # [hidden_channels, in_channels]
        
        # Compute cosine similarity matrix
        cosine_sim = torch.mm(W1_norm, W2_norm.T)  # [hidden_channels, hidden_channels]
        
        # Remove diagonal elements (self-similarity) if dimensions match
        if W1_weight.shape[0] == W2_weight.shape[0]:
            # Set diagonal to 0 to ignore self-similarity
            mask = torch.eye(cosine_sim.shape[0], device=cosine_sim.device)
            cosine_sim = cosine_sim * (1 - mask)
        
        cosine_loss = torch.norm(cosine_sim, p='fro') ** 2
        
        # Combine both losses (you can adjust weights as needed)
        orthogonal_loss = frobenius_loss + 0.1 * cosine_loss
        
        return orthogonal_loss

    def use_orthogonal_loss(self):
        """
        Check if orthogonal loss should be used in training.
        
        Returns:
            bool: True if using 'loss' method, False if using 'direct' method
        """
        return self.orthogonal_method == 'loss'
    
    def get_orthogonal_method(self):
        """
        Get the current orthogonalization method.
        
        Returns:
            str: Current orthogonalization method ('loss' or 'direct')
        """
        return self.orthogonal_method
    
    def apply_gru_to_sequences(self, H2, list_index_sequences_by_depth, depth):
        """
        Apply GRU to sequences at a specific depth and add results to parent nodes
        """
        H2_updated = H2.clone()
        
        if depth not in list_index_sequences_by_depth:
            return H2_updated
        
        sequences = list_index_sequences_by_depth[depth]
        
        for seq in sequences:
            if len(seq) < 2:  # Need at least 2 nodes for a meaningful sequence
                continue
                
            # Extract features for the sequence (excluding the parent node)
            seq_nodes = seq[:-1]  # All nodes except the last one (parent)
            parent_node = seq[-1]  # The last node is the parent
            
            if len(seq_nodes) == 0:
                continue
                
            # Get features for sequence nodes
            seq_features = H2[seq_nodes]  # [seq_len, hidden_channels]
            
            # Apply GRU
            seq_features = seq_features.unsqueeze(0)  # Add batch dimension: [1, seq_len, hidden_channels]
            gru_output, _ = self.gru(seq_features)  # [1, seq_len, gru_hidden_size]
            
            # Use the last output of GRU
            gru_final = gru_output[0, -1, :]  # [gru_hidden_size]
            
            # If GRU hidden size != hidden channels, we need to project
            if self.gru_hidden_size != self.hidden_channels:
                # Create a linear layer for projection if needed
                if not hasattr(self, 'gru_projection'):
                    self.gru_projection = nn.Linear(self.gru_hidden_size, self.hidden_channels).to(H2.device)
                gru_final = self.gru_projection(gru_final)
            
            # Add to parent node
            H2_updated[parent_node] = H2_updated[parent_node] + gru_final
        
        return H2_updated
    
    def hierarchical_propagation(self, H1, H2, edge_index_by_depth, list_index_sequences_by_depth):
        """
        Perform hierarchical information propagation from far to near
        """
        # Get all depths and sort from far to near (descending order)
        depths = sorted(edge_index_by_depth.keys(), reverse=True)
        
        H1_current = H1.clone()
        H2_current = H2.clone()
        
        for depth in depths:
            # Get edges for this depth
            edges = edge_index_by_depth[depth]
            if len(edges) == 0:
                continue
                
            # Convert to edge_index format [2, num_edges]
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(H1.device)
            
            # Apply GAT aggregation for H1 and H2
            H1_current = self.gat1(H1_current, edge_index)
            H2_current = self.gat2(H2_current, edge_index)
            
            # Apply GRU to sequences at this depth
            H2_current = self.apply_gru_to_sequences(H2_current, list_index_sequences_by_depth, depth)
        
        return H1_current, H2_current
    
    def ssagg_propagation(self, H, edge_index_by_depth):
        """
        Apply SSAgg propagation from far to near
        """
        # Get all depths and sort from far to near (descending order)
        depths = sorted(edge_index_by_depth.keys(), reverse=True)
        
        H_current = H.clone()
        
        for depth in depths:
            # Get edges for this depth
            edges = edge_index_by_depth[depth]
            if len(edges) == 0:
                continue
                
            # Convert to edge_index format [2, num_edges]
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(H.device)
            
            # Apply SSAgg
            H_current = self.ssagg(H_current, edge_index)
        
        return H_current
    
    def forward(self, x, edge_index_by_depth, list_index_sequences_by_depth):
        """
        Forward pass of UniSAGE model
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index_by_depth: Dict of depth -> List of [child, parent] edges
            list_index_sequences_by_depth: Dict of depth -> List of node sequences
        
        Returns:
            out: Node representations [num_nodes, out_channels]
        """
        # Step 1: Apply linear transformations directly on input
        H1 = self.W1(x)  # [num_nodes, hidden_channels]
        
        # Apply W2 based on orthogonal method
        if self.orthogonal_method == 'direct':
            # Use direct orthogonalization (original approach)
            W2_ortho = self.get_orthogonal_W2()
            H2 = F.linear(x, W2_ortho)  # [num_nodes, hidden_channels]
        else:
            # Use loss-based orthogonalization (default)
            # Apply W2 directly without orthogonalization
            # Orthogonality will be enforced through loss in training
            H2 = self.W2(x)  # [num_nodes, hidden_channels]
        
        # Step 2: Hierarchical propagation
        H1_prop, H2_prop = self.hierarchical_propagation(H1, H2, edge_index_by_depth, list_index_sequences_by_depth)
        
        # Step 3: Construct fusion representation by concatenation
        H_fusion = torch.cat([H1_prop, H2_prop], dim=1)  # [num_nodes, 2 * hidden_channels]
        
        # Step 4: Apply SSAgg propagation
        H_prime = self.ssagg_propagation(H_fusion, edge_index_by_depth)
        
        # Step 5: Concatenate H_fusion and H_prime and apply final linear transformation
        H_concat = torch.cat([H_fusion, H_prime], dim=1)  # [num_nodes, 4 * hidden_channels]
        
        # Final linear transformation
        out = self.final_linear(H_concat)  # [num_nodes, out_channels]
        
        return out

