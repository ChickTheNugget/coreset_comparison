import numpy as np
from sklearn.cluster import kmeans_plusplus
from sklearn.metrics import pairwise_distances_argmin

class BiasedCoreset:
    def __init__(self, m, k=None):
        self.m = m 
        self.k = k
        self.weights = None
        self.indices = None

    def generate(self, data):
        n = data.shape[0]
        k = self.k
        q = self.m
        
        # Fast Initial Approximation (Cython Backend)
        subset_size = min(n, max(1024, q))
        subset_indices = np.random.choice(n, size=subset_size, replace=False)
        subset = data[subset_indices]
        
        # Note: Removing random_state=42 so it respects the team's global seed in run_grid
        centers, _ = kmeans_plusplus(subset, n_clusters=k)
        labels = pairwise_distances_argmin(data, centers)
        
        # Contiguous Memory Sorting
        sort_indices = np.argsort(labels)
        sorted_labels = labels[sort_indices]
        sorted_data = data[sort_indices]
        original_indices_sorted = np.arange(n)[sort_indices]
        
        cluster_sizes = np.bincount(sorted_labels, minlength=k)
        
        coreset_points = []
        coreset_weights = []
        coreset_original_indices = []
        
        current_index = 0
        
        # Partitioning and Proportional Budgeting
        for i in range(k):
            size = cluster_sizes[i]
            if size == 0:
                continue
                
            # Slice continuous blocks
            cluster_points = sorted_data[current_index : current_index + size]
            cluster_orig_idx = original_indices_sorted[current_index : current_index + size]
            center = centers[i]
            
            # Vectorized Mean Radius
            distances_sq = np.sum((cluster_points - center) ** 2, axis=1)
            mean_dist_sq = np.mean(distances_sq)
            
            # Masks
            inner_mask = distances_sq <= mean_dist_sq
            outer_mask = ~inner_mask
            
            inner_pts = cluster_points[inner_mask]
            outer_pts = cluster_points[outer_mask]
            inner_orig_idx = cluster_orig_idx[inner_mask]
            outer_orig_idx = cluster_orig_idx[outer_mask]
            outer_dist_sq = distances_sq[outer_mask]
            
            # Dynamic Proportional Budgeting
            budget_per_cluster = q // k
            inner_ratio = len(inner_pts) / size
            inner_sample = max(1, int(budget_per_cluster * inner_ratio))
            outer_sample = max(1, budget_per_cluster - inner_sample)
            
            # Sample Inner Points
            if len(inner_pts) > 0:
                act_in = min(inner_sample, len(inner_pts))
                idx_in = np.random.choice(len(inner_pts), size=act_in, replace=False)
                
                coreset_points.append(inner_pts[idx_in])
                coreset_weights.append(np.full(act_in, len(inner_pts) / act_in))
                coreset_original_indices.append(inner_orig_idx[idx_in])
                
            # Sample Outer Points
            if len(outer_pts) > 0:
                act_out = min(outer_sample, len(outer_pts))
                cost_outer = np.sum(outer_dist_sq)
                
                if cost_outer == 0:
                    idx_out = np.random.choice(len(outer_pts), size=act_out, replace=False)
                    coreset_points.append(outer_pts[idx_out])
                    coreset_weights.append(np.full(act_out, len(outer_pts) / act_out))
                    coreset_original_indices.append(outer_orig_idx[idx_out])
                else:
                    probs = outer_dist_sq / cost_outer
                    idx_out = np.random.choice(len(outer_pts), size=act_out, replace=False, p=probs)
                    
                    coreset_points.append(outer_pts[idx_out])
                    sampled_dists = outer_dist_sq[idx_out]
                    coreset_weights.append(cost_outer / (act_out * sampled_dists))
                    coreset_original_indices.append(outer_orig_idx[idx_out])
                    
            current_index += size

        # Final Aggregation
        final_points = np.vstack(coreset_points)
        self.weights = np.concatenate(coreset_weights)
        self.indices = np.concatenate(coreset_original_indices)
        
        return final_points
