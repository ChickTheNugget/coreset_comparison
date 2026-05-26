import heapq

import numpy as np


class EGQCoreset:
    """Ranked exponential-grid/quadtree coreset.

    The splitter ranks cells by count * side_length^2 / reference_cost, then
    keeps splitting the highest-ranked active leaf until it reaches the target
    size. Representatives are one sampled point per leaf, weighted by leaf size.
    """

    def __init__(self, m, k=None):
        self.m = int(m)
        self.k = k
        self.weights = None
        self.indices = None
        self.sensitivities = None

        self.cells = None
        self.raw_size = None
        self.trimmed_to_target = False
        self.reference_centers = None
        self.reference_cost = None
        self.z_used = None
        self.eps_min = None
        self.eps_report = None
        self.beta_used = None
        self.rank_interval = None

    def generate(self, data):
        data = np.asarray(data, dtype=float)
        n, d = data.shape

        if n == 0:
            self.indices = np.empty(0, dtype=int)
            self.weights = np.empty(0, dtype=float)
            self.cells = []
            return data

        target_size = max(1, min(self.m, n))
        reference_k = self.k if self.k is not None else max(2, int(np.sqrt(target_size)))
        reference_k = max(1, min(int(reference_k), n))

        self.reference_centers = _reference_centers(data, reference_k, n_iterations=4)
        self.reference_cost = _kmeans_cost(data, self.reference_centers)

        leaves, z_used, interval = _build_cells_by_ranked_splits(
            data,
            target_size,
            self.reference_cost,
            reference_k,
            max_depth=64,
        )

        eps_min = z_used ** (1.0 / d) if z_used >= 0.0 else np.nan
        eps_report = _round_up_to_step(eps_min, 0.1)
        beta_used = z_used / (eps_report ** d) if eps_report > 0.0 else np.nan

        reps, weights, indices, cells = _representatives_from_cells(data, leaves)
        self.raw_size = int(reps.shape[0])

        if reps.shape[0] > target_size:
            keep = _top_weight_indices(weights, target_size)
            reps = reps[keep]
            weights = weights[keep]
            indices = indices[keep]
            cells = [cells[idx] for idx in keep]
            weight_sum = float(np.sum(weights))
            if weight_sum > 0.0:
                weights = weights * (n / weight_sum)
            self.trimmed_to_target = True

        self.weights = weights
        self.indices = indices
        self.sensitivities = None
        self.cells = cells
        self.z_used = z_used
        self.eps_min = eps_min
        self.eps_report = eps_report
        self.beta_used = beta_used
        self.rank_interval = interval

        return reps.reshape((-1, d))


def _reference_centers(data, k, n_iterations):
    centers = _kmeans_plus_plus_init(data, k)
    for _ in range(n_iterations):
        labels = _nearest_center_indices(data, centers)
        updated = centers.copy()
        for center_idx in range(k):
            members = data[labels == center_idx]
            if members.size == 0:
                updated[center_idx] = data[np.random.randint(data.shape[0])]
            else:
                updated[center_idx] = np.mean(members, axis=0)
        if np.allclose(updated, centers):
            break
        centers = updated
    return centers


def _kmeans_plus_plus_init(data, k):
    n = data.shape[0]
    centers = np.empty((k, data.shape[1]), dtype=float)
    centers[0] = data[np.random.randint(n)]
    closest_sq = _squared_distances_to_point(data, centers[0])
    for center_idx in range(1, k):
        total = float(np.sum(closest_sq))
        if total <= 0.0:
            next_idx = int(np.random.randint(n))
        else:
            next_idx = int(np.random.choice(n, p=closest_sq / total))
        centers[center_idx] = data[next_idx]
        closest_sq = np.minimum(closest_sq, _squared_distances_to_point(data, centers[center_idx]))
    return centers


def _squared_distances_to_point(data, point):
    diff = data - point
    return np.sum(diff * diff, axis=1)


def _nearest_center_indices(data, centers, chunk_size=250000):
    labels = np.empty(data.shape[0], dtype=int)
    for start in range(0, data.shape[0], chunk_size):
        chunk = data[start:start + chunk_size]
        distances = _squared_distances_to_centers(chunk, centers)
        labels[start:start + chunk.shape[0]] = np.argmin(distances, axis=1)
    return labels


def _kmeans_cost(data, centers, chunk_size=250000):
    cost = 0.0
    for start in range(0, data.shape[0], chunk_size):
        chunk = data[start:start + chunk_size]
        distances = _squared_distances_to_centers(chunk, centers)
        cost += float(np.sum(np.min(distances, axis=1)))
    return cost


def _squared_distances_to_centers(chunk, centers):
    diff = chunk[:, np.newaxis, :] - centers[np.newaxis, :, :]
    return np.sum(diff * diff, axis=2)


def _build_cells_by_ranked_splits(data, target_size, cost, k, max_depth):
    n, _ = data.shape
    root = _make_root_cell(data)
    leaves = [root]
    active_leaf_ids = {id(root)}
    active_leaf_count = 1

    if target_size <= 1 or cost <= 0.0:
        z_used, interval = _z_from_priority_interval(np.inf, 0.0, k, n)
        return leaves, z_used, interval

    heap = []
    serial = 0
    last_split_priority = None
    first_blocked_priority = None
    serial = _push_ranked_cell(heap, root, cost, serial)

    while active_leaf_count < target_size and heap:
        priority, cell = _pop_ranked_cell(heap, active_leaf_ids)
        if cell is None:
            break
        children = _split_cell(data, cell, max_depth)
        if len(children) <= 1:
            cell["splittable"] = False
            first_blocked_priority = priority
            continue
        active_leaf_ids.discard(id(cell))
        leaves.extend(children)
        active_leaf_count += len(children) - 1
        last_split_priority = priority
        for child in children:
            active_leaf_ids.add(id(child))
            serial = _push_ranked_cell(heap, child, cost, serial)

    if last_split_priority is not None:
        serial = _split_equal_priority_boundary(
            data,
            leaves,
            active_leaf_ids,
            heap,
            cost,
            max_depth,
            last_split_priority,
            serial,
        )

    next_priority, cell = _pop_ranked_cell(heap, active_leaf_ids)
    if cell is not None:
        heapq.heappush(heap, (-next_priority, serial, cell))
    if next_priority is None:
        next_priority = first_blocked_priority

    z_used, interval = _z_from_priority_interval(last_split_priority, next_priority, k, n)
    leaves = [leaf for leaf in leaves if id(leaf) in active_leaf_ids]
    return leaves, z_used, interval


def _push_ranked_cell(heap, cell, cost, serial):
    if cell["splittable"]:
        heapq.heappush(heap, (-_split_priority(cell, cost), serial, cell))
        serial += 1
    return serial


def _pop_ranked_cell(heap, active_leaf_ids):
    while heap:
        neg_priority, _, cell = heapq.heappop(heap)
        if cell["splittable"] and id(cell) in active_leaf_ids:
            return -neg_priority, cell
    return None, None


def _split_equal_priority_boundary(
    data,
    leaves,
    active_leaf_ids,
    heap,
    cost,
    max_depth,
    boundary_priority,
    serial,
):
    while True:
        priority, cell = _pop_ranked_cell(heap, active_leaf_ids)
        if cell is None:
            break
        if not np.isclose(priority, boundary_priority, rtol=1e-12, atol=1e-15):
            heapq.heappush(heap, (-priority, serial, cell))
            return serial + 1
        children = _split_cell(data, cell, max_depth)
        if len(children) <= 1:
            cell["splittable"] = False
            continue
        active_leaf_ids.discard(id(cell))
        leaves.extend(children)
        for child in children:
            active_leaf_ids.add(id(child))
            serial = _push_ranked_cell(heap, child, cost, serial)
    return serial


def _split_priority(cell, cost):
    if cost <= 0.0:
        return 0.0
    return cell["count"] * (cell["side_length"] ** 2) / cost


def _z_from_priority_interval(last_split_priority, next_priority, k, n):
    log_term = np.log(n) + 1.0
    lower_alpha = 0.0 if next_priority is None else next_priority
    upper_alpha = np.inf if last_split_priority is None else last_split_priority
    if lower_alpha is None:
        lower_alpha = 0.0
    if upper_alpha is None:
        upper_alpha = np.inf
    if not np.isfinite(upper_alpha):
        alpha = max(lower_alpha, 1.0)
    elif lower_alpha <= 0.0:
        alpha = 0.5 * upper_alpha
    elif lower_alpha < upper_alpha:
        alpha = np.sqrt(lower_alpha * upper_alpha)
    else:
        alpha = upper_alpha
    return alpha * k * log_term, (lower_alpha, upper_alpha)


def _round_up_to_step(value, step):
    if not np.isfinite(value) or value <= 0.0:
        return value
    return np.ceil(value / step) * step


def _make_root_cell(data):
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)
    side_length = float(np.max(maxs - mins))
    maxs = mins + side_length
    return {
        "indices": np.arange(data.shape[0], dtype=int),
        "count": int(data.shape[0]),
        "depth": 0,
        "bounds_min": mins,
        "bounds_max": maxs,
        "side_length": side_length,
        "splittable": side_length > 0.0 and data.shape[0] > 1,
    }


def _split_cell(data, cell, max_depth):
    if cell["depth"] >= max_depth or cell["side_length"] <= 0.0 or cell["count"] <= 1:
        return []
    d = data.shape[1]
    mid = 0.5 * (cell["bounds_min"] + cell["bounds_max"])
    children = []
    for subcube_idx in range(2 ** d):
        sub_min = cell["bounds_min"].copy()
        sub_max = cell["bounds_max"].copy()
        sub_indices = cell["indices"]
        for dim in range(d):
            if (subcube_idx >> dim) & 1 == 0:
                sub_max[dim] = mid[dim]
                mask = data[sub_indices, dim] <= mid[dim]
            else:
                sub_min[dim] = mid[dim]
                mask = data[sub_indices, dim] > mid[dim]
            sub_indices = sub_indices[mask]
        if sub_indices.size == 0:
            continue
        side_length = float(sub_max[0] - sub_min[0])
        children.append(
            {
                "indices": sub_indices,
                "count": int(sub_indices.size),
                "depth": cell["depth"] + 1,
                "bounds_min": sub_min,
                "bounds_max": sub_max,
                "side_length": side_length,
                "splittable": side_length > 0.0 and sub_indices.size > 1,
            }
        )
    if len(children) <= 1 or max(child["count"] for child in children) == cell["count"]:
        return []
    return children


def _representatives_from_cells(data, cells):
    reps = []
    weights = []
    indices = []
    bounds = []
    for cell in cells:
        rep_idx = int(np.random.choice(cell["indices"]))
        reps.append(data[rep_idx])
        weights.append(float(cell["count"]))
        indices.append(rep_idx)
        bounds.append(_bounds_to_tuple(cell["bounds_min"], cell["bounds_max"]))
    return np.vstack(reps), np.asarray(weights), np.asarray(indices), bounds


def _top_weight_indices(weights, target_size):
    order = np.lexsort((np.arange(weights.size), -weights))
    return np.sort(order[:target_size])


def _bounds_to_tuple(bounds_min, bounds_max):
    values = []
    for dim in range(bounds_min.size):
        values.extend([float(bounds_min[dim]), float(bounds_max[dim])])
    return tuple(values)
