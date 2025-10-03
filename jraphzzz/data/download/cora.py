import os.path as osp
import pickle
from typing import Dict, List, Tuple

import numpy as np
import jax.numpy as jnp

from jraphzzz import GraphsTuple

import os
import urllib.request

# make sure you already have:
# from jraph import GraphsTuple
# and read_planetoid_data(...) defined in this module

def read_file(folder: str, prefix: str, name: str):
    """
    Load one Planetoid file. Files are pickled Python objects except
    `test.index` which is a plain text list of ints in most releases.
    Returns numpy arrays or Python objects (dict for `graph`).
    """
    path = osp.join(folder, f"ind.{prefix.lower()}.{name}")

    if name == "test.index":
        # Most Planetoid releases provide this as a text file (one index per line).
        # Fall back to trying pickle if text read fails.
        try:
            with open(path, "r") as f:
                lines = [int(line.strip()) for line in f if line.strip()]
            return np.asarray(lines, dtype=np.int64)
        except Exception:
            # try pickle fallback
            with open(path, "rb") as f:
                obj = pickle.load(f, encoding="latin1")
            return np.asarray(obj, dtype=np.int64)

    # For everything else, it's stored as a pickle (sometimes scipy.sparse objects).
    with open(path, "rb") as f:
        obj = pickle.load(f, encoding="latin1")

    # Convert scipy sparse to dense numpy if necessary
    if hasattr(obj, "toarray"):
        obj = obj.toarray()

    # If it's a list/tuple/ndarray -> numpy array
    if isinstance(obj, (list, tuple, np.ndarray)):
        return np.asarray(obj)
    return obj  # e.g. graph dict


def index_to_mask(index: np.ndarray, size: int) -> np.ndarray:
    mask = np.zeros(size, dtype=bool)
    if index is None or index.size == 0:
        return mask
    idx = np.asarray(index, dtype=int)
    mask[idx] = True
    return mask


def edge_index_from_dict(graph_dict: Dict[int, List[int]], num_nodes: int) -> np.ndarray:
    """
    Convert adjacency dict {node: [nbrs,...], ...} to edge_index shape (2, E),
    where row 0 = senders (sources), row 1 = receivers (targets).
    """
    rows: List[int] = []
    cols: List[int] = []
    for src, nbrs in graph_dict.items():
        # sometimes nbrs can be a set/list/array
        for nbr in nbrs:
            rows.append(int(src))
            cols.append(int(nbr))
    if len(rows) == 0:
        return np.zeros((2, 0), dtype=np.int64)
    return np.vstack([np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64)])


def read_planetoid_data(folder: str, prefix: str
                       ) -> Tuple[GraphsTuple, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Load Planetoid dataset and return:
      - GraphsTuple (nodes, edges=None, receivers, senders, globals=None, n_node, n_edge)
      - y (labels as 1D jnp.ndarray of ints length N)
      - train_mask (jnp.bool_, shape (N,))
      - val_mask (jnp.bool_, shape (N,))
      - test_mask (jnp.bool_, shape (N,))

    This mirrors the original PyG loader behavior (including citeseer/nell special handling)
    but uses numpy + jax.numpy (no torch).
    """
    names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
    items = [read_file(folder, prefix, name) for name in names]
    x_train_file, tx, allx, y_train_file, ty, ally, graph, test_index = items

    # Ensure plain numpy arrays for matrix-like objects (and ints for indices)
    def _to_numpy(obj):
        if hasattr(obj, "toarray"):
            return obj.toarray()
        if isinstance(obj, jnp.ndarray):
            return np.array(obj)
        if isinstance(obj, np.ndarray):
            return obj
        if isinstance(obj, (list, tuple)):
            return np.array(obj)
        return obj

    x_train_file = _to_numpy(x_train_file)
    tx = _to_numpy(tx)
    allx = _to_numpy(allx)
    y_train_file = _to_numpy(y_train_file)
    ty = _to_numpy(ty)
    ally = _to_numpy(ally)
    test_index = np.asarray(test_index, dtype=int)

    # Indices for train/val as in original
    train_index = np.arange(y_train_file.shape[0], dtype=int)
    val_index = np.arange(y_train_file.shape[0], y_train_file.shape[0] + 500, dtype=int)
    sorted_test_index = np.sort(test_index)

    prefix_l = prefix.lower()

    # CITSEER: some test nodes are isolated; expand tx/ty to full contiguous range
    if prefix_l == 'citeseer':
        len_test_indices = int(test_index.max() - test_index.min()) + 1
        tx_ext = np.zeros((len_test_indices, tx.shape[1]), dtype=tx.dtype)
        tx_ext[sorted_test_index - test_index.min(), :] = tx
        ty_ext = np.zeros((len_test_indices, ty.shape[1]), dtype=ty.dtype)
        ty_ext[sorted_test_index - test_index.min(), :] = ty
        tx, ty = tx_ext, ty_ext

    # NELL special-case: mirror original behavior (dense-friendly handling here)
    if prefix_l == 'nell.0.001':
        total_nodes = len(graph)
        # feature dim from original x_train_file
        feat_dim = x_train_file.shape[1]
        tx_ext = np.zeros((total_nodes - allx.shape[0], feat_dim), dtype=x_train_file.dtype)
        tx_ext[sorted_test_index - allx.shape[0], :] = tx
        # label dim (ally / ty) assumed 2D one-hot in original files
        label_dim = ally.shape[1] if ally.ndim == 2 else 1
        ty_ext = np.zeros((total_nodes - ally.shape[0], label_dim), dtype=ty.dtype)
        ty_ext[sorted_test_index - ally.shape[0], :] = ty
        tx, ty = tx_ext, ty_ext

        x_combined = np.vstack([allx, tx])
        # reorder test rows to match sorted_test_index
        x_combined[test_index] = x_combined[sorted_test_index]
        # note: original created a CSR relation-feature matrix for isolated nodes.
        # we omit creating that CSR here (dense-friendly). Add scipy.sparse logic if needed.
    else:
        # default stacking and test-row reorder
        x_combined = np.vstack([allx, tx])
        x_combined[test_index] = x_combined[sorted_test_index]

    # labels: ally + ty -> convert from one-hot (if needed)
    y_all = np.vstack([ally, ty])
    if y_all.ndim == 2:
        y_labels = np.argmax(y_all, axis=1).astype(np.int64)
    else:
        y_labels = y_all.astype(np.int64).reshape(-1)

    # reorder labels for test indices
    y_labels[test_index] = y_labels[sorted_test_index]

    num_nodes = y_labels.shape[0]

    train_mask = index_to_mask(train_index, size=num_nodes)
    val_mask = index_to_mask(val_index, size=num_nodes)
    test_mask = index_to_mask(test_index, size=num_nodes)

    # Convert adjacency dict to edge_index (2, E)
    edge_index = edge_index_from_dict(graph_dict=graph, num_nodes=num_nodes)
    if edge_index.shape[1] == 0:
        senders_np = np.zeros((0,), dtype=np.int64)
        receivers_np = np.zeros((0,), dtype=np.int64)
    else:
        senders_np = edge_index[0, :].astype(np.int64)
        receivers_np = edge_index[1, :].astype(np.int64)

    # Convert everything to jax arrays
    nodes_j = jnp.asarray(x_combined)
    senders_j = jnp.asarray(senders_np, dtype=jnp.int32)
    receivers_j = jnp.asarray(receivers_np, dtype=jnp.int32)
    n_node_j = jnp.asarray(num_nodes, dtype=jnp.int32)
    n_edge_j = jnp.asarray(senders_np.shape[0], dtype=jnp.int32)

    graphs = GraphsTuple(
        nodes=nodes_j,
        edges=None,
        receivers=receivers_j,
        senders=senders_j,
        globals=None,
        n_node=n_node_j,
        n_edge=n_edge_j,
    )

    y_j = jnp.asarray(y_labels, dtype=jnp.int32)
    train_mask_j = jnp.asarray(train_mask)
    val_mask_j = jnp.asarray(val_mask)
    test_mask_j = jnp.asarray(test_mask)

    return graphs, y_j, train_mask_j, val_mask_j, test_mask_j


class Planetoid:
    url = 'https://github.com/kimiyoung/planetoid/raw/master/data'
    geom_gcn_url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/'
                    'geom-gcn/master')

    def __init__(
        self,
        root: str,
        name: str,
        split: str = "public",
        num_train_per_class: int = 20,
        num_val: int = 500,
        num_test: int = 1000,
        force_reload: bool = False,
    ) -> None:
        self.root = root
        self.name = name
        self.split = split.lower()
        assert self.split in ['public', 'full', 'geom-gcn', 'random']
        self.num_train_per_class = num_train_per_class
        self.num_val = num_val
        self.num_test = num_test

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        self._processed_path = osp.join(self.processed_dir, 'data.npz')

        if force_reload or not osp.exists(self._processed_path):
            self.download()
            self.process()
            self._save_processed()
        else:
            self._load_processed()

    # --- directories & filenames (same logic as original) ---
    @property
    def raw_dir(self) -> str:
        if self.split == 'geom-gcn':
            return osp.join(self.root, self.name, 'geom-gcn', 'raw')
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        if self.split == 'geom-gcn':
            return osp.join(self.root, self.name, 'geom-gcn', 'processed')
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
        return [f'ind.{self.name.lower()}.{name}' for name in names]
    
        ...
    @property
    def num_features(self) -> int:
        """Number of input features per node."""
        if self.graphs.nodes is None:
            return 0
        return self.graphs.nodes.shape[1]

    @property
    def num_classes(self) -> int:
        """Number of distinct node labels."""
        return int(jnp.max(self.y)) + 1

    def __getitem__(self, idx: int):
        """
        PyG-style indexing. We only have one graph,
        so dataset[0] returns the GraphsTuple + labels/masks.
        """
        if idx != 0:
            raise IndexError("Planetoid contains only one graph")
        return {
            "graph": self.graphs,
            "y": self.y,
            "train_mask": self.train_mask,
            "val_mask": self.val_mask,
            "test_mask": self.test_mask,
        }

    # --- downloading helpers ---
    def _download_file(self, url: str, dest: str) -> None:
        if osp.exists(dest):
            # skip existing file
            return
        os.makedirs(osp.dirname(dest), exist_ok=True)
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as e:
            # surface a helpful error
            raise RuntimeError(f"Failed to download {url} -> {dest}: {e}") from e

    def download(self) -> None:
        # Download core Planetoid raw files (if missing)
        for fname in self.raw_file_names:
            url = f'{self.url}/{fname}'
            dest = osp.join(self.raw_dir, fname)
            self._download_file(url, dest)

        # If geom-gcn splits requested, download the 10 npz split files
        if self.split == 'geom-gcn':
            os.makedirs(self.raw_dir, exist_ok=True)
            for i in range(10):
                fname = f'{self.name.lower()}_split_0.6_0.2_{i}.npz'
                url = f'{self.geom_gcn_url}/splits/{fname}'
                dest = osp.join(self.raw_dir, fname)
                self._download_file(url, dest)

    # --- processing (uses your read_planetoid_data function) ---
    def process(self) -> None:
        """
        Uses read_planetoid_data(self.raw_dir, self.name) which must return:
            graphs (jraph.GraphsTuple), y (jnp array), train_mask, val_mask, test_mask
        Then adjusts masks for 'full', 'random', or stacks geom-gcn splits.
        The final attributes set:
            self.graphs, self.y, self.train_mask, self.val_mask, self.test_mask
        """
        # read_planetoid_data is expected to exist in this module (you provided it earlier)
        graphs, y_j, train_mask_j, val_mask_j, test_mask_j = read_planetoid_data(self.raw_dir, self.name)

        # convert masks/labels to numpy for manipulation if needed
        y_np = np.asarray(y_j)
        train_mask = np.asarray(train_mask_j).astype(bool)
        val_mask = np.asarray(val_mask_j).astype(bool)
        test_mask = np.asarray(test_mask_j).astype(bool)

        if self.split == 'full':
            # all nodes except validation and test -> train
            train_mask = np.ones_like(train_mask, dtype=bool)
            train_mask[val_mask | test_mask] = False

        elif self.split == 'random':
            rng = np.random.default_rng()
            num_nodes = y_np.shape[0]
            num_classes = int(y_np.max()) + 1
            train_mask = np.zeros(num_nodes, dtype=bool)

            # select num_train_per_class random nodes per class
            for c in range(num_classes):
                idx = np.where(y_np == c)[0]
                if idx.size == 0:
                    continue
                perm = rng.permutation(idx)
                chosen = perm[: self.num_train_per_class]
                train_mask[chosen] = True

            # remaining nodes -> shuffled; pick val/test slices
            remaining = np.where(~train_mask)[0]
            remaining = rng.permutation(remaining)

            val_mask = np.zeros_like(val_mask)
            test_mask = np.zeros_like(test_mask)

            val_idx = remaining[: self.num_val]
            test_idx = remaining[self.num_val : self.num_val + self.num_test]

            val_mask[val_idx] = True
            test_mask[test_idx] = True

        elif self.split == 'geom-gcn':
            # load the 10 split npz files and stack masks along axis=1 -> shape (N, 10)
            train_masks = []
            val_masks = []
            test_masks = []
            for i in range(10):
                fname = f'{self.name.lower()}_split_0.6_0.2_{i}.npz'
                path = osp.join(self.raw_dir, fname)
                if not osp.exists(path):
                    raise FileNotFoundError(f"Expected geom-gcn split file not found: {path}")
                splits = np.load(path, allow_pickle=True)
                # keys assumed to be 'train_mask', 'val_mask', 'test_mask'
                train_masks.append(np.asarray(splits['train_mask']).astype(bool))
                val_masks.append(np.asarray(splits['val_mask']).astype(bool))
                test_masks.append(np.asarray(splits['test_mask']).astype(bool))

            # stack to (N, 10)
            train_mask = np.stack(train_masks, axis=1)
            val_mask = np.stack(val_masks, axis=1)
            test_mask = np.stack(test_masks, axis=1)

        # finalize and convert back to jnp
        self.graphs = graphs
        self.y = jnp.asarray(y_np)
        self.train_mask = jnp.asarray(train_mask)
        self.val_mask = jnp.asarray(val_mask)
        self.test_mask = jnp.asarray(test_mask)

    # --- saving / loading processed data ---
    def _save_processed(self) -> None:
        # convert jax arrays to numpy for saving
        nodes = np.asarray(self.graphs.nodes) if self.graphs.nodes is not None else np.array([])
        senders = np.asarray(self.graphs.senders) if self.graphs.senders is not None else np.array([], dtype=np.int32)
        receivers = np.asarray(self.graphs.receivers) if self.graphs.receivers is not None else np.array([], dtype=np.int32)
        n_node = np.asarray(self.graphs.n_node) if self.graphs.n_node is not None else np.array(0, dtype=np.int32)
        n_edge = np.asarray(self.graphs.n_edge) if self.graphs.n_edge is not None else np.array(0, dtype=np.int32)

        # masks & labels
        y = np.asarray(self.y)
        train_mask = np.asarray(self.train_mask)
        val_mask = np.asarray(self.val_mask)
        test_mask = np.asarray(self.test_mask)

        os.makedirs(self.processed_dir, exist_ok=True)
        np.savez_compressed(
            self._processed_path,
            nodes=nodes,
            senders=senders,
            receivers=receivers,
            n_node=n_node,
            n_edge=n_edge,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            name=self.name,
            split=self.split,
        )

    def _load_processed(self) -> None:
        if not osp.exists(self._processed_path):
            raise FileNotFoundError(f"Processed file not found: {self._processed_path}")

        npz = np.load(self._processed_path, allow_pickle=True)
        nodes = npz['nodes']
        senders = npz['senders']
        receivers = npz['receivers']
        n_node = npz['n_node'].tolist() if npz['n_node'].shape == () else npz['n_node']
        n_edge = npz['n_edge'].tolist() if npz['n_edge'].shape == () else npz['n_edge']

        # reconstruct GraphsTuple (use jnp arrays)
        graphs = GraphsTuple(
            nodes=jnp.asarray(nodes) if nodes.size != 0 else None,
            edges=None,
            receivers=jnp.asarray(receivers, dtype=jnp.int32) if receivers.size != 0 else None,
            senders=jnp.asarray(senders, dtype=jnp.int32) if senders.size != 0 else None,
            globals=None,
            n_node=jnp.asarray(n_node, dtype=jnp.int32),
            n_edge=jnp.asarray(n_edge, dtype=jnp.int32),
        )

        self.graphs = graphs
        self.y = jnp.asarray(npz['y'])
        self.train_mask = jnp.asarray(npz['train_mask'])
        self.val_mask = jnp.asarray(npz['val_mask'])
        self.test_mask = jnp.asarray(npz['test_mask'])

    def __repr__(self) -> str:
        return f"Planetoid(name={self.name}, split={self.split})"
