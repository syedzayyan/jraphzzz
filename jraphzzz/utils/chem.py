import jax.numpy as jnp 
from typing import Any, List, Dict
from ..data.graph import GraphsTuple

x_map: Dict[str, List[Any]] = {
    'atomic_num':
    list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'CHI_TETRAHEDRAL',
        'CHI_ALLENE',
        'CHI_SQUAREPLANAR',
        'CHI_TRIGONALBIPYRAMIDAL',
        'CHI_OCTAHEDRAL',
    ],
    'degree':
    list(range(0, 11)),
    'formal_charge':
    list(range(-5, 7)),
    'num_hs':
    list(range(0, 9)),
    'num_radical_electrons':
    list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map: Dict[str, List[Any]] = {
    'bond_type': [
        'UNSPECIFIED',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'QUADRUPLE',
        'QUINTUPLE',
        'HEXTUPLE',
        'ONEANDAHALF',
        'TWOANDAHALF',
        'THREEANDAHALF',
        'FOURANDAHALF',
        'FIVEANDAHALF',
        'AROMATIC',
        'IONIC',
        'HYDROGEN',
        'THREECENTER',
        'DATIVEONE',
        'DATIVE',
        'DATIVEL',
        'DATIVER',
        'OTHER',
        'ZERO',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOANY',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
    ],
    'is_conjugated': [False, True],
}

def from_smiles(smiles: str, with_hydrogen: bool = False, kekulize: bool = False) -> GraphsTuple:
    from rdkit import Chem, RDLogger 

    RDLogger.DisableLog('rdApp.*')
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles('')
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)

    xs: List[List[int]] = []
    for atom in mol.GetAtoms():
        row: List[int] = []
        row.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
        row.append(x_map['chirality'].index(str(atom.GetChiralTag())))
        row.append(x_map['degree'].index(atom.GetTotalDegree()))
        row.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
        row.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
        row.append(x_map['num_radical_electrons'].index(atom.GetNumRadicalElectrons()))
        row.append(x_map['hybridization'].index(str(atom.GetHybridization())))
        row.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
        row.append(x_map['is_in_ring'].index(atom.IsInRing()))
        xs.append(row)

    # node features as float32 (you could also keep int32 if treating as categorical indices)
    x = jnp.array(xs, dtype=jnp.float32) if len(xs) > 0 else jnp.zeros((0, 9), dtype=jnp.float32)

    edge_indices: List[List[int]] = []
    edge_attrs: List[List[Any]] = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        e = [
            e_map['bond_type'].index(str(bond.GetBondType())),
            e_map['stereo'].index(str(bond.GetStereo())),
            e_map['is_conjugated'].index(bond.GetIsConjugated()),
        ]
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    if len(edge_indices) == 0:
        senders = jnp.zeros((0,), dtype=jnp.int32)
        receivers = jnp.zeros((0,), dtype=jnp.int32)
        edge_attr = jnp.zeros((0, 3), dtype=jnp.float32)
        n_edge = jnp.array([0], dtype=jnp.int32)
    else:
        # edge_index shape (2, E)
        edge_index = jnp.array(edge_indices, dtype=jnp.int32).T
        edge_attr = jnp.array(edge_attrs, dtype=jnp.float32)
        perm = (edge_index[0] * max(1, x.shape[0]) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]
        senders = edge_index[0].astype(jnp.int32)
        receivers = edge_index[1].astype(jnp.int32)
        n_edge = jnp.array([edge_index.shape[1]], dtype=jnp.int32)

    n_node = jnp.array([x.shape[0]], dtype=jnp.int32)

    # single-graph globals
    globals = jnp.zeros((1, 1), dtype=jnp.float32)

    return GraphsTuple(
        nodes=x,                # (N, F)
        edges=edge_attr,        # (E, Fe)
        senders=senders,        # (E,)
        receivers=receivers,    # (E,)
        globals=globals,        # (n_graphs, Fg)
        n_node=n_node,          # (n_graphs,)
        n_edge=n_edge           # (n_graphs,)
    )
