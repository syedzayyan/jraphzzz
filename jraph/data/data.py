from jax.typing import Array
from graph import GraphsTuple
import networkx as nx

class Dataset:
    def __call__(nodes, edges, receivers, senders, globals = None):
        return GraphsTuple(
            nodes=nodes,
            edges=edges,
            senders=senders,
            receivers=receivers,
            globals=globals,
            n_node=nodes.shape[0],
            n_edge=edges.shape[0]
        )

    def from_networkx(G: nx.Graph) -> GraphsTuple:
        
        return GraphsTuple(
            nodes=nodes,
            edges=edges,
            senders=senders,
            receivers=receivers,
            globals=globals,
            n_node=G.number_of_nodes,
            n_edge=G.number_of_edges
        )