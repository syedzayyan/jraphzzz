# Jraphzzz - A library for graph neural networks in jax

This is a fork of the DeepMind Jraph library.

No connections to DeepMind whatsoever, except stealing all of their code. Horrible name (?): Jraphzzz, the extended Zs because the original Jraph library has gone to sleep. Geddit (?) Sorry, I'll shut up.

This is **not a spiritual successor** to Jraph. It is Jraph with additional code to make life easier for, especially, me. The intention is still the same as Graph_nets and Jraph:

* Keep things simple
* Help to write GNNs however you want, instead of forcing a certain way
* Remain an extremely forkable repo

If you want a PyTorch Geometric-like Jax Graph library, [JraphX](https://github.com/DBraun/jraphx) or [Haiku Geometric](https://github.com/alexOarga/haiku-geometric) is your friend.

Feel free to contribute, or use any part of the code.

![logo](logo.png)

---
## Todos

[ ] Examples, like lots of it
[ ] More Convolutional Layers
[ ] More Attention Layers
[ ] Not too keen on models, but hey, if you want please go ahead
[ ] Hypergraph Stuff
[ ] Higher Order Stuff
[ ] Temporal GCNs (?)


---

## Dev Installation (using uv)

To install the **dev version** of Jraphzzz, make sure you have [uv](https://docs.astral.sh/uv/guides/install-python/) installed, then run the following from the project root:

### Editable install (recommended for development)

```bash
uv install -e ".[dev]"
```

This makes the package **editable**, so changes you make to the source code are immediately available.

### Install with example extras

```bash
uv install -e ".[examples]"
```

---

## Citing Jraphzzz

To cite this repository:

```
@software{jraphzzz2025github,
  author = {Jonathan Godwin* and Thomas Keck* and Peter Battaglia and Victor Bapst and Thomas Kipf and Yujia Li and Kimberly Stachenfeld and Petar Veli\v{c}kovi\'c and Alvaro Sanchez-Gonzalez and Syed Zayyan Masud},
  title = {{J}raph: {A} library for graph neural networks in jax.},
  url = {http://github.com/deepmind/jraph},
  version = {0.0.1.dev},
  year = {2025},
}
```