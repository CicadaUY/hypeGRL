# Hyperbolic Embeddings

This repository provides tools to train, evaluate, and visualize graph embeddings in hyperbolic space.
Supported models include:

- Poincare Embeddings
- Lorentz Embeddings
- Poincare Maps
- D-Mercator
- Hydra
- Hydra Plus
- Hypermap


## Setup

### Requirements:

- Python 3.9+ is required
- Git (for cloning repositories)

### Build and install dependencies 

```bash
make
```

This will automatically:

1. Create a virtual environment (venv)
2. Clone required repositories (if not already present):
   - d-mercator
   - hypermap
   - lorentz-embeddings
   - PoincareMaps
3. Install required dependencies
4. Set up local packages like d-mercator and hypermap

**Note:** If you've already cloned the repositories manually, the Makefile will detect them and skip the cloning step.

## Usage

### Activate virtual env

```
source venv/bin/activate
```

### Initialize Embedding Manager

```
embedding_manager = HyperbolicEmbeddings(embedding_type="poincare_maps", config=config)
```

### Train embeddings

```
embedding_manager.train(adjacency_matrix=A, model_path=model_path)
```

### Plot embeddings

```
embedding_manager.plot_embeddings(labels=labels, edge_list=edge_list, save_path=plot_path)
```


## Examples


### Tree Test

```
python -m test.tree_test.main --embedding_type='hypermap' --output_space='poincare'
```