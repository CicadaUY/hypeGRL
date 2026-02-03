import logging
import os
import sys
import timeit
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

# ---- Threading guard (macOS segfault mitigation) ----
# Ideally these env vars are set BEFORE numpy/scipy/torch import in your entrypoint script,
# but keeping them here is still often helpful.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import torch
from scipy.sparse import csgraph, csr_matrix, identity, issparse
from scipy.sparse.linalg import splu
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models.base_hyperbolic_model import BaseHyperbolicModel
from models.PoincareMaps.data import compute_rfa
from models.PoincareMaps.model import PoincareDistance, PoincareEmbedding
from models.PoincareMaps.rsgd import RiemannianSGD
from utils.geometric_conversions import poincare_to_hyperboloid

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Disable PyTorch multiprocessing to prevent segfaults on macOS
torch.multiprocessing.set_sharing_strategy("file_system")
torch.set_num_threads(1)


class PoincareMapsModel(BaseHyperbolicModel):
    def __init__(self, config: Dict):
        self.dim = config.get("dim", 2)
        self.epochs = config.get("epochs", 50)
        self.batch_size = config.get("batch_size", -1)
        self.lr = config.get("lr", 0.3)
        self.sigma = config.get("sigma", 1.0)
        self.gamma = config.get("gamma", 1.0)
        self.burnin = config.get("burnin", 500)
        self.lrm = config.get("lrm", 1.0)
        self.earlystop = config.get("earlystop", 0.0001)
        self.mode = config.get("mode", "features")
        self.distlocal = config.get("distlocal", "minkowski")
        self.k_neighbours = config.get("k_neighbours", 15)

        self.device = config.get("device", "cpu" if not torch.cuda.is_available() else "cuda")
        self.logger = logging.getLogger(__name__)

    @property
    def native_space(self) -> str:
        return "poincare"

    def train(
        self,
        edge_list: Optional[List[tuple]] = None,
        adjacency_matrix: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
        model_path: str = "saved_models/model.bin",
    ):
        if features is not None:
            RFA = compute_rfa(
                features,
                mode=self.mode,
                k_neighbours=self.k_neighbours,
                distlocal=self.distlocal,
                distfn="MFIsym",
                connected=True,
                sigma=self.sigma,
            )
            # ensure torch tensor and contiguous
            if isinstance(RFA, np.ndarray):
                RFA = np.ascontiguousarray(RFA, dtype=np.float32)
                RFA = torch.from_numpy(RFA)
            else:
                RFA = RFA.contiguous() if torch.is_tensor(RFA) else torch.tensor(RFA, dtype=torch.float32)

        else:
            if edge_list is None and adjacency_matrix is None:
                raise ValueError("Either edge_list or adjacency_matrix must be provided")

            if adjacency_matrix is None:
                self.logger.info("Converting edge list into adjacency matrix")
                G = nx.Graph()
                G.add_edges_from(edge_list)
                # float dtype avoids integer laplacian/solve oddities
                adjacency_matrix = nx.to_numpy_array(G, dtype=np.float64)

            # ---- Make adjacency sparse to keep laplacian stable/fast ----
            A_adj = adjacency_matrix
            if not issparse(A_adj):
                A_adj = csr_matrix(A_adj)

            self.logger.info("Computing Laplacian")
            t0 = timeit.default_timer()

            L = csgraph.laplacian(A_adj, normed=False).tocsr()
            print(f"Laplacian computed in {(timeit.default_timer() - t0):.2f} sec")

            # ---- Build (L + I) in sparse, factorize, then (optionally) densify result ----
            M = (L + identity(L.shape[0], format="csr", dtype=np.float64)).tocsc()

            # Sparse LU (more robust than dense inv/solve for laplacians)
            lu = splu(M)
            print(f"LU factorization done in {(timeit.default_timer() - t0):.2f} sec")

            # WARNING: This creates a dense NxN matrix. Works for small/medium N only.
            n = M.shape[0]
            RFA = lu.solve(np.eye(n, dtype=np.float64))
            print(f"RFA computed in {(timeit.default_timer() - t0):.2f} sec")

            RFA = np.nan_to_num(RFA, nan=0.0, posinf=0.0, neginf=0.0)
            print(f"RFA cleaned in {(timeit.default_timer() - t0):.2f} sec")

            # ---- SAFEST conversion to torch (fixes your segfault line) ----
            RFA = np.ascontiguousarray(RFA, dtype=np.float32)
            RFA = torch.from_numpy(RFA)  # shares memory with numpy; stable conversion path
            # If you need it detached from numpy memory:
            # RFA = torch.from_numpy(RFA).clone()

            print(f"RFA torch conversion done in {(timeit.default_timer() - t0):.2f} sec")

        # If you later want to move to device:
        # RFA = RFA.to(self.device)

        if self.batch_size < 0:
            # Force batch_size=1 to avoid macOS/PyTorch edge issues
            self.batch_size = 1

        self.lr = self.batch_size / 16 * self.lr

        indices = torch.arange(len(RFA))
        dataset = TensorDataset(indices, RFA.contiguous())

        # Instantiate Embedding predictor
        self.model = PoincareEmbedding(
            size=len(dataset),
            dim=self.dim,
            dist=PoincareDistance,
            max_norm=1,
            Qdist="laplace",
            lossfn="klSym",
            gamma=self.gamma,
            cuda=0,  # Force CPU mode
        )

        optimizer = RiemannianSGD(self.model.parameters(), lr=self.lr)

        self.logger.info("Starting training...")
        self.logger.info(f"Using device: {self.device}, CUDA available: {torch.cuda.is_available()}")

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=None,
        )

        pbar = tqdm(range(self.epochs), ncols=80, file=sys.stdout, disable=False)

        epoch_loss = []
        earlystop_count = 0

        for epoch in pbar:
            lr = self.lr * self.lrm if epoch < self.burnin else self.lr

            epoch_error = 0.0
            grad_norm = []

            try:
                for batch_idx, (inputs, targets) in enumerate(loader):
                    loss = self.model.lossfn(self.model(inputs), targets)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step(lr=lr)

                    epoch_error += float(loss.item())
                    grad_norm.append(self.model.lt.weight.grad.data.norm().item())

            except Exception as e:
                self.logger.error(f"Error during training iteration: {e}")
                self.logger.error(f"Inputs shape: {inputs.shape if 'inputs' in locals() else 'N/A'}")
                self.logger.error(f"Targets shape: {targets.shape if 'targets' in locals() else 'N/A'}")
                raise

            epoch_error /= max(1, len(loader))
            epoch_loss.append(epoch_error)
            pbar.set_description(f"loss: {epoch_error:.5f}")

            if epoch > 10:
                delta = abs(epoch_loss[epoch] - epoch_loss[epoch - 1])
                if delta < self.earlystop:
                    earlystop_count += 1
                if earlystop_count > 50:
                    self.logger.info(f"\nStopped at epoch {epoch}")
                    break

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        self.logger.info(f"Model saved to {model_path}")

    def get_embedding(self, node_id: str, model_path: Optional[str] = None) -> np.ndarray:
        if model_path:
            state = torch.load(model_path)
            self.model.load_state_dict(state)

        embeddings = self.model.lt.weight.cpu().detach().numpy()

        return embeddings[int(node_id)]

    def get_all_embeddings(self, model_path: Optional[str] = None) -> np.ndarray:
        if model_path:
            state = torch.load(model_path)
            self.model.load_state_dict(state)

        return self.model.lt.weight.detach().cpu().numpy()

    def most_similar(self, node_id: str, topn: int = 5, model_path: Optional[str] = None) -> List[Tuple[str, float]]:
        pass

    def to_hyperboloid(self, model_path: Optional[str] = None) -> np.ndarray:
        """Convert Poincaré embeddings to hyperboloid coordinates."""
        poincare_embeddings = self.get_all_embeddings(model_path)
        return poincare_to_hyperboloid(poincare_embeddings)

    def to_poincare(self, model_path: Optional[str] = None) -> np.ndarray:
        """Return embeddings in Poincaré coordinates (already in Poincaré space)."""
        return self.get_all_embeddings(model_path)
