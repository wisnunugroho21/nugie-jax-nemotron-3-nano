"""
device_mesh.py — Shared JAX device mesh for data-parallel training.

Extracted into its own module so both nemotron.py (model) and
training_shared.py (training loop) can import it without circular deps.
"""

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def _setup_mesh() -> Mesh:
    """Create a 1-D data-parallel device mesh over all available devices."""
    return jax.make_mesh((len(jax.devices()),), ('data',))


MESH: Mesh = _setup_mesh()
NUM_DEVICES: int = len(jax.devices())

# Sharding helpers: shard batch dim over 'data', replicate everything else.
DATA_SHARDING: NamedSharding = NamedSharding(MESH, P("data"))
REPLICATED_SHARDING: NamedSharding = NamedSharding(MESH, P())
