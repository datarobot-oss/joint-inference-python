"""Pytest configuration shared by the whole suite."""

import torch

# Every xdist worker is its own process; torch's default intra-op thread count
# per worker oversubscribes the CPU and multiplies the suite's wall time.
torch.set_num_threads(1)
