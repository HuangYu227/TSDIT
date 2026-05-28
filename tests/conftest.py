# conftest.py
import pytest
import torch
import numpy as np


@pytest.fixture
def simple_motions():
    """Simple motion tensor for testing."""
    return torch.randn(10, 12, 3)  # (joints, timesteps, xyz)


@pytest.fixture
def complex_motions():
    """More complex motion data for testing."""
    return torch.randn(5, 25, 12, 3)  # (batch, timesteps, joints, xyz)


@pytest.fixture
def sample_parameters():
    """Sample text parameters for TSG."""
    return {
        "dancer_type": "female",
        "dance_type": "ballet",
        "motion_tempo": "moderate",
        "duration": "10 seconds"
    }


@pytest.fixture
def device():
    """Standard torch device."""
    return torch.device("cpu")


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    return 42
