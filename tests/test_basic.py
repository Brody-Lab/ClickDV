"""
Basic test to verify testing infrastructure works.
"""

def test_basic():
    """Test that basic assertions work."""
    assert 1 + 1 == 2


def test_imports():
    """Test that basic scientific computing imports work."""
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Basic functionality check
    arr = np.array([1, 2, 3])
    assert len(arr) == 3
    
    df = pd.DataFrame({'a': [1, 2, 3]})
    assert len(df) == 3