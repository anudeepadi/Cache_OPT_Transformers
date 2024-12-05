import numpy as np
import pytest
from src.model.attention import CacheAwareAttention, MemoryLayout

def test_attention_forward():
    # Test parameters
    batch_size = 2
    seq_len = 4
    dim = 8
    num_heads = 2
    
    # Create model
    model = CacheAwareAttention(dim=dim, num_heads=num_heads)
    
    # Create input
    x = np.random.normal(0, 1, (batch_size, seq_len, dim))
    
    # Forward pass
    output = model.forward(x)
    
    # Check output shape
    expected_shape = (batch_size, num_heads, seq_len, dim // num_heads)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"

def test_memory_layouts():
    # Test different memory layouts
    layouts = [
        MemoryLayout(contiguous_qkv=True, row_major=True),
        MemoryLayout(contiguous_qkv=False, row_major=True),
        MemoryLayout(contiguous_qkv=True, row_major=False),
        MemoryLayout(contiguous_qkv=False, row_major=False),
    ]
    
    batch_size = 2
    seq_len = 4
    dim = 8
    num_heads = 2
    x = np.random.normal(0, 1, (batch_size, seq_len, dim))
    
    for layout in layouts:
        model = CacheAwareAttention(dim=dim, num_heads=num_heads, memory_layout=layout)
        output = model.forward(x)
        
        # Check output shape
        expected_shape = (batch_size, num_heads, seq_len, dim // num_heads)
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"

def test_cache_alignment():
    # Test cache-aligned memory layout
    layout = MemoryLayout(cache_aligned=True, block_size=64)
    
    batch_size = 2
    seq_len = 4
    dim = 8
    num_heads = 2
    
    model = CacheAwareAttention(dim=dim, num_heads=num_heads, memory_layout=layout)
    x = np.random.normal(0, 1, (batch_size, seq_len, dim))
    
    output = model.forward(x)
    
    # Check output shape
    expected_shape = (batch_size, num_heads, seq_len, dim // num_heads)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"

def test_numerical_stability():
    # Test with larger values to check numerical stability
    batch_size = 2
    seq_len = 4
    dim = 8
    num_heads = 2
    
    model = CacheAwareAttention(dim=dim, num_heads=num_heads)
    
    # Create input with large values
    x = np.random.normal(0, 10, (batch_size, seq_len, dim))
    
    # Forward pass
    output = model.forward(x)
    
    # Check for NaN or inf values
    assert not np.any(np.isnan(output)), "Output contains NaN values"
    assert not np.any(np.isinf(output)), "Output contains infinite values"

if __name__ == '__main__':
    pytest.main([__file__])