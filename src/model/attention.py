import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class MemoryLayout:
    """Configuration for memory layout patterns"""
    contiguous_qkv: bool = True  # Whether Q,K,V matrices are stored contiguously
    row_major: bool = True       # Row vs column major storage
    cache_aligned: bool = False  # Whether to align data to cache lines
    block_size: int = 64        # Size of memory blocks for tiling

class CacheAwareAttention:
    def __init__(
        self, 
        dim: int,
        num_heads: int,
        memory_layout: Optional[MemoryLayout] = None
    ):
        if dim % num_heads != 0:
            raise ValueError(f"Dimension {dim} must be divisible by number of heads {num_heads}")
            
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.memory_layout = memory_layout or MemoryLayout()
        
        # Initialize weight matrices
        self.w_q = self._initialize_weights()
        self.w_k = self._initialize_weights()
        self.w_v = self._initialize_weights()
        
    def _initialize_weights(self) -> np.ndarray:
        """Initialize weight matrix with optimized memory layout"""
        # Base shape
        shape = (self.dim, self.dim)
        
        # Initialize weights
        if self.memory_layout.row_major:
            weights = np.random.normal(0, 0.02, shape)
        else:
            weights = np.random.normal(0, 0.02, shape[::-1]).T
        
        # Apply cache alignment if needed
        if self.memory_layout.cache_aligned and self.memory_layout.block_size > 0:
            # Pad to block size
            pad_size = self.memory_layout.block_size - (shape[0] % self.memory_layout.block_size)
            if pad_size != self.memory_layout.block_size:
                padded_shape = (shape[0] + pad_size, shape[1] + pad_size)
                padded = np.zeros(padded_shape)
                padded[:shape[0], :shape[1]] = weights
                weights = padded[:shape[0], :shape[1]]
        
        return weights
    
    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """Split the last dimension into (num_heads, head_dim)"""
        batch_size, seq_len, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"Input dimension {dim} does not match expected dimension {self.dim}")
            
        # Reshape: (batch_size, seq_len, dim) -> (batch_size, seq_len, num_heads, head_dim)
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        # Transpose: -> (batch_size, num_heads, seq_len, head_dim)
        return x.transpose(0, 2, 1, 3)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with cache-aware implementation"""
        batch_size, seq_len, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"Input dimension {dim} does not match model dimension {self.dim}")
        
        # Project inputs to Q, K, V with blocking
        if self.memory_layout.block_size > 0:
            block_size = min(self.memory_layout.block_size, seq_len)
            q = np.zeros((batch_size, seq_len, self.dim))
            k = np.zeros_like(q)
            v = np.zeros_like(q)
            
            for i in range(0, seq_len, block_size):
                i_end = min(i + block_size, seq_len)
                # Project each block
                q[:, i:i_end] = np.matmul(x[:, i:i_end], self.w_q)
                k[:, i:i_end] = np.matmul(x[:, i:i_end], self.w_k)
                v[:, i:i_end] = np.matmul(x[:, i:i_end], self.w_v)
        else:
            # Standard projection without blocking
            q = np.matmul(x, self.w_q)
            k = np.matmul(x, self.w_k)
            v = np.matmul(x, self.w_v)
        
        # Split heads
        q = self._split_heads(q)  # (batch, heads, seq_len, head_dim)
        k = self._split_heads(k)
        v = self._split_heads(v)
        
        # Scaled dot-product attention
        scale = np.sqrt(self.head_dim)
        
        if self.memory_layout.block_size > 0 and seq_len > self.memory_layout.block_size:
            # Blocked attention for long sequences
            output = np.zeros_like(q)
            block_size = self.memory_layout.block_size
            
            for i in range(0, seq_len, block_size):
                i_end = min(i + block_size, seq_len)
                q_block = q[:, :, i:i_end]
                
                # Compute attention for this block
                scores = np.matmul(q_block, k.transpose(0, 1, 3, 2)) / scale
                scores = scores - np.max(scores, axis=-1, keepdims=True)
                weights = np.exp(scores)
                weights = weights / np.sum(weights, axis=-1, keepdims=True)
                
                output[:, :, i:i_end] = np.matmul(weights, v)
        else:
            # Standard attention computation
            attention = np.matmul(q, k.transpose(0, 1, 3, 2)) / scale
            attention = attention - np.max(attention, axis=-1, keepdims=True)
            attention = np.exp(attention)
            attention = attention / np.sum(attention, axis=-1, keepdims=True)
            output = np.matmul(attention, v)
        
        return output