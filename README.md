# CPU Cache Optimization for Small-Scale Transformer Models

This project investigates CPU cache optimization techniques for running small-scale transformer models efficiently. It includes implementations of different memory layout patterns and tools for analyzing their impact on cache performance.

## Project Structure

```
cache_opt_transformer/
├── src/
│   ├── model/
│   │   └── attention.py         # Cache-aware attention implementation
│   ├── profiling/
│   │   └── cache_monitor.py     # Cache performance monitoring tools
│   └── utils/
├── tests/
│   └── test_attention.py        # Unit tests
└── benchmarks/
    └── benchmark_layouts.py     # Memory layout benchmarking
```

## Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd cache_opt_transformer
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Tests

To run the unit tests:
```bash
python -m pytest tests/
```

## Running Benchmarks

To run the memory layout benchmarks:
```bash
python benchmarks/benchmark_layouts.py
```

The benchmark results will be saved in the `benchmark_results` directory:
- JSON file with detailed results
- Plots showing execution times and cache hit rates

## Customizing Memory Layouts

You can create custom memory layouts by modifying the `MemoryLayout` configuration:

```python
from src.model.attention import MemoryLayout, CacheAwareAttention

# Create a custom memory layout
layout = MemoryLayout(
    contiguous_qkv=True,    # Store Q,K,V weights contiguously
    row_major=True,         # Use row-major memory layout
    cache_aligned=True,     # Align data to cache lines
    block_size=64          # Size of memory blocks for tiling
)

# Create attention module with custom layout
attention = CacheAwareAttention(
    dim=512,
    num_heads=8,
    memory_layout=layout
)
```

## Performance Monitoring

You can use the `profile_memory_access` decorator to monitor cache performance:

```python
from src.profiling.cache_monitor import profile_memory_access

@profile_memory_access
def run_model(model, input_data):
    return model.forward(input_data)
```

This will print cache hit rates and memory access pattern statistics.

## Benchmark Configuration

You can customize benchmark parameters in `benchmarks/benchmark_layouts.py`:

```python
config = BenchmarkConfig(
    model_dims=[256, 512],     # Model dimensions to test
    batch_sizes=[1, 8],        # Batch sizes to test
    seq_lengths=[128, 256],    # Sequence lengths to test
    num_heads=8                # Number of attention heads
)
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request