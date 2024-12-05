import os
import sys
import time
import psutil
from typing import List, Dict, Any
import numpy as np
from dataclasses import dataclass
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.attention import CacheAwareAttention, MemoryLayout
from src.profiling.cache_monitor import CacheMonitor, profile_memory_access

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs"""
    model_dims: List[int]
    batch_sizes: List[int]
    seq_lengths: List[int]
    num_heads: int
    num_runs: int = 3

@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    config: Dict[str, Any]
    execution_time: float
    l1_hit_rate: float
    l2_hit_rate: float
    memory_usage: int

class LayoutBenchmark:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = []
        self.process = psutil.Process()
    
    def get_memory_usage(self) -> int:
        """Get current memory usage in bytes"""
        return self.process.memory_info().rss
        
    def run_single_benchmark(self, 
                           dim: int, 
                           batch_size: int, 
                           seq_len: int,
                           memory_layout: MemoryLayout) -> BenchmarkResult:
        """Run a single benchmark configuration"""
        try:
            # Track initial memory
            initial_memory = self.get_memory_usage()
            
            # Create model
            model = CacheAwareAttention(
                dim=dim,
                num_heads=self.config.num_heads,
                memory_layout=memory_layout
            )
            
            # Create input data
            x = np.random.normal(0, 1, (batch_size, seq_len, dim))
            
            # Track memory after model creation
            model_memory = self.get_memory_usage() - initial_memory
            
            # Setup monitoring
            monitor = CacheMonitor()
            times = []
            
            for run in range(self.config.num_runs):
                print(f"  Run {run + 1}/{self.config.num_runs}")
                monitor.start_monitoring()
                start_time = time.perf_counter()
                
                # Forward pass
                _ = model.forward(x)
                
                end_time = time.perf_counter()
                stats = monitor.stop_monitoring()
                times.append(end_time - start_time)
            
            # Average results
            avg_time = sum(times) / len(times)
            
            return BenchmarkResult(
                config={
                    'dim': dim,
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                    'memory_layout': {
                        'contiguous_qkv': memory_layout.contiguous_qkv,
                        'row_major': memory_layout.row_major,
                        'cache_aligned': memory_layout.cache_aligned,
                        'block_size': memory_layout.block_size
                    }
                },
                execution_time=avg_time,
                l1_hit_rate=stats.l1_hit_rate,
                l2_hit_rate=stats.l2_hit_rate,
                memory_usage=model_memory
            )
        except Exception as e:
            print(f"Error in benchmark: {str(e)}")
            raise

    def run_benchmarks(self) -> List[BenchmarkResult]:
        """Run all benchmark configurations"""
        # Define memory layouts to test
        layouts = [
            MemoryLayout(contiguous_qkv=True, row_major=True, cache_aligned=False),
            MemoryLayout(contiguous_qkv=False, row_major=True, cache_aligned=False),
            MemoryLayout(contiguous_qkv=True, row_major=False, cache_aligned=True),
            MemoryLayout(contiguous_qkv=False, row_major=False, cache_aligned=True),
        ]
        
        layout_names = [
            'Standard (Row-major)',
            'Separated QKV (Row-major)',
            'Cache-aligned (Column-major)',
            'Fully Optimized (Column-major + Separated)'
        ]
        
        results = []
        total_runs = (len(self.config.model_dims) * 
                     len(self.config.batch_sizes) * 
                     len(self.config.seq_lengths) * 
                     len(layouts))
        
        print(f"Running {total_runs} benchmark configurations...")
        
        try:
            for dim in self.config.model_dims:
                for batch_size in self.config.batch_sizes:
                    for seq_len in self.config.seq_lengths:
                        for i, (layout, name) in enumerate(zip(layouts, layout_names)):
                            print(f"\nConfiguration {len(results) + 1}/{total_runs}")
                            print(f"Dimensions: dim={dim}, batch={batch_size}, seq_len={seq_len}")
                            print(f"Layout: {name}")
                            
                            result = self.run_single_benchmark(
                                dim=dim,
                                batch_size=batch_size,
                                seq_len=seq_len,
                                memory_layout=layout
                            )
                            results.append(result)
                            
                            print("Results:")
                            print(f"  Execution time: {result.execution_time:.4f}s")
                            print(f"  L1 hit rate: {result.l1_hit_rate:.2%}")
                            print(f"  L2 hit rate: {result.l2_hit_rate:.2%}")
                            print(f"  Memory usage: {result.memory_usage / 1024 / 1024:.2f} MB")
        except KeyboardInterrupt:
            print("\nBenchmark interrupted by user. Saving partial results...")
            self.results = results
            return results
            
        self.results = results
        return results
    
    def save_results(self, filename: str):
        """Save benchmark results to a JSON file"""
        results_dict = [
            {
                'config': r.config,
                'execution_time': r.execution_time,
                'l1_hit_rate': r.l1_hit_rate,
                'l2_hit_rate': r.l2_hit_rate,
                'memory_usage': r.memory_usage
            }
            for r in self.results
        ]
        
        output_dir = Path('benchmark_results')
        output_dir.mkdir(exist_ok=True)
        
        filepath = output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nResults saved to {filepath}")
    
    def plot_results(self, output_dir: str = 'benchmark_results'):
        """Generate plots from benchmark results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        layout_types = [
            'Standard',
            'Separated QKV',
            'Cache-aligned',
            'Fully Optimized'
        ]
        
        # Plot execution times
        plt.figure(figsize=(12, 6))
        for i, layout_type in enumerate(layout_types):
            times = [r.execution_time * 1000 for r in self.results[i::len(layout_types)]]
            plt.plot(times, label=layout_type, marker='o')
        
        plt.title('Execution Time by Layout Type')
        plt.xlabel('Configuration Index')
        plt.ylabel('Time (milliseconds)')
        plt.legend()
        plt.grid(True)
        
        plot_path = output_dir / 'execution_times.png'
        plt.savefig(plot_path)
        plt.close()
        print(f"Execution time plot saved to {plot_path}")
        
        # Plot memory usage
        plt.figure(figsize=(12, 6))
        for i, layout_type in enumerate(layout_types):
            memory = [r.memory_usage / 1024 / 1024 for r in self.results[i::len(layout_types)]]
            plt.plot(memory, label=layout_type, marker='o')
        
        plt.title('Memory Usage by Layout Type')
        plt.xlabel('Configuration Index')
        plt.ylabel('Memory Usage (MB)')
        plt.legend()
        plt.grid(True)
        
        plot_path = output_dir / 'memory_usage.png'
        plt.savefig(plot_path)
        plt.close()
        print(f"Memory usage plot saved to {plot_path}")
        
        # Plot cache hit rates
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        for i, layout_type in enumerate(layout_types):
            l1_rates = [r.l1_hit_rate * 100 for r in self.results[i::len(layout_types)]]
            l2_rates = [r.l2_hit_rate * 100 for r in self.results[i::len(layout_types)]]
            
            ax1.plot(l1_rates, label=layout_type, marker='o')
            ax2.plot(l2_rates, label=layout_type, marker='o')
        
        ax1.set_title('L1 Cache Hit Rate by Layout Type')
        ax1.set_ylabel('Hit Rate (%)')
        ax1.legend()
        ax1.grid(True)
        
        ax2.set_title('L2 Cache Hit Rate by Layout Type')
        ax2.set_xlabel('Configuration Index')
        ax2.set_ylabel('Hit Rate (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plot_path = output_dir / 'cache_hit_rates.png'
        plt.savefig(plot_path)
        plt.close()
        print(f"Cache hit rates plot saved to {plot_path}")

def main():
    # Use reasonable dimensions for testing
    config = BenchmarkConfig(
        model_dims=[64, 128],    # Model dimensions
        batch_sizes=[1, 4],      # Batch sizes
        seq_lengths=[32, 64],    # Sequence lengths
        num_heads=4              # Number of attention heads
    )
    
    print("Starting benchmarks with configuration:")
    print(f"Model dimensions: {config.model_dims}")
    print(f"Batch sizes: {config.batch_sizes}")
    print(f"Sequence lengths: {config.seq_lengths}")
    print(f"Number of heads: {config.num_heads}")
    print("\n" + "="*50 + "\n")
    
    benchmark = LayoutBenchmark(config)
    
    try:
        results = benchmark.run_benchmarks()
        
        # Save results
        benchmark.save_results('benchmark_results.json')
        benchmark.plot_results()
        
        print("\nBenchmarking complete! Results and plots have been saved.")
        
    except Exception as e:
        print(f"\nError during benchmarking: {str(e)}")
        raise
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
        if benchmark.results:
            print("Saving partial results...")
            benchmark.save_results('partial_benchmark_results.json')
            benchmark.plot_results()

if __name__ == "__main__":
    main()