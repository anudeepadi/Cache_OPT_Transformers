import os
import sys
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import psutil
import ctypes
from ctypes import windll, wintypes, byref
import platform

@dataclass
class CacheStats:
    """Container for cache performance statistics"""
    l1_hits: int = 0
    l1_misses: int = 0
    l2_hits: int = 0
    l2_misses: int = 0
    total_memory_accesses: int = 0
    cpu_cycles: int = 0
    instructions: int = 0
    
    @property
    def l1_hit_rate(self) -> float:
        """Calculate L1 cache hit rate"""
        total = self.l1_hits + self.l1_misses
        return self.l1_hits / total if total > 0 else 0.0
    
    @property
    def l2_hit_rate(self) -> float:
        """Calculate L2 cache hit rate"""
        total = self.l2_hits + self.l2_misses
        return self.l2_hits / total if total > 0 else 0.0
    
    @property
    def instructions_per_cycle(self) -> float:
        """Calculate instructions per cycle (IPC)"""
        return self.instructions / self.cpu_cycles if self.cpu_cycles > 0 else 0.0

class WindowsPerfCounter:
    def __init__(self):
        self.kernel32 = windll.kernel32
        self.pdh = windll.pdh
        
        # Initialize performance counters
        self.query = wintypes.HANDLE()
        self.pdh.PdhOpenQueryA(None, 0, byref(self.query))
        
        # Add relevant counters
        self.counters = {}
        self._add_counter("\\Processor(_Total)\\% Processor Time")
        self._add_counter("\\Memory\\Cache Bytes")
        self._add_counter("\\Memory\\Cache Faults/sec")
        
    def _add_counter(self, path: str):
        counter = wintypes.HANDLE()
        self.pdh.PdhAddCounterA(self.query, path.encode(), 0, byref(counter))
        self.counters[path] = counter
        
    def collect(self) -> Dict[str, float]:
        self.pdh.PdhCollectQueryData(self.query)
        results = {}
        
        for path, counter in self.counters.items():
            value = wintypes.DWORD()
            self.pdh.PdhGetFormattedCounterValue(counter, 0x8000, None, byref(value))
            results[path] = value.value
            
        return results
        
    def close(self):
        self.pdh.PdhCloseQuery(self.query)

class LinuxPerfCounter:
    def __init__(self):
        self.perf_cmd = ['perf', 'stat', '-e',
                        'cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses']
        
    def start(self):
        self.process = subprocess.Popen(
            self.perf_cmd + ['-p', str(os.getpid())],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
    def stop(self) -> Dict[str, int]:
        if hasattr(self, 'process'):
            self.process.terminate()
            _, stderr = self.process.communicate()
            return self._parse_perf_output(stderr.decode())
        return {}
        
    def _parse_perf_output(self, output: str) -> Dict[str, int]:
        results = {}
        for line in output.split('\n'):
            if any(event in line for event in ['cycles', 'instructions', 'cache']):
                parts = line.split()
                if len(parts) >= 1:
                    value = int(parts[0].replace(',', ''))
                    event = parts[-1]
                    results[event] = value
        return results

class CacheMonitor:
    def __init__(self):
        """Initialize platform-specific cache monitoring"""
        self.process = psutil.Process(os.getpid())
        self._reset_stats()
        
        if platform.system() == 'Windows':
            self.perf_counter = WindowsPerfCounter()
        else:
            self.perf_counter = LinuxPerfCounter()
        
    def _reset_stats(self):
        """Reset performance counters"""
        self.stats = CacheStats()
        
    def start_monitoring(self):
        """Begin monitoring cache performance"""
        self._reset_stats()
        self.initial_memory = self.process.memory_info()
        if platform.system() != 'Windows':
            self.perf_counter.start()
        
    def stop_monitoring(self) -> CacheStats:
        """Stop monitoring and return statistics"""
        final_memory = self.process.memory_info()
        
        if platform.system() == 'Windows':
            counters = self.perf_counter.collect()
            # Convert Windows performance counters to cache statistics
            self.stats.l1_hits = int(counters.get("\\Memory\\Cache Bytes", 0))
            self.stats.l1_misses = int(counters.get("\\Memory\\Cache Faults/sec", 0))
        else:
            perf_stats = self.perf_counter.stop()
            # Parse Linux perf statistics
            self.stats.cpu_cycles = perf_stats.get('cycles', 0)
            self.stats.instructions = perf_stats.get('instructions', 0)
            self.stats.l1_hits = perf_stats.get('L1-dcache-loads', 0)
            self.stats.l1_misses = perf_stats.get('L1-dcache-load-misses', 0)
            self.stats.l2_hits = perf_stats.get('LLC-loads', 0)
            self.stats.l2_misses = perf_stats.get('LLC-load-misses', 0)
            
        return self.stats

def profile_memory_access(func):
    """Decorator to profile memory access patterns of a function"""
    def wrapper(*args, **kwargs):
        monitor = CacheMonitor()
        
        # Start monitoring
        monitor.start_monitoring()
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Stop monitoring and analyze
        stats = monitor.stop_monitoring()
        
        # Print performance metrics
        print(f"\nPerformance Metrics:")
        print(f"L1 Cache Hit Rate: {stats.l1_hit_rate:.2%}")
        print(f"L2 Cache Hit Rate: {stats.l2_hit_rate:.2%}")
        if stats.cpu_cycles > 0:
            print(f"Instructions per Cycle: {stats.instructions_per_cycle:.2f}")
        
        return result
    
    return wrapper