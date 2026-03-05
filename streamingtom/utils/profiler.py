import os
from contextlib import contextmanager
from typing import Dict, List

import torch

class StreamingTOMProfiler:

    def __init__(self, enabled: bool = None):
        if enabled is None:
            self.enabled = os.getenv('STREAMINGTOM_PROFILE', '0') == '1'
        else:
            self.enabled = enabled

        self.timings: Dict[str, List[float]] = {}
        self.current_sample: Dict[str, float] = {}
        self.sample_count = 0
        self.sample_metrics: Dict[str, List[float]] = {}
        self.is_warmup = True

    @contextmanager
    def timer(self, stage: str):
        if not self.enabled or not torch.cuda.is_available():
            yield
            return

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        yield
        end_event.record()
        torch.cuda.synchronize()
        elapsed = start_event.elapsed_time(end_event) / 1000.0

        if stage in self.current_sample:
            self.current_sample[stage] += elapsed
        else:
            self.current_sample[stage] = elapsed

        if not self.is_warmup:
            if stage not in self.timings:
                self.timings[stage] = []
            self.timings[stage].append(elapsed)

    def record_time(self, stage: str, elapsed: float):
        if not self.enabled:
            return

        self.current_sample[stage] = elapsed

        if not self.is_warmup:
            if stage not in self.timings:
                self.timings[stage] = []
            self.timings[stage].append(elapsed)

    def record_metric(self, name: str, value: float):
        if not self.enabled:
            return

        if not self.is_warmup:
            if name not in self.sample_metrics:
                self.sample_metrics[name] = []
            self.sample_metrics[name].append(value)

    def new_sample(self):
        if not self.enabled:
            return

        if self.current_sample:
            self.sample_count += 1
            self._print_current_sample()
            if self.is_warmup:
                self.is_warmup = False
        self.current_sample = {}

    def _print_current_sample(self):
        warmup_tag = " (WARMUP - EXCLUDED)" if self.is_warmup else ""
        print(f"\n[STREAMINGTOM PROFILE] Sample #{self.sample_count} Time Breakdown{warmup_tag}:")
        print("-" * 60)

        total_time = sum(self.current_sample.values())
        sorted_stages = sorted(self.current_sample.items(), key=lambda x: x[1], reverse=True)

        for stage, elapsed in sorted_stages:
            percentage = (elapsed / total_time) * 100 if total_time > 0 else 0
            print(f"  {stage:25s}: {elapsed*1000:8.2f}ms ({percentage:5.1f}%)")

        print("-" * 60)
        print(f"  {'TOTAL':25s}: {total_time*1000:8.2f}ms")
        print()

    def get_summary(self) -> Dict:
        if not self.enabled or not self.timings:
            return {}

        summary = {}
        for stage, times in self.timings.items():
            summary[stage] = {
                'mean_ms': sum(times) / len(times) * 1000,
                'min_ms': min(times) * 1000,
                'max_ms': max(times) * 1000,
                'total_ms': sum(times) * 1000,
                'count': len(times)
            }
        return summary

    def print_summary(self):
        if not self.enabled or not self.timings:
            return

        if self.current_sample:
            self.sample_count += 1
            self._print_current_sample()
            self.current_sample = {}

        print("\n" + "="*70)
        print("STREAMINGTOM PERFORMANCE SUMMARY")
        print("="*70)

        if self.sample_metrics:
            num_samples = len(next(iter(self.sample_metrics.values())))
            print(f"\n=== Sample-Level Metrics (n={num_samples}, excluding warmup) ===")
            print("-" * 70)

            for metric_name, values in sorted(self.sample_metrics.items()):
                if len(values) > 0:
                    avg = sum(values) / len(values)
                    values_str = ', '.join([f'{v:.4f}' if metric_name != 'memory_gb' else f'{v:.2f}' for v in values])

                    if metric_name == 'ttft_sec':
                        print(f"  TTFT (sec)      : Avg={avg:.4f}, Values=[{values_str}]")
                    elif metric_name == 'throughput':
                        print(f"  Throughput (t/s): Avg={avg:.1f}, Values=[{values_str}]")
                    elif metric_name == 'memory_gb':
                        print(f"  Memory (GB)     : Avg={avg:.2f}, Values=[{values_str}]")

        summary = self.get_summary()

        batch_stages = set()
        for stage, stats in summary.items():
            if stats['count'] > self.sample_count:
                batch_stages.add(stage)

        print(f"\n=== Batch Operations (Total across all batches) ===")
        print("-" * 70)

        batch_total = 0
        for stage, stats in sorted(summary.items(), key=lambda x: x[1]['total_ms'], reverse=True):
            if stage in batch_stages:
                avg_per_call = stats['mean_ms']
                total = stats['total_ms']
                batch_total += total
                print(f"  {stage:25s}: Total: {total:8.1f}ms, Avg/call: {avg_per_call:6.1f}ms, Calls: {stats['count']}")

        print("-" * 70)
        print(f"  {'Batch Ops Total':25s}: {batch_total:8.1f}ms")

        print(f"\n=== Per-Query Operations ===")
        print("-" * 70)

        query_total = 0
        for stage, stats in sorted(summary.items(), key=lambda x: x[1]['mean_ms'], reverse=True):
            if stage not in batch_stages:
                query_total += stats['mean_ms']
                if stage in self.timings and len(self.timings[stage]) == self.sample_count:
                    values = self.timings[stage]
                    values_str = ', '.join([f'{v*1000:.2f}' for v in values])
                    print(f"  {stage:25s}: Avg: {stats['mean_ms']:8.2f}ms, Values=[{values_str}]ms")
                else:
                    print(f"  {stage:25s}: Avg: {stats['mean_ms']:8.2f}ms "
                          f"[min:{stats['min_ms']:.1f}, max:{stats['max_ms']:.1f}]")

        print("-" * 70)
        print(f"  {'Query Ops Total (avg)':25s}: {query_total:8.2f}ms")
        print("="*70)


_profiler_instance = None

def get_profiler() -> StreamingTOMProfiler:
    global _profiler_instance
    if _profiler_instance is None:
        _profiler_instance = StreamingTOMProfiler()
    return _profiler_instance

def reset_profiler():
    global _profiler_instance
    _profiler_instance = StreamingTOMProfiler()
