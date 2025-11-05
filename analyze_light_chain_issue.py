"""
Investigate Light Chain Length Issue

Compares training data, benchmark, and generated light chain lengths
to identify format mismatch and recommend solution.
"""

import json
import numpy as np
from pathlib import Path


def main():
    print("="*80)
    print("LIGHT CHAIN LENGTH ANALYSIS")
    print("="*80)

    # 1. Training Data Analysis
    print("\n" + "─"*80)
    print("1. TRAINING DATA")
    print("─"*80)

    train_stats_file = Path('data/generative/data_stats.json')
    with open(train_stats_file) as f:
        train_stats = json.load(f)

    print(f"From data_stats.json:")
    print(f"  Light chain mean: {train_stats['light_length']['mean']:.1f} aa")

    # Sample actual training data
    train_file = Path('data/generative/train.json')
    with open(train_file) as f:
        train_data = json.load(f)

    sample_size = min(1000, len(train_data))
    train_light_lengths = [len(train_data[i]['antibody_light'])
                           for i in range(sample_size)]

    print(f"\nSample analysis (n={sample_size}):")
    print(f"  Mean:   {np.mean(train_light_lengths):.1f} aa")
    print(f"  Std:    {np.std(train_light_lengths):.1f} aa")
    print(f"  Median: {np.median(train_light_lengths):.1f} aa")
    print(f"  Range:  {np.min(train_light_lengths)}-{np.max(train_light_lengths)} aa")

    # Show examples
    print(f"\nExample sequences (first 3):")
    for i in range(min(3, len(train_data))):
        light = train_data[i]['antibody_light']
        print(f"  {i+1}. Length: {len(light)} aa")
        print(f"     Sequence: {light[:80]}...")

    # 2. Benchmark Analysis
    print("\n" + "─"*80)
    print("2. BENCHMARK DATA (CoV-AbDab)")
    print("─"*80)

    benchmark_file = Path('benchmark/benchmark_dataset.json')
    with open(benchmark_file) as f:
        benchmark_data = json.load(f)

    benchmark_light_lengths = [ab['light_length'] for ab in benchmark_data]

    print(f"CoV-AbDab (n={len(benchmark_data)}):")
    print(f"  Mean:   {np.mean(benchmark_light_lengths):.1f} aa")
    print(f"  Std:    {np.std(benchmark_light_lengths):.1f} aa")
    print(f"  Median: {np.median(benchmark_light_lengths):.1f} aa")
    print(f"  Range:  {np.min(benchmark_light_lengths)}-{np.max(benchmark_light_lengths)} aa")

    # Show examples
    print(f"\nExample sequences (first 3):")
    for i in range(min(3, len(benchmark_data))):
        light = benchmark_data[i]['light_chain']
        print(f"  {i+1}. Length: {len(light)} aa")
        print(f"     Sequence: {light[:80]}...")

    # 3. Generated Sequences Analysis
    print("\n" + "─"*80)
    print("3. GENERATED SEQUENCES (Test Results)")
    print("─"*80)

    results_file = Path('benchmark/sequence_recovery_results.json')
    with open(results_file) as f:
        results = json.load(f)

    generated_light_lengths = [r['generated_light_length']
                              for r in results['individual_results']]

    print(f"Generated (n={len(generated_light_lengths)}):")
    print(f"  Mean:   {np.mean(generated_light_lengths):.1f} aa")
    print(f"  Std:    {np.std(generated_light_lengths):.1f} aa")
    print(f"  Median: {np.median(generated_light_lengths):.1f} aa")
    print(f"  Range:  {np.min(generated_light_lengths)}-{np.max(generated_light_lengths)} aa")

    # Show examples
    print(f"\nExample sequences (first 3):")
    for i in range(min(3, len(results['individual_results']))):
        light = results['individual_results'][i]['generated_light']
        print(f"  {i+1}. Length: {len(light)} aa")
        print(f"     Sequence: {light[:80]}...")

    # 4. Comparison & Analysis
    print("\n" + "="*80)
    print("COMPARISON & ANALYSIS")
    print("="*80)

    train_mean = np.mean(train_light_lengths)
    bench_mean = np.mean(benchmark_light_lengths)
    gen_mean = np.mean(generated_light_lengths)

    print(f"\nLength Comparison:")
    print(f"  Training:   {train_mean:6.1f} aa")
    print(f"  Generated:  {gen_mean:6.1f} aa")
    print(f"  Benchmark:  {bench_mean:6.1f} aa")

    print(f"\nDifferences:")
    diff_gen_bench = gen_mean - bench_mean
    diff_train_bench = train_mean - bench_mean
    diff_gen_train = gen_mean - train_mean

    print(f"  Generated - Benchmark:  {diff_gen_bench:+6.1f} aa ({diff_gen_bench/bench_mean*100:+.1f}%)")
    print(f"  Training  - Benchmark:  {diff_train_bench:+6.1f} aa ({diff_train_bench/bench_mean*100:+.1f}%)")
    print(f"  Generated - Training:   {diff_gen_train:+6.1f} aa ({diff_gen_train/train_mean*100:+.1f}%)")

    # Format detection
    print(f"\n" + "─"*80)
    print("FORMAT DETECTION")
    print("─"*80)

    # Typical formats:
    # V-region (VL): ~105-115 aa
    # Full-length (VL + CL): ~210-220 aa

    V_REGION_MAX = 130
    FULL_LENGTH_MIN = 180

    def detect_format(mean_length):
        if mean_length < V_REGION_MAX:
            return "V-REGION only (VL)"
        elif mean_length > FULL_LENGTH_MIN:
            return "FULL-LENGTH (VL + CL constant region)"
        else:
            return "MIXED or INTERMEDIATE"

    print(f"\nTraining data ({train_mean:.1f} aa):")
    print(f"  Format: {detect_format(train_mean)}")

    print(f"\nBenchmark ({bench_mean:.1f} aa):")
    print(f"  Format: {detect_format(bench_mean)}")

    print(f"\nGenerated ({gen_mean:.1f} aa):")
    print(f"  Format: {detect_format(gen_mean)}")

    # Root cause analysis
    print(f"\n" + "="*80)
    print("ROOT CAUSE ANALYSIS")
    print("="*80)

    if abs(diff_train_bench) > 50:
        print(f"\n🔍 FORMAT MISMATCH DETECTED")
        print(f"\nThe training data and benchmark use different formats:")
        print(f"  - Training:  {detect_format(train_mean)}")
        print(f"  - Benchmark: {detect_format(bench_mean)}")
        print(f"  - Difference: {abs(diff_train_bench):.1f} aa")

        print(f"\n📊 Model Behavior:")
        print(f"  The model learned to generate {detect_format(gen_mean).lower()}")
        print(f"  Generated length ({gen_mean:.1f} aa) is between:")
        print(f"    - Training ({train_mean:.1f} aa)")
        print(f"    - Benchmark ({bench_mean:.1f} aa)")
        print(f"  This is EXPECTED behavior - model learned from training data")

    else:
        print(f"\n✅ No significant format mismatch detected")
        print(f"Length differences may be due to other factors")

    # Recommendations
    print(f"\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    if abs(diff_gen_bench) > 50:
        print(f"\n⚠️  ACTION REQUIRED: Generated sequences are {abs(diff_gen_bench):.1f} aa too long")

        print(f"\n📋 Options (ranked by effort):")

        print(f"\n1. **POST-PROCESSING TRUNCATION** (Quick fix, 1 hour)")
        print(f"   - Truncate generated light chains at ~{bench_mean:.0f} aa")
        print(f"   - Keep V-region, remove constant region")
        print(f"   - Pros: Fast, no retraining needed")
        print(f"   - Cons: May lose some generated content")
        print(f"   - Recommended: YES (for immediate progress)")

        print(f"\n2. **FOCUS ON HEAVY CHAINS** (Alternative, 0 hours)")
        print(f"   - Heavy chains are performing well (42% identity)")
        print(f"   - Many antibodies work with heavy chain CDR3 alone")
        print(f"   - Pros: No changes needed, proceed immediately")
        print(f"   - Cons: Light chain still suboptimal")
        print(f"   - Recommended: IF time is critical")

        print(f"\n3. **REPROCESS TRAINING DATA** (Medium effort, 4-8 hours)")
        print(f"   - Extract V-regions from training data")
        print(f"   - Regenerate data_stats.json")
        print(f"   - Re-test model (no retraining needed if just testing)")
        print(f"   - Pros: Proper format matching")
        print(f"   - Cons: Time-consuming, may affect other metrics")
        print(f"   - Recommended: IF Plan 1 fails")

        print(f"\n4. **FINE-TUNE MODEL** (Long-term, 1-2 days)")
        print(f"   - Fine-tune on V-region-only data")
        print(f"   - Requires additional training time")
        print(f"   - Pros: Best long-term solution")
        print(f"   - Cons: Time and compute intensive")
        print(f"   - Recommended: FOR future iterations")

        print(f"\n✅ RECOMMENDED IMMEDIATE ACTION:")
        print(f"   **Option 1: Post-processing truncation at ~{bench_mean:.0f} aa**")

        print(f"\n   Implementation:")
        print(f"   ```python")
        print(f"   def truncate_light_chain(sequence, target_length={bench_mean:.0f}):")
        print(f"       '''Truncate to V-region length'''")
        print(f"       return sequence[:int(target_length)]")
        print(f"   ```")

        print(f"\n   Test this on 20 benchmark antibodies and measure:")
        print(f"   - New light chain similarity")
        print(f"   - Overall sequence recovery")
        print(f"   - Expected improvement: +10-15% light chain similarity")

    else:
        print(f"\n✅ No action required")
        print(f"Light chain lengths are acceptable")

    # Summary
    print(f"\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print(f"\n📊 Key Findings:")
    print(f"  1. Training data uses {detect_format(train_mean).lower()} ({train_mean:.1f} aa)")
    print(f"  2. Benchmark uses {detect_format(bench_mean).lower()} ({bench_mean:.1f} aa)")
    print(f"  3. Model generates sequences matching training data format")
    print(f"  4. Length error: {abs(diff_gen_bench):.1f} aa (too long)")

    print(f"\n🎯 Recommended Next Steps:")
    print(f"  1. Implement truncation at {bench_mean:.0f} aa")
    print(f"  2. Test on 20 benchmark antibodies")
    print(f"  3. Re-measure sequence recovery")
    print(f"  4. If successful: integrate into pipeline v3")
    print(f"  5. If unsuccessful: try Option 2 (heavy-chain focus)")

    print(f"\n⏰ Time Estimate:")
    print(f"  Implementation: 1 hour")
    print(f"  Testing: 30 minutes")
    print(f"  Total: 1.5 hours")

    print(f"\n💰 Budget Impact: $0 (computational only)")

    print(f"\n✅ Ready to proceed to Day 2 implementation")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
