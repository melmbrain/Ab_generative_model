"""
Training Monitor - View training progress in real-time

Reads log files and displays current training metrics
"""

import json
import sys
from pathlib import Path


def load_training_logs(log_file: str):
    """
    Load training logs from JSONL file

    Args:
        log_file: path to log file

    Returns:
        List of log entries
    """
    log_path = Path(log_file)

    if not log_path.exists():
        print(f"Error: Log file not found: {log_file}")
        return []

    logs = []
    with open(log_path, 'r') as f:
        for line in f:
            if line.strip():
                logs.append(json.loads(line))

    return logs


def display_training_progress(log_file: str):
    """
    Display training progress from log file

    Args:
        log_file: path to log file
    """
    logs = load_training_logs(log_file)

    if len(logs) == 0:
        print("No training logs found")
        return

    print("="*70)
    print("Training Progress")
    print("="*70)

    # Display overall stats
    print(f"\nTotal Epochs: {len(logs)}")

    # Get latest metrics
    latest = logs[-1]
    print(f"\nLatest Epoch ({latest['epoch']}):")
    print(f"  Train Loss: {latest['train_loss']:.4f}")
    print(f"  Val Loss:   {latest['val_loss']:.4f}")

    if 'val_metrics' in latest:
        val_metrics = latest['val_metrics']
        if 'validity' in val_metrics:
            print(f"  Validity:   {val_metrics['validity']:.1f}%")
        if 'diversity' in val_metrics:
            print(f"  Diversity:  {val_metrics['diversity'].get('unique_ratio', 0):.1f}%")

    # Find best epoch
    best_val_loss = min(log['val_loss'] for log in logs)
    best_epoch = next(log['epoch'] for log in logs if log['val_loss'] == best_val_loss)

    print(f"\nBest Validation Loss: {best_val_loss:.4f} (Epoch {best_epoch})")

    # Show recent history
    print(f"\nRecent History (last 5 epochs):")
    print("  Epoch | Train Loss | Val Loss  | Time")
    print("  ------|------------|-----------|-------")

    for log in logs[-5:]:
        epoch = log['epoch']
        train_loss = log['train_loss']
        val_loss = log['val_loss']
        epoch_time = log.get('epoch_time', 0)

        print(f"  {epoch:5d} | {train_loss:10.4f} | {val_loss:9.4f} | {epoch_time:5.1f}s")

    # Total training time
    total_time = sum(log.get('epoch_time', 0) for log in logs)
    print(f"\nTotal Training Time: {total_time/3600:.2f} hours")

    print("="*70)


def list_experiments(log_dir: str = 'logs'):
    """
    List all available experiments

    Args:
        log_dir: directory containing logs
    """
    log_path = Path(log_dir)

    if not log_path.exists():
        print(f"Log directory not found: {log_dir}")
        return

    log_files = list(log_path.glob('*.jsonl'))

    if len(log_files) == 0:
        print(f"No log files found in {log_dir}")
        return

    print("="*70)
    print("Available Experiments")
    print("="*70)

    for i, log_file in enumerate(log_files, 1):
        # Load logs to get info
        logs = load_training_logs(log_file)

        if len(logs) > 0:
            latest = logs[-1]
            print(f"\n{i}. {log_file.stem}")
            print(f"   Epochs: {len(logs)}")
            print(f"   Latest Val Loss: {latest['val_loss']:.4f}")

    print("="*70)


def compare_experiments(log_files: list):
    """
    Compare multiple experiments

    Args:
        log_files: list of log file paths
    """
    print("="*70)
    print("Experiment Comparison")
    print("="*70)

    results = []

    for log_file in log_files:
        logs = load_training_logs(log_file)

        if len(logs) == 0:
            continue

        latest = logs[-1]
        best_val_loss = min(log['val_loss'] for log in logs)

        results.append({
            'name': Path(log_file).stem,
            'epochs': len(logs),
            'final_val_loss': latest['val_loss'],
            'best_val_loss': best_val_loss
        })

    # Display comparison table
    print("\n  Experiment | Epochs | Final Val Loss | Best Val Loss")
    print("  -----------|--------|----------------|---------------")

    for r in results:
        print(f"  {r['name']:10s} | {r['epochs']:6d} | {r['final_val_loss']:14.4f} | {r['best_val_loss']:13.4f}")

    print("="*70)


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python monitor_training.py list")
        print("  python monitor_training.py <log_file>")
        print("  python monitor_training.py compare <log_file1> <log_file2> ...")
        return

    command = sys.argv[1]

    if command == 'list':
        list_experiments()

    elif command == 'compare':
        if len(sys.argv) < 3:
            print("Error: Please specify log files to compare")
            return
        compare_experiments(sys.argv[2:])

    else:
        # Assume it's a log file path
        display_training_progress(command)


if __name__ == '__main__':
    main()
