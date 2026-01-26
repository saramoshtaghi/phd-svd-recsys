import os
import time
from datetime import datetime

RESULTS_DIR = "/home/moshtasa/Research/phd-svd-recsys/SVD/Book/result/rec/top_re/0905/primary_analysis"
LOG_FILE = "/home/moshtasa/Research/phd-svd-recsys/SVD/Book/notebook/SVD_primary_nohup.log"

def monitor_progress():
    print(f"Monitoring progress at {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)
    
    # Check if process is running
    import subprocess
    try:
        result = subprocess.run(['pgrep', '-f', 'SVD_primary_only_no_original.py'], 
                              capture_output=True, text=True)
        if result.stdout.strip():
            print(f"✓ Process is running (PID: {result.stdout.strip()})")
        else:
            print("✗ Process is not running")
    except:
        print("? Could not check process status")
    
    # Count output files
    try:
        files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.csv')]
        print(f"✓ Output files generated: {len(files)}")
        if files:
            print("Recent files:")
            for f in sorted(files)[-5:]:  # Show last 5 files
                print(f"  - {f}")
    except:
        print("✗ Could not access output directory")
    
    # Check log file size
    try:
        log_size = os.path.getsize(LOG_FILE)
        print(f"✓ Log file size: {log_size:,} bytes")
    except:
        print("✗ Could not access log file")

if __name__ == "__main__":
    monitor_progress()
