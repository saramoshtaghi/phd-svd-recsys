import os

PRIMARY_GENRE_DIR = "/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/primary_genre_synthetic"
RESULTS_DIR = "/home/moshtasa/Research/phd-svd-recsys/SVD/Book/result/rec/top_re/0905/primary_analysis"
TOP_N_LIST = [15, 25, 35]

def check_remaining_datasets():
    if not os.path.isdir(PRIMARY_GENRE_DIR):
        print(f"Directory not found: {PRIMARY_GENRE_DIR}")
        return

    synthetic_files = sorted([f for f in os.listdir(PRIMARY_GENRE_DIR) if f.endswith(".csv")])
    print(f"Found {len(synthetic_files)} total datasets in primary_genre_synthetic")

    remaining_files = []
    completed_files = []

    for filename in synthetic_files:
        base_name = f"primary_{os.path.splitext(filename)[0]}"
        expected_outputs = [f"{base_name}_{n}recommendation.csv" for n in TOP_N_LIST]
        
        if all(os.path.exists(os.path.join(RESULTS_DIR, out)) for out in expected_outputs):
            completed_files.append(filename)
        else:
            remaining_files.append(filename)

    print(f"Completed: {len(completed_files)} datasets")
    print(f"Remaining: {len(remaining_files)} datasets")
    
    if completed_files:
        print("\nCompleted datasets:")
        for f in completed_files[:10]:
            print(f"  ✓ {f}")
        if len(completed_files) > 10:
            print(f"  ... and {len(completed_files) - 10} more")
    
    if remaining_files:
        print("\nRemaining datasets to process:")
        for f in remaining_files[:15]:
            print(f"  → {f}")
        if len(remaining_files) > 15:
            print(f"  ... and {len(remaining_files) - 15} more")
        
        print(f"\nSUMMARY: {len(remaining_files)} datasets remaining out of {len(synthetic_files)} total")
    else:
        print("\nAll datasets have been processed!")

if __name__ == "__main__":
    check_remaining_datasets()
