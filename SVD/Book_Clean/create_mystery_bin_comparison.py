"""
MYSTERY 1000 vs BASELINE - BIN COMPARISON FOR TOP-35
===================================================

Creates a focused visualization showing per-bin improvements in mystery books
when using Mystery 1000 extreme injection vs baseline.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def create_mystery_bin_comparison():
    """Create detailed bin comparison for Mystery 1000 vs Baseline"""
    
    print("🔍 CREATING MYSTERY 1000 vs BASELINE BIN COMPARISON")
    print("=" * 60)
    print("🎯 Focus: Top-35 recommendations")
    print("📊 Analysis: 10 user bins showing mystery book improvements")
    print("=" * 60)
    
    # Load genre mappings
    df_genres = pd.read_csv("/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/df_final_with_genres.csv")
    df_genre_filtered = df_genres.dropna(subset=['genres'])
    mystery_books = set(df_genre_filtered[df_genre_filtered['genres'].str.contains('Mystery', na=False)]['book_id'])
    
    print(f"🔍 Mystery books: {len(mystery_books)}")
    
    # Define datasets to compare
    datasets = {
        'Baseline': "/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/df_final_with_genres.csv",
        'Mystery 1000': "/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/synthetic_enhanced/enhanced_mystery_1000_any.csv"
    }
    
    # Process both datasets
    all_results = {}
    
    for dataset_name, dataset_path in datasets.items():
        print(f"\n🔬 Processing {dataset_name}...")
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        original_users_data = df[df['user_id'] <= 53424].copy()
        
        print(f"   📂 Loaded {len(df):,} ratings ({len(original_users_data):,} from original users)")
        
        # Train SVD model
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df[['user_id', 'book_id', 'rating']], reader)
        trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
        
        # Optimized parameters for Top-35
        algo = SVD(n_factors=100, n_epochs=15, lr_all=0.005, reg_all=0.02, random_state=42)
        algo.fit(trainset)
        
        print(f"   🤖 SVD model trained")
        
        # Generate Top-35 recommendations for users 1-3000
        sample_users = list(range(1, 3001))
        all_items = list(set(original_users_data['book_id'].unique()))
        
        user_results = []
        processed = 0
        
        for user_id in sample_users:
            if processed % 500 == 0:
                print(f"   Progress: {processed}/3000 users...")
            
            # Get user's rated books
            user_books = set(original_users_data[original_users_data['user_id'] == user_id]['book_id'])
            if len(user_books) == 0:
                processed += 1
                continue
            
            # Get candidate books (sample 300 for speed)
            candidate_items = [item for item in all_items if item not in user_books]
            if len(candidate_items) > 300:
                candidate_items = np.random.choice(candidate_items, 300, replace=False)
            
            # Predict ratings for Top-35
            predictions = []
            for item_id in candidate_items:
                try:
                    pred = algo.predict(user_id, item_id)
                    predictions.append((item_id, pred.est))
                except:
                    continue
            
            if predictions:
                # Sort by predicted rating and get top 35
                predictions.sort(key=lambda x: x[1], reverse=True)
                top_35 = [item_id for item_id, rating in predictions[:35]]
                
                # Count mystery books in top 35
                mystery_count = sum(1 for book_id in top_35 if book_id in mystery_books)
                
                user_results.append({
                    'user_id': user_id,
                    'mystery_count': mystery_count
                })
            
            processed += 1
            if processed >= 2000:  # Limit to 2000 users for speed
                break
        
        all_results[dataset_name] = pd.DataFrame(user_results)
        print(f"   ✅ Processed {len(user_results)} users")
    
    # Create bin analysis
    print(f"\n📊 Creating bin analysis...")
    
    # Sort baseline users by user_id for consistent binning
    baseline_df = all_results['Baseline'].sort_values('user_id')
    mystery_df = all_results['Mystery 1000'].sort_values('user_id')
    
    # Merge data on user_id to ensure same users in both datasets
    merged_df = pd.merge(baseline_df, mystery_df, on='user_id', suffixes=('_baseline', '_mystery'))
    
    print(f"   📋 Users in both datasets: {len(merged_df)}")
    
    # Create 10 bins
    n_bins = 10
    bin_size = len(merged_df) // n_bins
    
    bin_results = []
    
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(merged_df)
        bin_data = merged_df.iloc[start_idx:end_idx]
        
        # Calculate statistics for this bin
        baseline_avg = bin_data['mystery_count_baseline'].mean()
        mystery_avg = bin_data['mystery_count_mystery'].mean()
        improvement = mystery_avg - baseline_avg
        improvement_pct = (improvement / baseline_avg) * 100 if baseline_avg > 0 else 0
        
        # Count users who got more mystery books
        users_improved = (bin_data['mystery_count_mystery'] > bin_data['mystery_count_baseline']).sum()
        total_users = len(bin_data)
        improvement_rate = (users_improved / total_users) * 100
        
        bin_results.append({
            'bin': i + 1,
            'baseline_avg': baseline_avg,
            'mystery_avg': mystery_avg,
            'improvement': improvement,
            'improvement_pct': improvement_pct,
            'users_improved': users_improved,
            'total_users': total_users,
            'improvement_rate': improvement_rate,
            'user_range': f"{bin_data['user_id'].min()}-{bin_data['user_id'].max()}"
        })
    
    bin_df = pd.DataFrame(bin_results)
    
    # Create comprehensive visualization
    create_mystery_visualization(bin_df, merged_df)
    
    return bin_df, merged_df

def create_mystery_visualization(bin_df, merged_df):
    """Create focused Mystery 1000 vs Baseline visualization"""
    
    print(f"\n🎨 Creating Mystery 1000 vs Baseline visualization...")
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("Set2")
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Mystery 1000 Extreme vs Baseline: Top-35 Mystery Book Improvements by User Bins', 
                 fontsize=16, fontweight='bold')
    
    # Colors
    baseline_color = '#3498db'  # Blue
    mystery_color = '#e74c3c'   # Red
    improvement_color = '#2ecc71'  # Green
    
    # Plot 1: Average Mystery Books per Bin
    ax1 = axes[0, 0]
    
    x = np.arange(len(bin_df))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, bin_df['baseline_avg'], width, 
                   label='Baseline', alpha=0.8, color=baseline_color)
    bars2 = ax1.bar(x + width/2, bin_df['mystery_avg'], width,
                   label='Mystery 1000', alpha=0.8, color=mystery_color)
    
    ax1.set_xlabel('User Bins')
    ax1.set_ylabel('Average Mystery Books per User (Top-35)')
    ax1.set_title('Average Mystery Books by User Bin', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Bin {i}' for i in bin_df['bin']])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Improvement per Bin (absolute numbers)
    ax2 = axes[0, 1]
    
    bars3 = ax2.bar(x, bin_df['improvement'], alpha=0.8, color=improvement_color)
    
    ax2.set_xlabel('User Bins')
    ax2.set_ylabel('Improvement in Mystery Books')
    ax2.set_title('Mystery Book Improvement per Bin', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Bin {i}' for i in bin_df['bin']])
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (0.05 if height >= 0 else -0.1),
                f'{height:+.1f}', ha='center', va='bottom' if height >= 0 else 'top', 
                fontsize=9, fontweight='bold')
    
    # Plot 3: Percentage of Users Improved per Bin
    ax3 = axes[1, 0]
    
    bars4 = ax3.bar(x, bin_df['improvement_rate'], alpha=0.8, color='#f39c12')
    
    ax3.set_xlabel('User Bins')
    ax3.set_ylabel('% of Users with More Mystery Books')
    ax3.set_title('User Success Rate by Bin', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'Bin {i}' for i in bin_df['bin']])
    ax3.set_ylim(0, 100)
    ax3.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50% threshold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars4:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Scatter plot showing individual user improvements
    ax4 = axes[1, 1]
    
    # Create bins for color coding
    merged_df['bin'] = pd.cut(merged_df['user_id'], bins=10, labels=range(1, 11))
    
    # Scatter plot with different colors for each bin
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for bin_num in range(1, 11):
        bin_data = merged_df[merged_df['bin'] == bin_num]
        ax4.scatter(bin_data['mystery_count_baseline'], bin_data['mystery_count_mystery'],
                   label=f'Bin {bin_num}', alpha=0.6, s=20, color=colors[bin_num-1])
    
    # Add diagonal line (y=x) for reference
    max_val = max(merged_df['mystery_count_baseline'].max(), merged_df['mystery_count_mystery'].max())
    ax4.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='No change')
    
    ax4.set_xlabel('Baseline Mystery Books')
    ax4.set_ylabel('Mystery 1000 Mystery Books')
    ax4.set_title('Individual User Improvements', fontweight='bold')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mystery_1000_vs_baseline_top35.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"🎨 Visualization saved as: mystery_1000_vs_baseline_top35.png")
    
    # Print detailed statistics
    print(f"\n📊 DETAILED BIN STATISTICS:")
    print("=" * 80)
    
    total_improvement = bin_df['improvement'].mean()
    total_success_rate = bin_df['improvement_rate'].mean()
    
    print(f"📋 OVERALL PERFORMANCE:")
    print(f"   Average improvement: {total_improvement:+.2f} mystery books per user")
    print(f"   Average success rate: {total_success_rate:.1f}% of users improved")
    print()
    
    print(f"📊 PER-BIN BREAKDOWN:")
    for _, row in bin_df.iterrows():
        print(f"   Bin {row['bin']} (Users {row['user_range']}):")
        print(f"      Baseline: {row['baseline_avg']:.2f} → Mystery 1000: {row['mystery_avg']:.2f}")
        print(f"      Improvement: {row['improvement']:+.2f} books ({row['improvement_pct']:+.1f}%)")
        print(f"      Success rate: {row['users_improved']}/{row['total_users']} users ({row['improvement_rate']:.1f}%)")
        
        if row['improvement'] > 0.5:
            print(f"      ✅ STRONG SUCCESS")
        elif row['improvement'] > 0:
            print(f"      ✅ SUCCESS")
        else:
            print(f"      ❌ FAILED")
        print()
    
    # Identify best performing bins
    best_bin = bin_df.loc[bin_df['improvement'].idxmax()]
    worst_bin = bin_df.loc[bin_df['improvement'].idxmin()]
    
    print(f"🏆 BEST PERFORMING BIN:")
    print(f"   Bin {best_bin['bin']}: {best_bin['improvement']:+.2f} books improvement")
    print(f"   {best_bin['improvement_rate']:.1f}% of users improved")
    print()
    
    print(f"⚠️  WORST PERFORMING BIN:")
    print(f"   Bin {worst_bin['bin']}: {worst_bin['improvement']:+.2f} books change")
    print(f"   {worst_bin['improvement_rate']:.1f}% of users improved")

if __name__ == "__main__":
    bin_df, merged_df = create_mystery_bin_comparison()
    
    # Save detailed results
    bin_df.to_csv('mystery_bin_analysis.csv', index=False)
    merged_df.to_csv('mystery_user_comparison.csv', index=False)
    
    print(f"\n🎉 MYSTERY BIN COMPARISON COMPLETE!")
    print(f"📊 Analyzed {len(merged_df)} users across 10 bins")
    print(f"📁 Files saved:")
    print(f"   - mystery_1000_vs_baseline_top35.png (visualization)")
    print(f"   - mystery_bin_analysis.csv (bin statistics)")
    print(f"   - mystery_user_comparison.csv (individual user data)")
