"""
COMPREHENSIVE BIAS INJECTION VISUALIZATIONS
===========================================

Creates detailed per-user, per-bin, per-dataset visualizations to show
the improvement (or lack thereof) from bias injection strategies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def load_and_analyze_all_datasets():
    """Load and analyze all datasets with comprehensive visualization"""
    
    print("📊 COMPREHENSIVE BIAS INJECTION VISUALIZATION")
    print("=" * 60)
    
    # Load genre mappings
    df_genres = pd.read_csv("/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/df_final_with_genres.csv")
    df_genre_filtered = df_genres.dropna(subset=['genres'])
    adventure_books = set(df_genre_filtered[df_genre_filtered['genres'].str.contains('Adventure', na=False)]['book_id'])
    mystery_books = set(df_genre_filtered[df_genre_filtered['genres'].str.contains('Mystery', na=False)]['book_id'])
    
    print(f"🗺️  Adventure books: {len(adventure_books)}")
    print(f"🔍 Mystery books: {len(mystery_books)}")
    
    # Define all datasets to analyze
    datasets = [
        ("/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/df_final_with_genres.csv", "Baseline"),
        
        # Original extreme approach (failed results)
        ("/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/synthetic_enhanced/enhanced_adventure_1000_any.csv", "Adventure 1000 Extreme"),
        ("/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/synthetic_enhanced/enhanced_mystery_1000_any.csv", "Mystery 1000 Extreme"),
        
        # New moderate approach (promising results)
        ("/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/improved_synthetic/enhanced_adventure_moderate_300.csv", "Adventure 300 Moderate"),
        ("/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/improved_synthetic/enhanced_mystery_moderate_300.csv", "Mystery 300 Moderate"),
    ]
    
    all_results = []
    
    # Process each dataset
    for dataset_path, dataset_name in datasets:
        print(f"\\n🔬 Processing {dataset_name}...")
        
        try:
            # Load and prepare dataset
            df = pd.read_csv(dataset_path)
            original_users_data = df[df['user_id'] <= 53424].copy()
            
            print(f"   📂 Loaded {len(df):,} ratings ({len(original_users_data):,} from original users)")
            
            # Train SVD model
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(df[['user_id', 'book_id', 'rating']], reader)
            trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
            
            # Use optimized parameters for faster training
            algo = SVD(n_factors=50, n_epochs=8, lr_all=0.01, reg_all=0.05, random_state=42)
            algo.fit(trainset)
            
            print(f"   🤖 SVD model trained")
            
            # Generate recommendations for sample of original users (for speed)
            sample_users = list(range(1, 2001))  # First 2000 users
            all_items = list(set(original_users_data['book_id'].unique()))[::10]  # Sample items for speed
            
            user_results = []
            processed = 0
            
            for user_id in sample_users:
                if processed % 500 == 0:
                    print(f"   Progress: {processed}/2000 users...")
                
                # Get user's rated books
                user_books = set(original_users_data[original_users_data['user_id'] == user_id]['book_id'])
                if len(user_books) == 0:
                    processed += 1
                    continue
                
                # Get candidate books (sample for speed)
                candidate_items = [item for item in all_items if item not in user_books][:200]
                
                # Predict ratings
                predictions = []
                for item_id in candidate_items:
                    try:
                        pred = algo.predict(user_id, item_id)
                        predictions.append((item_id, pred.est))
                    except:
                        continue
                
                if predictions:
                    # Sort by predicted rating and get top 15
                    predictions.sort(key=lambda x: x[1], reverse=True)
                    top_15 = [item_id for item_id, rating in predictions[:15]]
                    
                    # Count genres
                    adventure_count = sum(1 for book_id in top_15 if book_id in adventure_books)
                    mystery_count = sum(1 for book_id in top_15 if book_id in mystery_books)
                    other_count = 15 - adventure_count - mystery_count
                    
                    user_results.append({
                        'user_id': user_id,
                        'dataset': dataset_name,
                        'adventure_count': adventure_count,
                        'mystery_count': mystery_count,
                        'other_count': other_count
                    })
                
                processed += 1
                if processed >= 1000:  # Limit for speed
                    break
            
            all_results.extend(user_results)
            print(f"   ✅ Processed {len(user_results)} users")
            
        except Exception as e:
            print(f"   ❌ Error processing {dataset_name}: {e}")
            continue
    
    # Convert to DataFrame
    if all_results:
        results_df = pd.DataFrame(all_results)
        print(f"\\n📊 Total results: {len(results_df)} user-dataset combinations")
        
        # Create visualizations
        create_comprehensive_visualizations(results_df)
        
        # Save raw results
        results_df.to_csv('comprehensive_bias_results.csv', index=False)
        print(f"💾 Saved raw results to: comprehensive_bias_results.csv")
        
        return results_df
    else:
        print("❌ No results to visualize!")
        return None

def create_comprehensive_visualizations(results_df):
    """Create comprehensive visualizations"""
    
    print("\\n🎨 Creating comprehensive visualizations...")
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 24))
    
    # 1. Per-Dataset Average Comparison
    ax1 = plt.subplot(4, 2, 1)
    dataset_means = results_df.groupby('dataset').agg({
        'adventure_count': 'mean',
        'mystery_count': 'mean'
    }).reset_index()
    
    x = np.arange(len(dataset_means))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, dataset_means['adventure_count'], width, 
                   label='Adventure', alpha=0.8, color='#e74c3c')
    bars2 = ax1.bar(x + width/2, dataset_means['mystery_count'], width,
                   label='Mystery', alpha=0.8, color='#3498db')
    
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Average Books per User (Top-15)')
    ax1.set_title('Average Genre Books per User by Dataset', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(dataset_means['dataset'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    # 2. Adventure Books - Distribution by Dataset
    ax2 = plt.subplot(4, 2, 2)
    datasets = results_df['dataset'].unique()
    adventure_data = [results_df[results_df['dataset'] == ds]['adventure_count'] for ds in datasets]
    
    box_plot = ax2.boxplot(adventure_data, labels=datasets, patch_artist=True)
    colors = ['#f39c12', '#e74c3c', '#9b59b6', '#2ecc71', '#3498db']
    for patch, color in zip(box_plot['boxes'], colors[:len(datasets)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Adventure Books per User')
    ax2.set_title('Adventure Books Distribution by Dataset', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 3. Mystery Books - Distribution by Dataset  
    ax3 = plt.subplot(4, 2, 3)
    mystery_data = [results_df[results_df['dataset'] == ds]['mystery_count'] for ds in datasets]
    
    box_plot2 = ax3.boxplot(mystery_data, labels=datasets, patch_artist=True)
    for patch, color in zip(box_plot2['boxes'], colors[:len(datasets)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_ylabel('Mystery Books per User')
    ax3.set_title('Mystery Books Distribution by Dataset', fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # 4. User Bins Analysis (Adventure)
    ax4 = plt.subplot(4, 2, 4)
    
    # Create user bins for baseline
    baseline_data = results_df[results_df['dataset'] == 'Baseline'].copy()
    baseline_data = baseline_data.sort_values('user_id')
    
    if len(baseline_data) > 0:
        n_bins = 10
        bin_size = len(baseline_data) // n_bins
        
        bin_data = []
        bin_labels = []
        
        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(baseline_data)
            bin_users = baseline_data.iloc[start_idx:end_idx]['user_id'].tolist()
            
            bin_results = {}
            for dataset in datasets:
                dataset_users = results_df[
                    (results_df['dataset'] == dataset) & 
                    (results_df['user_id'].isin(bin_users))
                ]
                avg_adventure = dataset_users['adventure_count'].mean() if len(dataset_users) > 0 else 0
                bin_results[dataset] = avg_adventure
            
            bin_data.append(bin_results)
            bin_labels.append(f'Bin {i+1}')
        
        # Plot binned data
        x_bins = np.arange(len(bin_labels))
        width = 0.15
        
        for idx, dataset in enumerate(datasets):
            values = [bin_results.get(dataset, 0) for bin_results in bin_data]
            ax4.bar(x_bins + (idx - 2) * width, values, width, 
                   label=dataset, alpha=0.8, color=colors[idx])
        
        ax4.set_xlabel('User Bins')
        ax4.set_ylabel('Avg Adventure Books')
        ax4.set_title('Adventure Books by User Bins', fontweight='bold')
        ax4.set_xticks(x_bins)
        ax4.set_xticklabels(bin_labels, rotation=45)
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
    
    # 5. User Bins Analysis (Mystery)
    ax5 = plt.subplot(4, 2, 5)
    
    if len(baseline_data) > 0:
        for idx, dataset in enumerate(datasets):
            values = []
            for bin_results in bin_data:
                # Recalculate for mystery
                bin_users = baseline_data.iloc[
                    (len(bin_data) - len(values) - 1) * bin_size:
                    (len(bin_data) - len(values)) * bin_size if len(values) < n_bins - 1 else len(baseline_data)
                ]['user_id'].tolist() if len(values) < len(bin_data) else []
                
                dataset_users = results_df[
                    (results_df['dataset'] == dataset) & 
                    (results_df['user_id'].isin(bin_users))
                ] if bin_users else pd.DataFrame()
                
                avg_mystery = dataset_users['mystery_count'].mean() if len(dataset_users) > 0 else 0
                values.append(avg_mystery)
            
            # Recalculate properly
            values = []
            for bin_idx in range(n_bins):
                start_idx = bin_idx * bin_size
                end_idx = (bin_idx + 1) * bin_size if bin_idx < n_bins - 1 else len(baseline_data)
                bin_users = baseline_data.iloc[start_idx:end_idx]['user_id'].tolist()
                
                dataset_users = results_df[
                    (results_df['dataset'] == dataset) & 
                    (results_df['user_id'].isin(bin_users))
                ]
                avg_mystery = dataset_users['mystery_count'].mean() if len(dataset_users) > 0 else 0
                values.append(avg_mystery)
            
            ax5.bar(x_bins + (idx - 2) * width, values, width, 
                   label=dataset, alpha=0.8, color=colors[idx])
        
        ax5.set_xlabel('User Bins')
        ax5.set_ylabel('Avg Mystery Books')
        ax5.set_title('Mystery Books by User Bins', fontweight='bold')
        ax5.set_xticks(x_bins)
        ax5.set_xticklabels(bin_labels, rotation=45)
        ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax5.grid(True, alpha=0.3)
    
    # 6. Improvement Heatmap
    ax6 = plt.subplot(4, 2, 6)
    
    # Calculate improvements vs baseline
    baseline_adventure = results_df[results_df['dataset'] == 'Baseline']['adventure_count'].mean()
    baseline_mystery = results_df[results_df['dataset'] == 'Baseline']['mystery_count'].mean()
    
    improvements = []
    dataset_names = []
    
    for dataset in datasets:
        if dataset != 'Baseline':
            data = results_df[results_df['dataset'] == dataset]
            avg_adv = data['adventure_count'].mean()
            avg_mys = data['mystery_count'].mean()
            
            adv_improvement = ((avg_adv - baseline_adventure) / baseline_adventure) * 100
            mys_improvement = ((avg_mys - baseline_mystery) / baseline_mystery) * 100
            
            improvements.append([adv_improvement, mys_improvement])
            dataset_names.append(dataset)
    
    if improvements:
        improvements = np.array(improvements)
        
        im = ax6.imshow(improvements.T, cmap='RdYlGn', aspect='auto', vmin=-50, vmax=50)
        
        ax6.set_xticks(range(len(dataset_names)))
        ax6.set_xticklabels(dataset_names, rotation=45, ha='right')
        ax6.set_yticks([0, 1])
        ax6.set_yticklabels(['Adventure', 'Mystery'])
        ax6.set_title('Improvement vs Baseline (%)', fontweight='bold')
        
        # Add text annotations
        for i in range(len(dataset_names)):
            for j in range(2):
                text = ax6.text(i, j, f'{improvements[i, j]:.1f}%',
                              ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=ax6, label='Improvement %')
    
    # 7. Scatter plot: Adventure vs Mystery per dataset
    ax7 = plt.subplot(4, 2, 7)
    
    for idx, dataset in enumerate(datasets):
        data = results_df[results_df['dataset'] == dataset]
        ax7.scatter(data['adventure_count'], data['mystery_count'], 
                   label=dataset, alpha=0.6, s=20, color=colors[idx])
    
    ax7.set_xlabel('Adventure Books per User')
    ax7.set_ylabel('Mystery Books per User')
    ax7.set_title('Adventure vs Mystery Books per User', fontweight='bold')
    ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax7.grid(True, alpha=0.3)
    
    # 8. Success Rate Analysis
    ax8 = plt.subplot(4, 2, 8)
    
    success_data = []
    
    for dataset in datasets:
        data = results_df[results_df['dataset'] == dataset]
        total_users = len(data)
        
        users_with_adventure = (data['adventure_count'] > 0).sum()
        users_with_mystery = (data['mystery_count'] > 0).sum()
        
        success_data.append({
            'dataset': dataset,
            'adventure_coverage': (users_with_adventure / total_users) * 100 if total_users > 0 else 0,
            'mystery_coverage': (users_with_mystery / total_users) * 100 if total_users > 0 else 0
        })
    
    success_df = pd.DataFrame(success_data)
    
    x = np.arange(len(success_df))
    bars1 = ax8.bar(x - width/2, success_df['adventure_coverage'], width, 
                   label='Adventure Coverage', alpha=0.8, color='#e74c3c')
    bars2 = ax8.bar(x + width/2, success_df['mystery_coverage'], width,
                   label='Mystery Coverage', alpha=0.8, color='#3498db')
    
    ax8.set_xlabel('Dataset')
    ax8.set_ylabel('User Coverage %')
    ax8.set_title('User Coverage by Dataset', fontweight='bold')
    ax8.set_xticks(x)
    ax8.set_xticklabels(success_df['dataset'], rotation=45, ha='right')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    ax8.set_ylim(0, 105)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('comprehensive_bias_injection_analysis.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print("🎨 Comprehensive visualization saved as: comprehensive_bias_injection_analysis.png")
    
    # Print summary statistics
    print("\\n📊 SUMMARY STATISTICS:")
    print("=" * 50)
    
    baseline_stats = results_df[results_df['dataset'] == 'Baseline'].agg({
        'adventure_count': 'mean',
        'mystery_count': 'mean'
    })
    
    print(f"📋 Baseline Performance:")
    print(f"   Adventure: {baseline_stats['adventure_count']:.2f} books per user")
    print(f"   Mystery: {baseline_stats['mystery_count']:.2f} books per user")
    print()
    
    for dataset in datasets:
        if dataset != 'Baseline':
            data_stats = results_df[results_df['dataset'] == dataset].agg({
                'adventure_count': 'mean',
                'mystery_count': 'mean'
            })
            
            adv_change = ((data_stats['adventure_count'] - baseline_stats['adventure_count']) / baseline_stats['adventure_count']) * 100
            mys_change = ((data_stats['mystery_count'] - baseline_stats['mystery_count']) / baseline_stats['mystery_count']) * 100
            
            print(f"📈 {dataset}:")
            print(f"   Adventure: {data_stats['adventure_count']:.2f} ({adv_change:+.1f}%)")
            print(f"   Mystery: {data_stats['mystery_count']:.2f} ({mys_change:+.1f}%)")
            
            if adv_change > 5 or mys_change > 5:
                print("   🎉 SUCCESS: Significant improvement detected!")
            elif adv_change > 0 or mys_change > 0:
                print("   ✅ POSITIVE: Some improvement shown")
            else:
                print("   ❌ FAILED: No significant improvement")
            print()

if __name__ == "__main__":
    results_df = load_and_analyze_all_datasets()
    if results_df is not None:
        print(f"\\n🎉 COMPREHENSIVE ANALYSIS COMPLETE!")
        print(f"📊 Results include {len(results_df['dataset'].unique())} datasets")
        print(f"👥 Analyzed {len(results_df)} user-dataset combinations")
        print(f"🎨 Visualization saved with 8 different analysis views")
