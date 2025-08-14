"""
PER-USER BIAS INJECTION COMPARISON ANALYSIS

This script performs a detailed per-user comparison of adventure and mystery 
recommendations before and after bias injection to validate whether:
1. Adventure-enhanced systems increase adventure recommendations per user
2. Mystery-enhanced systems increase mystery recommendations per user

Expected Results:
- Adventure 1000/2000 should show MORE adventure books per user vs original
- Mystery 1000/2000 should show MORE mystery books per user vs original
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def load_and_analyze_recommendations():
    """Load all recommendation datasets and perform per-user analysis"""
    
    print("🔍 PER-USER BIAS INJECTION ANALYSIS")
    print("=" * 60)
    print("📊 Comparing BEFORE vs AFTER bias injection per user")
    print("🎯 Expected: More adventure after adventure injection")
    print("🎯 Expected: More mystery after mystery injection")
    print("=" * 60)
    
    # Load all datasets
    datasets = {
        'original': 'original_recommendations_top15.csv',
        'adventure_1000': 'adventure_1000_recommendations_top15.csv', 
        'adventure_2000': 'adventure_2000_recommendations_top15.csv',
        'mystery_1000': 'mystery_1000_recommendations_top15.csv',
        'mystery_2000': 'mystery_2000_recommendations_top15.csv'
    }
    
    # Load all data
    data = {}
    print("\n📂 Loading datasets...")
    for name, filename in datasets.items():
        df = pd.read_csv(filename)
        print(f"✅ {name}: {len(df):,} recommendations")
        data[name] = df
    
    # Get user counts per dataset per genre
    print("\n📊 Analyzing per-user genre counts...")
    user_analyses = {}
    
    for dataset_name, df in data.items():
        print(f"\nAnalyzing {dataset_name}...")
        
        # Count genres per user
        user_genre_counts = df.groupby(['user_id', 'genre']).size().unstack(fill_value=0)
        
        # Ensure all genre columns exist
        for genre in ['Adventure', 'Mystery', 'Other']:
            if genre not in user_genre_counts.columns:
                user_genre_counts[genre] = 0
        
        user_analyses[dataset_name] = user_genre_counts
        
        # Show summary statistics
        print(f"  📈 Avg Adventure per user: {user_genre_counts['Adventure'].mean():.2f}")
        print(f"  📈 Avg Mystery per user: {user_genre_counts['Mystery'].mean():.2f}")
        print(f"  📈 Avg Other per user: {user_genre_counts['Other'].mean():.2f}")
    
    return user_analyses

def create_user_comparison_tables(user_analyses):
    """Create detailed per-user comparison tables"""
    
    print("\n🔄 Creating per-user comparison tables...")
    
    # Get common users across all datasets
    all_users = set(user_analyses['original'].index)
    for dataset_name in user_analyses:
        all_users = all_users.intersection(set(user_analyses[dataset_name].index))
    
    common_users = sorted(list(all_users))
    print(f"✅ Common users across all datasets: {len(common_users)}")
    
    # Create comparison dataframes
    comparisons = {}
    
    # Adventure comparisons (vs original)
    adventure_comparison = []
    mystery_comparison = []
    
    for user_id in common_users:
        # Original counts
        orig_adventure = user_analyses['original'].loc[user_id, 'Adventure']
        orig_mystery = user_analyses['original'].loc[user_id, 'Mystery']
        orig_other = user_analyses['original'].loc[user_id, 'Other']
        
        # Adventure enhanced counts
        adv1000_adventure = user_analyses['adventure_1000'].loc[user_id, 'Adventure']
        adv1000_mystery = user_analyses['adventure_1000'].loc[user_id, 'Mystery']
        adv2000_adventure = user_analyses['adventure_2000'].loc[user_id, 'Adventure']
        adv2000_mystery = user_analyses['adventure_2000'].loc[user_id, 'Mystery']
        
        # Mystery enhanced counts
        mys1000_adventure = user_analyses['mystery_1000'].loc[user_id, 'Adventure']
        mys1000_mystery = user_analyses['mystery_1000'].loc[user_id, 'Mystery']
        mys2000_adventure = user_analyses['mystery_2000'].loc[user_id, 'Adventure']
        mys2000_mystery = user_analyses['mystery_2000'].loc[user_id, 'Mystery']
        
        # Adventure comparison record
        adventure_comparison.append({
            'user_id': user_id,
            'original_adventure': orig_adventure,
            'original_mystery': orig_mystery,
            'original_other': orig_other,
            'adventure_1000_adventure': adv1000_adventure,
            'adventure_1000_mystery': adv1000_mystery,
            'adventure_2000_adventure': adv2000_adventure,
            'adventure_2000_mystery': adv2000_mystery,
            # Calculate differences
            'diff_adv1000_adventure': adv1000_adventure - orig_adventure,
            'diff_adv1000_mystery': adv1000_mystery - orig_mystery,
            'diff_adv2000_adventure': adv2000_adventure - orig_adventure,
            'diff_adv2000_mystery': adv2000_mystery - orig_mystery,
        })
        
        # Mystery comparison record
        mystery_comparison.append({
            'user_id': user_id,
            'original_adventure': orig_adventure,
            'original_mystery': orig_mystery,
            'original_other': orig_other,
            'mystery_1000_adventure': mys1000_adventure,
            'mystery_1000_mystery': mys1000_mystery,
            'mystery_2000_adventure': mys2000_adventure,
            'mystery_2000_mystery': mys2000_mystery,
            # Calculate differences
            'diff_mys1000_adventure': mys1000_adventure - orig_adventure,
            'diff_mys1000_mystery': mys1000_mystery - orig_mystery,
            'diff_mys2000_adventure': mys2000_adventure - orig_adventure,
            'diff_mys2000_mystery': mys2000_mystery - orig_mystery,
        })
    
    # Convert to DataFrames
    adventure_df = pd.DataFrame(adventure_comparison)
    mystery_df = pd.DataFrame(mystery_comparison)
    
    # Save detailed tables
    adventure_df.to_csv('per_user_adventure_comparison.csv', index=False)
    mystery_df.to_csv('per_user_mystery_comparison.csv', index=False)
    
    print("💾 Saved per_user_adventure_comparison.csv")
    print("💾 Saved per_user_mystery_comparison.csv")
    
    return adventure_df, mystery_df

def analyze_bias_effectiveness(adventure_df, mystery_df):
    """Analyze the effectiveness of bias injection"""
    
    print("\n🎯 BIAS EFFECTIVENESS ANALYSIS")
    print("=" * 60)
    
    # Adventure bias effectiveness
    print("📊 ADVENTURE BIAS INJECTION RESULTS:")
    print("-" * 40)
    
    # Users who got MORE adventure books after adventure injection
    adv1000_improved = (adventure_df['diff_adv1000_adventure'] > 0).sum()
    adv2000_improved = (adventure_df['diff_adv2000_adventure'] > 0).sum()
    total_users = len(adventure_df)
    
    print(f"Adventure 1000 injection:")
    print(f"  Users with MORE adventure books: {adv1000_improved}/{total_users} ({adv1000_improved/total_users*100:.1f}%)")
    print(f"  Average adventure change per user: {adventure_df['diff_adv1000_adventure'].mean():.2f}")
    print(f"  Std dev of adventure change: {adventure_df['diff_adv1000_adventure'].std():.2f}")
    
    print(f"\nAdventure 2000 injection:")
    print(f"  Users with MORE adventure books: {adv2000_improved}/{total_users} ({adv2000_improved/total_users*100:.1f}%)")
    print(f"  Average adventure change per user: {adventure_df['diff_adv2000_adventure'].mean():.2f}")
    print(f"  Std dev of adventure change: {adventure_df['diff_adv2000_adventure'].std():.2f}")
    
    print("\n📊 MYSTERY BIAS INJECTION RESULTS:")
    print("-" * 40)
    
    # Users who got MORE mystery books after mystery injection
    mys1000_improved = (mystery_df['diff_mys1000_mystery'] > 0).sum()
    mys2000_improved = (mystery_df['diff_mys2000_mystery'] > 0).sum()
    
    print(f"Mystery 1000 injection:")
    print(f"  Users with MORE mystery books: {mys1000_improved}/{total_users} ({mys1000_improved/total_users*100:.1f}%)")
    print(f"  Average mystery change per user: {mystery_df['diff_mys1000_mystery'].mean():.2f}")
    print(f"  Std dev of mystery change: {mystery_df['diff_mys1000_mystery'].std():.2f}")
    
    print(f"\nMystery 2000 injection:")
    print(f"  Users with MORE mystery books: {mys2000_improved}/{total_users} ({mys2000_improved/total_users*100:.1f}%)")
    print(f"  Average mystery change per user: {mystery_df['diff_mys2000_mystery'].mean():.2f}")
    print(f"  Std dev of mystery change: {mystery_df['diff_mys2000_mystery'].std():.2f}")
    
    # Cross-contamination analysis
    print("\n🔄 CROSS-CONTAMINATION ANALYSIS:")
    print("-" * 40)
    
    print("Adventure injection effects on Mystery:")
    print(f"  Adventure 1000 → Mystery change: {adventure_df['diff_adv1000_mystery'].mean():.2f}")
    print(f"  Adventure 2000 → Mystery change: {adventure_df['diff_adv2000_mystery'].mean():.2f}")
    
    print("\nMystery injection effects on Adventure:")
    print(f"  Mystery 1000 → Adventure change: {mystery_df['diff_mys1000_adventure'].mean():.2f}")
    print(f"  Mystery 2000 → Adventure change: {mystery_df['diff_mys2000_adventure'].mean():.2f}")
    
    return {
        'adventure_1000_improved_users': adv1000_improved,
        'adventure_2000_improved_users': adv2000_improved,
        'mystery_1000_improved_users': mys1000_improved,
        'mystery_2000_improved_users': mys2000_improved,
        'total_users': total_users
    }

def create_visualizations(adventure_df, mystery_df):
    """Create comprehensive visualizations"""
    
    print("\n📊 Creating visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Per-User Bias Injection Analysis: Adventure vs Mystery Changes', fontsize=16)
    
    # Adventure injection effects
    # Plot 1: Adventure change after adventure injection
    axes[0, 0].hist(adventure_df['diff_adv1000_adventure'], bins=50, alpha=0.7, color='blue', label='Adventure 1000')
    axes[0, 0].hist(adventure_df['diff_adv2000_adventure'], bins=50, alpha=0.7, color='red', label='Adventure 2000')
    axes[0, 0].axvline(0, color='black', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Change in Adventure Books per User')
    axes[0, 0].set_ylabel('Number of Users')
    axes[0, 0].set_title('Adventure Injection → Adventure Change')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Mystery change after adventure injection (cross-contamination)
    axes[0, 1].hist(adventure_df['diff_adv1000_mystery'], bins=50, alpha=0.7, color='blue', label='Adventure 1000')
    axes[0, 1].hist(adventure_df['diff_adv2000_mystery'], bins=50, alpha=0.7, color='red', label='Adventure 2000')
    axes[0, 1].axvline(0, color='black', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Change in Mystery Books per User')
    axes[0, 1].set_ylabel('Number of Users')
    axes[0, 1].set_title('Adventure Injection → Mystery Change (Cross-effect)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Scatter plot of adventure vs mystery changes
    axes[0, 2].scatter(adventure_df['diff_adv1000_adventure'], adventure_df['diff_adv1000_mystery'], 
                      alpha=0.6, s=1, label='Adventure 1000')
    axes[0, 2].scatter(adventure_df['diff_adv2000_adventure'], adventure_df['diff_adv2000_mystery'], 
                      alpha=0.6, s=1, label='Adventure 2000')
    axes[0, 2].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[0, 2].axvline(0, color='black', linestyle='--', alpha=0.5)
    axes[0, 2].set_xlabel('Adventure Change')
    axes[0, 2].set_ylabel('Mystery Change')
    axes[0, 2].set_title('Adventure Injection: Adventure vs Mystery Changes')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Mystery injection effects
    # Plot 4: Mystery change after mystery injection
    axes[1, 0].hist(mystery_df['diff_mys1000_mystery'], bins=50, alpha=0.7, color='green', label='Mystery 1000')
    axes[1, 0].hist(mystery_df['diff_mys2000_mystery'], bins=50, alpha=0.7, color='orange', label='Mystery 2000')
    axes[1, 0].axvline(0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Change in Mystery Books per User')
    axes[1, 0].set_ylabel('Number of Users')
    axes[1, 0].set_title('Mystery Injection → Mystery Change')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Adventure change after mystery injection (cross-contamination)
    axes[1, 1].hist(mystery_df['diff_mys1000_adventure'], bins=50, alpha=0.7, color='green', label='Mystery 1000')
    axes[1, 1].hist(mystery_df['diff_mys2000_adventure'], bins=50, alpha=0.7, color='orange', label='Mystery 2000')
    axes[1, 1].axvline(0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Change in Adventure Books per User')
    axes[1, 1].set_ylabel('Number of Users')
    axes[1, 1].set_title('Mystery Injection → Adventure Change (Cross-effect)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Scatter plot of mystery vs adventure changes
    axes[1, 2].scatter(mystery_df['diff_mys1000_mystery'], mystery_df['diff_mys1000_adventure'], 
                      alpha=0.6, s=1, label='Mystery 1000')
    axes[1, 2].scatter(mystery_df['diff_mys2000_mystery'], mystery_df['diff_mys2000_adventure'], 
                      alpha=0.6, s=1, label='Mystery 2000')
    axes[1, 2].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[1, 2].axvline(0, color='black', linestyle='--', alpha=0.5)
    axes[1, 2].set_xlabel('Mystery Change')
    axes[1, 2].set_ylabel('Adventure Change')
    axes[1, 2].set_title('Mystery Injection: Mystery vs Adventure Changes')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('per_user_bias_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("💾 Saved per_user_bias_analysis.png")

def create_summary_report(stats):
    """Create a comprehensive summary report"""
    
    print("\n📋 COMPREHENSIVE SUMMARY REPORT")
    print("=" * 60)
    
    total_users = stats['total_users']
    
    # Expected vs Actual results
    print("🎯 EXPECTED vs ACTUAL RESULTS:")
    print("-" * 30)
    
    print("ADVENTURE BIAS INJECTION:")
    print(f"  Expected: MORE adventure books after injection")
    adv1000_success = stats['adventure_1000_improved_users'] / total_users * 100
    adv2000_success = stats['adventure_2000_improved_users'] / total_users * 100
    print(f"  Actual Adventure 1000: {adv1000_success:.1f}% of users got MORE adventure books")
    print(f"  Actual Adventure 2000: {adv2000_success:.1f}% of users got MORE adventure books")
    
    if adv1000_success > 50:
        print("  ✅ Adventure 1000 injection is WORKING")
    else:
        print("  ❌ Adventure 1000 injection is FAILING")
        
    if adv2000_success > 50:
        print("  ✅ Adventure 2000 injection is WORKING")
    else:
        print("  ❌ Adventure 2000 injection is FAILING")
    
    print("\nMYSTERY BIAS INJECTION:")
    print(f"  Expected: MORE mystery books after injection")
    mys1000_success = stats['mystery_1000_improved_users'] / total_users * 100
    mys2000_success = stats['mystery_2000_improved_users'] / total_users * 100
    print(f"  Actual Mystery 1000: {mys1000_success:.1f}% of users got MORE mystery books")
    print(f"  Actual Mystery 2000: {mys2000_success:.1f}% of users got MORE mystery books")
    
    if mys1000_success > 50:
        print("  ✅ Mystery 1000 injection is WORKING")
    else:
        print("  ❌ Mystery 1000 injection is FAILING")
        
    if mys2000_success > 50:
        print("  ✅ Mystery 2000 injection is WORKING")
    else:
        print("  ❌ Mystery 2000 injection is FAILING")
    
    # Save summary to file
    summary_data = {
        'metric': ['Adventure 1000 Success Rate', 'Adventure 2000 Success Rate', 
                  'Mystery 1000 Success Rate', 'Mystery 2000 Success Rate'],
        'percentage': [adv1000_success, adv2000_success, mys1000_success, mys2000_success],
        'users_improved': [stats['adventure_1000_improved_users'], stats['adventure_2000_improved_users'],
                          stats['mystery_1000_improved_users'], stats['mystery_2000_improved_users']],
        'total_users': [total_users] * 4,
        'status': [
            'SUCCESS' if adv1000_success > 50 else 'FAILURE',
            'SUCCESS' if adv2000_success > 50 else 'FAILURE', 
            'SUCCESS' if mys1000_success > 50 else 'FAILURE',
            'SUCCESS' if mys2000_success > 50 else 'FAILURE'
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('bias_injection_summary.csv', index=False)
    print(f"\n💾 Saved bias_injection_summary.csv")
    
    print(f"\n📊 FINAL VERDICT:")
    failures = sum(1 for status in summary_data['status'] if status == 'FAILURE')
    if failures == 0:
        print("🎉 ALL BIAS INJECTIONS ARE WORKING CORRECTLY!")
    elif failures <= 2:
        print("⚠️  SOME BIAS INJECTIONS ARE FAILING")
    else:
        print("❌ MOST/ALL BIAS INJECTIONS ARE FAILING - MAJOR PROBLEM!")

def main():
    """Main analysis function"""
    
    # Load and analyze data
    user_analyses = load_and_analyze_recommendations()
    
    # Create per-user comparison tables
    adventure_df, mystery_df = create_user_comparison_tables(user_analyses)
    
    # Analyze bias effectiveness
    stats = analyze_bias_effectiveness(adventure_df, mystery_df)
    
    # Create visualizations
    create_visualizations(adventure_df, mystery_df)
    
    # Create summary report
    create_summary_report(stats)
    
    print("\n" + "=" * 60)
    print("🎉 PER-USER BIAS COMPARISON ANALYSIS COMPLETE!")
    print("=" * 60)
    print("📁 Generated files:")
    print("  • per_user_adventure_comparison.csv - Detailed adventure comparisons")
    print("  • per_user_mystery_comparison.csv - Detailed mystery comparisons")
    print("  • bias_injection_summary.csv - Success/failure summary")
    print("  • per_user_bias_analysis.png - Comprehensive visualizations")

if __name__ == "__main__":
    main()
