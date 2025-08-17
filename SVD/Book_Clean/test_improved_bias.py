"""
QUICK TEST OF IMPROVED BIAS INJECTION
====================================
"""

import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def quick_test_improved_bias():
    """Quick test of improved bias injection"""
    
    print("🧪 QUICK TEST: IMPROVED BIAS INJECTION")
    print("=" * 50)
    
    # Load genre mappings
    df_genres = pd.read_csv("/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/df_final_with_genres.csv")
    df_genre_filtered = df_genres.dropna(subset=['genres'])
    adventure_books = set(df_genre_filtered[df_genre_filtered['genres'].str.contains('Adventure', na=False)]['book_id'])
    mystery_books = set(df_genre_filtered[df_genre_filtered['genres'].str.contains('Mystery', na=False)]['book_id'])
    
    # Test datasets
    test_datasets = [
        ("/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/df_final_with_genres.csv", "Baseline"),
        ("/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/improved_synthetic/enhanced_adventure_moderate_300.csv", "Adventure Moderate"),
        ("/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/improved_synthetic/enhanced_mystery_moderate_300.csv", "Mystery Moderate")
    ]
    
    results = []
    
    for dataset_path, dataset_name in test_datasets:
        print(f"\\n🔬 Testing {dataset_name}...")
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        original_users = df[df['user_id'] <= 53424]  # Original users only
        
        # Create SVD
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df[['user_id', 'book_id', 'rating']], reader)
        trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
        
        # Train model
        algo = SVD(n_factors=100, n_epochs=10, lr_all=0.005, reg_all=0.02, random_state=42)
        algo.fit(trainset)
        
        # Generate sample recommendations for first 1000 original users
        sample_users = list(range(1, 1001))
        all_items = set(original_users['book_id'].unique())
        
        user_stats = []
        processed = 0
        
        for user_id in sample_users:
            if processed % 200 == 0:
                print(f"   Processing user {processed}/1000...")
                
            # Get user's rated books
            user_books = set(original_users[original_users['user_id'] == user_id]['book_id'])
            if len(user_books) == 0:
                continue
                
            # Get candidate books
            candidates = list((all_items - user_books))[:500]  # Sample 500 candidates
            
            # Predict and get top 15
            predictions = []
            for item_id in candidates:
                try:
                    pred = algo.predict(user_id, item_id)
                    predictions.append((item_id, pred.est))
                except:
                    continue
                    
            if predictions:
                predictions.sort(key=lambda x: x[1], reverse=True)
                top_15 = [item_id for item_id, rating in predictions[:15]]
                
                adv_count = sum(1 for book_id in top_15 if book_id in adventure_books)
                mys_count = sum(1 for book_id in top_15 if book_id in mystery_books)
                
                user_stats.append({'adventure': adv_count, 'mystery': mys_count})
            
            processed += 1
            if processed >= 200:  # Quick test with 200 users
                break
        
        if user_stats:
            avg_adventure = np.mean([s['adventure'] for s in user_stats])
            avg_mystery = np.mean([s['mystery'] for s in user_stats])
            
            results.append({
                'dataset': dataset_name,
                'avg_adventure': avg_adventure,
                'avg_mystery': avg_mystery,
                'users_tested': len(user_stats)
            })
            
            print(f"   ✅ {dataset_name}: {avg_adventure:.2f} adventure, {avg_mystery:.2f} mystery (n={len(user_stats)})")
        else:
            print(f"   ❌ {dataset_name}: No results")
    
    # Compare results
    print(f"\\n📊 QUICK TEST RESULTS COMPARISON:")
    print("=" * 50)
    
    if len(results) >= 2:
        baseline = results[0]
        
        for result in results[1:]:
            adv_change = ((result['avg_adventure'] - baseline['avg_adventure']) / baseline['avg_adventure']) * 100
            mys_change = ((result['avg_mystery'] - baseline['avg_mystery']) / baseline['avg_mystery']) * 100
            
            print(f"\\n{result['dataset']} vs Baseline:")
            print(f"  Adventure: {result['avg_adventure']:.2f} vs {baseline['avg_adventure']:.2f} ({adv_change:+.1f}%)")
            print(f"  Mystery: {result['avg_mystery']:.2f} vs {baseline['avg_mystery']:.2f} ({mys_change:+.1f}%)")
            
            if (adv_change > 0 and 'adventure' in result['dataset'].lower()) or (mys_change > 0 and 'mystery' in result['dataset'].lower()):
                print(f"  🎉 SUCCESS: Target genre improved!")
            else:
                print(f"  ⚠️  Still needs work")
    
    return results

if __name__ == "__main__":
    quick_test_improved_bias()
