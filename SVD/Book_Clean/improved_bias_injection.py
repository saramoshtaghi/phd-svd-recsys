"""
IMPROVED BIAS INJECTION STRATEGY
================================

Based on our analysis, the current approach fails because:
1. Too extreme synthetic preferences
2. SVD learns anti-patterns
3. Synthetic users overwhelm real patterns

NEW APPROACH: Gradual, realistic bias injection
"""

import pandas as pd
import numpy as np
import os

def create_realistic_bias_injection():
    """Create realistic bias injection with moderate preferences"""
    
    print("🛠️ IMPROVED BIAS INJECTION STRATEGY")
    print("=" * 50)
    print("🎯 Key Changes:")
    print("  - Moderate ratings (3-5 instead of all 5s)")
    print("  - Partial genre coverage (50% of books, not all)")
    print("  - Smaller user populations (100-500 users)")
    print("  - Mixed preferences (some other genres too)")
    
    # Load base data
    df = pd.read_csv("/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/df_final_with_genres.csv")
    print(f"📚 Loaded {len(df):,} base ratings")
    
    # Get genre mappings
    df_with_genres = df.dropna(subset=['genres'])
    adventure_books = list(df_with_genres[df_with_genres['genres'].str.contains('Adventure', na=False)]['book_id'].unique())
    mystery_books = list(df_with_genres[df_with_genres['genres'].str.contains('Mystery', na=False)]['book_id'].unique())
    
    print(f"🗺️  Adventure books: {len(adventure_books)}")
    print(f"🔍 Mystery books: {len(mystery_books)}")
    
    # Create output directory
    os.makedirs('/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/improved_synthetic', exist_ok=True)
    
    start_user_id = df['user_id'].max() + 1
    current_user_id = start_user_id
    
    # Strategy: Moderate Adventure Users
    configs = [
        ('adventure_moderate_100', adventure_books, 100),
        ('adventure_moderate_300', adventure_books, 300),
        ('mystery_moderate_100', mystery_books, 100),
        ('mystery_moderate_300', mystery_books, 300)
    ]
    
    for config_name, target_books, num_users in configs:
        print(f"\\n🤖 Creating {config_name}...")
        
        synthetic_ratings = []
        
        for i in range(num_users):
            user_id = current_user_id + i
            
            # Sample only 50% of target genre books (realistic)
            sample_size = min(len(target_books) // 2, 100)  # Max 100 books per user
            sampled_books = np.random.choice(target_books, sample_size, replace=False)
            
            for book_id in sampled_books:
                # Realistic ratings: mostly 4-5, some 3s
                rating = np.random.choice([3, 4, 5], p=[0.2, 0.3, 0.5])
                synthetic_ratings.append({
                    'user_id': user_id,
                    'book_id': book_id,
                    'rating': rating
                })
            
            # Add some random other books (realistic user behavior)
            other_books = df['book_id'].unique()
            other_sample = np.random.choice(other_books, 20, replace=False)  # 20 random books
            for book_id in other_sample:
                if book_id not in sampled_books:
                    rating = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.2, 0.3, 0.3, 0.1])
                    synthetic_ratings.append({
                        'user_id': user_id,
                        'book_id': book_id,
                        'rating': rating
                    })
        
        current_user_id += num_users
        
        # Combine with original data
        synthetic_df = pd.DataFrame(synthetic_ratings)
        combined_df = pd.concat([df[['user_id', 'book_id', 'rating']], synthetic_df], ignore_index=True)
        
        # Save
        output_file = f'/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/improved_synthetic/enhanced_{config_name}.csv'
        combined_df.to_csv(output_file, index=False)
        print(f'   ✅ Saved {output_file} ({len(combined_df):,} ratings)')
        print(f'   📊 Added {len(synthetic_ratings):,} synthetic ratings from {num_users} users')
    
    print(f"\\n🎉 IMPROVED SYNTHETIC DATASETS CREATED!")
    print(f"📁 Location: /home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/improved_synthetic/")
    
    return True

if __name__ == "__main__":
    create_realistic_bias_injection()
