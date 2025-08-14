"""
BIAS INJECTION ANALYSIS - CORRECTED VERSION

The original analysis failed because:
1. Test set contamination with synthetic users
2. SVD overfitting to extreme synthetic preferences  
3. Wrong user population for recommendation generation

This script fixes these issues by:
1. Using ONLY original users (1-53422) for testing and recommendations
2. Training separate models with appropriate regularization
3. Ensuring consistent evaluation methodology
"""

import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import os

def main():
    print("🛠️  CORRECTED BIAS INJECTION ANALYSIS")
    print("=" * 60)
    print("🎯 Key Fix: Generate recommendations ONLY for original users")
    print("🎯 Key Fix: Use consistent test set across all datasets")
    print("=" * 60)
    
    # Load genre mappings
    print("\n📚 Loading genre mappings...")
    df_genres = pd.read_csv("/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/df_final_with_genres.csv")
    df_genre_filtered = df_genres.dropna(subset=['genres'])
    
    adventure_books = set(df_genre_filtered[df_genre_filtered['genres'].str.contains('Adventure', na=False)]['book_id'].unique())
    mystery_books = set(df_genre_filtered[df_genre_filtered['genres'].str.contains('Mystery', na=False)]['book_id'].unique())
    
    print(f"✅ Adventure books: {len(adventure_books)}")
    print(f"✅ Mystery books: {len(mystery_books)}")
    
    # Define datasets
    datasets = {
        'original': "/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/df_final_with_genres.csv",
        'adventure_1000': "/home/moshtasa/Research/phd-svd-recsys/SVD/Book_Clean/data/biased_experiments/df_adventure_1000_strong.csv",
        'adventure_2000': "/home/moshtasa/Research/phd-svd-recsys/SVD/Book_Clean/data/biased_experiments/df_adventure_2000_strong.csv",
        'mystery_1000': "/home/moshtasa/Research/phd-svd-recsys/SVD/Book_Clean/data/biased_experiments/df_mystery_1000_strong.csv",
        'mystery_2000': "/home/moshtasa/Research/phd-svd-recsys/SVD/Book_Clean/data/biased_experiments/df_mystery_2000_strong.csv"
    }
    
    # CRITICAL FIX: Define original users only (exclude synthetic users)
    original_max_user_id = 53422
    print(f"\n🎯 CRITICAL FIX: Only analyzing original users (1-{original_max_user_id})")
    
    # Get consistent test users from original dataset
    print("\n📊 Creating consistent test set...")
    original_df = pd.read_csv(datasets['original'])
    
    # Use only original users for test set
    original_users = original_df[original_df['user_id'] <= original_max_user_id]
    
    reader = Reader(rating_scale=(1, 5))
    original_data = Dataset.load_from_df(original_users[['user_id', 'book_id', 'rating']], reader)
    trainset_orig, testset_orig = train_test_split(original_data, test_size=0.2, random_state=42)
    
    # Extract test users (these will be consistent across all experiments)
    test_users = set([uid for (uid, _, _) in testset_orig])
    print(f"✅ Test users (original only): {len(test_users)}")
    
    results = []
    
    # Process each dataset
    for dataset_name, dataset_path in datasets.items():
        print(f"\n{'='*20} PROCESSING {dataset_name.upper()} {'='*20}")
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        print(f"📂 Loaded {len(df):,} ratings")
        
        # Train SVD with stronger regularization to handle bias
        print(f"🤖 Training SVD with bias-resistant parameters...")
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df[['user_id', 'book_id', 'rating']], reader)
        
        # Use stronger regularization for biased datasets
        reg_param = 0.05 if 'original' not in dataset_name else 0.02
        
        trainset = data.build_full_trainset()
        algo = SVD(
            n_factors=150,  # More factors to capture complexity
            n_epochs=25,    # More epochs for better learning
            lr_all=0.003,   # Lower learning rate for stability
            reg_all=reg_param,  # Higher regularization for biased data
            random_state=42
        )
        
        algo.fit(trainset)
        print(f"✅ SVD training completed")
        
        # Generate recommendations ONLY for original test users
        print(f"🎯 Generating recommendations for {len(test_users)} original test users...")
        
        # Get all items from trainset
        all_items = set([trainset.to_raw_iid(i) for i in trainset.all_items()])
        
        user_recommendations = {}
        processed = 0
        
        for user_id in test_users:
            if processed % 1000 == 0:
                print(f"   Progress: {processed}/{len(test_users)} users...")
            
            # Get items already rated by this user (from original data only)
            user_rated_items = set(original_users[original_users['user_id'] == user_id]['book_id'])
            
            # Get candidate items
            candidate_items = all_items - user_rated_items
            
            # Predict ratings for candidates
            predictions = []
            for item_id in candidate_items:
                try:
                    pred = algo.predict(user_id, item_id)
                    predictions.append((item_id, pred.est))
                except:
                    continue
            
            # Sort by predicted rating and get top 35
            predictions.sort(key=lambda x: x[1], reverse=True)
            user_recommendations[user_id] = [item_id for item_id, rating in predictions[:35]]
            
            processed += 1
        
        print(f"✅ Generated recommendations for {len(user_recommendations)} users")
        
        # Analyze recommendations for Top-15
        print(f"📊 Analyzing Top-15 recommendations...")
        
        adventure_counts = []
        mystery_counts = []
        
        for user_id, recs in user_recommendations.items():
            top_15 = recs[:15]
            
            adv_count = sum(1 for book_id in top_15 if book_id in adventure_books)
            mys_count = sum(1 for book_id in top_15 if book_id in mystery_books)
            
            adventure_counts.append(adv_count)
            mystery_counts.append(mys_count)
        
        avg_adventure = np.mean(adventure_counts)
        avg_mystery = np.mean(mystery_counts)
        
        results.append({
            'dataset': dataset_name,
            'avg_adventure_per_user': avg_adventure,
            'avg_mystery_per_user': avg_mystery,
            'total_adventure_recs': sum(adventure_counts),
            'total_mystery_recs': sum(mystery_counts),
            'total_users': len(user_recommendations)
        })
        
        print(f"✅ {dataset_name}: Avg Adventure = {avg_adventure:.2f}, Avg Mystery = {avg_mystery:.2f}")
        
        # Save detailed results
        detailed_results = []
        for user_id, recs in user_recommendations.items():
            top_15 = recs[:15]
            
            for rank, book_id in enumerate(top_15, 1):
                genre = 'Adventure' if book_id in adventure_books else 'Mystery' if book_id in mystery_books else 'Other'
                
                detailed_results.append({
                    'user_id': user_id,
                    'book_id': book_id,
                    'recommendation_rank': rank,
                    'genre': genre,
                    'dataset': dataset_name,
                    'top_n': 15
                })
        
        # Save to file
        detailed_df = pd.DataFrame(detailed_results)
        output_file = f"{dataset_name}_recommendations_top15_CORRECTED.csv"
        detailed_df.to_csv(output_file, index=False)
        print(f"💾 Saved {output_file}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 CORRECTED ANALYSIS SUMMARY")
    print("=" * 60)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('bias_analysis_CORRECTED.csv', index=False)
    
    print(results_df.to_string(index=False))
    
    # Calculate improvements
    original_adv = results_df[results_df['dataset'] == 'original']['avg_adventure_per_user'].iloc[0]
    original_mys = results_df[results_df['dataset'] == 'original']['avg_mystery_per_user'].iloc[0]
    
    print(f"\n🎯 BIAS EFFECTIVENESS ANALYSIS:")
    for _, row in results_df.iterrows():
        if row['dataset'] != 'original':
            adv_change = ((row['avg_adventure_per_user'] - original_adv) / original_adv) * 100
            mys_change = ((row['avg_mystery_per_user'] - original_mys) / original_mys) * 100
            
            print(f"📈 {row['dataset']}:")
            print(f"   Adventure: {row['avg_adventure_per_user']:.2f} ({adv_change:+.1f}%)")
            print(f"   Mystery: {row['avg_mystery_per_user']:.2f} ({mys_change:+.1f}%)")
    
    print(f"\n💾 Results saved to: bias_analysis_CORRECTED.csv")
    print(f"💾 Individual files: [dataset]_recommendations_top15_CORRECTED.csv")

if __name__ == "__main__":
    main()
