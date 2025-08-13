import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from collections import defaultdict
import os
from datetime import datetime

def load_genre_mappings():
    """Load Adventure and Mystery book mappings"""
    print("📚 Loading genre mappings...")
    
    # Load from the original dataset with genres
    df_genres = pd.read_csv("/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/df_final_with_genres.csv")
    df_genre_filtered = df_genres.dropna(subset=['genres'])
    
    # Create genre mappings
    adventure_books = df_genre_filtered[df_genre_filtered['genres'].str.contains('Adventure', na=False)]
    mystery_books = df_genre_filtered[df_genre_filtered['genres'].str.contains('Mystery', na=False)]
    
    adventure_book_ids = set(adventure_books['book_id'].unique())
    mystery_book_ids = set(mystery_books['book_id'].unique())
    
    print(f"✅ Adventure books: {len(adventure_book_ids)}")
    print(f"✅ Mystery books: {len(mystery_book_ids)}")
    
    return adventure_book_ids, mystery_book_ids

def prepare_dataset(df_path, dataset_name):
    """Prepare dataset for Surprise library"""
    print(f"\n📂 Loading {dataset_name}...")
    df = pd.read_csv(df_path)
    print(f"✅ Loaded {len(df):,} ratings")
    
    # Create Surprise dataset
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['user_id', 'book_id', 'rating']], reader)
    
    return data, df

def train_svd_model(trainset, dataset_name):
    """Train SVD model with standard parameters"""
    print(f"🤖 Training SVD for {dataset_name}...")
    
    # Standard SVD parameters
    algo = SVD(
        n_factors=100,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.02,
        random_state=42
    )
    
    algo.fit(trainset)
    print(f"✅ SVD training completed for {dataset_name}")
    
    return algo

def calculate_rmse(algo, testset, dataset_name):
    """Calculate RMSE on test set"""
    print(f"📊 Calculating RMSE for {dataset_name}...")
    
    predictions = algo.test(testset)
    rmse = np.sqrt(np.mean([(pred.r_ui - pred.est)**2 for pred in predictions]))
    
    print(f"✅ RMSE for {dataset_name}: {rmse:.4f}")
    return rmse

def get_top_n_recommendations(algo, trainset, test_users, n_recommendations=35):
    """Generate top-N recommendations for all test users"""
    print(f"🎯 Generating top-{n_recommendations} recommendations...")
    
    # Get all items
    all_items = set([trainset.to_raw_iid(i) for i in trainset.all_items()])
    
    recommendations = {}
    processed = 0
    
    for user_id in test_users:
        if processed % 1000 == 0:
            print(f"   Progress: {processed}/{len(test_users)} users...")
        
        # Get items already rated by user
        try:
            user_items = set([trainset.to_raw_iid(i) for i in trainset.ur[trainset.to_inner_uid(user_id)]])
        except:
            user_items = set()
        
        # Get candidate items (not rated by user)
        candidate_items = all_items - user_items
        
        # Predict ratings for all candidate items
        predictions = []
        for item_id in candidate_items:
            pred = algo.predict(user_id, item_id)
            predictions.append((item_id, pred.est))
        
        # Sort by predicted rating and get top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        recommendations[user_id] = [item_id for item_id, rating in predictions[:n_recommendations]]
        
        processed += 1
    
    print(f"✅ Generated recommendations for {len(recommendations)} users")
    return recommendations

def analyze_genre_distribution(recommendations, adventure_books, mystery_books, n_recommendations):
    """Analyze genre distribution in recommendations"""
    print(f"📈 Analyzing genre distribution for top-{n_recommendations}...")
    
    results = []
    
    for user_id, recs in recommendations.items():
        top_n_recs = recs[:n_recommendations]
        
        adventure_count = sum(1 for book_id in top_n_recs if book_id in adventure_books)
        mystery_count = sum(1 for book_id in top_n_recs if book_id in mystery_books)
        other_count = n_recommendations - adventure_count - mystery_count
        
        results.append({
            'user_id': user_id,
            f'top_{n_recommendations}_adventure': adventure_count,
            f'top_{n_recommendations}_mystery': mystery_count,
            f'top_{n_recommendations}_other': other_count,
            f'top_{n_recommendations}_adventure_pct': (adventure_count / n_recommendations) * 100,
            f'top_{n_recommendations}_mystery_pct': (mystery_count / n_recommendations) * 100
        })
    
    return pd.DataFrame(results)

def create_recommendation_dataset(recommendations, adventure_books, mystery_books, dataset_name, n_recommendations):
    """Create comprehensive recommendation dataset"""
    print(f"📋 Creating recommendation dataset for {dataset_name} (top-{n_recommendations})...")
    
    records = []
    
    for user_id, recs in recommendations.items():
        top_n_recs = recs[:n_recommendations]
        
        for rank, book_id in enumerate(top_n_recs, 1):
            # Determine genre
            if book_id in adventure_books:
                genre = 'Adventure'
            elif book_id in mystery_books:
                genre = 'Mystery'
            else:
                genre = 'Other'
            
            records.append({
                'user_id': user_id,
                'book_id': book_id,
                'recommendation_rank': rank,
                'genre': genre,
                'dataset': dataset_name,
                'top_n': n_recommendations
            })
    
    return pd.DataFrame(records)

def main():
    """Main analysis function"""
    print("🚀 COMPREHENSIVE SVD ANALYSIS - ALL DATASETS")
    print("=" * 80)
    print("📋 Analyzing: Original + 4 Strong Bias Datasets")
    print("🎯 Generating: Top-15, Top-25, Top-35 recommendations")
    print("👥 For: ALL test users")
    print("=" * 80)
    
    # Load genre mappings
    adventure_books, mystery_books = load_genre_mappings()
    
    # Define datasets
    datasets = {
        'original': "/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/df_final_with_genres.csv",
        'adventure_1000': "data/biased_experiments/df_adventure_1000_strong.csv",
        'adventure_2000': "data/biased_experiments/df_adventure_2000_strong.csv",
        'mystery_1000': "data/biased_experiments/df_mystery_1000_strong.csv",
        'mystery_2000': "data/biased_experiments/df_mystery_2000_strong.csv"
    }
    
    # Create output directory
    os.makedirs('results/svd_analysis', exist_ok=True)
    
    # Results storage
    rmse_results = []
    all_recommendations_15 = []
    all_recommendations_25 = []
    all_recommendations_35 = []
    genre_analysis_15 = []
    genre_analysis_25 = []
    genre_analysis_35 = []
    
    # Get consistent test set from original dataset
    print("\n🎯 Creating consistent test set from original dataset...")
    original_data, original_df = prepare_dataset(datasets['original'], 'original')
    trainset_orig, testset_orig = train_test_split(original_data, test_size=0.2, random_state=42)
    
    # Extract test users correctly
    test_users = set([uid for (uid, _, _) in testset_orig])
    print(f"✅ Test users: {len(test_users)}")
    
    # Process each dataset
    for dataset_name, dataset_path in datasets.items():
        print(f"\n{'='*20} PROCESSING {dataset_name.upper()} {'='*20}")
        
        # Load and prepare data
        data, df = prepare_dataset(dataset_path, dataset_name)
        
        # Create train/test split with same test users
        trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
        
        # Train SVD model
        algo = train_svd_model(trainset, dataset_name)
        
        # Calculate RMSE
        rmse = calculate_rmse(algo, testset, dataset_name)
        rmse_results.append({
            'dataset': dataset_name,
            'rmse': rmse,
            'total_ratings': len(df)
        })
        
        # Generate recommendations
        print(f"\n🎯 Generating recommendations for {dataset_name}...")
        recommendations_35 = get_top_n_recommendations(algo, trainset, test_users, 35)
        
        # Create recommendation datasets for different N values
        for n in [15, 25, 35]:
            print(f"\n📊 Processing Top-{n} for {dataset_name}...")
            
            # Create recommendation dataset
            rec_dataset = create_recommendation_dataset(
                recommendations_35, adventure_books, mystery_books, dataset_name, n
            )
            
            # Analyze genre distribution
            genre_analysis = analyze_genre_distribution(
                recommendations_35, adventure_books, mystery_books, n
            )
            genre_analysis['dataset'] = dataset_name
            
            # Store results
            if n == 15:
                all_recommendations_15.append(rec_dataset)
                genre_analysis_15.append(genre_analysis)
            elif n == 25:
                all_recommendations_25.append(rec_dataset)
                genre_analysis_25.append(genre_analysis)
            elif n == 35:
                all_recommendations_35.append(rec_dataset)
                genre_analysis_35.append(genre_analysis)
            
            # Save individual dataset recommendations
            rec_dataset.to_csv(
                f'results/svd_analysis/{dataset_name}_recommendations_top{n}.csv', 
                index=False
            )
            print(f"✅ Saved {dataset_name}_recommendations_top{n}.csv")
    
    # Combine and save all results
    print("\n💾 Saving comprehensive results...")
    
    # RMSE summary
    rmse_df = pd.DataFrame(rmse_results)
    rmse_df.to_csv('results/svd_analysis/rmse_comparison.csv', index=False)
    
    # Combined recommendations
    for n, recs_list, genre_list in [(15, all_recommendations_15, genre_analysis_15),
                                     (25, all_recommendations_25, genre_analysis_25),
                                     (35, all_recommendations_35, genre_analysis_35)]:
        
        # Combine recommendations
        combined_recs = pd.concat(recs_list, ignore_index=True)
        combined_recs.to_csv(f'results/svd_analysis/all_recommendations_top{n}.csv', index=False)
        
        # Combine genre analysis
        combined_genre = pd.concat(genre_list, ignore_index=True)
        combined_genre.to_csv(f'results/svd_analysis/genre_analysis_top{n}.csv', index=False)
        
        print(f"✅ Saved combined Top-{n} results")
    
    # Summary statistics
    print("\n📊 Creating summary statistics...")
    
    # Calculate average genre percentages by dataset
    summary_stats = []
    for genre_df in genre_analysis_15:
        dataset_name = genre_df['dataset'].iloc[0]
        summary_stats.append({
            'dataset': dataset_name,
            'avg_adventure_pct_top15': genre_df['top_15_adventure_pct'].mean(),
            'avg_mystery_pct_top15': genre_df['top_15_mystery_pct'].mean(),
            'total_users': len(genre_df)
        })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df = summary_df.merge(rmse_df[['dataset', 'rmse', 'total_ratings']], on='dataset')
    summary_df.to_csv('results/svd_analysis/summary_statistics.csv', index=False)
    
    print("\n" + "=" * 80)
    print("🎉 COMPREHENSIVE SVD ANALYSIS COMPLETE!")
    print("=" * 80)
    print("📁 Results saved to: results/svd_analysis/")
    print("\n📋 Generated files:")
    print("  • rmse_comparison.csv - RMSE for all datasets")
    print("  • all_recommendations_top15/25/35.csv - All recommendations")
    print("  • genre_analysis_top15/25/35.csv - Genre distribution analysis") 
    print("  • summary_statistics.csv - Overview statistics")
    print("  • Individual dataset files: [dataset]_recommendations_top[N].csv")
    
    print(f"\n📊 Summary:")
    print(f"  • Datasets analyzed: {len(datasets)}")
    print(f"  • Test users: {len(test_users)}")
    print(f"  • Recommendation levels: Top-15, Top-25, Top-35")

if __name__ == "__main__":
    main()
