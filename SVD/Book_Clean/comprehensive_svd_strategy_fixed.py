"""
COMPREHENSIVE SVD STRATEGY IMPLEMENTATION - FIXED VERSION
=========================================================

Fixed version that properly handles user ID mapping and recommendation generation.
"""

import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import os
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SVDStrategyExperiment:
    def __init__(self, base_data_path, synthetic_data_path, results_path):
        self.base_data_path = base_data_path
        self.synthetic_data_path = synthetic_data_path
        self.results_path = results_path
        self.original_max_user_id = 53424
        self.genre_mappings = None
        self.results = []
        
        # Create results directory
        os.makedirs(results_path, exist_ok=True)
        
    def load_genre_mappings(self):
        """Load adventure and mystery book mappings"""
        print("📚 Loading genre mappings...")
        
        df_genres = pd.read_csv(self.base_data_path)
        df_genre_filtered = df_genres.dropna(subset=['genres'])
        
        # Strategy 1: Contains genre anywhere
        adventure_books_any = set(df_genre_filtered[df_genre_filtered['genres'].str.contains('Adventure', na=False)]['book_id'])
        mystery_books_any = set(df_genre_filtered[df_genre_filtered['genres'].str.contains('Mystery', na=False)]['book_id'])
        
        # Strategy 2: Primary genre only
        def get_primary_genre(genre_string):
            if pd.isna(genre_string):
                return None
            for delimiter in ['|', ',', ';', '//']:
                if delimiter in genre_string:
                    return genre_string.split(delimiter)[0].strip()
            return genre_string.strip()
        
        df_genre_filtered['primary_genre'] = df_genre_filtered['genres'].apply(get_primary_genre)
        adventure_books_primary = set(df_genre_filtered[df_genre_filtered['primary_genre'] == 'Adventure']['book_id'])
        mystery_books_primary = set(df_genre_filtered[df_genre_filtered['primary_genre'] == 'Mystery']['book_id'])
        
        self.genre_mappings = {
            'adventure_any': adventure_books_any,
            'mystery_any': mystery_books_any,
            'adventure_primary': adventure_books_primary,
            'mystery_primary': mystery_books_primary
        }
        
        print(f"✅ Adventure (any): {len(adventure_books_any)} books")
        print(f"✅ Mystery (any): {len(mystery_books_any)} books") 
        print(f"✅ Adventure (primary): {len(adventure_books_primary)} books")
        print(f"✅ Mystery (primary): {len(mystery_books_primary)} books")
        
    def get_optimized_svd_params(self, dataset_size, strategy_type):
        """Get optimized SVD parameters"""
        base_params = {
            'n_factors': 150,
            'n_epochs': 12,
            'lr_all': 0.005,
            'reg_all': 0.02,
            'random_state': 42
        }
        
        if dataset_size > 10_000_000:
            base_params['reg_all'] = 0.05
        elif dataset_size > 8_000_000:
            base_params['reg_all'] = 0.03
            
        if strategy_type == 'primary':
            base_params['reg_all'] *= 0.8
        else:
            base_params['reg_all'] *= 1.2
            
        return base_params
        
    def train_svd_with_monitoring(self, trainset, params, dataset_name):
        """Train SVD with monitoring"""
        print(f"🤖 Training SVD for {dataset_name}...")
        
        start_time = time.time()
        algo = SVD(**params)
        algo.fit(trainset)
        training_time = time.time() - start_time
        
        print(f"✅ Training completed in {training_time:.2f} seconds")
        return algo, training_time
        
    def evaluate_model(self, algo, testset, dataset_name):
        """Evaluate model performance"""
        print(f"📊 Evaluating {dataset_name}...")
        
        predictions = algo.test(testset)
        rmse = accuracy.rmse(predictions, verbose=False)
        
        print(f"   RMSE: {rmse:.4f}")
        return rmse
        
    def generate_recommendations_for_original_users(self, algo, full_trainset, original_user_data, top_n=35):
        """Generate recommendations for original users only - FIXED VERSION"""
        print(f"🎯 Generating top-{top_n} recommendations for original users...")
        
        # Get original users who actually have ratings
        original_users = list(original_user_data['user_id'].unique())
        original_users = [uid for uid in original_users if uid <= self.original_max_user_id]
        
        print(f"   Found {len(original_users)} original users with ratings")
        
        # Get all items from the full dataset
        all_items = set(original_user_data['book_id'].unique())
        
        recommendations = {}
        processed = 0
        
        for user_id in original_users[:5000]:  # Limit to 5000 users for faster processing
            if processed % 500 == 0:
                print(f"   Progress: {processed}/5000 users...")
                
            # Get items already rated by this user
            user_items = set(original_user_data[original_user_data['user_id'] == user_id]['book_id'])
            
            # Get candidate items
            candidate_items = all_items - user_items
            
            # Predict ratings for candidates
            predictions = []
            for item_id in list(candidate_items)[:1000]:  # Limit candidates for speed
                try:
                    pred = algo.predict(user_id, item_id)
                    predictions.append((item_id, pred.est))
                except:
                    continue
                    
            # Sort by predicted rating and get top N
            if predictions:
                predictions.sort(key=lambda x: x[1], reverse=True)
                recommendations[user_id] = [item_id for item_id, rating in predictions[:top_n]]
            
            processed += 1
            
        print(f"✅ Generated recommendations for {len(recommendations)} original users")
        return recommendations
        
    def analyze_genre_distribution(self, recommendations, genre_key, top_n):
        """Analyze genre distribution in recommendations"""
        adventure_books = self.genre_mappings[f'adventure_{genre_key}']
        mystery_books = self.genre_mappings[f'mystery_{genre_key}']
        
        user_stats = []
        
        for user_id, recs in recommendations.items():
            top_recs = recs[:top_n]
            
            adventure_count = sum(1 for book_id in top_recs if book_id in adventure_books)
            mystery_count = sum(1 for book_id in top_recs if book_id in mystery_books)
            other_count = top_n - adventure_count - mystery_count
            
            user_stats.append({
                'user_id': user_id,
                'adventure_count': adventure_count,
                'mystery_count': mystery_count,
                'other_count': other_count,
                'total_count': top_n
            })
            
        return pd.DataFrame(user_stats)
        
    def run_single_experiment(self, dataset_path, dataset_name, strategy_type, genre_key):
        """Run a single SVD experiment"""
        print(f"\n{'='*60}")
        print(f"🧪 EXPERIMENT: {dataset_name}")
        print(f"{'='*60}")
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        print(f"📂 Loaded {len(df):,} ratings")
        
        # Get original user data for recommendation generation
        original_user_data = df[df['user_id'] <= self.original_max_user_id].copy()
        print(f"📊 Original user data: {len(original_user_data):,} ratings from {original_user_data['user_id'].nunique()} users")
        
        # Create Surprise dataset
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df[['user_id', 'book_id', 'rating']], reader)
        
        # Create train/test split
        trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
        
        # Get optimized parameters
        params = self.get_optimized_svd_params(len(df), strategy_type)
        
        # Train model
        algo, training_time = self.train_svd_with_monitoring(trainset, params, dataset_name)
        
        # Evaluate model
        rmse = self.evaluate_model(algo, testset, dataset_name)
        
        # Generate recommendations for different top-N sizes
        experiment_results = []
        
        # Generate recommendations once for top-35, then slice for smaller sizes
        recommendations = self.generate_recommendations_for_original_users(
            algo, trainset, original_user_data, 35
        )
        
        for top_n in [35, 25, 15]:
            print(f"\n📊 Analyzing Top-{top_n} recommendations...")
            
            # Analyze genre distribution
            genre_stats = self.analyze_genre_distribution(recommendations, genre_key, top_n)
            
            if len(genre_stats) == 0:
                print("❌ No recommendations generated!")
                continue
                
            # Calculate summary statistics
            avg_adventure = genre_stats['adventure_count'].mean()
            avg_mystery = genre_stats['mystery_count'].mean()
            avg_other = genre_stats['other_count'].mean()
            
            users_with_adventure = (genre_stats['adventure_count'] > 0).sum()
            users_with_mystery = (genre_stats['mystery_count'] > 0).sum()
            total_users = len(genre_stats)
            
            print(f"   📈 Average per user: {avg_adventure:.2f} adventure, {avg_mystery:.2f} mystery, {avg_other:.2f} other")
            print(f"   👥 User coverage: {users_with_adventure}/{total_users} adventure ({users_with_adventure/total_users*100:.1f}%), {users_with_mystery}/{total_users} mystery ({users_with_mystery/total_users*100:.1f}%)")
            
            # Store results
            result = {
                'dataset_name': dataset_name,
                'strategy_type': strategy_type,
                'genre_key': genre_key,
                'top_n': top_n,
                'total_ratings': len(df),
                'training_time': training_time,
                'rmse': rmse,
                'avg_adventure_per_user': avg_adventure,
                'avg_mystery_per_user': avg_mystery,
                'avg_other_per_user': avg_other,
                'users_with_adventure': users_with_adventure,
                'users_with_mystery': users_with_mystery,
                'total_users': total_users,
                'adventure_coverage_pct': users_with_adventure/total_users*100,
                'mystery_coverage_pct': users_with_mystery/total_users*100,
                'timestamp': datetime.now().isoformat()
            }
            
            experiment_results.append(result)
            
        return experiment_results
        
    def run_comprehensive_experiment(self):
        """Run comprehensive SVD strategy experiment"""
        print("🚀 COMPREHENSIVE SVD STRATEGY EXPERIMENT - FIXED VERSION")
        print("="*70)
        
        # Load genre mappings
        self.load_genre_mappings()
        
        # Define experiments (reduced set for testing)
        experiments = [
            {
                'path': self.base_data_path,
                'name': 'baseline',
                'strategy': 'original',
                'genre_key': 'any'
            },
            {
                'path': f"{self.synthetic_data_path}/enhanced_adventure_1000_any.csv",
                'name': 'adventure_1000_any',
                'strategy': 'contains',
                'genre_key': 'any'
            },
            {
                'path': f"{self.synthetic_data_path}/enhanced_mystery_1000_any.csv",
                'name': 'mystery_1000_any', 
                'strategy': 'contains',
                'genre_key': 'any'
            },
            {
                'path': f"{self.synthetic_data_path}/enhanced_adventure_1000_primary.csv",
                'name': 'adventure_1000_primary',
                'strategy': 'primary',
                'genre_key': 'primary'
            },
            {
                'path': f"{self.synthetic_data_path}/enhanced_mystery_1000_primary.csv",
                'name': 'mystery_1000_primary',
                'strategy': 'primary',
                'genre_key': 'primary'
            }
        ]
        
        # Run experiments
        all_results = []
        
        for i, exp in enumerate(experiments, 1):
            print(f"\n🔬 RUNNING EXPERIMENT {i}/{len(experiments)}")
            
            if not os.path.exists(exp['path']):
                print(f"❌ Dataset not found: {exp['path']}")
                continue
                
            try:
                exp_results = self.run_single_experiment(
                    exp['path'], exp['name'], exp['strategy'], exp['genre_key']
                )
                all_results.extend(exp_results)
                
            except Exception as e:
                print(f"❌ Experiment failed: {e}")
                continue
                
        # Save results
        if all_results:
            results_df = pd.DataFrame(all_results)
            results_file = f"{self.results_path}/comprehensive_svd_results_fixed.csv"
            results_df.to_csv(results_file, index=False)
            print(f"\n💾 Saved results: {results_file}")
            
            # Create comparison
            self.create_strategy_comparison(results_df)
        else:
            print("❌ No results to save!")
            
    def create_strategy_comparison(self, results_df):
        """Create strategy comparison"""
        print(f"\n📊 CREATING STRATEGY COMPARISON...")
        
        baseline_results = results_df[results_df['dataset_name'] == 'baseline']
        
        comparison_data = []
        
        for top_n in [15, 25, 35]:
            if len(baseline_results[baseline_results['top_n'] == top_n]) == 0:
                continue
                
            baseline_row = baseline_results[baseline_results['top_n'] == top_n].iloc[0]
            baseline_adv = baseline_row['avg_adventure_per_user']
            baseline_mys = baseline_row['avg_mystery_per_user']
            
            enhanced_results = results_df[
                (results_df['dataset_name'] != 'baseline') & 
                (results_df['top_n'] == top_n)
            ]
            
            for _, row in enhanced_results.iterrows():
                adv_improvement = ((row['avg_adventure_per_user'] - baseline_adv) / baseline_adv) * 100
                mys_improvement = ((row['avg_mystery_per_user'] - baseline_mys) / baseline_mys) * 100
                
                comparison_data.append({
                    'dataset': row['dataset_name'],
                    'strategy': row['strategy_type'],
                    'top_n': top_n,
                    'baseline_adventure': baseline_adv,
                    'enhanced_adventure': row['avg_adventure_per_user'],
                    'adventure_improvement_pct': adv_improvement,
                    'baseline_mystery': baseline_mys,
                    'enhanced_mystery': row['avg_mystery_per_user'],
                    'mystery_improvement_pct': mys_improvement,
                    'adventure_coverage': row['adventure_coverage_pct'],
                    'mystery_coverage': row['mystery_coverage_pct'],
                    'rmse': row['rmse']
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_file = f"{self.results_path}/strategy_comparison_fixed.csv"
            comparison_df.to_csv(comparison_file, index=False)
            print(f"💾 Saved comparison: {comparison_file}")
            
            # Print summary
            self.print_experiment_summary(comparison_df)
        
    def print_experiment_summary(self, comparison_df):
        """Print experiment summary"""
        print(f"\n🏆 EXPERIMENT SUMMARY")
        print("="*50)
        
        for top_n in [15, 25, 35]:
            top_n_results = comparison_df[comparison_df['top_n'] == top_n]
            
            if len(top_n_results) == 0:
                continue
                
            print(f"\n📊 TOP-{top_n} RESULTS:")
            print("-" * 30)
            
            for _, row in top_n_results.iterrows():
                print(f"{row['dataset']} ({row['strategy']}):")
                print(f"  Adventure: {row['adventure_improvement_pct']:+.1f}% ({row['adventure_coverage']:.1f}% coverage)")
                print(f"  Mystery: {row['mystery_improvement_pct']:+.1f}% ({row['mystery_coverage']:.1f}% coverage)")
                print(f"  RMSE: {row['rmse']:.4f}")
                print()

def main():
    """Main execution"""
    base_data_path = "/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/df_final_with_genres.csv"
    synthetic_data_path = "/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/synthetic_enhanced"
    results_path = "/home/moshtasa/Research/phd-svd-recsys/SVD/Book_Clean/results/svd_strategy_comparison"
    
    experiment = SVDStrategyExperiment(base_data_path, synthetic_data_path, results_path)
    experiment.run_comprehensive_experiment()
    
    print(f"\n🎉 EXPERIMENT COMPLETE!")

if __name__ == "__main__":
    main()
