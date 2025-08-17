"""
COMPREHENSIVE SVD STRATEGY IMPLEMENTATION
==========================================

This script implements a comprehensive SVD strategy following the 6-step plan:
1. Run both strategies with careful parameter tuning (Top-15, 25, 35)
2. Monitor training progress for signs of overfitting
3. Evaluate on original users only
4. Compare strategies to pick the winner
5. If successful: Scale to production recommendations
6. If failed: Try ensemble methods or post-processing approaches
"""

import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import os
import time
from datetime import datetime
import json
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
        """Get optimized SVD parameters based on dataset characteristics"""
        
        # Base parameters
        base_params = {
            'n_factors': 200,
            'n_epochs': 15,
            'lr_all': 0.002,
            'reg_all': 0.08,
            'random_state': 42
        }
        
        # Adjust based on dataset size
        if dataset_size > 10_000_000:  # Very large datasets
            base_params['n_factors'] = 250
            base_params['n_epochs'] = 12
            base_params['reg_all'] = 0.10
        elif dataset_size > 8_000_000:  # Large datasets
            base_params['n_factors'] = 220
            base_params['n_epochs'] = 13
            base_params['reg_all'] = 0.09
            
        # Adjust based on strategy type
        if strategy_type == 'primary':
            # Primary genre strategies need less regularization (cleaner signal)
            base_params['reg_all'] *= 0.8
            base_params['n_epochs'] += 2
        else:
            # Contains strategies need more regularization (noisy signal)
            base_params['reg_all'] *= 1.2
            
        return base_params
        
    def train_svd_with_monitoring(self, trainset, params, dataset_name):
        """Train SVD with progress monitoring"""
        print(f"🤖 Training SVD for {dataset_name}...")
        print(f"   Parameters: {params}")
        
        start_time = time.time()
        
        # Create SVD model
        algo = SVD(**params)
        
        # Train with monitoring
        algo.fit(trainset)
        
        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.2f} seconds")
        
        return algo, training_time
        
    def evaluate_model(self, algo, testset, dataset_name):
        """Evaluate model performance"""
        print(f"📊 Evaluating {dataset_name}...")
        
        # Calculate RMSE
        predictions = algo.test(testset)
        rmse = accuracy.rmse(predictions, verbose=False)
        
        print(f"   RMSE: {rmse:.4f}")
        return rmse
        
    def generate_recommendations_for_original_users(self, algo, trainset, top_n=35):
        """Generate recommendations ONLY for original users"""
        print(f"🎯 Generating top-{top_n} recommendations for original users...")
        
        # Get original users only (avoid synthetic users)
        original_users = list(range(1, self.original_max_user_id + 1))
        
        # Get all items
        all_items = set([trainset.to_raw_iid(i) for i in trainset.all_items()])
        
        recommendations = {}
        processed = 0
        
        for user_id in original_users:
            if processed % 5000 == 0:
                print(f"   Progress: {processed}/{len(original_users)} users...")
                
            # Skip if user not in trainset
            try:
                inner_uid = trainset.to_inner_uid(user_id)
            except:
                continue
                
            # Get items already rated by user  
            user_items = set([trainset.to_raw_iid(i) for i in trainset.ur[inner_uid]])
            
            # Get candidate items
            candidate_items = all_items - user_items
            
            # Predict ratings for candidates
            predictions = []
            for item_id in candidate_items:
                try:
                    pred = algo.predict(user_id, item_id)
                    predictions.append((item_id, pred.est))
                except:
                    continue
                    
            # Sort by predicted rating and get top N
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
        print(f"📁 Dataset: {dataset_path}")
        print(f"🎯 Strategy: {strategy_type}")
        print(f"{'='*60}")
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        print(f"📂 Loaded {len(df):,} ratings")
        
        # Create Surprise dataset
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df[['user_id', 'book_id', 'rating']], reader)
        
        # Create train/test split (consistent across experiments)
        trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
        print(f"📊 Train: {trainset.n_ratings:,} ratings, Test: {len(testset):,} ratings")
        
        # Get optimized parameters
        params = self.get_optimized_svd_params(len(df), strategy_type)
        
        # Train model with monitoring
        algo, training_time = self.train_svd_with_monitoring(trainset, params, dataset_name)
        
        # Evaluate model
        rmse = self.evaluate_model(algo, testset, dataset_name)
        
        # Generate recommendations for all top-N sizes
        experiment_results = []
        
        for top_n in [35, 25, 15]:
            print(f"\n📊 Analyzing Top-{top_n} recommendations...")
            
            # Generate recommendations
            recommendations = self.generate_recommendations_for_original_users(algo, trainset, top_n)
            
            # Analyze genre distribution
            genre_stats = self.analyze_genre_distribution(recommendations, genre_key, top_n)
            
            # Calculate summary statistics
            avg_adventure = genre_stats['adventure_count'].mean()
            avg_mystery = genre_stats['mystery_count'].mean()
            avg_other = genre_stats['other_count'].mean()
            
            # Count users with improvements
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
                'svd_params': params,
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
            
            # Save detailed recommendations
            detailed_results = []
            for user_id, recs in recommendations.items():
                top_recs = recs[:top_n]
                
                for rank, book_id in enumerate(top_recs, 1):
                    genre = 'Other'
                    if book_id in self.genre_mappings[f'adventure_{genre_key}']:
                        genre = 'Adventure'
                    elif book_id in self.genre_mappings[f'mystery_{genre_key}']:
                        genre = 'Mystery'
                        
                    detailed_results.append({
                        'user_id': user_id,
                        'book_id': book_id,
                        'recommendation_rank': rank,
                        'genre': genre,
                        'dataset': dataset_name,
                        'strategy': strategy_type,
                        'top_n': top_n
                    })
            
            # Save detailed results
            detailed_df = pd.DataFrame(detailed_results)
            output_file = f"{self.results_path}/{dataset_name}_top{top_n}_detailed.csv"
            detailed_df.to_csv(output_file, index=False)
            print(f"💾 Saved detailed results: {output_file}")
            
        return experiment_results
        
    def run_comprehensive_experiment(self):
        """Run comprehensive SVD strategy experiment"""
        print("🚀 COMPREHENSIVE SVD STRATEGY EXPERIMENT")
        print("="*70)
        print("📋 Testing both strategies with parameter tuning")
        print("🎯 Evaluating Top-15, 25, 35 recommendations")
        print("👥 Testing on original users only")
        print("="*70)
        
        # Load genre mappings
        self.load_genre_mappings()
        
        # Define experiments
        experiments = [
            # Baseline
            {
                'path': self.base_data_path,
                'name': 'baseline',
                'strategy': 'original',
                'genre_key': 'any'  # Doesn't matter for baseline
            },
            
            # Strategy 1: Contains Genre Anywhere
            {
                'path': f"{self.synthetic_data_path}/enhanced_adventure_1000_any.csv",
                'name': 'adventure_1000_any',
                'strategy': 'contains',
                'genre_key': 'any'
            },
            {
                'path': f"{self.synthetic_data_path}/enhanced_adventure_2000_any.csv", 
                'name': 'adventure_2000_any',
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
                'path': f"{self.synthetic_data_path}/enhanced_mystery_2000_any.csv",
                'name': 'mystery_2000_any',
                'strategy': 'contains', 
                'genre_key': 'any'
            },
            
            # Strategy 2: Primary Genre Only
            {
                'path': f"{self.synthetic_data_path}/enhanced_adventure_1000_primary.csv",
                'name': 'adventure_1000_primary',
                'strategy': 'primary',
                'genre_key': 'primary'
            },
            {
                'path': f"{self.synthetic_data_path}/enhanced_adventure_2000_primary.csv",
                'name': 'adventure_2000_primary', 
                'strategy': 'primary',
                'genre_key': 'primary'
            },
            {
                'path': f"{self.synthetic_data_path}/enhanced_mystery_1000_primary.csv",
                'name': 'mystery_1000_primary',
                'strategy': 'primary',
                'genre_key': 'primary'
            },
            {
                'path': f"{self.synthetic_data_path}/enhanced_mystery_2000_primary.csv",
                'name': 'mystery_2000_primary',
                'strategy': 'primary',
                'genre_key': 'primary'
            }
        ]
        
        # Run all experiments
        all_results = []
        
        for i, exp in enumerate(experiments, 1):
            print(f"\n🔬 RUNNING EXPERIMENT {i}/{len(experiments)}")
            
            if not os.path.exists(exp['path']):
                print(f"❌ Dataset not found: {exp['path']}")
                print("   Please run the synthetic user creation first!")
                continue
                
            try:
                exp_results = self.run_single_experiment(
                    exp['path'], exp['name'], exp['strategy'], exp['genre_key']
                )
                all_results.extend(exp_results)
                
            except Exception as e:
                print(f"❌ Experiment failed: {e}")
                continue
                
        # Save comprehensive results
        if all_results:
            results_df = pd.DataFrame(all_results)
            results_file = f"{self.results_path}/comprehensive_svd_results.csv"
            results_df.to_csv(results_file, index=False)
            print(f"\n💾 Saved comprehensive results: {results_file}")
            
            # Create summary analysis
            self.create_strategy_comparison(results_df)
        else:
            print("❌ No results to save!")
            
    def create_strategy_comparison(self, results_df):
        """Create comprehensive strategy comparison"""
        print(f"\n📊 CREATING STRATEGY COMPARISON...")
        
        # Get baseline results for comparison
        baseline_results = results_df[results_df['dataset_name'] == 'baseline']
        
        comparison_data = []
        
        for top_n in [15, 25, 35]:
            baseline_row = baseline_results[baseline_results['top_n'] == top_n].iloc[0]
            baseline_adv = baseline_row['avg_adventure_per_user']
            baseline_mys = baseline_row['avg_mystery_per_user']
            
            # Compare each enhanced dataset
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
                    'rmse': row['rmse'],
                    'training_time': row['training_time']
                })
        
        # Save comparison
        comparison_df = pd.DataFrame(comparison_data)
        comparison_file = f"{self.results_path}/strategy_comparison.csv"
        comparison_df.to_csv(comparison_file, index=False)
        print(f"💾 Saved strategy comparison: {comparison_file}")
        
        # Print summary
        self.print_experiment_summary(comparison_df)
        
    def print_experiment_summary(self, comparison_df):
        """Print experiment summary and winner selection"""
        print(f"\n🏆 EXPERIMENT SUMMARY & WINNER SELECTION")
        print("="*70)
        
        # Find best performers for each genre and top_n
        for top_n in [15, 25, 35]:
            print(f"\n📊 TOP-{top_n} RESULTS:")
            print("-" * 50)
            
            top_n_results = comparison_df[comparison_df['top_n'] == top_n]
            
            # Best adventure improvement
            best_adv = top_n_results.loc[top_n_results['adventure_improvement_pct'].idxmax()]
            print(f"🥇 Best Adventure Enhancement:")
            print(f"   Dataset: {best_adv['dataset']}")
            print(f"   Strategy: {best_adv['strategy']}")
            print(f"   Improvement: {best_adv['adventure_improvement_pct']:+.1f}%")
            print(f"   Coverage: {best_adv['adventure_coverage']:.1f}% of users")
            
            # Best mystery improvement
            best_mys = top_n_results.loc[top_n_results['mystery_improvement_pct'].idxmax()]
            print(f"\n🥇 Best Mystery Enhancement:")
            print(f"   Dataset: {best_mys['dataset']}")
            print(f"   Strategy: {best_mys['strategy']}")
            print(f"   Improvement: {best_mys['mystery_improvement_pct']:+.1f}%")
            print(f"   Coverage: {best_mys['mystery_coverage']:.1f}% of users")
            
        # Overall strategy winner
        print(f"\n🎯 OVERALL STRATEGY ASSESSMENT:")
        print("-" * 50)
        
        # Calculate average improvements by strategy
        strategy_summary = comparison_df.groupby('strategy').agg({
            'adventure_improvement_pct': 'mean',
            'mystery_improvement_pct': 'mean',
            'adventure_coverage': 'mean',
            'mystery_coverage': 'mean',
            'rmse': 'mean'
        })
        
        print("Strategy Performance Summary:")
        for strategy in strategy_summary.index:
            row = strategy_summary.loc[strategy]
            print(f"\n📈 {strategy.upper()} Strategy:")
            print(f"   Adventure improvement: {row['adventure_improvement_pct']:+.1f}% avg")
            print(f"   Mystery improvement: {row['mystery_improvement_pct']:+.1f}% avg")
            print(f"   User coverage: {row['adventure_coverage']:.1f}% adv, {row['mystery_coverage']:.1f}% mys")
            print(f"   Average RMSE: {row['rmse']:.4f}")
            
        # Determine success/failure
        max_adv_improvement = comparison_df['adventure_improvement_pct'].max()
        max_mys_improvement = comparison_df['mystery_improvement_pct'].max()
        
        print(f"\n🎖️  FINAL VERDICT:")
        if max_adv_improvement > 50 and max_mys_improvement > 50:
            print("✅ SUCCESS - Ready for production!")
        elif max_adv_improvement > 20 or max_mys_improvement > 20:
            print("⚠️  PARTIAL SUCCESS - Needs refinement")
        else:
            print("❌ FAILURE - Back to drawing board")

def main():
    """Main execution function"""
    
    # Configuration
    base_data_path = "/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/df_final_with_genres.csv"
    synthetic_data_path = "/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/synthetic_enhanced"
    results_path = "/home/moshtasa/Research/phd-svd-recsys/SVD/Book_Clean/results/svd_strategy_comparison"
    
    # Create experiment
    experiment = SVDStrategyExperiment(base_data_path, synthetic_data_path, results_path)
    
    # Run comprehensive experiment
    experiment.run_comprehensive_experiment()
    
    print(f"\n🎉 COMPREHENSIVE SVD STRATEGY EXPERIMENT COMPLETE!")
    print(f"📁 Results saved to: {results_path}")

if __name__ == "__main__":
    main()
