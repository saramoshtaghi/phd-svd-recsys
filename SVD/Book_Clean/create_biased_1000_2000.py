import pandas as pd
import numpy as np
import os

def main():
    """Create strongly biased datasets for 1000 and 2000 users only"""
    print("🚀 CREATING STRONG BIASED DATASETS - 1000 & 2000 USERS ONLY")
    print("=" * 80)
    print("📋 Strategy: Each synthetic user rates ALL books of their preferred genre with rating 5")
    print("=" * 80)
    
    # Step 1: Load the original dataframe with genres
    print("📂 Loading original dataset with genres...")
    df = pd.read_csv("/home/moshtasa/Research/phd-svd-recsys/SVD/Book/data/df_final_with_genres.csv")
    print(f"✅ Loaded {len(df):,} rows from original dataset")
    
    # Create directory for biased experiments
    os.makedirs('data/biased_experiments', exist_ok=True)
    
    # Step 2: Filter books that contain genres
    print("\n🔍 Filtering books by genres...")
    df_genre_filtered = df.dropna(subset=['genres'])
    
    # Adventure books
    adventure_books = df_genre_filtered[df_genre_filtered['genres'].str.contains('Adventure', na=False)]
    adventure_book_ids = adventure_books['book_id'].unique()
    print(f"✅ Total 'Adventure' books found: {len(adventure_book_ids)}")
    
    # Mystery books
    mystery_books = df_genre_filtered[df_genre_filtered['genres'].str.contains('Mystery', na=False)]
    mystery_book_ids = mystery_books['book_id'].unique()
    print(f"✅ Total 'Mystery' books found: {len(mystery_book_ids)}")
    
    # Function to generate synthetic users
    def generate_synthetic_users(start_user_id, num_users, book_ids, genre='Adventure', rating=5):
        print(f"👥 Generating {num_users} synthetic {genre} users...")
        print(f"📚 Each user rates ALL {len(book_ids)} {genre} books with rating {rating}")
        
        rows = []
        for user_id in range(start_user_id, start_user_id + num_users):
            for book_id in book_ids:
                rows.append({
                    'user_id': user_id,
                    'book_id': book_id,
                    'rating': rating,
                    'decade': None,
                    'original_title': None,
                    'authors': None,
                    'genres': genre
                })
        return pd.DataFrame(rows)
    
    # Only generate for 1000 and 2000 users
    user_counts = [1000, 2000]
    original_max_user_id = df['user_id'].max()
    print(f"🔢 Original max user_id: {original_max_user_id}")
    
    results = []
    
    # Process both genres
    for genre_name, book_ids in [('Adventure', adventure_book_ids), ('Mystery', mystery_book_ids)]:
        print(f"\n🎯 --- Processing {genre_name.upper()} genre ---")
        print(f"📚 Books in genre: {len(book_ids)}")
        
        current_max_user_id = original_max_user_id
        
        for count in user_counts:
            print(f"\n--- {genre_name} - {count} users ---")
            start_user_id = current_max_user_id + 1
            
            # Generate synthetic users
            new_users_df = generate_synthetic_users(start_user_id, count, book_ids, genre_name, 5)
            
            # Combine with original data
            df_biased = pd.concat([df, new_users_df], ignore_index=True)
            
            print(f"🆕 User IDs: {start_user_id} → {start_user_id + count - 1}")
            print(f"🧾 Biased rows added: {len(new_users_df):,}")
            print(f"💾 Total dataset size: {len(df_biased):,} rows")
            
            # Save result
            output_file = f"data/biased_experiments/df_{genre_name.lower()}_{count}_strong.csv"
            print(f"💾 Saving to: {output_file}")
            df_biased.to_csv(output_file, index=False)
            print(f"✅ Saved successfully!")
            
            results.append({
                'genre': genre_name.lower(),
                'num_users': count,
                'biased_ratings': len(new_users_df),
                'total_ratings': len(df_biased),
                'original_ratings': len(df),
                'books_per_user': len(book_ids)
            })
            
            # Update max user_id for next round
            current_max_user_id += count
    
    # Save summary
    summary_df = pd.DataFrame(results)
    summary_df.to_csv('data/biased_experiments/strong_datasets_1000_2000_summary.csv', index=False)
    
    print("\n" + "=" * 80)
    print("🎉 SUMMARY - STRONG BIAS DATASETS COMPLETE")
    print("=" * 80)
    print(f"📊 Original dataset: {len(df):,} ratings")
    print(f"📁 Generated datasets: 4 total (Adventure 1000/2000, Mystery 1000/2000)")
    
    print(f"\n📚 Adventure books: {len(adventure_book_ids):,}")
    print(f"📚 Mystery books: {len(mystery_book_ids):,}")
    
    print("\n📈 Dataset details:")
    for _, row in summary_df.iterrows():
        biased_pct = (row['biased_ratings'] / row['total_ratings']) * 100
        print(f"🎯 {row['genre'].upper()} {row['num_users']} users:")
        print(f"   📊 Total: {row['total_ratings']:,} ratings (+{row['biased_ratings']:,} biased = {biased_pct:.1f}%)")
        print(f"   📚 Each user rated {row['books_per_user']:,} books with rating 5")
    
    print(f"\n💾 All datasets saved to: data/biased_experiments/")
    print("💾 Summary: data/biased_experiments/strong_datasets_1000_2000_summary.csv")
    
    print(f"\n🚨 STRONG BIAS CHARACTERISTICS:")
    print(f"   ✅ Each synthetic user rates ALL books of their preferred genre")
    print(f"   ✅ All synthetic ratings are 5 stars (maximum rating)")
    print(f"   ✅ Creates maximum possible genre bias")

if __name__ == "__main__":
    main()
