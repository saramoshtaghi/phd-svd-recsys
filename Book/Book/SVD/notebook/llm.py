import os
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import multiprocessing as mp
import re

# === Load dataset once globally ===
data_path = '../data/goodbooks-10k'
books = pd.read_csv(os.path.join(data_path, 'books.csv'))
df_books = books[['book_id', 'original_title', 'authors']].dropna(subset=['original_title'])

GENRE_LIST = [
    "Fantasy", "Science Fiction", "Romance", "Mystery", "Thriller",
    "Historical", "Adult", "Horror", "Children's",
    "Adventure", "Classics", "Nonfiction", "Drama"
]
genre_options_str = ", ".join(GENRE_LIST)

# === Global pipe for multiprocessing
global_pipe = None

def init_model(model_name):
    """Initialize model pipeline in each worker process."""
    global global_pipe
    # Use AutoModelForSeq2SeqLM for encoder-decoder models like T5
    if "flan" in model_name.lower() or "t5" in model_name.lower():
        from transformers import AutoModelForSeq2SeqLM
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    global_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)


def process_book(row):
    """Process a single book, returns exactly two predicted genres."""
    global global_pipe
    title = row['original_title']
    author = row['authors']

    prompt = f"""
You are a book genre classifier. Your job is to pick exactly two genres from the list below that best match the book title (and author, if given).

Examples:
- "The Hobbit" → Fantasy, Adventure
- "Pride and Prejudice" → Romance, Classics
- "The Shining" → Horror, Thriller
- "Dune" → Science Fiction, Adventure
- "The Diary of a Young Girl" → Nonfiction, Historical
- "Charlotte's Web" → Children's, Classics
- "The Da Vinci Code" → Mystery, Thriller
- "Dracula" → Horror, Classics
- "War and Peace" → Historical, Drama
- "Fifty Shades of Grey" → Romance, Adult

Now choose the two most appropriate genres for:
Title: "{title}"{f" by {author}" if author else ""}

Genres: {genre_options_str}
Respond ONLY with two genres separated by a comma.
"""
    result = global_pipe(prompt.strip(), max_new_tokens=50, temperature=0.6)
    raw_output = result[0]['generated_text'].strip().lower()

    # Extract genres from the output
    genres = [genre for genre in GENRE_LIST if genre.lower() in raw_output]

    # Return the first two genres if more than two are found
    if len(genres) > 2:
        genres = genres[:2]

    # If less than two genres are found, return the raw output for inspection
    if len(genres) < 2:
        return raw_output

    return ", ".join(genres)

def process_and_save_parallel(model_name, output_filename, num_workers=2):
    print(f"\n🚀 Loading model: {model_name} on {num_workers} parallel workers...")

    # For testing: sample first 100 books
    df_books_sample = df_books.head(100).copy()

    with mp.Pool(
        processes=num_workers,
        initializer=init_model,
        initargs=(model_name,)
    ) as pool:
        genres = list(tqdm(
            pool.imap(process_book, df_books_sample.to_dict(orient="records")),
            total=len(df_books_sample)
        ))

    df_books_sample['genres'] = genres
    df_books_sample.to_csv(output_filename, index=False)
    print(f"✅ Saved genres dataset: {output_filename}")

if __name__ == "__main__":
    tasks = [
        # Uncomment or add models as needed:
        # ("HuggingFaceH4/zephyr-7b-beta", "../data/2/books_genres_zephyr.csv"),
        # ("mistralai/Mistral-7B-Instruct-v0.2", "../data/2/books_genres_mistral.csv"),
        # ("meta-llama/Llama-3-8b-instruct", "../data/2/books_genres_llama3.csv"),
        # ("microsoft/phi-2", "../data/2/books_genres_phi2.csv"),
        ("google/flan-t5-xl", "../data/2/books_genres_flan_t5.csv"),
    ]
    num_workers = 2  # reduce workers to avoid overloading your server

    for model_name, output_filename in tasks:
        process_and_save_parallel(model_name, output_filename, num_workers)

    print("🎉 All models finished processing!")
