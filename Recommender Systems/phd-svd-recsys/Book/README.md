# Goodbooks-10k Dataset

This dataset was originally sourced from [Kaggle - Goodbooks-10k](https://www.kaggle.com/datasets/zygmunt/goodbooks-10k).

It contains ratings for 10,000 popular books, and was collected from the [goodreads.com](https://www.goodreads.com/) website.

---

## Dataset Summary

- **Number of books:** 10,000
- **Number of users:** ~53,000
- **Number of ratings:** ~9.38 million

The dataset can be used for building and evaluating recommender systems, collaborative filtering, and matrix factorization methods such as SVD.

---

## Files

### `books.csv`

Contains metadata for the books.

| Column | Description |
|--------|-------------|
| book_id | Unique identifier for each book |
| title | Title of the book |
| authors | Author(s) of the book |
| average_rating | Average rating on Goodreads |
| isbn | ISBN number |
| isbn13 | 13-digit ISBN number |
| language_code | Language of the book |
| num_pages | Number of pages |
| ratings_count | Total number of ratings |
| text_reviews_count | Number of text reviews |
| publication_date | Date of publication |
| publisher | Publisher of the book |
| original_publication_year | Year of original publication |

---

### `ratings.csv`

Contains the user ratings.

| Column | Description |
|--------|-------------|
| user_id | Unique identifier for each user |
| book_id | ID of the book (foreign key to `books.csv`) |
| rating | Rating provided by the user (1-5 scale) |

---

### `tags.csv` and `book_tags.csv`

- `tags.csv` contains tag definitions (tag_id, tag_name).
- `book_tags.csv` maps tags to books and includes tag counts.

---

## Licensing

This dataset is provided for educational and research purposes.

Source: [Kaggle Goodbooks-10k Dataset](https://www.kaggle.com/datasets/zygmunt/goodbooks-10k)

---

## Notes

- The dataset was originally created by Zygmunt Zając.
- The dataset is widely used for recommendation system research and experiments.
