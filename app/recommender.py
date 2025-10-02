from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import csv
import os

try:
    # Optional: use sklearn if available for speed
    from sklearn.neighbors import NearestNeighbors  # type: ignore
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

import numpy as np


@dataclass
class Recommendation:
    title: str
    distance: float


class BookKNNRecommender:
    def __init__(self,
                 books_csv: Optional[str] = None,
                 ratings_csv: Optional[str] = None,
                 min_user_ratings: int = 200,
                 min_book_ratings: int = 100) -> None:
        self.books_csv = books_csv
        self.ratings_csv = ratings_csv
        self.min_user_ratings = min_user_ratings
        self.min_book_ratings = min_book_ratings

        self.title_to_isbn: Dict[str, str] = {}
        self.isbn_to_title: Dict[str, str] = {}

        self.book_index_to_isbn: List[str] = []
        self.user_index_to_user_id: List[str] = []
        self.ratings_matrix: Optional[np.ndarray] = None

        self.nn_model = None

    def _load_csv(self, path: str) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows

    def _load_sample(self) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        # Minimal sample dataset to make the app usable without files
        books = [
            {"ISBN": "0001", "Book-Title": "The Hobbit"},
            {"ISBN": "0002", "Book-Title": "The Lord of the Rings"},
            {"ISBN": "0003", "Book-Title": "The Silmarillion"},
            {"ISBN": "0004", "Book-Title": "The Witcher"},
            {"ISBN": "0005", "Book-Title": "A Game of Thrones"},
        ]
        ratings = [
            {"User-ID": "U1", "ISBN": "0001", "Book-Rating": "9"},
            {"User-ID": "U1", "ISBN": "0002", "Book-Rating": "9"},
            {"User-ID": "U1", "ISBN": "0003", "Book-Rating": "8"},
            {"User-ID": "U2", "ISBN": "0001", "Book-Rating": "8"},
            {"User-ID": "U2", "ISBN": "0002", "Book-Rating": "9"},
            {"User-ID": "U2", "ISBN": "0005", "Book-Rating": "6"},
            {"User-ID": "U3", "ISBN": "0003", "Book-Rating": "7"},
            {"User-ID": "U3", "ISBN": "0004", "Book-Rating": "9"},
            {"User-ID": "U3", "ISBN": "0005", "Book-Rating": "8"},
        ]
        return books, ratings

    def _filter_and_pivot(self,
                           books: List[Dict[str, str]],
                           ratings: List[Dict[str, str]]) -> None:
        # Build counts
        from collections import Counter

        user_counts = Counter(r["User-ID"] for r in ratings)
        book_counts = Counter(r["ISBN"] for r in ratings)

        filtered = [
            r for r in ratings
            if user_counts[r["User-ID"]] >= self.min_user_ratings
            and book_counts[r["ISBN"]] >= self.min_book_ratings
        ]

        # If filtering removes all (likely on sample), fall back to unfiltered
        if len(filtered) == 0:
            filtered = ratings

        # Build index maps
        unique_isbns = sorted({r["ISBN"] for r in filtered})
        unique_users = sorted({r["User-ID"] for r in filtered})

        isbn_to_index = {isbn: i for i, isbn in enumerate(unique_isbns)}
        user_to_index = {uid: i for i, uid in enumerate(unique_users)}

        # Populate dense matrix
        matrix = np.zeros((len(unique_isbns), len(unique_users)), dtype=np.float32)
        for r in filtered:
            i = isbn_to_index[r["ISBN"]]
            j = user_to_index[r["User-ID"]]
            try:
                rating = float(r["Book-Rating"]) if r["Book-Rating"] != '' else 0.0
            except ValueError:
                rating = 0.0
            matrix[i, j] = rating

        # Save state
        self.book_index_to_isbn = unique_isbns
        self.user_index_to_user_id = unique_users

        # Title maps
        self.title_to_isbn = {b["Book-Title"]: b["ISBN"] for b in books}
        self.isbn_to_title = {b["ISBN"]: b["Book-Title"] for b in books}

        self.ratings_matrix = matrix

    def fit(self) -> None:
        # Load data
        books: List[Dict[str, str]]
        ratings: List[Dict[str, str]]
        if self.books_csv and os.path.exists(self.books_csv) and self.ratings_csv and os.path.exists(self.ratings_csv):
            books = self._load_csv(self.books_csv)
            ratings = self._load_csv(self.ratings_csv)
        else:
            books, ratings = self._load_sample()

        self._filter_and_pivot(books, ratings)

        # Fit model
        if SKLEARN_AVAILABLE:
            self.nn_model = NearestNeighbors(metric="cosine", algorithm="brute")
            self.nn_model.fit(self.ratings_matrix)
        else:
            # No explicit model; we'll compute cosine distances on demand
            self.nn_model = None

    def _cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 1.0
        cosine_sim = float(np.dot(a, b) / (a_norm * b_norm))
        return 1.0 - cosine_sim

    def recommend(self, title: str, k: int = 5) -> List[Recommendation]:
        if self.ratings_matrix is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        isbn = self.title_to_isbn.get(title)
        if isbn is None:
            raise ValueError(f"Book '{title}' not found.")

        try:
            book_idx = self.book_index_to_isbn.index(isbn)
        except ValueError:
            raise ValueError(f"Book '{title}' not found in ratings matrix.")

        target_vec = self.ratings_matrix[book_idx]

        if SKLEARN_AVAILABLE and self.nn_model is not None:
            distances, indices = self.nn_model.kneighbors(target_vec.reshape(1, -1), n_neighbors=min(k + 1, len(self.book_index_to_isbn)))
            indices = indices.flatten().tolist()
            distances = distances.flatten().tolist()
        else:
            # Manual cosine distances
            indices = list(range(len(self.book_index_to_isbn)))
            distances = [self._cosine_distance(target_vec, self.ratings_matrix[i]) for i in indices]

        # Pair and sort (skip self index)
        pairs = [
            (idx, dist) for idx, dist in zip(indices, distances)
            if idx != book_idx
        ]
        pairs.sort(key=lambda x: x[1])
        top = pairs[:k]

        results: List[Recommendation] = []
        for idx, dist in top:
            rec_isbn = self.book_index_to_isbn[idx]
            rec_title = self.isbn_to_title.get(rec_isbn, "Unknown Title")
            results.append(Recommendation(title=rec_title, distance=float(dist)))
        return results

    def suggest(self, prefix: str, limit: int = 10) -> List[str]:
        if not self.title_to_isbn:
            return []
        p = prefix.lower()
        titles = [t for t in self.title_to_isbn.keys() if p in t.lower()]
        titles.sort()
        return titles[:limit]
