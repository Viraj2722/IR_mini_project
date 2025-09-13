"""
FAQ IR Retrieval System
File: faq_ir_system.py

Instructions:
1. Place your CSV at '/mnt/data/Mental_Health_FAQ.csv' or update the path variable below.
2. Install dependencies: pip install pandas scikit-learn nltk streamlit
3. Run: streamlit run faq_ir_system.py

This script builds a TF-IDF index over FAQ questions, provides cosine-similarity
retrieval, a simple evaluation method, and a Streamlit UI to query the system.
It demonstrates core IR concepts: preprocessing, indexing, query processing,
ranking, similarity, and evaluation.

Columns expected in CSV: 'Question' and 'Answer'

"""

import os
import sys
from typing import List, Tuple

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# For optional preprocessing
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
except Exception:
    nltk = None

# --------------------------- Config ---------------------------------
CSV_PATH = "Mental_Health_FAQ.csv"  # change if needed
TFIDF_MAX_FEATURES = 5000
RANDOM_STATE = 42

# --------------------------- Utilities -------------------------------

def ensure_nltk_resources():
    """Download necessary NLTK resources if they are missing."""
    global nltk
    if nltk is None:
        try:
            import nltk as _nltk
            nltk = _nltk
        except Exception:
            print("NLTK not available. Falling back to simple tokenization.")
            return

    resources = ["punkt", "wordnet", "omw-1.4", "stopwords"]
    for res in resources:
        try:
            nltk.data.find(res)
        except Exception:
            try:
                nltk.download(res)
            except Exception as e:
                print(f"Could not download NLTK resource {res}: {e}")


# --------------------------- Preprocessing ---------------------------

class Preprocessor:
    """Tokenize, remove stopwords, and lemmatize text for IR.

    Uses NLTK if available; otherwise falls back to simple rules.
    """

    def __init__(self):
        self.use_nltk = nltk is not None
        if self.use_nltk:
            ensure_nltk_resources()
            self.stopwords = set(stopwords.words("english"))
            self.lemmatizer = WordNetLemmatizer()
        else:
            # Minimal English stopword list
            self.stopwords = set([
                "the", "a", "an", "in", "on", "and", "is", "are", "of",
                "for", "to", "with", "that", "this", "it", "as", "be"
            ])
            self.lemmatizer = None

    def preprocess(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        # Remove simple punctuation by replacing non-alphanumerics with space
        text = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
        if self.use_nltk:
            tokens = word_tokenize(text)
            tokens = [t for t in tokens if t not in self.stopwords and t.isalpha()]
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
            return " ".join(tokens)
        else:
            tokens = [t for t in text.split() if t not in self.stopwords]
            return " ".join(tokens)


# --------------------------- IR Model --------------------------------

class FAQIR:
    """Information Retrieval system for FAQ data using TF-IDF + cosine similarity."""

    def __init__(self, questions: List[str], answers: List[str], preprocessor: Preprocessor):
        self.raw_questions = questions
        self.raw_answers = answers
        self.preprocessor = preprocessor

        # Preprocessed text for indexing
        print("Preprocessing questions...")
        self.processed_questions = [self.preprocessor.preprocess(q) for q in self.raw_questions]

        # Build TF-IDF index
        print("Building TF-IDF index...")
        self.vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.processed_questions)

    def search(self, query: str, top_n: int = 5) -> List[Tuple[int, float, str, str]]:
        """Search the indexed FAQs and return top_n results.

        Returns a list of tuples: (index, score, question, answer)
        """
        if not query:
            return []
        processed_query = self.preprocessor.preprocess(query)
        query_vec = self.vectorizer.transform([processed_query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        ranked_idx = np.argsort(scores)[::-1][:top_n]
        results = [(int(i), float(scores[i]), self.raw_questions[i], self.raw_answers[i]) for i in ranked_idx if scores[i] > 0]
        return results

    def explain(self, query: str, doc_idx: int, top_k_terms: int = 5) -> List[Tuple[str, float]]:
        """Return contribution of top terms from the vectorizer for a specific document.

        This gives a rough explanation: top TF-IDF features that overlap between query and doc.
        """
        processed_query = self.preprocessor.preprocess(query)
        query_vec = self.vectorizer.transform([processed_query]).toarray().flatten()
        doc_vec = self.tfidf_matrix[doc_idx].toarray().flatten()
        # element-wise product to see overlapping high-weight terms
        overlap = query_vec * doc_vec
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        top_indices = np.argsort(overlap)[::-1][:top_k_terms]
        terms_scores = [(feature_names[i], float(overlap[i])) for i in top_indices if overlap[i] > 0]
        return terms_scores


# --------------------------- Evaluation -------------------------------

def simple_evaluate(ir_system: FAQIR, sample_queries: List[Tuple[str, List[int]]], k: int = 5):
    """
    Very small evaluation harness.

    sample_queries: list of tuples (query_text, relevant_indices)
    relevant_indices: list of ground-truth FAQ indices (0-based) relevant to the query

    Computes precision@k for each sample and returns average precision.
    """
    precisions = []
    for q, relevant in sample_queries:
        results = ir_system.search(q, top_n=k)
        retrieved_indices = [r[0] for r in results]
        if len(retrieved_indices) == 0:
            precisions.append(0.0)
            continue
        num_relevant_retrieved = sum(1 for idx in retrieved_indices if idx in relevant)
        prec = num_relevant_retrieved / min(k, len(retrieved_indices))
        precisions.append(prec)
    avg_prec = float(np.mean(precisions)) if precisions else 0.0
    return {
        "average_precision_at_{}".format(k): avg_prec,
        "detailed": precisions,
    }


# --------------------------- Loading Data -----------------------------

def load_faq_data(csv_path: str) -> Tuple[List[str], List[str]]:
    if not os.path.exists(csv_path):
        print(f"CSV file not found at {csv_path}. Please update CSV_PATH at the top of the script.")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    # Tolerate different column names
    q_col = None
    a_col = None
    for col in df.columns:
        if col.strip().lower() in ("question", "questions", "q"):
            q_col = col
        if col.strip().lower() in ("answer", "answers", "a"):
            a_col = col
    if q_col is None or a_col is None:
        raise ValueError("CSV must contain 'Question' and 'Answer' columns (case-insensitive). Found: " + ",".join(df.columns))
    questions = df[q_col].fillna("").astype(str).tolist()
    answers = df[a_col].fillna("").astype(str).tolist()
    return questions, answers


# --------------------------- Streamlit UI -----------------------------

def run_streamlit_app(ir_system: FAQIR):
    try:
        import streamlit as st
    except Exception:
        print("Streamlit is not installed. Install it with: pip install streamlit")
        return

    st.set_page_config(page_title="Mental Health FAQ IR", layout="wide")

    st.title("Mental Health FAQ — IR Retrieval Demo")
    st.markdown(
        """
        Enter a query (a word like `depression` or a full question). The system uses TF-IDF + cosine
        similarity to retrieve the most relevant FAQs. This demo shows core IR building blocks: preprocessing,
        indexing, query processing, ranking, and simple evaluation.
        """
    )

    with st.sidebar:
        st.header("Controls")
        top_k = st.slider("Top N results", 1, 10, 5)
        show_explain = st.checkbox("Show explanation (overlapping terms)", value=True)
        sample_eval = st.checkbox("Run example evaluation", value=False)

    query = st.text_input("Type your query here", value="depression")

    if st.button("Search") and query:
        results = ir_system.search(query, top_n=top_k)
        if not results:
            st.warning("No matching FAQs found. Try a different query or check preprocessing settings.")
        else:
            for rank, (idx, score, q_text, a_text) in enumerate(results, start=1):
                st.subheader(f"{rank}. {q_text} (score: {score:.4f})")
                st.write(a_text)
                if show_explain:
                    terms = ir_system.explain(query, idx, top_k_terms=5)
                    if terms:
                        st.markdown("**Top overlapping terms (query vs doc):**")
                        st.write(pd.DataFrame(terms, columns=["term", "overlap_score"]))

    st.markdown("---")
    st.subheader("Index statistics")
    st.write(f"Number of FAQs indexed: {len(ir_system.raw_questions)}")
    st.write(f"TF-IDF vocabulary size: {len(ir_system.vectorizer.get_feature_names_out())}")

    if sample_eval:
        st.markdown("### Example evaluation")
        # The UI cannot know ground-truth; show a tiny demonstration using hard-coded samples
        sample_queries = [
            ("depression", find_relevant_indices_for_keyword(ir_system, "depression", top_n=10)),
            ("anxiety attacks", find_relevant_indices_for_keyword(ir_system, "anxiety", top_n=10)),
        ]
        eval_result = simple_evaluate(ir_system, sample_queries, k=top_k)
        st.write(eval_result)


# --------------------------- Helper for sample eval ------------------

def find_relevant_indices_for_keyword(ir_system: FAQIR, keyword: str, top_n: int = 10) -> List[int]:
    """Heuristic to construct "ground truth" indices for demo evaluation: find any FAQ containing the keyword."""
    keyword = keyword.lower()
    indices = [i for i, q in enumerate(ir_system.raw_questions) if keyword in q.lower()]
    # fallback: search and take top
    if not indices:
        results = ir_system.search(keyword, top_n=top_n)
        indices = [r[0] for r in results]
    return indices


# --------------------------- Main ------------------------------------

def main():
    pre = Preprocessor()
    questions, answers = load_faq_data(CSV_PATH)
    ir_system = FAQIR(questions, answers, pre)

    # If this script is run directly (not via streamlit), show a small CLI demo
    if "STREAMLIT_RUN" not in os.environ and (len(sys.argv) > 1 and sys.argv[1] == "cli"):
        print("CLI demo mode. Type queries (empty to exit).")
        while True:
            q = input("Query> ")
            if not q:
                break
            res = ir_system.search(q, top_n=5)
            for idx, score, q_text, a_text in res:
                print(f"[{idx}] ({score:.4f}) {q_text}\n-> {a_text}\n")
        return

    # Otherwise try to run as a Streamlit app
    # Streamlit sets environment variables when run with `streamlit run` so we call the app UI always.
    run_streamlit_app(ir_system)


if __name__ == "__main__":
    main()
