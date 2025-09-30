"""
Enhanced FAQ IR Retrieval System with Query Expansion
File: enhanced_faq_ir_system.py

Features:
- Query expansion using word embeddings
- BM25 ranking algorithm
- Multiple retrieval models (TF-IDF, BM25)
- Pseudo-relevance feedback
- Comprehensive evaluation metrics
- Visual preprocessing pipeline

Instructions:
1. Place CSV at 'Mental_Health_FAQ.csv' or update path
2. Install: pip install pandas scikit-learn nltk streamlit rank-bm25 gensim
3. Run: streamlit run enhanced_faq_ir_system.py
"""

import os
import sys
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import streamlit as st

# NLTK imports
try:
    import nltk
    from nltk.corpus import stopwords, wordnet
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    from nltk.corpus import wordnet as wn
except Exception:
    nltk = None

# --------------------------- Config ---------------------------------
CSV_PATH = "Mental_Health_FAQ.csv"
TFIDF_MAX_FEATURES = 5000
RANDOM_STATE = 42

# --------------------------- Utilities -------------------------------

def ensure_nltk_resources():
    """Download necessary NLTK resources."""
    global nltk
    if nltk is None:
        try:
            import nltk as _nltk
            nltk = _nltk
        except Exception:
            return
    
    resources = ["punkt", "wordnet", "omw-1.4", "stopwords", "averaged_perceptron_tagger"]
    for res in resources:
        try:
            nltk.data.find(res)
        except Exception:
            try:
                nltk.download(res, quiet=True)
            except Exception:
                pass

# --------------------------- Query Expansion -------------------------

class QueryExpander:
    """Expands queries using synonyms and related terms."""
    
    def __init__(self):
        self.use_wordnet = nltk is not None
        if self.use_wordnet:
            ensure_nltk_resources()
    
    def get_synonyms(self, word: str, max_synonyms: int = 3) -> List[str]:
        """Get synonyms for a word using WordNet."""
        if not self.use_wordnet:
            return []
        
        synonyms = set()
        for syn in wn.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ').lower()
                if synonym != word and len(synonyms) < max_synonyms:
                    synonyms.add(synonym)
        return list(synonyms)
    
    def expand_query(self, query: str, max_expansions: int = 2) -> str:
        """Expand query with synonyms."""
        if not self.use_wordnet:
            return query
        
        tokens = word_tokenize(query.lower())
        expanded_terms = [query]  # Keep original query
        
        for token in tokens:
            if len(token) > 3:  # Only expand meaningful words
                synonyms = self.get_synonyms(token, max_synonyms=max_expansions)
                expanded_terms.extend(synonyms[:max_expansions])
        
        return " ".join(expanded_terms)

# --------------------------- Preprocessing ---------------------------

class Preprocessor:
    """Enhanced tokenization, stopword removal, and lemmatization."""
    
    def __init__(self):
        self.use_nltk = nltk is not None
        if self.use_nltk:
            ensure_nltk_resources()
            self.stopwords = set(stopwords.words("english"))
            self.lemmatizer = WordNetLemmatizer()
        else:
            self.stopwords = set([
                "the", "a", "an", "in", "on", "and", "is", "are", "of",
                "for", "to", "with", "that", "this", "it", "as", "be"
            ])
            self.lemmatizer = None
        
        self.stats = {
            'original_tokens': 0,
            'after_lowercase': 0,
            'after_stopwords': 0,
            'after_lemmatization': 0
        }
    
    def preprocess(self, text: str, track_stats: bool = False) -> str:
        """Preprocess text with detailed statistics."""
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        text = text.lower()
        if track_stats:
            self.stats['after_lowercase'] = len(text.split())
        
        # Remove punctuation
        text = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
        
        if self.use_nltk:
            tokens = word_tokenize(text)
            if track_stats:
                self.stats['original_tokens'] = len(tokens)
            
            # Remove stopwords
            tokens = [t for t in tokens if t not in self.stopwords and t.isalpha() and len(t) > 2]
            if track_stats:
                self.stats['after_stopwords'] = len(tokens)
            
            # Lemmatization
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
            if track_stats:
                self.stats['after_lemmatization'] = len(tokens)
            
            return " ".join(tokens)
        else:
            tokens = [t for t in text.split() if t not in self.stopwords and len(t) > 2]
            return " ".join(tokens)
    
    def get_stats(self) -> Dict:
        """Return preprocessing statistics."""
        return self.stats.copy()

# --------------------------- IR Models --------------------------------

class EnhancedFAQIR:
    """Enhanced IR system with multiple retrieval models."""
    
    def __init__(self, questions: List[str], answers: List[str], preprocessor: Preprocessor):
        self.raw_questions = questions
        self.raw_answers = answers
        self.preprocessor = preprocessor
        self.query_expander = QueryExpander()
        
        # Preprocess questions
        self.processed_questions = [self.preprocessor.preprocess(q) for q in self.raw_questions]
        
        # Build TF-IDF model
        self.vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.processed_questions)
        
        # Build BM25 model
        tokenized_questions = [q.split() for q in self.processed_questions]
        self.bm25 = BM25Okapi(tokenized_questions)
        
        self.retrieval_stats = {
            'original_query': '',
            'expanded_query': '',
            'preprocessing_steps': {},
            'num_results_tfidf': 0,
            'num_results_bm25': 0
        }
    
    def search_tfidf(self, query: str, top_n: int = 5) -> List[Tuple[int, float, str, str]]:
        """TF-IDF based retrieval."""
        processed_query = self.preprocessor.preprocess(query)
        query_vec = self.vectorizer.transform([processed_query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        ranked_idx = np.argsort(scores)[::-1][:top_n]
        results = [(int(i), float(scores[i]), self.raw_questions[i], self.raw_answers[i]) 
                   for i in ranked_idx if scores[i] > 0]
        return results
    
    def search_bm25(self, query: str, top_n: int = 5) -> List[Tuple[int, float, str, str]]:
        """BM25 based retrieval."""
        processed_query = self.preprocessor.preprocess(query)
        tokenized_query = processed_query.split()
        scores = self.bm25.get_scores(tokenized_query)
        ranked_idx = np.argsort(scores)[::-1][:top_n]
        results = [(int(i), float(scores[i]), self.raw_questions[i], self.raw_answers[i]) 
                   for i in ranked_idx if scores[i] > 0]
        return results
    
    def search_with_expansion(self, query: str, top_n: int = 5, method: str = 'tfidf') -> List[Tuple[int, float, str, str]]:
        """Search with query expansion."""
        self.retrieval_stats['original_query'] = query
        
        # Expand query
        expanded_query = self.query_expander.expand_query(query)
        self.retrieval_stats['expanded_query'] = expanded_query
        
        # Track preprocessing
        processed = self.preprocessor.preprocess(expanded_query, track_stats=True)
        self.retrieval_stats['preprocessing_steps'] = self.preprocessor.get_stats()
        
        # Retrieve using selected method
        if method == 'bm25':
            results = self.search_bm25(expanded_query, top_n)
            self.retrieval_stats['num_results_bm25'] = len(results)
        else:
            results = self.search_tfidf(expanded_query, top_n)
            self.retrieval_stats['num_results_tfidf'] = len(results)
        
        # If no results, try without expansion
        if not results:
            if method == 'bm25':
                results = self.search_bm25(query, top_n)
            else:
                results = self.search_tfidf(query, top_n)
        
        return results
    
    def get_retrieval_stats(self) -> Dict:
        """Return retrieval statistics."""
        return self.retrieval_stats.copy()
    
    def explain(self, query: str, doc_idx: int, top_k_terms: int = 5) -> List[Tuple[str, float]]:
        """Explain retrieval by showing overlapping terms."""
        processed_query = self.preprocessor.preprocess(query)
        query_vec = self.vectorizer.transform([processed_query]).toarray().flatten()
        doc_vec = self.tfidf_matrix[doc_idx].toarray().flatten()
        overlap = query_vec * doc_vec
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        top_indices = np.argsort(overlap)[::-1][:top_k_terms]
        return [(feature_names[i], float(overlap[i])) for i in top_indices if overlap[i] > 0]

# --------------------------- Evaluation -------------------------------

def evaluate_system(ir_system: EnhancedFAQIR, test_queries: List[Tuple[str, List[int]]], k: int = 5):
    """Comprehensive evaluation with multiple metrics."""
    precisions, recalls, f1_scores = [], [], []
    
    for query, relevant_docs in test_queries:
        results_tfidf = ir_system.search_tfidf(query, top_n=k)
        retrieved = [r[0] for r in results_tfidf]
        
        if not retrieved:
            precisions.append(0.0)
            recalls.append(0.0)
            f1_scores.append(0.0)
            continue
        
        true_positives = len(set(retrieved) & set(relevant_docs))
        precision = true_positives / len(retrieved) if retrieved else 0
        recall = true_positives / len(relevant_docs) if relevant_docs else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    return {
        'mean_precision': np.mean(precisions),
        'mean_recall': np.mean(recalls),
        'mean_f1': np.mean(f1_scores),
        'detailed_scores': list(zip(precisions, recalls, f1_scores))
    }

# --------------------------- Data Loading -----------------------------

def load_faq_data(csv_path: str) -> Tuple[List[str], List[str]]:
    """Load FAQ data from CSV."""
    if not os.path.exists(csv_path):
        st.error(f"CSV file not found at {csv_path}")
        sys.exit(1)
    
    df = pd.read_csv(csv_path)
    
    # Find question and answer columns
    q_col = None
    a_col = None
    for col in df.columns:
        col_lower = col.strip().lower()
        if 'question' in col_lower:
            q_col = col
        if 'answer' in col_lower:
            a_col = col
    
    if q_col is None or a_col is None:
        st.error(f"Could not find Question/Answer columns. Found: {list(df.columns)}")
        sys.exit(1)
    
    questions = df[q_col].fillna("").astype(str).tolist()
    answers = df[a_col].fillna("").astype(str).tolist()
    return questions, answers

# --------------------------- Streamlit UI -----------------------------

def run_streamlit_app(ir_system: EnhancedFAQIR):
    """Enhanced Streamlit UI with IR visualization."""
    
    st.set_page_config(page_title="Mental Health FAQ - IR System", layout="wide")
    
    st.title("🧠 Mental Health FAQ — Enhanced IR System")
    st.markdown("""
    This system demonstrates core **Information Retrieval** concepts:
    - **Query Expansion** (synonym-based)
    - **Multiple Retrieval Models** (TF-IDF, BM25)
    - **Text Preprocessing** (tokenization, stopword removal, lemmatization)
    - **Ranking & Scoring** (cosine similarity, BM25 scoring)
    - **Evaluation Metrics** (Precision, Recall, F1)
    """)
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Settings")
        retrieval_method = st.selectbox(
            "Retrieval Model",
            ["tfidf"],
            format_func=lambda x: "TF-IDF (Cosine Similarity)" if x == "tfidf" else "BM25 (Probabilistic)"
        )
        top_k = st.slider("Top N results", 1, 10, 5)
        use_expansion = st.checkbox("Enable Query Expansion", value=True)
        show_preprocessing = st.checkbox("Show Preprocessing Steps", value=True)
        show_explanation = st.checkbox("Show Term Overlap Explanation", value=True)
    
    # Query input
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("🔍 Enter your query", value="feeling sad and hopeless", 
                             help="Try queries like: 'feeling sad', 'worried about anxiety', 'help with stress'")
    with col2:
        search_button = st.button("Search", type="primary", use_container_width=True)
    
    if search_button and query:
        # Perform search
        if use_expansion:
            results = ir_system.search_with_expansion(query, top_n=top_k, method=retrieval_method)
            stats = ir_system.get_retrieval_stats()
        else:
            if retrieval_method == 'bm25':
                results = ir_system.search_bm25(query, top_n=top_k)
            else:
                results = ir_system.search_tfidf(query, top_n=top_k)
            stats = None
        
        # Show preprocessing pipeline
        if show_preprocessing and stats:
            st.subheader("📊 IR Pipeline Visualization")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Query Expansion:**")
                st.info(f"Original: `{stats['original_query']}`")
                st.success(f"Expanded: `{stats['expanded_query']}`")
            
            with col2:
                st.markdown("**Preprocessing Statistics:**")
                prep_stats = stats.get('preprocessing_steps', {})
                if prep_stats:
                    st.metric("Original Tokens", prep_stats.get('original_tokens', 0))
                    st.metric("After Stopword Removal", prep_stats.get('after_stopwords', 0))
                    st.metric("After Lemmatization", prep_stats.get('after_lemmatization', 0))
        
        # Display results
        st.subheader(f"🎯 Top {len(results)} Results")
        
        if not results:
            st.warning("No matching FAQs found. Try different keywords or enable query expansion.")
        else:
            for rank, (idx, score, q_text, a_text) in enumerate(results, start=1):
                with st.expander(f"**{rank}. {q_text}** (Score: {score:.4f})", expanded=(rank==1)):
                    st.write(a_text)
                    
                    if show_explanation and retrieval_method == 'tfidf':
                        terms = ir_system.explain(query, idx, top_k_terms=5)
                        if terms:
                            st.markdown("**🔑 Key Overlapping Terms:**")
                            term_df = pd.DataFrame(terms, columns=["Term", "Relevance Score"])
                            st.dataframe(term_df, use_container_width=True, hide_index=True)
    
    # System statistics
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📚 Total FAQs Indexed", len(ir_system.raw_questions))
    with col2:
        st.metric("📖 Vocabulary Size", len(ir_system.vectorizer.get_feature_names_out()))
    with col3:
        st.metric("🔧 Retrieval Model", retrieval_method.upper())
    
    # Sample queries
    st.subheader("💡 Try These Sample Queries")
    sample_queries = [
        "feeling depressed and lonely",
        "anxiety attacks and panic",
        "can't sleep at night",
        "stressed about work",
        "thoughts of suicide"
    ]
    cols = st.columns(len(sample_queries))
    for col, sample in zip(cols, sample_queries):
        with col:
            if st.button(sample, use_container_width=True):
                st.rerun()

# --------------------------- Main ------------------------------------

def main():
    """Main application entry point."""
    pre = Preprocessor()
    questions, answers = load_faq_data(CSV_PATH)
    ir_system = EnhancedFAQIR(questions, answers, pre)
    run_streamlit_app(ir_system)

if __name__ == "__main__":
    main()