"""
Comprehensive FAQ IR System - Syllabus-Aligned Implementation
File: comprehensive_faq_ir_system.py

Implemented IR Concepts from Syllabus:
1. Vector Space Model (TF-IDF + Cosine Similarity)
2. Boolean Retrieval Model (AND/OR/NOT)
3. Inverted Index
4. Query Expansion (Global Analysis)
5. Relevance Feedback (Rocchio Algorithm)
6. Zone/Parametric Indexing
7. Evaluation Metrics (Precision, Recall, F1, MAP)
8. Text Preprocessing Pipeline

Instructions:
1. Place CSV at 'Mental_Health_FAQ.csv'
2. Install: pip install pandas scikit-learn nltk streamlit
3. Run: streamlit run comprehensive_faq_ir_system.py
"""

import os
import sys
from typing import List, Tuple, Dict, Set
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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

# --------------------------- Inverted Index --------------------------

class InvertedIndex:
    """
    IR CONCEPT: Inverted Index
    Core data structure mapping terms to document IDs
    Enables efficient Boolean and ranked retrieval
    """
    
    def __init__(self, documents: List[str]):
        self.index = defaultdict(set)  # term -> set of doc_ids
        self.term_frequencies = defaultdict(lambda: defaultdict(int))  # term -> doc_id -> freq
        self.doc_lengths = {}
        self.build_index(documents)
    
    def build_index(self, documents: List[str]):
        """Build inverted index from documents."""
        for doc_id, doc in enumerate(documents):
            terms = doc.split()
            self.doc_lengths[doc_id] = len(terms)
            
            # Build posting lists
            for term in terms:
                self.index[term].add(doc_id)
                self.term_frequencies[term][doc_id] += 1
    
    def get_postings(self, term: str) -> Set[int]:
        """Get posting list for a term."""
        return self.index.get(term, set())
    
    def get_term_frequency(self, term: str, doc_id: int) -> int:
        """Get term frequency in document."""
        return self.term_frequencies.get(term, {}).get(doc_id, 0)
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        return {
            'vocabulary_size': len(self.index),
            'num_documents': len(self.doc_lengths),
            'avg_doc_length': np.mean(list(self.doc_lengths.values())),
            'total_postings': sum(len(postings) for postings in self.index.values())
        }

# --------------------------- Boolean Retrieval -----------------------

class BooleanRetrieval:
    """
    IR CONCEPT: Boolean Model
    Exact-match retrieval using AND, OR, NOT operations
    """
    
    def __init__(self, inverted_index: InvertedIndex):
        self.index = inverted_index
    
    def search_and(self, terms: List[str]) -> Set[int]:
        """Boolean AND: documents containing ALL terms."""
        if not terms:
            return set()
        result = self.index.get_postings(terms[0])
        for term in terms[1:]:
            result = result.intersection(self.index.get_postings(term))
        return result
    
    def search_or(self, terms: List[str]) -> Set[int]:
        """Boolean OR: documents containing ANY term."""
        result = set()
        for term in terms:
            result = result.union(self.index.get_postings(term))
        return result
    
    def search_not(self, include_terms: List[str], exclude_terms: List[str]) -> Set[int]:
        """Boolean NOT: documents with include_terms but NOT exclude_terms."""
        all_docs = set(range(self.index.get_stats()['num_documents']))
        include_docs = self.search_or(include_terms) if include_terms else all_docs
        exclude_docs = self.search_or(exclude_terms)
        return include_docs - exclude_docs
    
    def parse_boolean_query(self, query: str) -> Set[int]:
        """Parse and execute boolean query."""
        query = query.lower()
        
        if ' and ' in query:
            terms = [t.strip() for t in query.split(' and ')]
            return self.search_and(terms)
        elif ' or ' in query:
            terms = [t.strip() for t in query.split(' or ')]
            return self.search_or(terms)
        elif ' not ' in query:
            parts = query.split(' not ')
            include = [parts[0].strip()]
            exclude = [parts[1].strip()]
            return self.search_not(include, exclude)
        else:
            return self.search_or([query])

# --------------------------- Zone Indexing ---------------------------

class ZoneIndex:
    """
    IR CONCEPT: Zone/Parametric Indexing
    Separate indexes for different document fields (zones)
    Enables field-specific searching
    """
    
    def __init__(self, questions: List[str], answers: List[str]):
        self.question_index = InvertedIndex(questions)
        self.answer_index = InvertedIndex(answers)
    
    def search_zone(self, query: str, zone: str = 'both') -> Set[int]:
        """Search in specific zone (question/answer/both)."""
        tokens = query.split()
        
        if zone == 'question':
            result = set()
            for token in tokens:
                result = result.union(self.question_index.get_postings(token))
            return result
        elif zone == 'answer':
            result = set()
            for token in tokens:
                result = result.union(self.answer_index.get_postings(token))
            return result
        else:  # both
            q_results = set()
            a_results = set()
            for token in tokens:
                q_results = q_results.union(self.question_index.get_postings(token))
                a_results = a_results.union(self.answer_index.get_postings(token))
            return q_results.union(a_results)
    
    def weighted_zone_search(self, query: str, question_weight: float = 0.7, 
                           answer_weight: float = 0.3) -> List[Tuple[int, float]]:
        """
        IR CONCEPT: Weighted Zone Scoring
        Combine scores from different zones with weights
        """
        tokens = query.split()
        doc_scores = defaultdict(float)
        
        # Score from question zone
        for token in tokens:
            for doc_id in self.question_index.get_postings(token):
                doc_scores[doc_id] += question_weight
        
        # Score from answer zone
        for token in tokens:
            for doc_id in self.answer_index.get_postings(token):
                doc_scores[doc_id] += answer_weight
        
        # Sort by score
        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked

# --------------------------- Query Expansion -------------------------

class QueryExpander:
    """
    IR CONCEPT: Automatic Global Analysis (Query Expansion)
    Expands queries using thesaurus (WordNet) for better recall
    """
    
    def __init__(self):
        self.use_wordnet = nltk is not None
        if self.use_wordnet:
            ensure_nltk_resources()
    
    def get_synonyms(self, word: str, max_synonyms: int = 3) -> List[str]:
        """Get synonyms for a word using WordNet thesaurus."""
        if not self.use_wordnet:
            return []
        
        synonyms = set()
        try:
            for syn in wn.synsets(word):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ').lower()
                    if synonym != word and len(synonyms) < max_synonyms:
                        synonyms.add(synonym)
        except Exception:
            pass
        return list(synonyms)
    
    def expand_query(self, query: str, max_expansions: int = 2) -> str:
        """Expand query with synonyms."""
        if not self.use_wordnet:
            return query
        
        try:
            tokens = word_tokenize(query.lower())
            expanded_terms = [query]
            
            for token in tokens:
                if len(token) > 3:
                    synonyms = self.get_synonyms(token, max_synonyms=max_expansions)
                    expanded_terms.extend(synonyms[:max_expansions])
            
            return " ".join(expanded_terms)
        except Exception:
            return query

# --------------------------- Rocchio Feedback ------------------------

class RocchioFeedback:
    """
    IR CONCEPT: User Relevance Feedback (Rocchio Algorithm)
    Modifies query vector based on relevant/non-relevant documents
    
    Formula: Q_new = α*Q + β*(1/|Dr|)*Σ(Dr) - γ*(1/|Dnr|)*Σ(Dnr)
    where Dr = relevant docs, Dnr = non-relevant docs
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.75, gamma: float = 0.15):
        self.alpha = alpha  # Original query weight
        self.beta = beta    # Relevant docs weight
        self.gamma = gamma  # Non-relevant docs weight
    
    def modify_query(self, query_vec: np.ndarray, relevant_docs: List[np.ndarray], 
                     non_relevant_docs: List[np.ndarray]) -> np.ndarray:
        """Apply Rocchio algorithm to modify query vector."""
        # Ensure query_vec is 1D
        if query_vec.ndim > 1:
            query_vec = query_vec.flatten()
        
        modified = self.alpha * query_vec
        
        if relevant_docs:
            # Stack all relevant docs and compute mean along axis 0
            relevant_stack = np.vstack([doc.flatten() for doc in relevant_docs])
            relevant_centroid = np.mean(relevant_stack, axis=0)
            modified = modified + self.beta * relevant_centroid
        
        if non_relevant_docs:
            # Stack all non-relevant docs and compute mean along axis 0
            non_relevant_stack = np.vstack([doc.flatten() for doc in non_relevant_docs])
            non_relevant_centroid = np.mean(non_relevant_stack, axis=0)
            modified = modified - self.gamma * non_relevant_centroid
        
        return modified

# --------------------------- Preprocessing ---------------------------

class Preprocessor:
    """
    IR CONCEPT: Text Preprocessing Pipeline
    Steps: Tokenization → Stopword Removal → Lemmatization
    """
    
    def __init__(self):
        self.use_nltk = nltk is not None
        if self.use_nltk:
            ensure_nltk_resources()
            try:
                self.stopwords = set(stopwords.words("english"))
                self.lemmatizer = WordNetLemmatizer()
            except Exception:
                self.use_nltk = False
        
        if not self.use_nltk:
            self.stopwords = set([
                "the", "a", "an", "in", "on", "and", "is", "are", "of",
                "for", "to", "with", "that", "this", "it", "as", "be"
            ])
            self.lemmatizer = None
        
        self.stats = {
            'original_tokens': 0,
            'after_stopwords': 0,
            'after_lemmatization': 0
        }
    
    def preprocess(self, text: str, track_stats: bool = False) -> str:
        """Preprocess text with detailed statistics."""
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
        
        if self.use_nltk:
            try:
                tokens = word_tokenize(text)
                if track_stats:
                    self.stats['original_tokens'] = len(tokens)
                
                tokens = [t for t in tokens if t not in self.stopwords and t.isalpha() and len(t) > 2]
                if track_stats:
                    self.stats['after_stopwords'] = len(tokens)
                
                tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
                if track_stats:
                    self.stats['after_lemmatization'] = len(tokens)
                
                return " ".join(tokens)
            except Exception:
                tokens = [t for t in text.split() if t not in self.stopwords and len(t) > 2]
                return " ".join(tokens)
        else:
            tokens = [t for t in text.split() if t not in self.stopwords and len(t) > 2]
            return " ".join(tokens)
    
    def get_stats(self) -> Dict:
        """Return preprocessing statistics."""
        return self.stats.copy()

# --------------------------- Main IR System --------------------------

class ComprehensiveFAQIR:
    """
    Comprehensive IR system implementing multiple retrieval models
    """
    
    def __init__(self, questions: List[str], answers: List[str], preprocessor: Preprocessor):
        self.raw_questions = questions
        self.raw_answers = answers
        self.preprocessor = preprocessor
        self.query_expander = QueryExpander()
        
        # Preprocess
        print("Preprocessing documents...")
        self.processed_questions = [self.preprocessor.preprocess(q) for q in self.raw_questions]
        self.processed_answers = [self.preprocessor.preprocess(a) for a in self.raw_answers]
        
        # Build Inverted Index
        print("Building inverted index...")
        self.inverted_index = InvertedIndex(self.processed_questions)
        
        # Build Zone Index
        print("Building zone index...")
        self.zone_index = ZoneIndex(self.processed_questions, self.processed_answers)
        
        # Build Boolean retrieval
        self.boolean_retrieval = BooleanRetrieval(self.inverted_index)
        
        # Build TF-IDF Vector Space Model
        print("Building TF-IDF vector space model...")
        self.vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.processed_questions)
        
        # Rocchio feedback
        self.rocchio = RocchioFeedback()
        
        self.retrieval_stats = {
            'original_query': '',
            'expanded_query': '',
            'preprocessing_steps': {},
            'retrieval_model': '',
            'num_results': 0
        }
    
    def search_boolean(self, query: str) -> List[Tuple[int, float, str, str]]:
        """
        IR MODEL: Boolean Retrieval
        Exact-match retrieval without ranking
        """
        processed_query = self.preprocessor.preprocess(query)
        doc_ids = self.boolean_retrieval.parse_boolean_query(processed_query)
        results = [(int(i), 1.0, self.raw_questions[i], self.raw_answers[i]) 
                   for i in sorted(doc_ids)]
        return results
    
    def search_vector_space(self, query: str, top_n: int = 5) -> List[Tuple[int, float, str, str]]:
        """
        IR MODEL: Vector Space Model with TF-IDF + Cosine Similarity
        Ranked retrieval based on term weights
        """
        processed_query = self.preprocessor.preprocess(query)
        query_vec = self.vectorizer.transform([processed_query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        ranked_idx = np.argsort(scores)[::-1][:top_n]
        results = [(int(i), float(scores[i]), self.raw_questions[i], self.raw_answers[i]) 
                   for i in ranked_idx if scores[i] > 0]
        return results
    
    def search_zone_based(self, query: str, zone: str = 'both', top_n: int = 5) -> List[Tuple[int, float, str, str]]:
        """
        IR CONCEPT: Zone/Parametric Indexing
        Search in specific document fields
        """
        processed_query = self.preprocessor.preprocess(query)
        
        if zone == 'weighted':
            ranked = self.zone_index.weighted_zone_search(processed_query)
            results = [(int(doc_id), float(score), self.raw_questions[doc_id], self.raw_answers[doc_id]) 
                      for doc_id, score in ranked[:top_n]]
        else:
            doc_ids = self.zone_index.search_zone(processed_query, zone)
            results = [(int(i), 1.0, self.raw_questions[i], self.raw_answers[i]) 
                      for i in sorted(doc_ids)[:top_n]]
        
        return results
    
    def search_with_expansion(self, query: str, top_n: int = 5, method: str = 'vector') -> List[Tuple[int, float, str, str]]:
        """
        Search with automatic query expansion (global analysis)
        """
        self.retrieval_stats['original_query'] = query
        self.retrieval_stats['retrieval_model'] = method.upper()
        
        # Expand query
        expanded_query = self.query_expander.expand_query(query)
        self.retrieval_stats['expanded_query'] = expanded_query
        
        # Track preprocessing
        processed = self.preprocessor.preprocess(expanded_query, track_stats=True)
        self.retrieval_stats['preprocessing_steps'] = self.preprocessor.get_stats()
        
        # Retrieve
        if method == 'boolean':
            results = self.search_boolean(expanded_query)[:top_n]
        elif method == 'zone':
            results = self.search_zone_based(expanded_query, zone='weighted', top_n=top_n)
        else:
            results = self.search_vector_space(expanded_query, top_n)
        
        self.retrieval_stats['num_results'] = len(results)
        
        # Fallback if no results
        if not results:
            if method == 'boolean':
                results = self.search_boolean(query)[:top_n]
            elif method == 'zone':
                results = self.search_zone_based(query, zone='weighted', top_n=top_n)
            else:
                results = self.search_vector_space(query, top_n)
        
        return results
    
    def search_with_rocchio(self, query: str, relevant_ids: List[int], 
                           non_relevant_ids: List[int], top_n: int = 5) -> List[Tuple[int, float, str, str]]:
        """
        IR CONCEPT: Relevance Feedback using Rocchio Algorithm
        User marks documents as relevant/non-relevant to refine search
        """
        processed_query = self.preprocessor.preprocess(query)
        query_vec = self.vectorizer.transform([processed_query]).toarray()
        
        relevant_docs = [self.tfidf_matrix[i].toarray() for i in relevant_ids]
        non_relevant_docs = [self.tfidf_matrix[i].toarray() for i in non_relevant_ids]
        
        modified_query_vec = self.rocchio.modify_query(query_vec[0], relevant_docs, non_relevant_docs)
        
        scores = cosine_similarity([modified_query_vec], self.tfidf_matrix).flatten()
        ranked_idx = np.argsort(scores)[::-1][:top_n]
        results = [(int(i), float(scores[i]), self.raw_questions[i], self.raw_answers[i]) 
                   for i in ranked_idx if scores[i] > 0]
        return results
    
    def get_retrieval_stats(self) -> Dict:
        """Return retrieval statistics."""
        return self.retrieval_stats.copy()
    
    def get_index_stats(self) -> Dict:
        """Get inverted index statistics."""
        return self.inverted_index.get_stats()
    
    def explain_scoring(self, query: str, doc_idx: int, top_k_terms: int = 5) -> List[Tuple[str, float]]:
        """Explain TF-IDF scoring for a document."""
        processed_query = self.preprocessor.preprocess(query)
        query_vec = self.vectorizer.transform([processed_query]).toarray().flatten()
        doc_vec = self.tfidf_matrix[doc_idx].toarray().flatten()
        overlap = query_vec * doc_vec
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        top_indices = np.argsort(overlap)[::-1][:top_k_terms]
        return [(feature_names[i], float(overlap[i])) for i in top_indices if overlap[i] > 0]

# --------------------------- Evaluation ------------------------------

def evaluate_system(ir_system: ComprehensiveFAQIR, test_queries: List[Tuple[str, List[int]]], 
                   k: int = 5):
    """
    IR CONCEPT: System Evaluation
    Metrics: Precision@K, Recall@K, F1@K, Mean Average Precision (MAP)
    """
    precisions, recalls, f1_scores, avg_precisions = [], [], [], []
    
    for query, relevant_docs in test_queries:
        results = ir_system.search_vector_space(query, top_n=k)
        retrieved = [r[0] for r in results]
        
        if not retrieved:
            precisions.append(0.0)
            recalls.append(0.0)
            f1_scores.append(0.0)
            avg_precisions.append(0.0)
            continue
        
        # Precision & Recall
        true_positives = len(set(retrieved) & set(relevant_docs))
        precision = true_positives / len(retrieved) if retrieved else 0
        recall = true_positives / len(relevant_docs) if relevant_docs else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        
        # Average Precision
        ap = 0.0
        relevant_found = 0
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant_docs:
                relevant_found += 1
                ap += relevant_found / (i + 1)
        ap = ap / len(relevant_docs) if relevant_docs else 0
        avg_precisions.append(ap)
    
    return {
        'Precision@K': np.mean(precisions),
        'Recall@K': np.mean(recalls),
        'F1@K': np.mean(f1_scores),
        'MAP': np.mean(avg_precisions),
        'detailed': list(zip(precisions, recalls, f1_scores, avg_precisions))
    }

# --------------------------- Data Loading ----------------------------

def load_faq_data(csv_path: str) -> Tuple[List[str], List[str]]:
    """Load FAQ data from CSV."""
    if not os.path.exists(csv_path):
        st.error(f"CSV file not found at {csv_path}")
        sys.exit(1)
    
    df = pd.read_csv(csv_path)
    
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

# --------------------------- Streamlit UI ----------------------------

def run_streamlit_app(ir_system: ComprehensiveFAQIR):
    """Streamlit UI showcasing all IR concepts."""
    
    st.set_page_config(page_title="IR System - Syllabus Implementation", layout="wide")
    
    st.title("Information Retrieval System - Syllabus Concepts")
    st.markdown("""
    **Implemented Concepts:** Vector Space Model, Boolean Model, Inverted Index, 
    Zone Indexing, Query Expansion, Rocchio Feedback, TF-IDF, Evaluation Metrics
    """)
    
    # Initialize session state
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""
    if 'current_results' not in st.session_state:
        st.session_state.current_results = []
    if 'relevant_docs' not in st.session_state:
        st.session_state.relevant_docs = []
    if 'non_relevant_docs' not in st.session_state:
        st.session_state.non_relevant_docs = []
    if 'show_refined' not in st.session_state:
        st.session_state.show_refined = False
    if 'refined_results' not in st.session_state:
        st.session_state.refined_results = []
    
    # Sidebar
    with st.sidebar:
        st.header("IR Model Selection")
        
        st.markdown("**Choose Retrieval Model:**")
        retrieval_method = st.radio(
            "Model",
            ["vector", "boolean", "zone"],
            format_func=lambda x: {
                "vector": "Vector Space Model (TF-IDF)",
                "boolean": "Boolean Model (AND/OR/NOT)",
                "zone": "Zone-Based Search"
            }[x],
            help="Compare different IR models"
        )
        
        st.markdown("---")
        st.info(f"**Active Model:** {retrieval_method.upper()}")
        
        if retrieval_method == 'zone':
            st.markdown("**Zone Selection:**")
            zone_type = st.radio(
                "Search In",
                ["question", "answer", "both", "weighted"],
                help="Search in specific document fields"
            )
        else:
            zone_type = None
        
        st.markdown("---")
        st.subheader("Query Options")
        top_k = st.slider("Top K Results", 1, 10, 5)
        use_expansion = st.checkbox("Enable Query Expansion", value=True, 
                                    help="Expands query with synonyms for better recall")
        show_preprocessing = st.checkbox("Show Preprocessing Steps", value=True)
        show_explanation = st.checkbox("Show Scoring Details", value=True)
        
        st.markdown("---")
        st.subheader("Inverted Index Stats")
        idx_stats = ir_system.get_index_stats()
        st.metric("Vocabulary Size", idx_stats['vocabulary_size'])
        st.metric("Total Documents", idx_stats['num_documents'])
        st.metric("Total Postings", idx_stats['total_postings'])
        st.metric("Avg Doc Length", f"{idx_stats['avg_doc_length']:.1f} terms")
        
        st.markdown("---")
        st.subheader("Rocchio Feedback Status")
        st.info(f"✅ Relevant: {len(st.session_state.relevant_docs)}")
        st.info(f"❌ Non-relevant: {len(st.session_state.non_relevant_docs)}")
        
        if st.button("Clear All Feedback"):
            st.session_state.relevant_docs = []
            st.session_state.non_relevant_docs = []
            st.session_state.show_refined = False
            st.session_state.refined_results = []
            st.rerun()
    
    # Main area with tabs
    tab1, tab2 = st.tabs(["🔍 Search", "🔄 Relevance Feedback"])
    
    with tab1:
        st.subheader("Search Interface")
        
        col1, col2 = st.columns([4, 1])
        with col1:
            query = st.text_input(
                "Enter Your Query",
                value="feeling sad and hopeless",
                placeholder="Try: 'fitness', 'stress', 'depression AND anxiety'...",
                help="Enter any health-related query. System will find semantically related FAQs."
            )
        with col2:
            search_btn = st.button("🔍 Search", type="primary", use_container_width=True)
        
        # Show example queries
        with st.expander("💡 Try queries with terms NOT in the dataset"):
            st.write("The system handles out-of-vocabulary words through query expansion:")
            example_cols = st.columns(4)
            example_queries = ["fitness", "exercise", "loneliness", "medication"]
            for col, ex_query in zip(example_cols, example_queries):
                if col.button(ex_query, use_container_width=True):
                    query = ex_query
                    search_btn = True
        
        if search_btn and query:
            # Store query
            st.session_state.current_query = query
            
            # Reset feedback states
            st.session_state.show_refined = False
            st.session_state.refined_results = []
            
            st.markdown("---")
            
            # Perform search
            if use_expansion:
                results = ir_system.search_with_expansion(query, top_n=top_k, method=retrieval_method)
                stats = ir_system.get_retrieval_stats()
                
                # Show if query had out-of-vocabulary terms
                original_terms = set(query.lower().split())
                vocab = set(ir_system.vectorizer.get_feature_names_out())
                oov_terms = original_terms - vocab
                
                if oov_terms:
                    st.info(f"⚠️ Terms not in vocabulary: **{', '.join(oov_terms)}** - Using query expansion to find related documents")
            else:
                if retrieval_method == 'boolean':
                    results = ir_system.search_boolean(query)[:top_k]
                elif retrieval_method == 'zone':
                    results = ir_system.search_zone_based(query, zone=zone_type, top_n=top_k)
                else:
                    results = ir_system.search_vector_space(query, top_n=top_k)
                stats = None
            
            # Show preprocessing pipeline
            if show_preprocessing and stats:
                st.subheader("🔧 IR Processing Pipeline")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**1️⃣ Query Expansion**")
                    st.code(f"Original:\n{stats['original_query']}", language="text")
                    if stats['expanded_query'] != stats['original_query']:
                        st.code(f"Expanded:\n{stats['expanded_query']}", language="text")
                        st.success("✅ Query expanded with synonyms")
                    else:
                        st.info("ℹ️ No synonyms found")
                
                with col2:
                    st.markdown("**2️⃣ Preprocessing**")
                    prep = stats.get('preprocessing_steps', {})
                    st.metric("Original Tokens", prep.get('original_tokens', 0))
                    st.metric("After Stopwords", prep.get('after_stopwords', 0))
                    st.metric("After Lemmatization", prep.get('after_lemmatization', 0))
                
                with col3:
                    st.markdown("**3️⃣ Retrieval**")
                    st.metric("Model", stats['retrieval_model'])
                    st.metric("Results Found", stats['num_results'])
                    st.metric("Similarity", "Cosine")
            
            st.markdown("---")
            st.subheader(f"📋 Search Results ({len(results)} documents)")
            
            if not results:
                st.warning("⚠️ No matching documents found.")
                st.write("**Suggestions:**")
                st.write("- Enable **Query Expansion** to find semantically related documents")
                st.write("- Try broader terms")
                st.write("- Use **Rocchio Feedback** by marking documents as relevant/non-relevant")
            else:
                # Store results
                st.session_state.current_results = results
                
                for rank, (idx, score, q_text, a_text) in enumerate(results, start=1):
                    with st.expander(f"**{rank}. {q_text}** | Score: {score:.4f}", expanded=(rank==1)):
                        st.markdown(f"**Answer:**")
                        st.write(a_text)
                        
                        # Show scoring explanation
                        if show_explanation and retrieval_method == 'vector':
                            terms = ir_system.explain_scoring(query, idx, top_k_terms=5)
                            if terms:
                                st.markdown("---")
                                st.markdown("**🎯 TF-IDF Overlap (Why this matched):**")
                                term_df = pd.DataFrame(terms, columns=["Term", "TF-IDF Score"])
                                st.dataframe(term_df, hide_index=True, use_container_width=True)
    
    # Relevance Feedback Tab
    with tab2:
        st.subheader("Rocchio Relevance Feedback")
        st.info("""
        **How it works:** 
        1. Perform a search in the Search tab
        2. Mark documents as relevant (👍) or not relevant (👎)
        3. Click "Apply Rocchio Algorithm" to refine your search
        
        The system will adjust your query using: Q_new = αQ + β(Relevant Docs) - γ(Non-Relevant Docs)
        """)
        
        # Check if search has been performed
        if not st.session_state.current_query:
            st.warning("⚠️ Please perform a search first in the Search tab.")
        elif not st.session_state.current_results:
            st.warning("⚠️ No search results available. Please search first.")
        else:
            st.markdown(f"**Current Query:** `{st.session_state.current_query}`")
            st.markdown("---")
            
            st.subheader("📋 Mark Documents for Feedback")
            
            # Display results with feedback checkboxes
            for rank, (idx, score, q_text, a_text) in enumerate(st.session_state.current_results, start=1):
                col1, col2, col3 = st.columns([6, 1, 1])
                
                with col1:
                    st.markdown(f"**{rank}. {q_text}**")
                    with st.expander("View Answer"):
                        st.write(a_text)
                    st.caption(f"Relevance Score: {score:.4f}")
                
                with col2:
                    # Check if already marked
                    is_relevant = idx in st.session_state.relevant_docs
                    if st.checkbox("👍 Relevant", key=f"rel_{idx}", value=is_relevant):
                        if idx not in st.session_state.relevant_docs:
                            st.session_state.relevant_docs.append(idx)
                        if idx in st.session_state.non_relevant_docs:
                            st.session_state.non_relevant_docs.remove(idx)
                    else:
                        if idx in st.session_state.relevant_docs:
                            st.session_state.relevant_docs.remove(idx)
                
                with col3:
                    # Check if already marked
                    is_non_relevant = idx in st.session_state.non_relevant_docs
                    if st.checkbox("👎 Not Relevant", key=f"nonrel_{idx}", value=is_non_relevant):
                        if idx not in st.session_state.non_relevant_docs:
                            st.session_state.non_relevant_docs.append(idx)
                        if idx in st.session_state.relevant_docs:
                            st.session_state.relevant_docs.remove(idx)
                    else:
                        if idx in st.session_state.non_relevant_docs:
                            st.session_state.non_relevant_docs.remove(idx)
                
                st.markdown("---")
            
            # Apply Rocchio button
            st.markdown("### Apply Feedback")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Marked Relevant", len(st.session_state.relevant_docs))
            with col2:
                st.metric("Marked Non-Relevant", len(st.session_state.non_relevant_docs))
            
            if st.session_state.relevant_docs or st.session_state.non_relevant_docs:
                if st.button("🔄 Apply Rocchio Algorithm", type="primary", use_container_width=True):
                    with st.spinner("Applying Rocchio algorithm to refine results..."):
                        # Apply Rocchio
                        refined_results = ir_system.search_with_rocchio(
                            st.session_state.current_query,
                            st.session_state.relevant_docs,
                            st.session_state.non_relevant_docs,
                            top_n=top_k
                        )
                        
                        st.session_state.refined_results = refined_results
                        st.session_state.show_refined = True
                    
                    st.success("✅ Rocchio algorithm applied successfully!")
            else:
                st.warning("⚠️ Please mark at least one document as relevant or non-relevant.")
            
            # Show refined results
            if st.session_state.show_refined and st.session_state.refined_results:
                st.markdown("---")
                st.subheader("🎯 Refined Search Results")
                st.success(f"Query refined using {len(st.session_state.relevant_docs)} relevant and {len(st.session_state.non_relevant_docs)} non-relevant documents")
                
                for rank, (idx, score, q_text, a_text) in enumerate(st.session_state.refined_results, start=1):
                    # Check if this is a new result
                    original_indices = [r[0] for r in st.session_state.current_results]
                    is_new = idx not in original_indices
                    
                    with st.expander(
                        f"{'🆕 ' if is_new else ''}{rank}. {q_text} | Score: {score:.4f}",
                        expanded=(rank <= 3)
                    ):
                        if is_new:
                            st.info("🆕 This is a new result found through feedback!")
                        st.markdown("**Answer:**")
                        st.write(a_text)
                        
                        # Show TF-IDF explanation
                        if show_explanation:
                            terms = ir_system.explain_scoring(st.session_state.current_query, idx, top_k_terms=5)
                            if terms:
                                st.markdown("---")
                                st.markdown("**🎯 TF-IDF Overlap:**")
                                term_df = pd.DataFrame(terms, columns=["Term", "TF-IDF Score"])
                                st.dataframe(term_df, hide_index=True, use_container_width=True)
                
                # Option to start new search
                if st.button("🔄 Start New Search", use_container_width=True):
                    st.session_state.current_query = ""
                    st.session_state.current_results = []
                    st.session_state.relevant_docs = []
                    st.session_state.non_relevant_docs = []
                    st.session_state.show_refined = False
                    st.session_state.refined_results = []
                    st.rerun()

# --------------------------- Main ------------------------------------

def main():
    """Main entry point."""
    pre = Preprocessor()
    questions, answers = load_faq_data(CSV_PATH)
    ir_system = ComprehensiveFAQIR(questions, answers, pre)
    run_streamlit_app(ir_system)

if __name__ == "__main__":
    main()