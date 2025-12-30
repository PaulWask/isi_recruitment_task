"""Advanced Retrieval Strategies for RAG.

This module provides professional retrieval enhancements:

1. Hybrid Search (BM25 + Vector)
   - Combines keyword matching (BM25) with semantic search (vectors)
   - Better for queries with specific terms + semantic meaning
   - Typically improves recall by 15-20%

2. Reranking (Cross-Encoder)
   - Re-scores retrieved documents with a more accurate model
   - Slower but significantly more accurate than bi-encoder
   - Typically improves precision by 20-25%

3. Query Expansion
   - Generates alternative query phrasings
   - Helps capture different ways to express the same concept

Usage:
    from knowledge_base_rag.engine.retrieval import HybridRetriever, Reranker
    
    # Hybrid search
    retriever = HybridRetriever(index, embed_model)
    results = retriever.retrieve(query, top_k=10)
    
    # Rerank results
    reranker = Reranker()
    reranked = reranker.rerank(query, results, top_k=5)
"""

import logging
from dataclasses import dataclass
from typing import Optional, List, Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieval result with metadata."""
    text: str
    score: float
    metadata: dict
    node_id: str = ""
    
    # Detailed scores for hybrid search
    vector_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float = 0.0


class Reranker:
    """Cross-encoder reranker for improving retrieval precision.
    
    Uses a cross-encoder model to re-score query-document pairs.
    Cross-encoders are more accurate than bi-encoders but slower.
    
    Best used as a second stage after initial retrieval:
    1. Retrieve top-K candidates with fast bi-encoder (e.g., K=20)
    2. Rerank to get top-N best matches (e.g., N=5)
    
    Example:
        reranker = Reranker()
        
        # Initial retrieval
        candidates = retriever.retrieve(query, top_k=20)
        
        # Rerank to get best 5
        best = reranker.rerank(query, candidates, top_k=5)
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
    ):
        """Initialize the reranker.
        
        Args:
            model_name: Cross-encoder model to use. Options:
                - "cross-encoder/ms-marco-MiniLM-L-6-v2" (fast, good quality)
                - "cross-encoder/ms-marco-MiniLM-L-12-v2" (slower, better)
                - "BAAI/bge-reranker-base" (excellent quality)
            device: Device to run on ("cpu", "cuda", "mps"). Auto-detects if None.
        """
        self.model_name = model_name
        self._model = None
        self._device = device
        
        logger.info(f"Reranker initialized: model={model_name}")
    
    @property
    def model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                # Only pass device if explicitly set
                if self._device:
                    self._model = CrossEncoder(self.model_name, device=self._device)
                else:
                    self._model = CrossEncoder(self.model_name)
                logger.info(f"Loaded reranker model: {self.model_name}")
            except ImportError:
                logger.error("sentence-transformers required for reranking")
                raise
        return self._model
    
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """Rerank results using cross-encoder.
        
        Args:
            query: The search query
            results: List of retrieval results to rerank
            top_k: Number of top results to return (None = all)
            
        Returns:
            Reranked list of results with updated scores (normalized to 0-1)
        """
        if not results:
            return []
        
        # Prepare query-document pairs as list of lists (required by CrossEncoder)
        pairs = [[query, r.text] for r in results]
        
        # Get cross-encoder scores (raw logits, can be negative: typically -10 to +10)
        raw_scores = self.model.predict(pairs)
        
        # CRITICAL: Normalize scores to 0-1 range using sigmoid
        # Raw cross-encoder scores are logits, NOT probabilities!
        # Without normalization, a score of -1.7 would show as "-170%" in UI
        normalized_scores = 1.0 / (1.0 + np.exp(-np.array(raw_scores)))
        
        # Update results with normalized rerank scores
        reranked = []
        for result, raw_score, norm_score in zip(results, raw_scores, normalized_scores):
            result.rerank_score = float(norm_score)  # Store normalized (0-1)
            result.score = float(norm_score)  # Use normalized as primary score
            reranked.append(result)
        
        # Sort by rerank score (descending)
        reranked.sort(key=lambda x: x.rerank_score, reverse=True)
        
        # Return top_k if specified
        if top_k is not None:
            reranked = reranked[:top_k]
        
        best_score = reranked[0].rerank_score if reranked else 0
        logger.info(f"Reranked {len(results)} → {len(reranked)}, best={best_score:.1%}")
        
        return reranked


class BM25Index:
    """BM25 index for keyword-based retrieval.
    
    BM25 (Best Matching 25) is a bag-of-words retrieval function.
    Excellent for exact keyword matching, complements semantic search.
    """
    
    def __init__(self, documents: Optional[List[str]] = None):
        """Initialize BM25 index.
        
        Args:
            documents: List of document texts to index
        """
        self._bm25 = None
        self._documents = []
        self._tokenized_docs = []
        
        if documents:
            self.index(documents)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        # Lowercase and split on non-alphanumeric
        import re
        tokens = re.findall(r'\w+', text.lower())
        return tokens
    
    def index(self, documents: List[str]) -> None:
        """Index documents for BM25 search.
        
        Args:
            documents: List of document texts
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.error("rank-bm25 required for BM25 indexing")
            raise
        
        self._documents = documents
        self._tokenized_docs = [self._tokenize(doc) for doc in documents]
        self._bm25 = BM25Okapi(self._tokenized_docs)
        
        logger.info(f"BM25 index built with {len(documents)} documents")
    
    def search(self, query: str, top_k: int = 10) -> List[tuple]:
        """Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (doc_index, score) tuples
        """
        if self._bm25 is None:
            logger.warning("BM25 index not built")
            return []
        
        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        return [(int(idx), float(scores[idx])) for idx in top_indices]


class HybridRetriever:
    """Hybrid retriever combining BM25 and vector search.
    
    Fusion strategy: Reciprocal Rank Fusion (RRF)
    - Combines rankings from multiple retrieval methods
    - Robust to score scale differences
    - Score = sum(1 / (k + rank)) for each method
    
    Example:
        retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            documents=corpus_texts,
        )
        
        results = retriever.retrieve(query, top_k=10)
    """
    
    def __init__(
        self,
        vector_retriever: Any,
        documents: Optional[List[str]] = None,
        node_texts: Optional[dict] = None,
        alpha: float = 0.5,
        rrf_k: int = 60,
    ):
        """Initialize hybrid retriever.
        
        Args:
            vector_retriever: LlamaIndex vector retriever
            documents: List of document texts for BM25
            node_texts: Dict mapping node_id to text (alternative to documents)
            alpha: Weight for vector vs BM25 (0.5 = equal weight)
            rrf_k: RRF constant (higher = less aggressive fusion)
        """
        self.vector_retriever = vector_retriever
        self.alpha = alpha
        self.rrf_k = rrf_k
        
        # Build BM25 index if documents provided
        self.bm25_index = None
        self._doc_to_node = {}
        
        if documents:
            self.bm25_index = BM25Index(documents)
        elif node_texts:
            docs = list(node_texts.values())
            self.bm25_index = BM25Index(docs)
            self._doc_to_node = {i: node_id for i, node_id in enumerate(node_texts.keys())}
        
        logger.info(f"HybridRetriever initialized: alpha={alpha}")
    
    def _rrf_score(self, ranks: List[int]) -> float:
        """Compute Reciprocal Rank Fusion score."""
        return sum(1.0 / (self.rrf_k + r) for r in ranks if r > 0)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        vector_top_k: Optional[int] = None,
        bm25_top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """Retrieve documents using hybrid search.
        
        Args:
            query: Search query
            top_k: Number of final results to return
            vector_top_k: Number of vector results (default: 2 * top_k)
            bm25_top_k: Number of BM25 results (default: 2 * top_k)
            
        Returns:
            List of RetrievalResult with hybrid scores
        """
        vector_top_k = vector_top_k or top_k * 2
        bm25_top_k = bm25_top_k or top_k * 2
        
        # Vector search
        vector_nodes = self.vector_retriever.retrieve(query)[:vector_top_k]
        
        # BM25 search
        bm25_results = []
        if self.bm25_index:
            bm25_results = self.bm25_index.search(query, top_k=bm25_top_k)
        
        # Build result dict with scores from both methods
        results_dict = {}
        
        # Add vector results
        for rank, node in enumerate(vector_nodes, 1):
            node_id = node.node_id
            results_dict[node_id] = {
                "text": node.text,
                "metadata": node.metadata,
                "vector_rank": rank,
                "vector_score": node.score or 0.0,
                "bm25_rank": 0,
                "bm25_score": 0.0,
            }
        
        # Add BM25 results (if we have node mapping)
        if self._doc_to_node:
            for rank, (doc_idx, score) in enumerate(bm25_results, 1):
                node_id = self._doc_to_node.get(doc_idx)
                if node_id:
                    if node_id in results_dict:
                        results_dict[node_id]["bm25_rank"] = rank
                        results_dict[node_id]["bm25_score"] = score
                    # Note: If not in vector results, we could add it
                    # but we need the text/metadata from vector store
        
        # Compute hybrid scores using RRF
        final_results = []
        for node_id, data in results_dict.items():
            vector_rrf = 1.0 / (self.rrf_k + data["vector_rank"]) if data["vector_rank"] > 0 else 0
            bm25_rrf = 1.0 / (self.rrf_k + data["bm25_rank"]) if data["bm25_rank"] > 0 else 0
            
            # Weighted combination
            hybrid_score = self.alpha * vector_rrf + (1 - self.alpha) * bm25_rrf
            
            # Normalize to 0-1 range (approximate)
            normalized_score = min(1.0, hybrid_score * self.rrf_k)
            
            result = RetrievalResult(
                text=data["text"],
                score=normalized_score,
                metadata=data["metadata"],
                node_id=node_id,
                vector_score=data["vector_score"],
                bm25_score=data["bm25_score"],
            )
            final_results.append(result)
        
        # Sort by hybrid score
        final_results.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(
            f"Hybrid search: {len(vector_nodes)} vector + "
            f"{len(bm25_results)} BM25 → {len(final_results[:top_k])} results"
        )
        
        return final_results[:top_k]


# Convenience function for easy integration
def create_enhanced_retriever(
    index,
    embed_model,
    enable_hybrid: bool = True,
    enable_reranking: bool = True,
    rerank_top_k: int = 20,
    final_top_k: int = 6,
) -> dict:
    """Create an enhanced retrieval pipeline.
    
    Args:
        index: LlamaIndex VectorStoreIndex
        embed_model: Embedding model
        enable_hybrid: Enable hybrid BM25+Vector search
        enable_reranking: Enable cross-encoder reranking
        rerank_top_k: Number of candidates for reranking
        final_top_k: Final number of results
        
    Returns:
        Dict with retriever and reranker components
    """
    from llama_index.core.retrievers import VectorIndexRetriever
    
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=rerank_top_k if enable_reranking else final_top_k,
    )
    
    components = {
        "vector_retriever": vector_retriever,
        "hybrid_retriever": None,
        "reranker": None,
        "config": {
            "enable_hybrid": enable_hybrid,
            "enable_reranking": enable_reranking,
            "rerank_top_k": rerank_top_k,
            "final_top_k": final_top_k,
        }
    }
    
    if enable_reranking:
        components["reranker"] = Reranker()
        logger.info("Reranker enabled")
    
    if enable_hybrid:
        logger.info("Hybrid search available (requires document corpus)")
    
    return components


# =============================================================================
# Query Expansion
# =============================================================================

# Domain-specific synonyms for financial/economic documents
# Comprehensive mapping for acronyms, terminology, and multi-language support
DOMAIN_SYNONYMS = {
    # =========================================================================
    # ACRONYMS - Common financial/economic abbreviations
    # =========================================================================
    "cpi": ["consumer price index", "price index", "inflation index", "cost of living index"],
    "gdp": ["gross domestic product", "economic output", "national output", "domestic product"],
    "gnp": ["gross national product", "national product"],
    "yoy": ["year over year", "year-on-year", "annual change", "yearly comparison"],
    "mom": ["month over month", "month-on-month", "monthly change"],
    "qoq": ["quarter over quarter", "quarterly change", "quarter-on-quarter"],
    "ytd": ["year to date", "year-to-date"],
    "imf": ["international monetary fund"],
    "ecb": ["european central bank"],
    "fed": ["federal reserve", "fed reserve", "federal reserve system"],
    "bsp": ["bangko sentral ng pilipinas", "central bank of the philippines"],
    "nso": ["national statistics office"],
    "psa": ["philippine statistics authority"],
    "dof": ["department of finance"],
    "neda": ["national economic development authority"],
    "bop": ["balance of payments"],
    "fdi": ["foreign direct investment", "direct investment"],
    "ofw": ["overseas filipino workers", "overseas workers"],
    "ppe": ["property plant equipment", "fixed assets"],
    "roa": ["return on assets"],
    "roe": ["return on equity"],
    "eps": ["earnings per share"],
    "p/e": ["price to earnings", "price-to-earnings ratio", "pe ratio"],
    
    # =========================================================================
    # ECONOMIC TERMS - Key concepts with synonyms
    # =========================================================================
    "inflation": ["price increase", "price growth", "rising prices", "cost of living", "price inflation"],
    "deflation": ["price decrease", "falling prices", "price decline"],
    "disinflation": ["slowing inflation", "declining inflation rate"],
    "recession": ["economic downturn", "economic contraction", "negative growth", "economic decline"],
    "growth": ["expansion", "increase", "rise", "economic growth"],
    "stagnation": ["flat growth", "no growth", "zero growth"],
    "stagflation": ["stagnation with inflation", "high inflation low growth"],
    "monetary policy": ["interest rate policy", "central bank policy", "money supply policy"],
    "fiscal policy": ["government spending", "taxation policy", "budget policy"],
    "interest rate": ["lending rate", "borrowing rate", "policy rate"],
    "exchange rate": ["forex rate", "currency rate", "fx rate"],
    "unemployment": ["joblessness", "jobless rate", "unemployment rate"],
    "employment": ["jobs", "hiring", "labor market"],
    "remittances": ["money transfers", "overseas remittances", "ofw remittances"],
    "exports": ["foreign sales", "overseas sales", "outbound trade"],
    "imports": ["foreign purchases", "inbound trade", "overseas purchases"],
    "trade deficit": ["negative trade balance", "trade gap"],
    "trade surplus": ["positive trade balance", "trade excess"],
    "budget deficit": ["fiscal deficit", "government deficit"],
    "budget surplus": ["fiscal surplus", "government surplus"],
    
    # =========================================================================
    # TIME PERIODS - Quarter and date formats
    # =========================================================================
    "q1": ["first quarter", "jan-mar", "january to march", "1st quarter"],
    "q2": ["second quarter", "apr-jun", "april to june", "2nd quarter"],
    "q3": ["third quarter", "jul-sep", "july to september", "3rd quarter"],
    "q4": ["fourth quarter", "oct-dec", "october to december", "4th quarter"],
    "h1": ["first half", "first semester", "1st half"],
    "h2": ["second half", "second semester", "2nd half"],
    "fy": ["fiscal year", "financial year"],
    
    # =========================================================================
    # MULTI-LANGUAGE SUPPORT - Spanish/Tagalog terms (for Philippine documents)
    # =========================================================================
    "presyo": ["price", "prices", "pricing"],
    "ekonomiya": ["economy", "economic"],
    "trabaho": ["employment", "jobs", "work"],
    "sahod": ["salary", "wages", "income"],
    "utang": ["debt", "borrowing", "loan"],
    "buwis": ["tax", "taxes", "taxation"],
}


class QueryExpander:
    """Query expansion for improved retrieval recall.
    
    Supports multiple expansion strategies:
    1. Synonym Expansion: Add domain-specific synonyms
    2. LLM Rewriting: Use LLM to generate query variations
    3. HyDE: Hypothetical Document Embeddings
    
    Example:
        expander = QueryExpander()
        
        # Simple synonym expansion (fast, no LLM)
        variations = expander.expand_with_synonyms("What is CPI in 2024?")
        # → ["What is CPI in 2024?", "What is consumer price index in 2024?"]
        
        # LLM-based expansion (better quality)
        variations = expander.expand_with_llm("CPI trends", llm, num_variations=3)
        
        # HyDE for semantic search (best for vague queries)
        hypothetical_doc = expander.generate_hyde("inflation outlook", llm)
    """
    
    def __init__(
        self,
        custom_synonyms: Optional[dict] = None,
        max_expansions: int = 3,
    ):
        """Initialize query expander.
        
        Args:
            custom_synonyms: Additional domain-specific synonyms
            max_expansions: Maximum number of expanded queries to generate
        """
        self.synonyms = {**DOMAIN_SYNONYMS}
        if custom_synonyms:
            self.synonyms.update(custom_synonyms)
        self.max_expansions = max_expansions
        
        logger.info(f"QueryExpander initialized with {len(self.synonyms)} synonym groups")
    
    def expand_with_synonyms(self, query: str) -> List[str]:
        """Expand query using domain synonyms.
        
        Fast, no LLM required. Good for acronym expansion.
        
        Args:
            query: Original search query
            
        Returns:
            List of expanded queries (including original)
        """
        expanded = [query]
        query_lower = query.lower()
        
        for term, synonyms in self.synonyms.items():
            if term in query_lower:
                # Add variations with synonym substitutions
                for synonym in synonyms[:2]:  # Limit to 2 synonyms per term
                    # Case-insensitive replacement
                    import re
                    pattern = re.compile(re.escape(term), re.IGNORECASE)
                    expanded_query = pattern.sub(synonym, query)
                    if expanded_query != query and expanded_query not in expanded:
                        expanded.append(expanded_query)
                        if len(expanded) >= self.max_expansions:
                            break
                if len(expanded) >= self.max_expansions:
                    break
        
        logger.debug(f"Synonym expansion: {query} → {len(expanded)} variations")
        return expanded
    
    def expand_with_llm(
        self,
        query: str,
        llm: Any,
        num_variations: int = 3,
    ) -> List[str]:
        """Expand query using LLM to generate variations.
        
        Better quality than synonyms, captures semantic variations.
        Adds ~0.5-1s latency per query.
        
        Args:
            query: Original search query
            llm: LlamaIndex LLM instance
            num_variations: Number of variations to generate
            
        Returns:
            List of query variations (including original)
        """
        prompt = f"""Generate {num_variations} alternative ways to ask this question.
Keep the same meaning but use different words and phrasing.
Be concise - each alternative should be a search query, not a full sentence.

Original question: {query}

Alternative phrasings (one per line, no numbering):"""

        try:
            response = llm.complete(prompt)
            lines = response.text.strip().split("\n")
            
            # Parse variations
            variations = [query]  # Always include original
            for line in lines:
                line = line.strip()
                # Remove any numbering or bullet points
                line = line.lstrip("0123456789.-) ")
                if line and line != query and len(line) > 5:
                    variations.append(line)
                    if len(variations) >= num_variations + 1:
                        break
            
            logger.info(f"LLM expansion: {query} → {len(variations)} variations")
            return variations
            
        except Exception as e:
            logger.warning(f"LLM expansion failed: {e}")
            return [query]
    
    def generate_hyde(
        self,
        query: str,
        llm: Any,
        max_tokens: int = 150,
    ) -> str:
        """Generate Hypothetical Document Embedding (HyDE).
        
        Creates a hypothetical answer that can be used for semantic search.
        Particularly effective for vague or conceptual queries.
        
        Args:
            query: Original search query
            llm: LlamaIndex LLM instance
            max_tokens: Maximum length of generated document
            
        Returns:
            Hypothetical document text for embedding
        """
        prompt = f"""Write a short, factual paragraph that directly answers this question.
Write as if you are quoting from an official financial or economic report.
Be specific with numbers and dates where appropriate.
Keep it under {max_tokens} words.

Question: {query}

Answer:"""

        try:
            response = llm.complete(prompt)
            hyde_doc = response.text.strip()
            
            logger.info(f"HyDE generated: {len(hyde_doc)} chars for query: {query[:50]}...")
            return hyde_doc
            
        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}")
            return query  # Fallback to original query
    
    def expand_multi_query(
        self,
        query: str,
        llm: Any,
        num_subqueries: int = 3,
    ) -> List[str]:
        """Break complex query into sub-queries.
        
        Useful for multi-part questions that require information
        from different parts of the knowledge base.
        
        Args:
            query: Original complex query
            llm: LlamaIndex LLM instance
            num_subqueries: Number of sub-queries to generate
            
        Returns:
            List of sub-queries (including original)
        """
        prompt = f"""Break this complex question into {num_subqueries} simpler, focused sub-questions.
Each sub-question should target a specific piece of information needed to answer the main question.

Main question: {query}

Sub-questions (one per line, no numbering):"""

        try:
            response = llm.complete(prompt)
            lines = response.text.strip().split("\n")
            
            subqueries = [query]  # Always include original
            for line in lines:
                line = line.strip()
                line = line.lstrip("0123456789.-) ")
                if line and len(line) > 10:
                    subqueries.append(line)
                    if len(subqueries) >= num_subqueries + 1:
                        break
            
            logger.info(f"Multi-query expansion: {query} → {len(subqueries)} sub-queries")
            return subqueries
            
        except Exception as e:
            logger.warning(f"Multi-query expansion failed: {e}")
            return [query]


class MultiQueryRetriever:
    """Retriever that uses query expansion for better recall.
    
    Combines multiple retrieval strategies:
    1. Original query retrieval
    2. Synonym-expanded queries
    3. LLM-rewritten queries (optional)
    4. HyDE retrieval (optional)
    
    Results are deduplicated and optionally reranked.
    
    Example:
        retriever = MultiQueryRetriever(
            vector_retriever=base_retriever,
            expander=QueryExpander(),
            enable_llm_expansion=True,
        )
        
        results = retriever.retrieve(query, llm=llm, top_k=10)
    """
    
    def __init__(
        self,
        vector_retriever: Any,
        expander: Optional[QueryExpander] = None,
        enable_synonym_expansion: bool = True,
        enable_llm_expansion: bool = False,
        enable_hyde: bool = False,
        reranker: Optional[Reranker] = None,
    ):
        """Initialize multi-query retriever.
        
        Args:
            vector_retriever: Base vector retriever
            expander: QueryExpander instance (created if None)
            enable_synonym_expansion: Use synonym expansion
            enable_llm_expansion: Use LLM for query variations
            enable_hyde: Use HyDE for semantic search
            reranker: Optional reranker for final results
        """
        self.vector_retriever = vector_retriever
        self.expander = expander or QueryExpander()
        self.enable_synonym_expansion = enable_synonym_expansion
        self.enable_llm_expansion = enable_llm_expansion
        self.enable_hyde = enable_hyde
        self.reranker = reranker
        
        logger.info(
            f"MultiQueryRetriever: synonyms={enable_synonym_expansion}, "
            f"llm={enable_llm_expansion}, hyde={enable_hyde}"
        )
    
    def retrieve(
        self,
        query: str,
        llm: Optional[Any] = None,
        top_k: int = 10,
        per_query_k: int = 5,
    ) -> List[RetrievalResult]:
        """Retrieve documents using query expansion.
        
        Args:
            query: Search query
            llm: LLM for expansion (required if LLM expansion enabled)
            top_k: Final number of results
            per_query_k: Results per expanded query
            
        Returns:
            Deduplicated and ranked results
        """
        all_queries = [query]
        
        # Synonym expansion (fast, no LLM)
        if self.enable_synonym_expansion:
            all_queries.extend(self.expander.expand_with_synonyms(query)[1:])
        
        # LLM expansion (requires LLM)
        if self.enable_llm_expansion and llm:
            all_queries.extend(self.expander.expand_with_llm(query, llm)[1:])
        
        # HyDE (requires LLM)
        if self.enable_hyde and llm:
            hyde_doc = self.expander.generate_hyde(query, llm)
            all_queries.append(hyde_doc)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in all_queries:
            q_lower = q.lower().strip()
            if q_lower not in seen:
                seen.add(q_lower)
                unique_queries.append(q)
        
        logger.info(f"Expanded to {len(unique_queries)} unique queries")
        
        # Retrieve for each query
        all_results: dict = {}  # node_id -> RetrievalResult
        
        for i, expanded_query in enumerate(unique_queries):
            try:
                nodes = self.vector_retriever.retrieve(expanded_query)[:per_query_k]
                
                for rank, node in enumerate(nodes):
                    node_id = node.node_id
                    score = node.score or 0.0
                    
                    if node_id in all_results:
                        # Boost score for documents appearing in multiple queries
                        all_results[node_id].score = max(
                            all_results[node_id].score,
                            score * (1.0 - i * 0.1)  # Slight penalty for later queries
                        )
                    else:
                        all_results[node_id] = RetrievalResult(
                            text=node.text,
                            score=score,
                            metadata=node.metadata,
                            node_id=node_id,
                            vector_score=score,
                        )
            except Exception as e:
                logger.warning(f"Retrieval failed for query '{expanded_query[:50]}...': {e}")
        
        # Sort by score
        results = sorted(all_results.values(), key=lambda x: x.score, reverse=True)
        
        # Rerank if available
        if self.reranker and len(results) > top_k:
            results = self.reranker.rerank(query, results, top_k=top_k)
        else:
            results = results[:top_k]
        
        logger.info(f"Multi-query retrieval: {len(unique_queries)} queries → {len(results)} results")
        
        return results

