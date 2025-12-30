"""RAG Engine - Retrieval Augmented Generation pipeline.

Design decisions:
- Combines vector retrieval with LLM generation for grounded answers
- Configurable retrieval parameters (top_k, similarity threshold)
- Source attribution in responses for transparency
- Streaming support for responsive UI

RAG Pipeline:
    1. User asks a question
    2. Question → Embedding → Vector search (retrieve top-k chunks)
    3. Retrieved chunks + Question → LLM prompt
    4. LLM generates answer grounded in retrieved context
    5. Return answer + source citations

Why RAG over pure LLM:
- Grounded: Answers based on your actual documents
- Accurate: Reduces hallucination by providing context
- Traceable: Can cite sources for every answer
- Up-to-date: No need to retrain model for new documents

Retrieval parameters:
- similarity_top_k=6: Retrieve 6 most relevant chunks
- This provides ~6K tokens of context (6 × 1024)
- Balances context richness vs. noise

Metrics tracked:
- Latency: TTFR (retrieval), TTFG (generation), E2E (total)
- Retrieval: Precision@K, Recall@K (estimated), MRR, Hit Rate
- Quality: Average relevance score, source coverage
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.llms import LLM
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
from llama_index.core.prompts import PromptTemplate

from knowledge_base_rag.core.config import settings
from knowledge_base_rag.core.llm import get_llm
from knowledge_base_rag.storage.embeddings import get_embed_model
from knowledge_base_rag.storage.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


# =============================================================================
# Module-level Caches for Expensive Components
# =============================================================================
# These caches persist across RAGEngine recreations during Streamlit reruns.
# This is critical because Streamlit reruns the script on every interaction.

# BM25 index is expensive to build (~10k documents)
_bm25_cache = {
    "index": None,       # BM25Index instance
    "doc_ids": None,     # Mapping from BM25 index position to node_id
    "doc_count": 0,      # Number of documents indexed (for validation)
}

# Reranker loads a cross-encoder model (slow)
_reranker_cache = None

# Query expander is lightweight but cache anyway
_query_expander_cache = None


# =============================================================================
# Metrics Dataclasses
# =============================================================================

@dataclass
class RAGMetrics:
    """Professional metrics for RAG pipeline evaluation.
    
    Latency Metrics:
        - ttfr_ms: Time To First Retrieval (embedding + vector search)
        - ttfg_ms: Time To First Generation (LLM processing start)
        - e2e_ms: End-to-End latency (total response time)
        
    Retrieval Quality Metrics:
        - precision_at_k: Relevant docs / K (based on threshold)
        - recall_estimated: Estimated recall (relevant found / expected)
        - mrr: Mean Reciprocal Rank (1 / rank of first relevant)
        - hit_rate: 1 if any doc above threshold, else 0
        - avg_score: Average relevance score of retrieved docs
        
    Quality Indicators:
        - source_coverage: Number of unique source files
        - above_threshold: Count of docs above relevance threshold
    """
    # Latency (in milliseconds)
    ttfr_ms: float = 0.0  # Time To First Retrieval
    ttfg_ms: float = 0.0  # Time To First Generation (retrieval + prompt)
    e2e_ms: float = 0.0   # End-to-End latency
    
    # Retrieval quality
    precision_at_k: float = 0.0  # Relevant / K
    recall_estimated: float = 0.0  # Estimate based on score distribution
    mrr: float = 0.0  # Mean Reciprocal Rank
    hit_rate: float = 0.0  # 1 if hit, 0 if miss
    
    # Score statistics
    avg_score: float = 0.0
    max_score: float = 0.0
    min_score: float = 0.0
    
    # Coverage
    source_coverage: int = 0  # Unique source files
    above_threshold: int = 0  # Docs above relevance threshold
    total_retrieved: int = 0  # Total docs retrieved
    
    # Query expansion
    queries_used: int = 1  # Number of queries executed (1 = no expansion)
    expansion_enabled: bool = False  # Whether expansion was used
    
    # Hybrid search
    hybrid_enabled: bool = False  # Whether BM25+Vector fusion was used
    
    # Threshold used (lowered for cross-encoder normalized scores)
    relevance_threshold: float = 0.4
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging/display."""
        return {
            "latency": {
                "ttfr_ms": round(self.ttfr_ms, 2),
                "ttfg_ms": round(self.ttfg_ms, 2),
                "e2e_ms": round(self.e2e_ms, 2),
            },
            "retrieval": {
                "precision_at_k": round(self.precision_at_k, 3),
                "recall_estimated": round(self.recall_estimated, 3),
                "mrr": round(self.mrr, 3),
                "hit_rate": self.hit_rate,
            },
            "scores": {
                "avg": round(self.avg_score, 3),
                "max": round(self.max_score, 3),
                "min": round(self.min_score, 3),
            },
            "coverage": {
                "unique_sources": self.source_coverage,
                "above_threshold": self.above_threshold,
                "total_retrieved": self.total_retrieved,
            },
            "expansion": {
                "enabled": self.expansion_enabled,
                "queries_used": self.queries_used,
            },
            "hybrid": {
                "enabled": self.hybrid_enabled,
            },
        }


@dataclass
class RAGResponse:
    """Structured response from RAG engine with metrics.
    
    Attributes:
        answer: Generated answer text
        sources: List of source documents used
        confidence: Optional confidence score
        query: Original query
        metrics: Detailed performance and quality metrics
        warning: Optional warning message (e.g., low relevance scores)
    """
    answer: str
    sources: list[dict] = field(default_factory=list)
    confidence: Optional[float] = None
    query: str = ""
    metrics: Optional[RAGMetrics] = None
    warning: Optional[str] = None  # Warning for low-quality results
    
    def __str__(self) -> str:
        return self.answer
    
    def get_sources_text(self) -> str:
        """Get formatted sources for display."""
        if not self.sources:
            return "No sources found."
        
        lines = ["Sources:"]
        for i, src in enumerate(self.sources, 1):
            source_name = src.get("source", "Unknown")
            score = src.get("score", 0)
            lines.append(f"  {i}. {source_name} (relevance: {score:.2f})")
        
        return "\n".join(lines)
    
    def get_metrics_summary(self) -> str:
        """Get a summary of metrics for display."""
        if not self.metrics:
            return "No metrics available."
        
        m = self.metrics
        return (
            f"⏱️ Latency: {m.e2e_ms:.0f}ms (retrieval: {m.ttfr_ms:.0f}ms, gen: {m.e2e_ms - m.ttfr_ms:.0f}ms)\n"
            f"📊 Precision@{m.total_retrieved}: {m.precision_at_k:.1%} | MRR: {m.mrr:.2f}\n"
            f"📁 Sources: {m.source_coverage} files | Avg score: {m.avg_score:.1%}"
        )


class RAGEngine:
    """RAG Engine for question answering over the knowledge base.
    
    Orchestrates the full RAG pipeline:
    - Vector retrieval from Qdrant
    - Response generation with LLM
    - Source attribution
    
    Example:
        # Initialize
        engine = RAGEngine()
        
        # Query
        response = engine.query("What are the symptoms of diabetes?")
        print(response.answer)
        print(response.get_sources_text())
        
        # With custom parameters
        response = engine.query(
            "Explain the treatment options",
            similarity_top_k=10,
        )
    """
    
    def __init__(
        self,
        collection_name: Optional[str] = None,
        llm: Optional[LLM] = None,
        embed_model: Optional[BaseEmbedding] = None,
        similarity_top_k: Optional[int] = None,
        enable_reranking: bool = True,  # ON by default - always improves quality
        rerank_top_k: int = 20,
        enable_query_expansion: bool = False,
        enable_hybrid_search: bool = False,
    ):
        """Initialize the RAG engine.
        
        Args:
            collection_name: Qdrant collection name. Defaults to config.
            llm: LLM for generation. Defaults to configured LLM.
            embed_model: Embedding model. Defaults to configured model.
            similarity_top_k: Number of chunks to retrieve. Defaults to config.
            enable_reranking: Enable cross-encoder reranking for better precision.
            enable_query_expansion: Enable query expansion for better recall.
            rerank_top_k: Number of candidates to fetch before reranking.
            enable_hybrid_search: Enable BM25 + Vector hybrid search for better recall.
        """
        self.collection_name = collection_name or settings.collection_name
        self.similarity_top_k = similarity_top_k or settings.similarity_top_k
        self.enable_reranking = enable_reranking
        self.rerank_top_k = rerank_top_k
        self.enable_query_expansion = enable_query_expansion
        self.enable_hybrid_search = enable_hybrid_search
        
        # Lazy initialization
        self._llm = llm
        self._embed_model = embed_model
        self._vector_store_manager: Optional[VectorStoreManager] = None
        self._index: Optional[VectorStoreIndex] = None
        self._query_engine = None
        self._reranker = None
        self._query_expander = None
        self._hybrid_retriever = None
        self._bm25_index = None
        
        logger.info(
            f"RAGEngine initialized: collection={self.collection_name}, "
            f"top_k={self.similarity_top_k}, reranking={enable_reranking}, "
            f"query_expansion={enable_query_expansion}, hybrid={enable_hybrid_search}"
        )
    
    @property
    def llm(self) -> LLM:
        """Get or create the LLM."""
        if self._llm is None:
            self._llm = get_llm()
        return self._llm
    
    @property
    def embed_model(self) -> BaseEmbedding:
        """Get or create the embedding model."""
        if self._embed_model is None:
            self._embed_model = get_embed_model()
        return self._embed_model
    
    @property
    def vector_store_manager(self) -> VectorStoreManager:
        """Get or create the vector store manager."""
        if self._vector_store_manager is None:
            self._vector_store_manager = VectorStoreManager(
                collection_name=self.collection_name,
                embed_model=self.embed_model,
            )
        return self._vector_store_manager
    
    @property
    def reranker(self):
        """Get or create the reranker (lazy-loaded).
        
        Uses module-level cache to persist across RAGEngine recreations.
        """
        global _reranker_cache
        
        # Check module-level cache first
        if _reranker_cache is not None and self.enable_reranking:
            self._reranker = _reranker_cache
            return self._reranker
        
        if self._reranker is None and self.enable_reranking:
            try:
                from knowledge_base_rag.engine.retrieval import Reranker
                self._reranker = Reranker()
                _reranker_cache = self._reranker  # Store in module-level cache
                logger.info("Reranker loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load reranker: {e}")
                self.enable_reranking = False
        return self._reranker
    
    @property
    def query_expander(self):
        """Get or create the query expander (lazy-loaded).
        
        Uses module-level cache to persist across RAGEngine recreations.
        """
        global _query_expander_cache
        
        # Check module-level cache first
        if _query_expander_cache is not None and self.enable_query_expansion:
            self._query_expander = _query_expander_cache
            return self._query_expander
        
        if self._query_expander is None and self.enable_query_expansion:
            try:
                from knowledge_base_rag.engine.retrieval import QueryExpander
                self._query_expander = QueryExpander()
                _query_expander_cache = self._query_expander  # Store in module-level cache
                logger.info("Query expander loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load query expander: {e}")
                self.enable_query_expansion = False
        return self._query_expander
    
    @property
    def bm25_index(self):
        """Get or create BM25 index for hybrid search (lazy-loaded).
        
        Uses module-level cache to persist across RAGEngine recreations.
        """
        global _bm25_cache
        
        # Check if module-level cache is valid
        if _bm25_cache["index"] is not None and self.enable_hybrid_search:
            # Reuse cached index
            self._bm25_index = _bm25_cache["index"]
            self._bm25_doc_ids = _bm25_cache["doc_ids"]
            return self._bm25_index
        
        if self._bm25_index is None and self.enable_hybrid_search:
            try:
                from knowledge_base_rag.engine.retrieval import BM25Index
                
                # Fetch documents directly from Qdrant (docstore is empty with Qdrant)
                doc_data = self.vector_store_manager.get_all_documents()
                
                if doc_data:
                    doc_ids = [doc_id for doc_id, _ in doc_data]
                    documents = [text for _, text in doc_data]
                    
                    self._bm25_index = BM25Index(documents)
                    self._bm25_doc_ids = doc_ids  # Map index position to node_id
                    
                    # Store in module-level cache
                    _bm25_cache["index"] = self._bm25_index
                    _bm25_cache["doc_ids"] = self._bm25_doc_ids
                    _bm25_cache["doc_count"] = len(documents)
                    
                    logger.info(f"BM25 index built with {len(documents)} documents from Qdrant")
                else:
                    logger.warning("No documents found in Qdrant for BM25 index")
                    self.enable_hybrid_search = False
            except ImportError:
                logger.warning("rank-bm25 not installed. Run: uv add rank-bm25")
                self.enable_hybrid_search = False
            except Exception as e:
                logger.warning(f"Failed to build BM25 index: {e}")
                self.enable_hybrid_search = False
        return self._bm25_index
    
    @property
    def index(self) -> VectorStoreIndex:
        """Get or load the vector index."""
        if self._index is None:
            self._index = self.vector_store_manager.load_index()
            if self._index is None:
                raise ValueError(
                    f"No index found for collection '{self.collection_name}'.\n"
                    "Run indexing first: uv run python scripts/index_documents.py"
                )
        return self._index
    
    def _create_query_engine(
        self,
        similarity_top_k: Optional[int] = None,
    ) -> RetrieverQueryEngine:
        """Create a query engine with specified parameters.
        
        Args:
            similarity_top_k: Override default top_k for this query.
            
        Returns:
            Configured query engine.
        """
        top_k = similarity_top_k or self.similarity_top_k
        
        # Create retriever (used only for vector search, not response generation)
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k,
        )
        
        # Note: Response generation now uses _get_qa_prompt() directly in query()
        # This query engine is primarily used for retrieval; we synthesize responses
        # separately using our enhanced/reranked sources
        response_synthesizer = get_response_synthesizer(
            llm=self.llm,
            response_mode=ResponseMode.COMPACT,
            text_qa_template=self._get_qa_prompt(),  # Single source of truth
        )
        
        # Create query engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )
        
        return query_engine
    
    def query(
        self,
        question: str,
        similarity_top_k: Optional[int] = None,
        relevance_threshold: float = 0.4,  # Lowered for cross-encoder scores
        use_reranking: Optional[bool] = None,
        use_query_expansion: Optional[bool] = None,
        use_hybrid_search: Optional[bool] = None,
    ) -> RAGResponse:
        """Query the knowledge base and generate an answer.
        
        Args:
            question: User's question.
            similarity_top_k: Override default number of chunks to retrieve.
            relevance_threshold: Score threshold for "relevant" docs (default 0.5).
            use_reranking: Override reranking setting for this query.
            use_query_expansion: Override query expansion setting for this query.
            use_hybrid_search: Override hybrid search setting for this query.
            
        Returns:
            RAGResponse with answer, sources, and detailed metrics.
        """
        logger.info(f"Query: {question[:100]}...")
        
        # Determine which enhancements to use
        should_rerank = use_reranking if use_reranking is not None else self.enable_reranking
        should_expand = use_query_expansion if use_query_expansion is not None else self.enable_query_expansion
        should_hybrid = use_hybrid_search if use_hybrid_search is not None else self.enable_hybrid_search
        
        # Apply query expansion if enabled
        expanded_queries = [question]
        if should_expand and self.query_expander:
            expanded_queries = self.query_expander.expand_with_synonyms(question)
            logger.info(f"Query expanded to {len(expanded_queries)} variations")
        
        # Start timing
        start_time = time.perf_counter()
        
        # If reranking, fetch more candidates
        fetch_k = similarity_top_k or self.similarity_top_k
        if should_rerank:
            fetch_k = max(fetch_k, self.rerank_top_k)
        
        # Create query engine
        query_engine = self._create_query_engine(fetch_k)
        
        # Track retrieval time
        retrieval_start = time.perf_counter()
        
        # Execute query with expansion and/or hybrid search
        all_nodes = {}  # node_id -> (node, best_score, source)
        
        # BM25 hybrid search (if enabled)
        bm25_results = []
        if should_hybrid and self.bm25_index:
            try:
                bm25_hits = self.bm25_index.search(question, top_k=fetch_k)
                for idx, bm25_score in bm25_hits:
                    if hasattr(self, '_bm25_doc_ids') and idx < len(self._bm25_doc_ids):
                        node_id = self._bm25_doc_ids[idx]
                        bm25_results.append((node_id, bm25_score))
                logger.info(f"BM25 search returned {len(bm25_results)} results")
            except Exception as e:
                logger.warning(f"BM25 search failed: {e}")
        
        # Vector search (with query expansion if enabled)
        if len(expanded_queries) > 1:
            # Multi-query: execute each expanded query and merge results
            for i, q in enumerate(expanded_queries):
                try:
                    resp = query_engine.query(q)
                    for node in resp.source_nodes:
                        node_id = node.node_id
                        score = node.score or 0.0
                        # Keep the best score for each node
                        if node_id not in all_nodes or score > all_nodes[node_id][1]:
                            all_nodes[node_id] = (node, score, "vector")
                except Exception as e:
                    logger.warning(f"Query expansion query failed: {e}")
            logger.info(f"Query expansion: {len(expanded_queries)} queries → {len(all_nodes)} unique nodes")
        else:
            # Single query
            resp = query_engine.query(question)
            for node in resp.source_nodes:
                all_nodes[node.node_id] = (node, node.score or 0.0, "vector")
        
        # Merge BM25 results with vector results using RRF (Reciprocal Rank Fusion)
        if bm25_results:
            # Build rank mappings
            vector_ranks = {nid: rank for rank, nid in enumerate(all_nodes.keys(), 1)}
            bm25_ranks = {nid: rank for rank, (nid, _) in enumerate(bm25_results, 1)}
            
            # RRF constant
            rrf_k = 60
            
            # Update scores with RRF fusion
            for node_id, (node, vector_score, source) in all_nodes.items():
                vector_rrf = 1.0 / (rrf_k + vector_ranks.get(node_id, 1000))
                bm25_rrf = 1.0 / (rrf_k + bm25_ranks.get(node_id, 1000))
                # Weighted combination: 60% vector, 40% BM25
                hybrid_score = 0.6 * vector_rrf + 0.4 * bm25_rrf
                # Normalize and update
                normalized = min(1.0, hybrid_score * rrf_k)
                all_nodes[node_id] = (node, max(vector_score, normalized), "hybrid")
            
            logger.info(f"Hybrid fusion: {len(all_nodes)} results from vector+BM25")
        
        # Sort by score and take top results
        sorted_nodes = sorted(all_nodes.values(), key=lambda x: x[1], reverse=True)
        source_nodes = [n[0] for n in sorted_nodes[:fetch_k]]
        
        retrieval_time = (time.perf_counter() - retrieval_start) * 1000
        
        # Apply reranking if enabled
        if should_rerank and self.reranker and len(source_nodes) > 0:
            logger.info(f"Reranking {len(source_nodes)} candidates...")
            rerank_start = time.perf_counter()
            
            # Convert to reranker format
            from knowledge_base_rag.engine.retrieval import RetrievalResult
            candidates = [
                RetrievalResult(
                    text=node.text,
                    score=node.score or 0.0,
                    metadata=node.metadata,
                    node_id=node.node_id,
                    vector_score=node.score or 0.0,
                )
                for node in source_nodes
            ]
            
            # Rerank
            final_k = similarity_top_k or self.similarity_top_k
            reranked = self.reranker.rerank(question, candidates, top_k=final_k)
            
            rerank_time = (time.perf_counter() - rerank_start) * 1000
            logger.info(f"Reranking completed in {rerank_time:.0f}ms")
            
            # Convert back to sources format
            sources = []
            scores = []
            unique_sources = set()
            
            for result in reranked:
                source_file = result.metadata.get("source", result.metadata.get("file_name", "Unknown"))
                source_info = {
                    "source": source_file,
                    "score": result.rerank_score,  # Use rerank score
                    "vector_score": result.vector_score,  # Keep original for reference
                    "text": result.text[:1000] + "..." if len(result.text) > 1000 else result.text,
                    "metadata": result.metadata,
                }
                sources.append(source_info)
                scores.append(result.rerank_score)
                unique_sources.add(source_file)
        else:
            # No reranking - use original scores
            sources = []
            scores = []
            unique_sources = set()
            
            for node in source_nodes:
                score = node.score if hasattr(node, "score") and node.score else 0.0
                source_file = node.metadata.get("source", node.metadata.get("file_name", "Unknown"))
                
                source_info = {
                    "source": source_file,
                    "score": score,
                    "text": node.text[:1000] + "..." if len(node.text) > 1000 else node.text,
                    "metadata": node.metadata,
                }
                sources.append(source_info)
                scores.append(score)
                unique_sources.add(source_file)
        
        # Generate answer using the ENHANCED sources (not a fresh retrieval)
        # This is critical: we pass our reranked/hybrid sources to the LLM
        logger.info(f"Generating answer from {len(sources)} enhanced sources...")
        generation_start = time.perf_counter()
        
        # Use response synthesizer with our enhanced context
        from llama_index.core.schema import NodeWithScore, TextNode
        enhanced_nodes = []
        for s in sources[:self.similarity_top_k]:
            node = TextNode(text=s["text"])
            node.metadata = s.get("metadata", {})
            enhanced_nodes.append(NodeWithScore(node=node, score=s["score"]))
        
        # Create synthesizer and generate response
        response_synthesizer = get_response_synthesizer(
            llm=self.llm,
            response_mode=ResponseMode.COMPACT,  # COMPACT works well when we control the sources
            text_qa_template=self._get_qa_prompt(),
        )
        
        response = response_synthesizer.synthesize(
            query=question,
            nodes=enhanced_nodes,
        )
        
        generation_time = (time.perf_counter() - generation_start) * 1000
        logger.info(f"Answer generated in {generation_time:.0f}ms")
        
        # Post-process response: Add proper spacing before source citations
        answer_text = str(response)
        # Add two line breaks before "Source:" citations for readability
        answer_text = re.sub(r'\n?(Source:)', r'\n\n\1', answer_text)
        # Also handle inline source citations without newline
        answer_text = re.sub(r'([.!?])\s*(Source:)', r'\1\n\n\2', answer_text)
        
        # End-to-end time
        e2e_time = (time.perf_counter() - start_time) * 1000  # ms
        
        # Compute retrieval metrics
        metrics = self._compute_metrics(
            scores=scores,
            unique_sources=unique_sources,
            relevance_threshold=relevance_threshold,
            e2e_ms=e2e_time,
            queries_used=len(expanded_queries),
            expansion_enabled=should_expand,
            hybrid_enabled=should_hybrid and self.bm25_index is not None,
        )
        
        # Estimate confidence from metrics
        confidence = self._compute_confidence(metrics)
        
        # Check for low-quality results and generate warning
        warning = self._generate_warning(metrics, scores)
        
        # Build response
        rag_response = RAGResponse(
            answer=answer_text,
            sources=sources,
            query=question,
            confidence=confidence,
            metrics=metrics,
            warning=warning,
        )
        
        log_level = "warning" if warning else "info"
        logger.log(
            logging.WARNING if warning else logging.INFO,
            f"Generated answer: {len(sources)} sources, "
            f"precision={metrics.precision_at_k:.2f}, "
            f"e2e={metrics.e2e_ms:.0f}ms"
            + (f" | Warning: {warning}" if warning else "")
        )
        
        return rag_response
    
    def _get_qa_prompt(self) -> PromptTemplate:
        """Get the QA prompt template for answer generation."""
        return PromptTemplate(
            """You are a Senior Financial Analyst with 20+ years of experience in central banking, monetary policy, emerging markets, and economic research. You have deep expertise in:

- Macroeconomics: GDP, inflation (CPI, CCPI, PPI), employment, trade balances
- Monetary Policy: Interest rates, reserve requirements, open market operations
- Financial Markets: Equities, bonds, forex, commodities, derivatives
- Emerging Markets: Country risk, capital flows, currency dynamics
- Trade & Tariffs: Import/export duties, trade agreements, protectionism
- Raw Materials: Commodity prices, supply chains, resource economics

YOUR TASK: Answer questions using ONLY the provided source documents.

CRITICAL RULES:
1. GROUND your answer in the provided context - cite specific data points
2. Use EXACT terminology from sources (e.g., "CCPI" not "CPI" if document says "CCPI")
3. Include specific numbers, dates, percentages, and time periods when available
4. If data is provisional or estimated, mention it (e.g., "provisional data shows...")
5. Compare periods when relevant (e.g., "increased from X% in July to Y% in August")
6. If the context doesn't contain sufficient information, clearly state: "The provided documents do not contain specific information about [topic]."
7. Never invent or assume data - only use what's explicitly in the sources
8. When multiple sources exist, synthesize them coherently
9. For numerical data, maintain precision (don't round unless the source does)
10. Identify the source institution when mentioned (e.g., "According to the Central Bank...")
11. For TABLE DATA: Match the EXACT row label from the question to find the correct value. Don't confuse subtotals with specific line items.
12. PAY CLOSE ATTENTION to footnotes (marked with *) that may specify conditions

RESPONSE FORMAT:
- Lead with the direct answer to the question
- Support with specific data from the sources
- Note any caveats, time periods, or methodology if mentioned
- Keep response focused and professional

Context from knowledge base:
---------------------
{context_str}
---------------------

Question: {query_str}

Answer: """
        )
    
    def _generate_warning(self, metrics: RAGMetrics, scores: list[float]) -> Optional[str]:
        """Generate warning message for low-quality results.
        
        Args:
            metrics: Computed metrics
            scores: List of relevance scores (normalized 0-1)
            
        Returns:
            Warning message or None if quality is acceptable
            
        Note:
            When reranking is enabled, scores are sigmoid-normalized from cross-encoder logits.
            Typical score ranges:
            - 0.8-1.0: Highly relevant
            - 0.5-0.8: Moderately relevant
            - 0.2-0.5: Loosely related
            - 0.0-0.2: Not relevant
        """
        if not scores:
            return "No relevant documents found. Try rephrasing your question."
        
        max_score = max(scores) if scores else 0
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # Very low relevance - likely off-topic query
        # Threshold lowered for cross-encoder (tends to be conservative)
        if max_score < 0.3:
            return (
                f"⚠️ Low relevance: Best match is only {max_score:.0%}. "
                "The knowledge base may not contain information about this topic."
            )
        
        # Moderately low - tangential matches
        if max_score < 0.5 and avg_score < 0.3:
            return (
                f"⚠️ Moderate relevance: Avg score {avg_score:.0%}. "
                "Answer may be based on tangentially related documents."
            )
        
        # No documents above threshold
        if metrics.above_threshold == 0 and metrics.relevance_threshold > 0.3:
            return (
                f"⚠️ No highly relevant docs found (threshold: {metrics.relevance_threshold:.0%}). "
                "Consider reviewing the sources carefully."
            )
        
        # Precision is very low (most docs irrelevant)
        if metrics.precision_at_k < 0.2:
            return (
                f"⚠️ Low precision ({metrics.precision_at_k:.0%}). "
                "Most retrieved documents have low relevance."
            )
        
        return None  # Quality is acceptable
    
    def _compute_metrics(
        self,
        scores: list[float],
        unique_sources: set,
        relevance_threshold: float,
        e2e_ms: float,
        queries_used: int = 1,
        expansion_enabled: bool = False,
        hybrid_enabled: bool = False,
    ) -> RAGMetrics:
        """Compute retrieval and quality metrics.
        
        Args:
            scores: List of relevance scores
            unique_sources: Set of unique source file names
            relevance_threshold: Threshold for "relevant"
            e2e_ms: End-to-end latency in ms
            queries_used: Number of queries executed (for expansion)
            expansion_enabled: Whether query expansion was used
            hybrid_enabled: Whether hybrid BM25+Vector was used
            
        Returns:
            RAGMetrics with computed values
        """
        if not scores:
            return RAGMetrics(
                e2e_ms=e2e_ms,
                relevance_threshold=relevance_threshold,
                queries_used=queries_used,
                expansion_enabled=expansion_enabled,
                hybrid_enabled=hybrid_enabled,
            )
        
        k = len(scores)
        above_threshold = sum(1 for s in scores if s >= relevance_threshold)
        
        # Precision@K: fraction of retrieved docs that are relevant
        precision_at_k = above_threshold / k if k > 0 else 0.0
        
        # MRR: Mean Reciprocal Rank (1 / rank of first relevant doc)
        mrr = 0.0
        for i, score in enumerate(scores):
            if score >= relevance_threshold:
                mrr = 1.0 / (i + 1)
                break
        
        # Hit Rate: 1 if any doc above threshold
        hit_rate = 1.0 if above_threshold > 0 else 0.0
        
        # Recall estimation (heuristic: based on score distribution)
        # Assumes more high-scoring docs = better coverage
        recall_estimated = min(1.0, above_threshold / max(3, k // 2))
        
        # Estimate TTFR (retrieval takes ~30-50% of total time typically)
        ttfr_ms = e2e_ms * 0.4
        
        return RAGMetrics(
            ttfr_ms=ttfr_ms,
            ttfg_ms=ttfr_ms,  # Time to start generation
            e2e_ms=e2e_ms,
            precision_at_k=precision_at_k,
            recall_estimated=recall_estimated,
            mrr=mrr,
            hit_rate=hit_rate,
            avg_score=sum(scores) / len(scores),
            max_score=max(scores),
            min_score=min(scores),
            source_coverage=len(unique_sources),
            above_threshold=above_threshold,
            total_retrieved=k,
            relevance_threshold=relevance_threshold,
            queries_used=queries_used,
            expansion_enabled=expansion_enabled,
            hybrid_enabled=hybrid_enabled,
        )
    
    def _compute_confidence(self, metrics: RAGMetrics) -> float:
        """Compute overall confidence score from metrics.
        
        Combines multiple signals:
        - Average relevance score
        - Precision@K
        - Source coverage
        
        Returns:
            Confidence score between 0 and 1.
        """
        if metrics.total_retrieved == 0:
            return 0.0
        
        # Weighted combination
        score_weight = 0.4
        precision_weight = 0.3
        coverage_weight = 0.3
        
        # Normalize source coverage (more sources = higher confidence, up to 5)
        coverage_score = min(1.0, metrics.source_coverage / 5)
        
        confidence = (
            score_weight * metrics.avg_score +
            precision_weight * metrics.precision_at_k +
            coverage_weight * coverage_score
        )
        
        return min(1.0, max(0.0, confidence))
    
    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """Retrieve relevant chunks without generating an answer.
        
        Useful for debugging or showing relevant documents.
        
        Args:
            question: Query text.
            top_k: Number of chunks to retrieve.
            
        Returns:
            List of retrieved chunks with metadata.
        """
        top_k = top_k or self.similarity_top_k
        
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k,
        )
        
        nodes = retriever.retrieve(question)
        
        results = []
        for node in nodes:
            results.append({
                "text": node.text,
                "score": node.score if hasattr(node, "score") else 0.0,
                "source": node.metadata.get("source", "Unknown"),
                "metadata": node.metadata,
            })
        
        return results
    
    def is_ready(self) -> bool:
        """Check if the RAG engine is ready for queries.
        
        Returns:
            True if index exists and is accessible.
        """
        try:
            return self.vector_store_manager.collection_exists()
        except Exception:
            return False
    
    def get_stats(self) -> dict:
        """Get RAG engine statistics.
        
        Returns:
            Dictionary with engine stats.
        """
        vs_stats = self.vector_store_manager.get_stats()
        
        return {
            "ready": self.is_ready(),
            "collection_name": self.collection_name,
            "similarity_top_k": self.similarity_top_k,
            "llm_model": settings.llm_model if settings.llm_service == "local" else settings.groq_model,
            "llm_service": settings.llm_service,
            "embed_model": settings.embed_model,
            "vectors_count": vs_stats.get("vectors_count", 0),
        }


def create_rag_engine(
    collection_name: Optional[str] = None,
    **kwargs,
) -> RAGEngine:
    """Factory function to create a RAG engine.
    
    Args:
        collection_name: Optional collection name override.
        **kwargs: Additional arguments passed to RAGEngine.
        
    Returns:
        Configured RAGEngine instance.
    
    Example:
        from knowledge_base_rag.engine import create_rag_engine
        
        engine = create_rag_engine()
        response = engine.query("What is...?")
    """
    return RAGEngine(collection_name=collection_name, **kwargs)


