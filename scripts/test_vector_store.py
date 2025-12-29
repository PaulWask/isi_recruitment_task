#!/usr/bin/env python3
"""Test Vector Store (Qdrant).

Run with: uv run python scripts/test_vector_store.py
"""

from llama_index.core import Document

from knowledge_base_rag.core.llm import get_llm
from knowledge_base_rag.storage.embeddings import get_embed_model
from knowledge_base_rag.storage.vector_store import VectorStoreManager


def main():
    print("=" * 60)
    print("🗄️  Vector Store (Qdrant) Test")
    print("=" * 60)

    # Initialize embedding model
    print("\n📋 Loading embedding model...")
    embed_model = get_embed_model()
    print(f"   ✅ Model loaded: {embed_model.model_name}")

    # Initialize vector store
    print("\n📋 Initializing vector store...")
    manager = VectorStoreManager(
        collection_name="test_collection",
        embed_model=embed_model,
    )

    # Health check
    healthy = manager.health_check()
    print(f"   ✅ Qdrant healthy: {healthy}")

    # Create test documents
    print("\n📄 Creating test documents...")
    test_docs = [
        Document(
            text="Machine learning is a subset of artificial intelligence.",
            metadata={"source": "test1.txt", "topic": "ML"}
        ),
        Document(
            text="Deep learning uses neural networks with many layers.",
            metadata={"source": "test2.txt", "topic": "DL"}
        ),
        Document(
            text="Natural language processing enables computers to understand text.",
            metadata={"source": "test3.txt", "topic": "NLP"}
        ),
        Document(
            text="Computer vision allows machines to interpret images.",
            metadata={"source": "test4.txt", "topic": "CV"}
        ),
        Document(
            text="Reinforcement learning trains agents through rewards.",
            metadata={"source": "test5.txt", "topic": "RL"}
        ),
    ]
    print(f"   Created {len(test_docs)} test documents")

    # Create index
    print("\n🔨 Creating index...")
    index = manager.create_index(test_docs, show_progress=False)
    print("   ✅ Index created")

    # Get stats
    stats = manager.get_stats()
    print(f"\n📊 Collection stats:")
    print(f"   - Exists: {stats.get('exists', False)}")
    print(f"   - Vectors: {stats.get('vectors_count', 'N/A')}")
    print(f"   - Dimension: {stats.get('vector_size', 'N/A')}")
    print(f"   - Status: {stats.get('status', 'N/A')}")

    # Test query
    print("\n🔍 Testing search...")
    query = "What is deep learning?"
    print(f"   Query: '{query}'")

    # Use our configured LLM (Ollama), not the default OpenAI
    llm = get_llm()
    query_engine = index.as_query_engine(
        similarity_top_k=2,
        llm=llm,
    )
    response = query_engine.query(query)

    print(f"\n   Response: {str(response)[:200]}...")

    # Show source nodes
    print(f"\n   📚 Retrieved sources:")
    for i, node in enumerate(response.source_nodes, 1):
        score = node.score if hasattr(node, 'score') else 'N/A'
        source = node.metadata.get('source', 'unknown')
        print(f"      {i}. [{score:.3f}] {source}: {node.text[:50]}...")

    # Cleanup test collection
    print("\n🧹 Cleaning up test collection...")
    manager.delete_collection()
    print("   ✅ Test collection deleted")

    print("\n" + "=" * 60)
    print("✅ Vector Store test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()


