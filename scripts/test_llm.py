#!/usr/bin/env python3
"""Test LLM service.

Run with: uv run python scripts/test_llm.py

Prerequisites:
- For Ollama: Install Ollama and run 'ollama pull llama3.2:3b'
- For Groq: Set GROQ_API_KEY in .env file
"""

from knowledge_base_rag.core.llm import LLMService


def main():
    print("=" * 60)
    print("🤖 LLM Service Test")
    print("=" * 60)

    # Initialize service
    service = LLMService()
    info = service.get_model_info()

    print(f"\n📋 Configuration:")
    print(f"   Provider: {info['provider']}")
    print(f"   Model: {info['model_name']}")
    print(f"   Available: {info['available']}")

    if not info['available']:
        if info['provider'] == 'Ollama':
            print("\n⚠️  Ollama not available!")
            print("   1. Install Ollama: https://ollama.ai/")
            print("   2. Start Ollama")
            print(f"   3. Pull model: ollama pull {info['model_name']}")
        else:
            print("\n⚠️  Groq not configured!")
            print("   1. Get API key: https://console.groq.com/")
            print("   2. Add to .env: GROQ_API_KEY=your-key")
        return

    # Test completion
    print("\n🧪 Testing completion...")
    prompt = "What is machine learning in one sentence?"
    print(f"   Prompt: {prompt}")

    try:
        response = service.complete(prompt)
        print(f"   Response: {response[:200]}...")
        print("\n✅ LLM service is working!")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()


