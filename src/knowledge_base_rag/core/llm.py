"""LLM service supporting local (Ollama) and cloud (Groq) providers.

Design decisions:
- Ollama (local): FREE, private, no rate limits, requires local setup
- Groq (cloud): FREE tier (14,400 req/day), fast inference, no GPU needed
- Abstract interface allows seamless switching between providers
- Both support streaming for responsive UI

Why these providers:
1. Ollama (default):
   - Runs completely locally on your machine
   - No data sent to external servers (privacy)
   - No API costs ever
   - Works offline
   - Recommended model: llama3.2:3b (3B params, fits 8GB RAM)

2. Groq (alternative):
   - Fastest LLM inference available (500+ tokens/sec)
   - Free tier: 14,400 requests/day, 6,000 tokens/min
   - No GPU required on your machine
   - Great for demos and testing
   - Recommended model: llama-3.1-70b-versatile

Switching between providers:
    # In .env file:
    LLM_SERVICE=local    # Use Ollama
    LLM_SERVICE=groq     # Use Groq (requires GROQ_API_KEY)
"""

import logging
from typing import Optional

from llama_index.core.llms import LLM

from knowledge_base_rag.core.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Service for LLM interactions with multiple provider support.

    Supports:
    - local: Ollama (free, private, runs locally)
    - groq: Groq Cloud (free tier, fast inference)

    Example:
        # Using default (from config)
        service = LLMService()
        response = service.complete("Explain machine learning")

        # Force specific provider
        service = LLMService(service_type="groq")
    """

    def __init__(
        self,
        service_type: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """Initialize the LLM service.

        Args:
            service_type: "local" (Ollama) or "groq". Defaults to config.
            model_name: Model name override. Defaults to config.
        """
        self.service_type = service_type or settings.llm_service
        self._llm: Optional[LLM] = None

        # Select appropriate model based on service
        if model_name:
            self.model_name = model_name
        elif self.service_type == "groq":
            self.model_name = settings.groq_model
        else:
            self.model_name = settings.llm_model

    @property
    def llm(self) -> LLM:
        """Get or create the LLM instance (lazy loading)."""
        if self._llm is None:
            self._llm = self._create_llm()
        return self._llm

    def _create_llm(self) -> LLM:
        """Create LLM instance based on service type."""
        if self.service_type == "groq":
            return self._create_groq_llm()
        else:
            return self._create_ollama_llm()

    def _create_ollama_llm(self) -> LLM:
        """Create Ollama LLM instance."""
        from llama_index.llms.ollama import Ollama

        logger.info(f"Initializing Ollama LLM: {self.model_name}")

        return Ollama(
            model=self.model_name,
            base_url=settings.ollama_base_url,
            request_timeout=settings.ollama_timeout,
            context_window=4096,
        )

    def _create_groq_llm(self) -> LLM:
        """Create Groq LLM instance."""
        from llama_index.llms.groq import Groq

        # Validate API key
        if not settings.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY required for Groq LLM.\n"
                "Get free key at: https://console.groq.com/"
            )

        logger.info(f"Initializing Groq LLM: {self.model_name}")

        return Groq(
            model=self.model_name,
            api_key=settings.groq_api_key,
        )

    def complete(self, prompt: str) -> str:
        """Generate completion for a prompt.

        Args:
            prompt: Text prompt to complete.

        Returns:
            Generated text response.
        """
        response = self.llm.complete(prompt)
        return str(response)

    def chat(self, message: str, system_prompt: Optional[str] = None) -> str:
        """Generate chat response.

        Args:
            message: User message.
            system_prompt: Optional system instructions.

        Returns:
            Assistant's response.
        """
        from llama_index.core.llms import ChatMessage, MessageRole

        messages = []

        if system_prompt:
            messages.append(
                ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)
            )

        messages.append(
            ChatMessage(role=MessageRole.USER, content=message)
        )

        response = self.llm.chat(messages)
        return str(response.message.content)

    def is_available(self) -> bool:
        """Check if the LLM service is available.

        Returns:
            True if LLM can be reached, False otherwise.
        """
        try:
            if self.service_type == "local":
                # Check if Ollama is running
                import httpx

                response = httpx.get(
                    f"{settings.ollama_base_url}/api/tags",
                    timeout=5.0,
                )
                if response.status_code == 200:
                    # Check if required model is available
                    models = response.json().get("models", [])
                    model_names = [m.get("name", "") for m in models]
                    return any(self.model_name in name for name in model_names)
                return False
            else:
                # For Groq, just check API key exists
                return bool(settings.groq_api_key)

        except Exception as e:
            logger.warning(f"LLM availability check failed: {e}")
            return False

    def get_model_info(self) -> dict:
        """Get information about the LLM configuration.

        Returns:
            Dictionary with model details.
        """
        return {
            "service_type": self.service_type,
            "model_name": self.model_name,
            "provider": "Ollama" if self.service_type == "local" else "Groq",
            "available": self.is_available(),
        }


def get_llm(
    service_type: Optional[str] = None,
    model_name: Optional[str] = None,
) -> LLM:
    """Get LLM instance (convenience function).

    This is the primary way to get an LLM for use with LlamaIndex.

    Args:
        service_type: "local" or "groq". Defaults to config.
        model_name: Model name override.

    Returns:
        Configured LLM instance.

    Example:
        from knowledge_base_rag.core.llm import get_llm

        llm = get_llm()
        # Use with LlamaIndex query engine
        query_engine = index.as_query_engine(llm=llm)
    """
    service = LLMService(service_type, model_name)
    return service.llm


def is_ollama_available() -> bool:
    """Check if Ollama LLM service is available.

    Convenience function to check Ollama availability without
    creating an LLMService instance.

    Returns:
        True if Ollama is running and model is available, False otherwise.
    """
    try:
        import httpx

        # Short timeout - if Ollama doesn't respond in 1.5s on localhost, it's slow/down
        response = httpx.get(
            f"{settings.ollama_base_url}/api/tags",
            timeout=1.5,
        )
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            return any(settings.llm_model in name for name in model_names)
        return False
    except Exception as e:
        logger.warning(f"Ollama availability check failed: {e}")
        return False


