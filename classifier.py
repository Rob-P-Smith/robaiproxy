"""
Intent Classification Module using LLM-based Research Detection

Classifies user queries to determine if research mode should be triggered.
Uses Qwen3-0.6B running on vLLM (port 8092) for fast, accurate classification.

Returns confidence score 0.0-1.0 that query requires research.
Falls back to regex-based classification if LLM unavailable.
"""

import logging
from typing import Optional, Tuple, Dict

logger = logging.getLogger("researchProxy.classifier")


class IntentClassifier:
    """
    LLM-based research intent classifier using Qwen3-0.6B.

    Calls Qwen3-0.6B running on vLLM (port 8092) to get confidence score
    that a query requires research. Falls back to regex if LLM unavailable.

    For multimodal queries, uses VLM to describe images first, then classifies
    based on the combined text + image description.
    """

    # System prompt with 60 examples for LLM classification
    SYSTEM_PROMPT = """/no_think /no_think /no_think /no_think /no_think

Rate your confidence that this is a research task requiring looking up information, gathering data, or analyzing complex topics.

Return ONLY a confidence score between 0.0 and 1.0:
- 0.0 = definitely NOT research (greeting, simple fact, debugging help)
- 1.0 = definitely IS research (needs analysis, comparison, deep explanation)

Examples:

Research queries (high confidence):
"explain kubernetes architecture" -> 0.92
"compare REST vs GraphQL" -> 0.91
"analyze microservices patterns" -> 0.90
"how does OAuth work" -> 0.89
"explain distributed consensus" -> 0.91
"compare SQL vs NoSQL" -> 0.90
"explain event sourcing" -> 0.88
"analyze load balancing methods" -> 0.89
"explain CAP theorem" -> 0.92
"compare message queues" -> 0.90
"explain how Kubernetes handles pod scheduling and resource management" -> 0.93
"compare the performance implications of REST vs GraphQL for mobile apps" -> 0.92
"what are the trade-offs between microservices and monolithic architecture" -> 0.91
"how does OAuth 2.0 authorization code flow work with PKCE" -> 0.90
"analyze the differences between optimistic and pessimistic locking" -> 0.89
"explain how eventual consistency works in distributed databases" -> 0.91
"what's the difference between server-side and client-side rendering" -> 0.90
"compare different approaches to implementing authentication in microservices" -> 0.92
"how does gRPC compare to REST for inter-service communication" -> 0.91
"explain the CAP theorem and its implications for database selection" -> 0.93
"I'm trying to understand the architectural differences between Kubernetes and Docker Swarm, specifically how they handle networking and service discovery. Can you explain?" -> 0.94
"I need to decide between microservices versus monolithic architecture for an e-commerce platform. What are the main trade-offs?" -> 0.93
"Can you help me understand how OAuth 2.0 authorization code flow works with PKCE, and explain why it's more secure than implicit flow?" -> 0.92
"I'm researching state management solutions for React and trying to decide between TanStack Query and Redux. What are the main differences?" -> 0.91
"I want to understand the performance implications of GraphQL versus REST APIs for a mobile app with limited bandwidth." -> 0.90
"Can you explain the difference between optimistic and pessimistic locking in distributed databases?" -> 0.89
"I'm trying to understand how eventual consistency works in DynamoDB. What strategies can I use to handle conflicts?" -> 0.91
"What's the difference between server-side rendering, static site generation, and client-side rendering in Next.js?" -> 0.92
"I'm researching approaches to implementing authentication in microservices. What are the best practices?" -> 0.93
"Can you explain how gRPC compares to REST for microservices? I'm interested in performance and type safety." -> 0.91

Non-research queries (low confidence):
"thanks" -> 0.02
"hello" -> 0.01
"got it" -> 0.03
"ok" -> 0.02
"perfect" -> 0.03
"great" -> 0.02
"bye" -> 0.01
"yes" -> 0.02
"no" -> 0.02
"awesome" -> 0.03
"what is Docker" -> 0.15
"what is REST" -> 0.14
"what is OAuth" -> 0.15
"what is GraphQL" -> 0.14
"what is Kubernetes" -> 0.16
"what is Redis" -> 0.15
"what is Git" -> 0.14
"what is JSON" -> 0.15
"help me debug this" -> 0.18
"fix this error" -> 0.19
"Thanks so much for the detailed explanation" -> 0.03
"Perfect, I appreciate the clarification" -> 0.04
"Got it, I'll try that approach now" -> 0.03
"Awesome, exactly what I needed" -> 0.04
"What does CORS stand for in web development" -> 0.16
"What is TypeScript and how does it relate to JavaScript" -> 0.15
"What is Docker in simple terms" -> 0.16
"Can you take a look at this error message I'm getting" -> 0.19
"I'm seeing some weird behavior, can you help me debug" -> 0.20
"What's wrong with this code snippet" -> 0.18
"Thank you so much for that detailed explanation." -> 0.04
"Perfect, that makes complete sense now." -> 0.03
"What does CORS stand for, and can you give me a basic definition?" -> 0.16
"Can you take a look at this error? I'm not sure what it means." -> 0.21

RETURN ONLY THE CONFIDENCE SCORE AS A NUMBER.

Example: 0.92"""

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        llm_url: str = "http://localhost:8092",
        vllm_url: Optional[str] = None
    ):
        """
        Initialize the intent classifier.

        Args:
            confidence_threshold: Minimum confidence score (0-1) to trigger research
            llm_url: URL of Qwen3-0.6B vLLM instance for classification
            vllm_url: vLLM base URL for image description (required for multimodal)
        """
        self.confidence_threshold = confidence_threshold
        self.llm_url = llm_url
        self.vllm_url = vllm_url
        self.fallback_mode = False

        logger.info(f"IntentClassifier initialized: llm_url={llm_url}, threshold={confidence_threshold}")

    def _truncate_message(self, text: str, max_chars: int = 1000) -> str:
        """
        Truncate long messages to first 500 + last 500 chars.

        Args:
            text: Input text
            max_chars: Maximum characters (default 1000 = 500 + 500)

        Returns:
            Truncated text with middle section replaced by "..."
        """
        if len(text) <= max_chars:
            return text

        half = max_chars // 2
        return f"{text[:half]}\n...\n{text[-half:]}"

    async def _describe_images_with_vllm(self, message_content: list) -> str:
        """
        Use VLM to describe images and extract text from multimodal content.

        Args:
            message_content: OpenAI-style multimodal content list

        Returns:
            Combined description of all images

        Example:
            >>> content = [
            ...     {"type": "text", "text": "Fix this error"},
            ...     {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
            ... ]
            >>> description = await classifier._describe_images_with_vllm(content)
            >>> # "Screenshot showing Python ImportError traceback..."
        """
        import httpx

        if not self.vllm_url:
            logger.warning("VLM URL not configured, cannot describe images")
            return ""

        try:
            # Build VLM request with image description prompt
            vllm_messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Briefly describe this image in 1-2 sentences. If it contains text (like code, error messages, or logs), transcribe the key parts."},
                    *[item for item in message_content if item.get("type") == "image_url"]
                ]
            }]

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.vllm_url}/chat/completions",
                    json={
                        "model": "default",
                        "messages": vllm_messages,
                        "max_tokens": 200,
                        "temperature": 0.1
                    }
                )
                response.raise_for_status()
                result = response.json()

                description = result["choices"][0]["message"]["content"]
                logger.debug(f"Image description from VLM: {description[:100]}...")
                return description

        except Exception as e:
            logger.error(f"Failed to get image description from VLM: {e}", exc_info=True)
            return ""

    async def classify_intent_async(self, message_content, original_text: str = "") -> Tuple[str, float]:
        """
        Classify intent with async support for multimodal content.

        For multimodal queries:
        1. Extracts images and text from content
        2. Calls VLM to describe images
        3. Augments text with image description
        4. Classifies the combined text

        Args:
            message_content: Can be string (text only) or list (multimodal)
            original_text: Original text if message_content is list

        Returns:
            Tuple of (intent_label, confidence_score)

        Examples:
            >>> intent, conf = await classifier.classify_intent_async("What is Python?")
            >>> # ("factual", 0.92)

            >>> content = [
            ...     {"type": "text", "text": "Fix this error"},
            ...     {"type": "image_url", "image_url": {"url": "..."}}
            ... ]
            >>> intent, conf = await classifier.classify_intent_async(content)
            >>> # ("research", 0.88) - because image shows code error
        """
        # Handle simple text case
        if isinstance(message_content, str):
            text = message_content
            has_images = False
        else:
            # Multimodal content - extract text and check for images
            text_parts = []
            has_images = False

            for item in message_content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif item.get("type") == "image_url":
                        has_images = True

            text = original_text or " ".join(text_parts)

        # If has images, get VLM description and augment text
        if has_images and isinstance(message_content, list):
            logger.debug("Multimodal query detected - getting image description from VLM...")
            image_description = await self._describe_images_with_vllm(message_content)

            if image_description:
                # Augment text with image description
                augmented_text = f"{text}\n\nImage content: {image_description}"
                logger.debug(f"Augmented text with image description: {augmented_text[:100]}...")
                text = augmented_text
            else:
                logger.warning("Failed to get image description, classifying with text only")

        # Now classify with full context (text + image description)
        # Truncate to first 500 + last 500 chars
        truncated_text = self._truncate_message(text)

        # Try LLM classification first
        if not self.fallback_mode:
            intent, confidence = await self._classify_with_llm(truncated_text)
            if confidence is not None:
                return (intent, confidence)

        # Fallback to regex
        return self._classify_with_regex(truncated_text)

    async def _classify_with_llm(self, text: str) -> Tuple[str, Optional[float]]:
        """
        Classify using Qwen3-0.6B LLM running on vLLM.

        Args:
            text: The user's message text (already truncated)

        Returns:
            Tuple of (intent_label, confidence_score)
            Returns ("research", None) if LLM fails
        """
        import httpx
        import re

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.llm_url}/v1/chat/completions",
                    json={
                        "model": "Qwen3-0.6",
                        "messages": [
                            {"role": "system", "content": self.SYSTEM_PROMPT},
                            {"role": "user", "content": text}
                        ],
                        "temperature": 0.1,
                        "max_tokens": 10
                    }
                )

                if response.status_code != 200:
                    logger.warning(f"LLM classification failed: HTTP {response.status_code}")
                    self.fallback_mode = True
                    return ("research", None)

                result = response.json()
                content = result["choices"][0]["message"]["content"].strip()

                # Strip <think></think> tags
                clean_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

                # Parse confidence score
                confidence = float(clean_content)
                confidence = max(0.0, min(1.0, confidence))  # Clamp to 0-1

                # Determine intent based on confidence
                # >= 0.5 = research, < 0.5 = not research
                intent = "research" if confidence >= 0.5 else "factual"

                logger.debug(f"LLM classification: '{text[:50]}...' -> {intent} (confidence: {confidence:.3f})")
                return (intent, confidence)

        except Exception as e:
            logger.error(f"LLM classification failed: {e}", exc_info=True)
            logger.warning("Falling back to regex classification")
            self.fallback_mode = True
            return ("research", None)

    def _classify_with_regex(self, text: str) -> Tuple[str, float]:
        """
        Fallback regex-based classification.

        Args:
            text: The user's message text

        Returns:
            Tuple of (intent_label, confidence_score)
        """
        import re

        text_lower = text.lower().strip()

        # Conversational patterns (negative)
        conversational_patterns = [
            r'^(hello|hi|hey|greetings)',
            r'^(thank you|thanks|goodbye|bye)',
            r'^(yes|no|okay|ok|sure)',
        ]

        for pattern in conversational_patterns:
            if re.match(pattern, text_lower):
                return ("conversational", 0.9)

        # Research patterns (high complexity)
        research_patterns = [
            r'^(what|who|when|where|why|how)\s+(is|are|does|did|do|can|will|would|should|could)',
            r'^(explain|describe|tell me about)',
            r'\b(compare|comparison|versus|vs\.?|difference)',
            r'\b(analyze|analysis|evaluate|pros and cons)',
            r'\b(tutorial|guide|how to)',
        ]

        for pattern in research_patterns:
            if re.search(pattern, text_lower):
                # Check if it's a simple factual question
                if len(text_lower.split()) < 5 and '?' not in text:
                    return ("factual", 0.6)
                return ("research", 0.85)

        # Check for image-related keywords
        if any(word in text_lower for word in ['image content:', 'screenshot', 'picture shows', 'error message']):
            return ("vision", 0.8)

        # Default to factual if has question mark
        if '?' in text:
            return ("factual", 0.7)

        # Default to conversational for very short messages
        if len(text_lower.split()) < 3:
            return ("conversational", 0.5)

        # Default to factual
        return ("factual", 0.6)

    async def should_trigger_research(
        self,
        message_content,
        original_text: str = ""
    ) -> Tuple[bool, float, str]:
        """
        Determine if research mode should be triggered.

        Args:
            message_content: Text string or multimodal content list
            original_text: Original text if message_content is list

        Returns:
            Tuple of (should_research, confidence, reason)

        Examples:
            >>> should, conf, reason = await classifier.should_trigger_research("What is Python?")
            >>> # (True, 0.92, "Research intent detected (confidence: 0.92)")

            >>> content = [{"type": "text", "text": "What is this"}, {"type": "image_url", ...}]
            >>> should, conf, reason = await classifier.should_trigger_research(content)
            >>> # Depends on image content - may trigger research if shows error
        """
        # Classify intent (with VLM augmentation if multimodal)
        intent, confidence = await self.classify_intent_async(message_content, original_text)

        # Vision queries never trigger research (they go to VLM directly)
        if intent == "vision":
            return (False, confidence, "Vision query - routing to VLM")

        # Conversational queries never trigger research
        if intent == "conversational":
            return (False, confidence, f"Conversational intent (confidence: {confidence:.2f})")

        # Research intent triggers if above threshold
        if intent == "research" and confidence >= self.confidence_threshold:
            return (True, confidence, f"Research intent detected (confidence: {confidence:.2f})")

        # Factual queries might trigger if very high confidence
        if intent == "factual" and confidence >= 0.9:
            return (True, confidence, f"High-confidence factual query (confidence: {confidence:.2f})")

        # Default: no research
        return (False, confidence, f"{intent.capitalize()} intent below threshold (confidence: {confidence:.2f})")

    def get_stats(self) -> Dict[str, any]:
        """Get classifier statistics."""
        return {
            "classifier_type": "LLM-based (Qwen3-0.6B)",
            "llm_url": self.llm_url,
            "fallback_mode": self.fallback_mode,
            "confidence_threshold": self.confidence_threshold,
            "vllm_configured": self.vllm_url is not None
        }


# Global singleton instance
_classifier: Optional[IntentClassifier] = None


def get_classifier(
    confidence_threshold: float = 0.5,
    llm_url: str = "http://localhost:8092",
    vllm_url: Optional[str] = None
) -> IntentClassifier:
    """
    Get global classifier instance (singleton pattern).

    Args:
        confidence_threshold: Minimum confidence for research trigger (default 0.5)
        llm_url: URL of Qwen3-0.6B vLLM instance (default http://localhost:8092)
        vllm_url: vLLM base URL for image description

    Returns:
        Global IntentClassifier instance

    Usage:
        >>> classifier = get_classifier()
        >>> intent, conf = await classifier.classify_intent_async("What is Python?")
    """
    global _classifier

    if _classifier is None:
        _classifier = IntentClassifier(
            confidence_threshold=confidence_threshold,
            llm_url=llm_url,
            vllm_url=vllm_url
        )

    return _classifier


def initialize_classifier(
    confidence_threshold: float = 0.5,
    llm_url: str = "http://localhost:8092",
    vllm_url: Optional[str] = None
) -> IntentClassifier:
    """
    Initialize and return the global classifier instance.

    Call this on application startup.

    Args:
        confidence_threshold: Minimum confidence for research trigger (default 0.5)
        llm_url: URL of Qwen3-0.6B vLLM instance
        vllm_url: vLLM base URL for image description

    Returns:
        Initialized IntentClassifier instance
    """
    logger.info("🔧 Initializing LLM-based intent classifier...")
    classifier = get_classifier(confidence_threshold, llm_url, vllm_url)
    logger.info(f"✅ Intent classifier ready: {classifier.get_stats()}")
    return classifier
