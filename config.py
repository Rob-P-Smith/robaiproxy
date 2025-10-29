"""
Configuration management for Research Proxy.

Loads all configuration from environment variables (.env file).
Provides centralized access to settings with validation and defaults.
"""

import os
import sys
import logging
from typing import Optional
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()


class Config:
    """Centralized configuration class with validation."""

    # ========================================================================
    # Model Backend Configuration
    # ========================================================================
    # Research model endpoint (used by research agent)
    RESEARCH_MODEL_URL: str = os.getenv("RESEARCH_MODEL_URL", "http://localhost:8078/v1")

    # Legacy aliases for backward compatibility
    VLLM_BASE_URL: str = RESEARCH_MODEL_URL  # Alias to RESEARCH_MODEL_URL
    VLLM_BACKEND_URL: str = RESEARCH_MODEL_URL.replace("/v1", "")  # Without /v1 suffix
    VLLM_TIMEOUT: int = int(os.getenv("VLLM_TIMEOUT", "300"))

    # ========================================================================
    # MCP Server Configuration (mcpragcrawl4ai)
    # ========================================================================
    REST_API_URL: str = os.getenv("REST_API_URL", "http://localhost:8080/api/v1")
    REST_API_KEY: str = os.getenv("REST_API_KEY", "")
    MCP_TIMEOUT: int = int(os.getenv("MCP_TIMEOUT", "60"))

    # ========================================================================
    # External API Keys
    # ========================================================================
    SERPER_API_KEY: str = os.getenv("SERPER_API_KEY", "")
    SERPER_TIMEOUT: int = int(os.getenv("SERPER_TIMEOUT", "30"))

    # ========================================================================
    # Research Queue Limits
    # ========================================================================
    MAX_STANDARD_RESEARCH: int = int(os.getenv("MAX_STANDARD_RESEARCH", "3"))
    MAX_DEEP_RESEARCH: int = int(os.getenv("MAX_DEEP_RESEARCH", "1"))

    # ========================================================================
    # Server Configuration
    # ========================================================================
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8079"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    # ========================================================================
    # Feature Flags
    # ========================================================================
    AUTO_DETECT_MODEL: bool = os.getenv("AUTO_DETECT_MODEL", "true").lower() == "true"
    MODEL_POLL_INTERVAL: int = int(os.getenv("MODEL_POLL_INTERVAL", "2"))

    @classmethod
    def validate(cls) -> None:
        """
        Validate configuration and warn about missing values.

        Raises warnings for missing critical API keys but doesn't fail,
        allowing the service to start for non-research requests.
        """
        warnings = []

        if not cls.REST_API_KEY:
            warnings.append("⚠️  REST_API_KEY not set - MCP tools will fail")

        if not cls.SERPER_API_KEY:
            warnings.append("⚠️  SERPER_API_KEY not set - Web search will fail")

        if warnings:
            print("\n" + "=" * 80)
            print("CONFIGURATION WARNINGS")
            print("=" * 80)
            for warning in warnings:
                print(warning)
            print("=" * 80 + "\n")

    @classmethod
    def display(cls) -> None:
        """Display current configuration (with secrets masked)."""
        print("\n" + "=" * 80)
        print("RESEARCH PROXY CONFIGURATION")
        print("=" * 80)
        print(f"Research Model:      {cls.RESEARCH_MODEL_URL}")
        print(f"MCP Server:          {cls.REST_API_URL}")
        print(f"MCP API Key:         {'***' + cls.REST_API_KEY[-8:] if cls.REST_API_KEY else 'NOT SET'}")
        print(f"Serper API Key:      {'***' + cls.SERPER_API_KEY[-8:] if cls.SERPER_API_KEY else 'NOT SET'}")
        print(f"Standard Research:   Max {cls.MAX_STANDARD_RESEARCH} concurrent")
        print(f"Deep Research:       Max {cls.MAX_DEEP_RESEARCH} concurrent")
        print(f"Server:              {cls.HOST}:{cls.PORT}")
        print(f"Log Level:           {cls.LOG_LEVEL}")
        print(f"Auto-Detect Model:   {cls.AUTO_DETECT_MODEL}")
        print("=" * 80 + "\n")


# Create singleton instance
config = Config()

# Validate on import
config.validate()


# ============================================================================
# Logging Configuration
# ============================================================================

def setup_logging():
    """
    Configure logging for the Research Proxy.

    Sets up dual output:
    - Console: Colored, formatted logs for development
    - File: Detailed logs to proxy.log for production
    """
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    log_file = script_dir / "proxy.log"

    # Create logger
    logger = logging.getLogger("researchProxy")
    logger.setLevel(getattr(logging, config.LOG_LEVEL))

    # Prevent duplicate handlers if setup_logging is called multiple times
    if logger.handlers:
        return logger

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )

    # File handler (detailed logs)
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # Log everything to file
    file_handler.setFormatter(detailed_formatter)

    # Console handler (less verbose)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, config.LOG_LEVEL))
    console_handler.setFormatter(console_formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Setup logging and create global logger instance
logger = setup_logging()


# Log startup configuration
logger.info("=" * 80)
logger.info("RESEARCH PROXY STARTING")
logger.info("=" * 80)
logger.info(f"Research Model:      {config.RESEARCH_MODEL_URL}")
logger.info(f"MCP Server:          {config.REST_API_URL}")
logger.info(f"MCP API Key:         {'***' + config.REST_API_KEY[-8:] if config.REST_API_KEY else 'NOT SET'}")
logger.info(f"Serper API Key:      {'***' + config.SERPER_API_KEY[-8:] if config.SERPER_API_KEY else 'NOT SET'}")
logger.info(f"Standard Research:   Max {config.MAX_STANDARD_RESEARCH} concurrent")
logger.info(f"Deep Research:       Max {config.MAX_DEEP_RESEARCH} concurrent")
logger.info(f"Server:              {config.HOST}:{config.PORT}")
logger.info(f"Log Level:           {config.LOG_LEVEL}")
logger.info(f"Log File:            {Path(__file__).parent / 'proxy.log'}")
logger.info("=" * 80)
