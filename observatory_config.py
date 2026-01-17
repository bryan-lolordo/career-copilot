"""
Observatory Configuration - Career Copilot
Location: career-copilot/observatory_config.py

This is the ONLY file needed in your application to use Observatory.
All logic lives in the observatory package - this just configures it.

SETUP:
1. Replace your existing observatory_config.py with this file
2. All Career Copilot-specific settings are already configured
3. Import and use: from observatory_config import obs, track_call

Usage:
    from observatory_config import obs, judge, cache, router, prompts, track_llm_call
    
    # In your code
    quality = await judge.maybe_evaluate(operation, prompt, response, client)
    track_llm_call(model, tokens, latency, operation=op, quality_evaluation=quality)

UPDATED: Now supports all 139 fields including conversation linking, model config,
         tool tracking, streaming, error details, experiments, and observability.
         
ADDED (Dec 2025): classify_error() and generate_cache_key() helpers for 
                  error classification and cache key generation.
                  
UPDATED (Dec 2025): Added chat_history_count auto-extraction for conversation tracking.
"""

import os
import re
import logging
from typing import Optional, Dict, List, Any
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# =============================================================================
# OBSERVATORY SDK IMPORTS
# =============================================================================

# Core components
from observatory import (
    # Version
    __version__ as OBSERVATORY_VERSION,
    
    # Core classes
    Observatory,
    ModelProvider,
    AgentRole,
    
    # Main tracking function (renamed to avoid conflict with wrapper)
    track_llm_call as _sdk_track_llm_call,
    
    # Utilities
    estimate_tokens,
)

# Optimization components
from observatory import (
    LLMJudge,
    CacheManager,
    PersistentCacheManager,  # SQLite-backed cross-session cache
    PrefixCacheDetector,
    ModelRouter,
    PromptManager,
    PromptOptimizer,
    BatchDetector,
    ParallelDetector,
    StreamingDetector,
    SequentialCallDetector,
    ContextGrowthDetector,
    TokenEfficiencyDetector,
    BatchProcessor,
    ParallelExecutor,
)

# Data models (for type hints)
from observatory import (
    LLMCall,
    RoutingDecision,
    CacheMetadata,
    QualityEvaluation,
    PromptBreakdown,
    PromptMetadata,
    ModelConfig,
    StreamingMetrics,
    ExperimentMetadata,
    ErrorDetails,
    SemanticCacheResult,
)

# Helper functions
from observatory import (
    create_routing_decision,
    create_cache_metadata,
    create_quality_evaluation,
    create_prompt_metadata,
    create_prompt_breakdown,
    create_semantic_cache_metadata,
)

# Optional: SemanticCache (requires chromadb)
try:
    from observatory import SemanticCache
    SEMANTIC_CACHE_AVAILABLE = True
except ImportError:
    SemanticCache = None
    SEMANTIC_CACHE_AVAILABLE = False
    logger.warning("⚠️ SemanticCache unavailable - install with: pip install chromadb")

# =============================================================================
# IMPORT VALIDATION
# =============================================================================

MIN_REQUIRED_VERSION = "0.1.0"

# Validate critical components
REQUIRED_COMPONENTS = {
    'Observatory': Observatory,
    'track_llm_call': _sdk_track_llm_call,
    'ModelProvider': ModelProvider,
    'LLMJudge': LLMJudge,
    'CacheManager': CacheManager,
}

for name, component in REQUIRED_COMPONENTS.items():
    if component is None:
        raise ImportError(f"❌ Critical component '{name}' not available")

# Version check (warning only, not blocking)
try:
    if OBSERVATORY_VERSION < MIN_REQUIRED_VERSION:
        logger.warning(f"⚠️ Observatory {MIN_REQUIRED_VERSION}+ recommended, found {OBSERVATORY_VERSION}")
except (TypeError, AttributeError):
    logger.warning(f"⚠️ Could not validate Observatory version")

logger.info(f"✅ Observatory SDK loaded (version: {OBSERVATORY_VERSION})")

# =============================================================================
# PHASE CONFIGURATION
# =============================================================================
# Controls whether optimizations are detected (baseline) or applied (optimized)
#
# BASELINE MODE (default):
#   - Track all metrics with full 139-field schema
#   - Detect optimization opportunities (cache hits, routing, compression)
#   - Store opportunities in metadata for analysis
#   - NO changes to application behavior
#
# OPTIMIZED MODE:
#   - Apply all detected optimizations
#   - Two-tier caching (SQLite exact + ChromaDB semantic)
#   - Model routing (complexity-based)
#   - Prompt compression (simple/medium/complex variants)
#   - Token efficiency (operation-specific max_tokens)
#
# Set via environment variable or .env file:
#   OBSERVATORY_PHASE=baseline   (default - detect only)
#   OBSERVATORY_PHASE=optimized  (apply optimizations)
#
# Or run with: OBSERVATORY_PHASE=optimized python your_app.py

CURRENT_PHASE = os.getenv("OBSERVATORY_PHASE", "baseline")

# Validate phase
if CURRENT_PHASE not in ("baseline", "optimized"):
    logger.warning(f"⚠️ Invalid OBSERVATORY_PHASE '{CURRENT_PHASE}', defaulting to 'baseline'")
    CURRENT_PHASE = "baseline"

# Phase descriptions for clarity
PHASE_DESCRIPTIONS = {
    "baseline": "Tracking metrics and detecting optimization opportunities (no changes applied)",
    "optimized": "Applying optimizations (caching, routing, compression, token efficiency)",
}

logger.info(f"🎚️ Observatory Phase: {CURRENT_PHASE.upper()}")
logger.info(f"   {PHASE_DESCRIPTIONS[CURRENT_PHASE]}")

# =============================================================================
# PROJECT CONFIGURATION
# =============================================================================
# Universal settings - customize via environment variables or .env file

# Project identification
PROJECT_NAME = os.getenv("PROJECT_NAME", "Career Copilot")

# Model configuration
DEFAULT_PROVIDER = ModelProvider(os.getenv("MODEL_PROVIDER", "azure"))
DEFAULT_MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") or os.getenv("DEFAULT_MODEL", "gpt-4o-mini")

# Database configuration
if 'DATABASE_URL' in os.environ:
    # Use explicit DATABASE_URL if provided
    OBSERVATORY_DB_PATH = os.environ['DATABASE_URL'].replace('sqlite:///', '')
else:
    # Default: observatory.db in parent directory's ai-agent-observatory folder
    db_dir = os.getenv("OBSERVATORY_DB_DIR", os.path.join(os.path.dirname(__file__), "..", "ai-agent-observatory"))
    db_name = os.getenv("OBSERVATORY_DB_NAME", "observatory.db")
    OBSERVATORY_DB_PATH = os.path.abspath(os.path.join(db_dir, db_name))
    os.environ['DATABASE_URL'] = f"sqlite:///{OBSERVATORY_DB_PATH}"

# Ensure database directory exists
Path(OBSERVATORY_DB_PATH).parent.mkdir(parents=True, exist_ok=True)

# Log configuration
logger.info("="*70)
logger.info(f"📦 Observatory Configuration")
logger.info(f"   Project: {PROJECT_NAME}")
logger.info(f"   Provider: {DEFAULT_PROVIDER.value}")
logger.info(f"   Model: {DEFAULT_MODEL}")
logger.info(f"   Database: {OBSERVATORY_DB_PATH}")
logger.info(f"   Phase: {CURRENT_PHASE.upper()}")
logger.info("="*70)

# =============================================================================
# INITIALIZE OBSERVATORY
# =============================================================================

obs = Observatory(
    project_name=PROJECT_NAME,
    enabled=os.getenv("OBSERVATORY_ENABLED", "true").lower() == "true",
)

logger.info(f"✅ Observatory initialized (enabled={obs.collector.enabled})")

# =============================================================================
# CONFIGURE OPTIMIZATION TRACKER
# =============================================================================
# Tracks optimization impact by comparing baseline vs optimized phases.
#
# Automatically aggregates all calls tagged with phase='baseline' vs phase='optimized'
# to measure cost, latency, token, and quality improvements.
#
# Works in both baseline and optimized phases - data is automatically tagged.

from observatory import OptimizationTracker

optimization_tracker = OptimizationTracker(
    observatory=obs,
    enabled=os.getenv("OPTIMIZATION_TRACKER_ENABLED", "true").lower() == "true",
)

logger.info(f"✅ OptimizationTracker configured (enabled={optimization_tracker.enabled})")
if optimization_tracker.enabled:
    logger.info(f"   Database: {optimization_tracker.db_path}")
    logger.info(f"   Tracks phase comparisons automatically")

# =============================================================================
# CONFIGURE LLM JUDGE - CAREER COPILOT DOMAIN
# =============================================================================
# Quality evaluation with LLM-as-judge
# For other projects: Update operations, criteria, and domain_context

judge = LLMJudge(
    observatory=obs,
    
    # Your existing operations (KEEP AS-IS)
    operations={
        "improve_bullet",
        "generate_change_report",
        "deep_analyze_job",
        "deep_analyze_with_guidance",
        "critique_match",
        "streamlit_chat",
        "cli_chat_message",
        "chat",
        "generate_sql",
        "explain_recent_match",  
        "refine_analysis", 
    },
    
    # Your existing skip_operations (KEEP AS-IS)
    skip_operations={
        "job_search",
        "save_jobs",
        "list_resumes",
        "quick_score_job",
        "generate_refinements",
    },
    
    # 100% sampling for both phases to ensure full quality tracking
    sample_rate=1.0,  # Always evaluate quality
    
    # Your existing criteria (KEEP AS-IS)
    criteria={
        "relevance": 0.25,
        "accuracy": 0.25,
        "helpfulness": 0.25,
        "professionalism": 0.15,
        "clarity": 0.10,
    },
    
    # Your existing domain context (KEEP AS-IS)
    domain_context="career advice, resume optimization, and job matching",
    
    # Your existing judge model (KEEP AS-IS)
    judge_model=os.getenv("JUDGE_MODEL", DEFAULT_MODEL),
    
    # Your existing tracking setting (KEEP AS-IS)
    track_judge_calls=True,
    
    # Your existing enabled setting (KEEP AS-IS)
    enabled=os.getenv("JUDGE_ENABLED", "true").lower() == "true",
    
    # NEW: Add these 3 lines (optional but recommended)
    min_confidence=0.7,           # ← NEW: Reject low-confidence evaluations
    max_prompt_chars=2000,        # ← NEW: Increase from default 1000
    max_response_chars=3000,      # ← NEW: Increase from default 1500
)

logger.info(f"✅ LLMJudge configured (enabled={judge.enabled}, sample_rate={judge.sample_rate})")
logger.info(f"   Phase: {CURRENT_PHASE} (tracked in metadata)")  
logger.info(f"   Evaluating: {len(judge.operations)} operations")
logger.info(f"   Skipping: {len(judge.skip_operations)} operations")
logger.info(f"   Min confidence: {judge.min_confidence}") 

# =============================================================================
# CONFIGURE CACHE MANAGER - CAREER COPILOT OPERATIONS
# =============================================================================
# Exact hash-based caching with TTL
# 
# BASELINE MODE: Detects exact match opportunities (logs would-be hits)
# OPTIMIZED MODE: Returns cached responses for exact matches
#
# For other projects: Update operations with appropriate TTL and clustering

cache = CacheManager(
    observatory=obs,
    
    # Career Copilot operations to cache with TTL settings
    operations={
        # Job operations
        "find_jobs": {"ttl": 3600, "normalize": True, "cluster_id": "job_searches"},
        "get_job_details": {"ttl": 1800, "normalize": False, "cluster_id": "job_details"},
        "get_saved_jobs": {"ttl": 600, "normalize": False, "cluster_id": "saved_jobs"},
        
        # Database operations
        "query_database": {"ttl": 300, "normalize": True, "cluster_id": "db_queries"},
        "generate_sql": {"ttl": 3600, "normalize": True, "cluster_id": "sql_generation"},
        
        # Resume operations
        "list_resumes": {"ttl": 600, "normalize": False, "cluster_id": "resume_list"},
        
        # Matching operations
        "resume_job_matching": {"ttl": 3600, "normalize": False, "cluster_id": "resume_matching"},
        "resume_job_deep_analysis": {"ttl": 3600, "normalize": False, "cluster_id": "deep_analysis"},
        
        # Chat operations
        "cli_chat": {"ttl": 1800, "normalize": False, "cluster_id": "cli_chat"},
        "streamlit_chat": {"ttl": 1800, "normalize": False, "cluster_id": "streamlit_chat"},
    },
    
    # Defaults
    default_ttl=int(os.getenv("CACHE_DEFAULT_TTL", "3600")),
    max_entries=int(os.getenv("CACHE_MAX_ENTRIES", "1000")),
    normalize_prompts=os.getenv("CACHE_NORMALIZE", "true").lower() == "true",
    
    # Enable in both phases (behavior differs based on detection_only)
    enabled=os.getenv("CACHE_ENABLED", "true").lower() == "true",
    
    # NEW: Detection-only mode for baseline
    # Baseline: Check cache, track opportunities, DON'T return cached responses
    # Optimized: Return cached responses
    detection_only=(CURRENT_PHASE == "baseline"),
)

logger.info(f"✅ CacheManager configured (enabled={cache.enabled})")
if cache.enabled:
    mode = "detection only (tracking opportunities)" if cache.detection_only else "active caching"
    logger.info(f"   Mode: {mode}")
    logger.info(f"   Caching: {len(cache.operations)} operations")
    logger.info(f"   Default TTL: {cache.default_ttl}s, Max entries: {cache.max_entries}")

# =============================================================================
# CONFIGURE PERSISTENT CACHE - SQLite-backed cross-session caching
# =============================================================================
# Unlike CacheManager (in-memory), PersistentCacheManager stores data in SQLite
# so it survives application restarts. Perfect for:
# - Resume-job match results that shouldn't be recomputed
# - Deep analysis results (expensive to regenerate)
# - Any result where the same resume + job should return the same match
#
# Use case: User matches resume to jobs on Day 1, comes back Day 3 -
# cached results are still available without re-running expensive LLM calls.

persistent_cache = PersistentCacheManager(
    db_path=os.getenv("PERSISTENT_CACHE_PATH", "./cache/career_copilot_cache.db"),
    observatory=obs,

    # Operations that benefit from cross-session persistence
    operations={
        # Match results - expensive to recompute, stable over time
        "quick_score_job": {"ttl": 604800},  # 7 days - fast initial scoring
        "deep_analyze_job": {"ttl": 604800},  # 7 days - detailed analysis

        # Self-improving match iterations - valuable to keep
        "deep_analyze_with_guidance": {"ttl": 604800},  # 7 days
        "refine_analysis": {"ttl": 604800},  # 7 days
        "critique_match": {"ttl": 604800},  # 7 days
    },

    # Defaults
    default_ttl=604800,  # 7 days default for persistent cache
    max_entries=int(os.getenv("PERSISTENT_CACHE_MAX_ENTRIES", "5000")),

    # Enable based on environment
    enabled=os.getenv("PERSISTENT_CACHE_ENABLED", "true").lower() == "true",

    # Detection-only mode for baseline (same pattern as CacheManager)
    detection_only=(CURRENT_PHASE == "baseline"),
)

logger.info(f"✅ PersistentCacheManager configured (enabled={persistent_cache.enabled})")
if persistent_cache.enabled:
    mode = "detection only (tracking opportunities)" if persistent_cache.detection_only else "active caching"
    logger.info(f"   Mode: {mode}")
    logger.info(f"   DB path: {persistent_cache.db_path}")
    logger.info(f"   Operations: {len(persistent_cache.operations)}")
    logger.info(f"   Default TTL: {persistent_cache.default_ttl}s (7 days)")

# =============================================================================
# CONFIGURE PREFIX CACHE DETECTOR - AZURE/ANTHROPIC PREFIX CACHING
# =============================================================================
# Detects opportunities for Azure/Anthropic prefix caching (~50% cost savings on cached prefix)
#
# BASELINE MODE: Tracks prefix reuse and calculates potential savings
# OPTIMIZED MODE: Application code can use Azure/Anthropic prefix caching API
#
# Perfect for large system prompts that rarely change (like Career Copilot's 1,555 token prompt)

prefix_cache = PrefixCacheDetector(
    observatory=obs,
    
    # Track first 500 characters as prefix (captures system prompt start)
    prefix_length=500,
    
    # Only track prompts with 100+ tokens (smaller prompts don't benefit)
    min_prefix_tokens=100,
    
    # Enable in both phases
    enabled=os.getenv("PREFIX_CACHE_ENABLED", "true").lower() == "true",
    
    # Detection-only mode for baseline
    # Baseline: Track prefix reuse, calculate savings, DON'T apply caching
    # Optimized: Application can use Azure/Anthropic prefix caching API
    detection_only=(CURRENT_PHASE == "baseline"),
)

logger.info(f"✅ PrefixCacheDetector configured (enabled={prefix_cache.enabled})")
if prefix_cache.enabled:
    mode = "detection only (tracking opportunities)" if prefix_cache.detection_only else "API integration ready"
    logger.info(f"   Mode: {mode}")
    logger.info(f"   Prefix length: {prefix_cache.prefix_length} chars")
    logger.info(f"   Min tokens: {prefix_cache.min_prefix_tokens}")

# =============================================================================
# CONFIGURE SEMANTIC CACHE - VECTOR-BASED SIMILARITY MATCHING
# =============================================================================
# Unlike CacheManager (exact hash match), SemanticCache uses embeddings to find
# semantically similar prompts. "Find Python jobs" ≈ "Search for Python positions"
#
# BASELINE MODE: Detects semantic similarity opportunities (logs would-be hits)
# OPTIMIZED MODE: Actually returns cached responses
#
# Requires: pip install chromadb
# For other projects: Update operations with domain-specific similarity thresholds

# Store semantic cache in Career Copilot's data folder (not in observatory package)
SEMANTIC_CACHE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "data", "semantic_cache")
)

# Only initialize if ChromaDB is available
if SEMANTIC_CACHE_AVAILABLE:
    semantic_cache = SemanticCache(
        observatory=obs,
        db_path=SEMANTIC_CACHE_PATH,  # Store in project's data folder

        operations={
            # ═══════════════════════════════════════════════════════════════
            # HIGH VALUE - SQL Generation
            # ═══════════════════════════════════════════════════════════════
            "generate_sql": {
                "ttl": 86400,       # 24 hours (SQL patterns are stable)
                "threshold": 0.95,  # 95% - high precision (exact SQL matters)
                "cluster_id": "sql_generation",
            },
            
            # ═══════════════════════════════════════════════════════════════
            # HIGH VALUE - Quick Scoring
            # ═══════════════════════════════════════════════════════════════
            "quick_score_job": {
                "ttl": 3600,        # 1 hour (job relevance can change)
                "threshold": 0.90,  # 90% - jobs in same domain match well
                "cluster_id": "resume_matching",
            },
            
            # ═══════════════════════════════════════════════════════════════
            # MEDIUM VALUE - Deep Analysis
            # ═══════════════════════════════════════════════════════════════
            "deep_analyze_job": {
                "ttl": 3600,        # 1 hour
                "threshold": 0.88,  # 88% - lower threshold for expensive ops
                "cluster_id": "deep_analysis",
            },
        },
        
        # Defaults for any operation not explicitly configured
        default_ttl=int(os.getenv("SEMANTIC_CACHE_DEFAULT_TTL", "3600")),
        default_threshold=float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92")),
        
        # Enable in both modes (behavior differs based on detection_only)
        enabled=os.getenv("SEMANTIC_CACHE_ENABLED", "true").lower() == "true",
        
        # ✅ NEW: Detection-only mode for baseline
        # Baseline: Calculate similarity, track opportunities, DON'T return cached responses
        # Optimized: Return cached responses
        detection_only=(CURRENT_PHASE == "baseline"),
    )
    
    logger.info(f"✅ SemanticCache configured (enabled={semantic_cache.enabled})")
    if semantic_cache.enabled:
        mode = "detection only (tracking opportunities)" if semantic_cache.detection_only else "active caching"
        logger.info(f"   Mode: {mode}")
        logger.info(f"   Semantic matching: {len(semantic_cache.operations)} operations")
        logger.info(f"   Default threshold: {semantic_cache.default_threshold}")
else:
    semantic_cache = None
    logger.info("⏭️  SemanticCache skipped (ChromaDB not available)")

# =============================================================================
# CONFIGURE MODEL ROUTER - CAREER COPILOT ROUTING RULES
# =============================================================================
# Intelligent model selection based on complexity, operation, and token count
#
# BASELINE MODE: Calculates routing opportunities (logs would-be upgrades/downgrades)
# OPTIMIZED MODE: Actually routes to different models
#
# For other projects: Update rules to match your operations and model strategy

router = ModelRouter(
    observatory=obs,
    default_model=DEFAULT_MODEL,
    fallback_model=os.getenv("FALLBACK_MODEL", "gpt-4o-mini"),
    
    # Enable in both phases (behavior differs based on detection_only)
    enabled=os.getenv("ROUTER_ENABLED", "true").lower() == "true",
    
    # ✅ NEW: Detection-only mode for baseline
    # Baseline: Calculate routing decision, log opportunities, return default_model
    # Optimized: Actually return routed model
    detection_only=(CURRENT_PHASE == "baseline"),
    
    # Routing rules for Career Copilot operations (evaluated in order)
    rules=[
        # Simple retrieval operations → cheap model
        {
            "name": "simple_retrieval",
            "operations": ["find_jobs", "list_resumes", "get_job_details", "get_saved_jobs"],
            "model": "gpt-4o-mini",
            "reason": "Simple retrieval - cheap model sufficient",
        },
        
        # Database queries → cheap model
        {
            "name": "db_queries",
            "operations": ["query_database", "generate_sql"],
            "model": "gpt-4o-mini",
            "reason": "SQL generation - structured output",
        },
        
        # Complex analysis → premium model
        {
            "name": "complex_analysis",
            "operations": ["deep_analyze_job", "deep_analyze_with_guidance", "critique_match"],
            "model": "gpt-4o",
            "reason": "Complex analysis requires premium model",
        },
        
        # Self-improving operations → premium model
        {
            "name": "self_improving",
            "operations": ["improve_bullet", "generate_change_report"],
            "model": "gpt-4o",
            "reason": "Quality-critical content generation",
        },
        
        # High complexity → premium model
        {
            "name": "high_complexity",
            "min_complexity": 0.7,
            "model": "gpt-4o",
            "reason": "High complexity score detected",
        },
        
        # Low token count → cheap model
        {
            "name": "short_requests",
            "max_tokens": 500,
            "model": "gpt-4o-mini",
            "reason": "Short request - cheap model sufficient",
        },
    ],
)

logger.info(f"✅ ModelRouter configured (enabled={router.enabled})")
if router.enabled:
    mode = "detection only (tracking opportunities)" if router.detection_only else "active routing"
    logger.info(f"   Mode: {mode}")
    logger.info(f"   Default model: {router.default_model}")
    logger.info(f"   Fallback model: {router.fallback_model}")
    logger.info(f"   Routing rules: {len(router._rules)}")

# =============================================================================
# CONFIGURE PROMPT MANAGER (A/B TESTING)
# =============================================================================
# Template versioning and A/B testing for prompts
# Works identically in both baseline and optimized phases
# For other projects: Register your prompt templates with variants

prompts = PromptManager(observatory=obs)

# Example: Register system prompt with variants for Career Copilot
# Uncomment and customize as needed:
#
# prompts.register(
#     template_id="career_copilot_system",
#     version="2.0.0",
#     content=SYSTEM_PROMPT,  # Your main system prompt
#     variants={
#         "control": SYSTEM_PROMPT,
#         "concise": CONCISE_SYSTEM_PROMPT,
#         "structured": STRUCTURED_SYSTEM_PROMPT,
#     },
#     experiment_id="system_prompt_test_dec_2024",
#     weights={"control": 0.5, "concise": 0.25, "structured": 0.25},
#     description="Testing different system prompt styles for career advice",
# )

logger.info(f"✅ PromptManager configured")
logger.info(f"   Templates registered: {len(prompts.list_templates())}")
logger.info(f"   Note: A/B testing works in both baseline and optimized phases")

# =============================================================================
# PROMPT VARIANTS - CAREER COPILOT
# =============================================================================
# Define 3 prompt variants for compression optimization
# System prompt is 99.8% of input tokens (1,555 avg) - huge optimization opportunity!
#
# For other projects: Replace with your prompt variants

# TODO: Define your actual prompts here
# FULL_SYSTEM_PROMPT = """Your complete 11K token system prompt..."""
# SIMPLE_SYSTEM_PROMPT = """Compressed 50-token version for simple tasks..."""
# MEDIUM_SYSTEM_PROMPT = """Compressed 200-token version for medium tasks..."""

# Prompt variant configuration
PROMPT_VARIANTS = {
    "passthrough": {
        "content": None,  # None = use the default_prompt passed to get_optimized_prompt()
        "max_tokens": None,  # None = no limit override
        "description": "Passthrough - use the plugin's own task-specific prompt (for scoring, SQL, etc.)"
    },
    "simple": {
        "content": "You are a helpful career advisor. Provide concise responses.",  # Placeholder - replace with actual
        "max_tokens": 150,
        "description": "Minimal prompt for simple tasks (quick scoring, lists)"
    },
    "medium": {
        "content": "You are an experienced career advisor specializing in resume optimization and job matching.",  # Placeholder
        "max_tokens": 500,
        "description": "Medium prompt for structured tasks (SQL generation, bullet improvements)"
    },
    "complex": {
        "content": "REPLACE_WITH_FULL_SYSTEM_PROMPT",  # TODO: Use your actual full system prompt
        "max_tokens": 1000,
        "description": "Full system prompt for complex analysis"
    },
}

# Map operations to complexity levels
OPERATION_COMPLEXITY = {
    # Passthrough operations - plugins have their own task-specific prompts
    # These should NOT be replaced by generic prompts
    "quick_score_job": "passthrough",      # ResumeMatchingPlugin - needs JSON scoring format
    "deep_analyze_job": "passthrough",     # ResumeMatchingPlugin - needs semantic analysis format
    "deep_analyze_with_guidance": "passthrough",
    "critique_match": "passthrough",
    "refine_analysis": "passthrough",
    "generate_sql": "passthrough",         # QueryDatabasePlugin - needs schema + SQL syntax
    "query_database": "passthrough",
    "improve_bullet": "passthrough",       # ResumeTailoringPlugin - needs tailoring instructions
    "tailor_resume": "passthrough",
    "generate_change_report": "passthrough",

    # Simple operations (use compressed prompt) - listing/fetching only
    "list_resumes": "simple",
    "find_jobs": "simple",
    "get_job_details": "simple",
    "get_saved_jobs": "simple",

    # Complex operations (use full Career Copilot prompt) - main chat flow
    "streamlit_chat": "complex",
    "cli_chat_message": "complex",
}

logger.info(f"✅ Prompt variants defined")
logger.info(f"   Variants: {', '.join(PROMPT_VARIANTS.keys())}")
logger.info(f"   Operations mapped: {len(OPERATION_COMPLEXITY)}")

# =============================================================================
# CONFIGURE PROMPT OPTIMIZER
# =============================================================================
# Reduces input tokens through prompt compression and operation-specific max_tokens
#
# BASELINE MODE: Detects compression opportunities (logs potential savings)
# OPTIMIZED MODE: Returns compressed prompts based on operation complexity
#
# Expected savings: 77% token reduction on input (1,555 → 350 avg)

prompt_optimizer = PromptOptimizer(
    observatory=obs,
    
    # Prompt variants with max_tokens limits
    prompt_variants=PROMPT_VARIANTS,
    
    # Operation → complexity mapping
    operation_complexity=OPERATION_COMPLEXITY,
    
    # Enable in both phases
    enabled=os.getenv("PROMPT_OPTIMIZER_ENABLED", "true").lower() == "true",
    
    # Detection-only mode for baseline
    # Baseline: Calculate savings, log opportunities, return default prompt
    # Optimized: Return compressed prompt variant
    detection_only=(CURRENT_PHASE == "baseline"),
)

logger.info(f"✅ PromptOptimizer configured (enabled={prompt_optimizer.enabled})")
if prompt_optimizer.enabled:
    mode = "detection only (tracking savings)" if prompt_optimizer.detection_only else "active compression"
    logger.info(f"   Mode: {mode}")
    logger.info(f"   Variants: {len(PROMPT_VARIANTS)}")
    logger.info(f"   Operations: {len(OPERATION_COMPLEXITY)}")

# =============================================================================
# EXECUTION OPTIMIZATION
# =============================================================================
# Detects opportunities for batching, parallelism, and streaming
#
# BASELINE MODE: Identifies patterns and calculates potential savings
# OPTIMIZED MODE: Application code can implement batching/parallel/streaming
#
# Expected savings: 60% latency reduction from batching, 50-70% from parallelism

# Batch Detector - Groups rapid sequential calls
batch_detector = BatchDetector(
    observatory=obs,
    
    # Group calls within 100ms window
    time_window_ms=100,
    
    # At least 2 calls to qualify as batch
    min_batch_size=2,
    
    # Monitor specific operations (None = all operations)
    operations={"quick_score_job", "generate_sql", "find_jobs"},
    
    # Enable in both phases
    enabled=os.getenv("BATCH_DETECTOR_ENABLED", "true").lower() == "true",
    
    # Detection-only mode for baseline
    detection_only=(CURRENT_PHASE == "baseline"),
)

logger.info(f"✅ BatchDetector configured (enabled={batch_detector.enabled})")
if batch_detector.enabled:
    mode = "detection only (tracking opportunities)" if batch_detector.detection_only else "implementation ready"
    logger.info(f"   Mode: {mode}")
    logger.info(f"   Time window: {batch_detector.time_window_ms}ms")
    logger.info(f"   Min batch size: {batch_detector.min_batch_size}")

# Parallel Detector - Identifies independent operations
parallel_detector = ParallelDetector(
    observatory=obs,
    
    # Look for parallelism within 5 second window
    time_window_s=5.0,
    
    # At least 2 calls to parallelize
    min_parallel_count=2,
    
    # Enable in both phases
    enabled=os.getenv("PARALLEL_DETECTOR_ENABLED", "true").lower() == "true",
    
    # Detection-only mode for baseline
    detection_only=(CURRENT_PHASE == "baseline"),
)

logger.info(f"✅ ParallelDetector configured (enabled={parallel_detector.enabled})")
if parallel_detector.enabled:
    mode = "detection only (tracking opportunities)" if parallel_detector.detection_only else "implementation ready"
    logger.info(f"   Mode: {mode}")
    logger.info(f"   Time window: {parallel_detector.time_window_s}s")
    logger.info(f"   Min parallel count: {parallel_detector.min_parallel_count}")

# Streaming Detector - Flags high-latency or large-output calls
streaming_detector = StreamingDetector(
    observatory=obs,
    
    # Flag calls over 2 seconds
    latency_threshold_ms=2000,
    
    # Flag outputs over 500 tokens
    token_threshold=500,
    
    # Monitor specific operations (None = all operations)
    operations={"deep_analyze_job", "deep_analyze_with_guidance", "critique_match", "streamlit_chat"},
    
    # Enable in both phases
    enabled=os.getenv("STREAMING_DETECTOR_ENABLED", "true").lower() == "true",
    
    # Detection-only mode for baseline
    detection_only=(CURRENT_PHASE == "baseline"),
)

logger.info(f"✅ StreamingDetector configured (enabled={streaming_detector.enabled})")
if streaming_detector.enabled:
    mode = "detection only (flagging candidates)" if streaming_detector.detection_only else "implementation ready"
    logger.info(f"   Mode: {mode}")
    logger.info(f"   Latency threshold: {streaming_detector.latency_threshold_ms}ms")
    logger.info(f"   Token threshold: {streaming_detector.token_threshold}")

# =============================================================================
# PROMPT PATTERN DETECTORS (NEW)
# =============================================================================
# Detects context growth and token inefficiency in prompts

# Sequential Call Detector - Identifies sequential patterns over longer timeframes
sequential_detector = SequentialCallDetector(
    observatory=obs,
    
    # Look for patterns within 60 second window
    sequence_window_s=60.0,
    
    # At least 5 sequential calls to qualify
    min_sequence_count=5,
    
    # Monitor specific operations (None = all operations)
    operations={"quick_score_job", "deep_analyze_job", "generate_sql"},
    
    # Enable in both phases
    enabled=os.getenv("SEQUENTIAL_DETECTOR_ENABLED", "true").lower() == "true",
    
    # Detection-only mode for baseline
    detection_only=(CURRENT_PHASE == "baseline"),
)

logger.info(f"✅ SequentialCallDetector configured (enabled={sequential_detector.enabled})")
if sequential_detector.enabled:
    mode = "detection only (tracking patterns)" if sequential_detector.detection_only else "implementation ready"
    logger.info(f"   Mode: {mode}")
    logger.info(f"   Sequence window: {sequential_detector.sequence_window_s}s")
    logger.info(f"   Min sequence count: {sequential_detector.min_sequence_count}")

# =============================================================================
# UNIVERSAL BATCH + PARALLEL EXECUTION
# =============================================================================
# Execute batching and parallel processing with automatic Observatory tracking

batch_processor = BatchProcessor(
    observatory=obs,
    
    # Default batch size (can override per operation)
    default_batch_size=3,
    
    # Track batch creation
    track_batching=True,
    
    # Enable in both phases
    enabled=os.getenv("BATCH_PROCESSOR_ENABLED", "true").lower() == "true",
)

logger.info(f"✅ BatchProcessor configured (enabled={batch_processor.enabled})")
logger.info(f"   Default batch size: {batch_processor.default_batch_size}")
logger.info(f"   Universal batching available for all operations")

parallel_executor = ParallelExecutor(
    observatory=obs,
    
    # Default concurrency limit
    default_max_concurrent=3,
    
    # Semaphore type: 'count' (fixed limit) or 'rate' (requests/second)
    semaphore_type='count',
    
    # Track parallel execution
    track_parallelism=True,
    
    # Enable in both phases
    enabled=os.getenv("PARALLEL_EXECUTOR_ENABLED", "true").lower() == "true",
)

logger.info(f"✅ ParallelExecutor configured (enabled={parallel_executor.enabled})")
logger.info(f"   Default concurrency: {parallel_executor.default_max_concurrent}")
logger.info(f"   Semaphore type: {parallel_executor.semaphore_type}")
logger.info(f"   Universal parallel execution available for all operations")

# Context Growth Detector - Detects when chat history grows too large
context_growth_detector = ContextGrowthDetector(
    observatory=obs,
    
    # Alert when history > 50% of total prompt
    threshold_percentage=50.0,
    
    # Monitor chat operations (None = all operations)
    operations={"streamlit_chat", "cli_chat_message"},
    
    # Enable in both phases
    enabled=os.getenv("CONTEXT_GROWTH_DETECTOR_ENABLED", "true").lower() == "true",
    
    # Detection-only mode for baseline
    detection_only=(CURRENT_PHASE == "baseline"),
)

logger.info(f"✅ ContextGrowthDetector configured (enabled={context_growth_detector.enabled})")
if context_growth_detector.enabled:
    mode = "detection only (tracking growth)" if context_growth_detector.detection_only else "context limiting ready"
    logger.info(f"   Mode: {mode}")
    logger.info(f"   Threshold: {context_growth_detector.threshold_percentage}%")
    logger.info(f"   Monitored operations: {len(context_growth_detector.operations) if context_growth_detector.operations else 'all'}")

# Token Efficiency Detector - Detects inefficient token usage patterns
token_efficiency_detector = TokenEfficiencyDetector(
    observatory=obs,
    
    # Alert when prompt/completion ratio > 50:1
    threshold_ratio=50.0,
    
    # Monitor specific operations (None = all operations)
    operations={"streamlit_chat", "cli_chat_message", "deep_analyze_job"},
    
    # Enable in both phases
    enabled=os.getenv("TOKEN_EFFICIENCY_DETECTOR_ENABLED", "true").lower() == "true",
    
    # Detection-only mode for baseline
    detection_only=(CURRENT_PHASE == "baseline"),
)

logger.info(f"✅ TokenEfficiencyDetector configured (enabled={token_efficiency_detector.enabled})")
if token_efficiency_detector.enabled:
    mode = "detection only (tracking inefficiency)" if token_efficiency_detector.detection_only else "compression ready"
    logger.info(f"   Mode: {mode}")
    logger.info(f"   Threshold ratio: {token_efficiency_detector.threshold_ratio}:1")
    logger.info(f"   Monitored operations: {len(token_efficiency_detector.operations) if token_efficiency_detector.operations else 'all'}")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def classify_error(error: Exception, operation: str = None) -> dict:
    """
    Classify errors for tracking. Returns dict with error_type, error_code.
    
    Universal error classifier - works across different LLM providers and application types.
    Add project-specific patterns at the end for custom error handling.
    
    Use with ** unpacking in track_llm_call:
        track_llm_call(
            ...
            **classify_error(e, operation="generate_sql"),
            retry_count=0,
            ...
        )
    
    Categories:
        - authentication: API key, auth token errors
        - throttling: Rate limit, quota errors
        - network: Timeout, connection errors
        - llm_provider: Provider-specific errors (context length, content filter)
        - database: Database operation errors
        - database_schema: Schema/structure errors
        - data_missing: Not found, missing data
        - input_error: Validation, malformed input
        - response_format: Parsing, JSON decode errors
        - unclassified: Unknown errors
    
    Args:
        error: The exception that was caught
        operation: Optional operation name for context
        
    Returns:
        Dict with error_type, error_code
    """
    error_str = str(error).lower()
    error_type = type(error).__name__
    
    # Authentication errors (universal - all providers)
    if any(x in error_str for x in ["api key", "api_key", "unauthorized", "401", "authentication failed"]):
        return {
            "error_type": error_type,
            "error_code": "AUTH_ERROR",
        }
    
    # Throttling errors (universal - all providers)
    elif any(x in error_str for x in ["rate limit", "429", "quota exceeded", "too many requests"]):
        return {
            "error_type": error_type,
            "error_code": "RATE_LIMIT",
        }
    
    # Network errors (universal)
    elif any(x in error_str for x in ["timeout", "timed out", "connection", "network"]):
        return {
            "error_type": error_type,
            "error_code": "TIMEOUT" if "timeout" in error_str else "CONNECTION_ERROR",
        }
    
    # LLM provider errors (OpenAI, Azure, Anthropic, etc.)
    elif any(x in error_str for x in ["context length", "token limit", "max tokens", "context_length"]):
        return {
            "error_type": error_type,
            "error_code": "CONTEXT_LENGTH_EXCEEDED",
        }
    
    elif any(x in error_str for x in ["content filter", "content_filter", "policy violation"]):
        return {
            "error_type": error_type,
            "error_code": "CONTENT_FILTER",
        }
    
    elif any(x in error_str for x in ["model not found", "deployment not found", "invalid model"]):
        return {
            "error_type": error_type,
            "error_code": "MODEL_NOT_FOUND",
        }
    
    # Database errors (universal - SQLite, PostgreSQL, MySQL, etc.)
    elif any(x in error_str for x in ["no such column", "no such table", "unknown column", "unknown table"]):
        return {
            "error_type": error_type,
            "error_code": "SCHEMA_ERROR",
        }
    
    elif any(x in error_type.lower() for x in ["sqlite", "psycopg", "mysql", "database"]) or "database" in error_str:
        return {
            "error_type": error_type,
            "error_code": "DB_ERROR",
        }
    
    # Data errors (universal)
    elif re.search(r"not found|no .* found|404", error_str):
        return {
            "error_type": error_type,
            "error_code": "NOT_FOUND",
        }
    
    # Input validation errors (universal)
    elif any(x in error_str for x in ["invalid", "validation", "malformed", "bad request", "400"]):
        return {
            "error_type": error_type,
            "error_code": "VALIDATION_ERROR",
        }
    
    # Response format errors (universal)
    elif any(x in error_str for x in ["json", "parse", "decode", "unmarshal", "serialization"]):
        return {
            "error_type": error_type,
            "error_code": "PARSE_ERROR",
        }
    
    # Unclassified (catch-all)
    else:
        return {
            "error_type": error_type,
            "error_code": "UNKNOWN",
        }


def extract_azure_cache_metrics(response, system_prompt: str = None, prefix_length: int = 500) -> dict:
    """
    Extract Azure prompt cache metrics from an LLM response.

    Returns a dict ready to be spread into track_llm_call():
        cached_prompt_tokens: int - Tokens served from Azure's prompt cache
        cached_token_savings: float - Cost savings from cached tokens
        prompt_prefix_hash: str - Hash for grouping by stable prefix category

    Usage in plugins:
        cache_metrics = extract_azure_cache_metrics(response, system_prompt)
        track_llm_call(
            ...other params...,
            **cache_metrics,  # Spread the cache metrics
        )

    Args:
        response: The LLM response object (Semantic Kernel ChatMessageContent, etc.)
        system_prompt: The system prompt used (for computing prefix hash)
        prefix_length: Characters to use for prefix hash (default: 500)

    Returns:
        Dict with cached_prompt_tokens, cached_token_savings, prompt_prefix_hash
    """
    import hashlib

    result = {
        "cached_prompt_tokens": 0,
        "cached_token_savings": 0.0,
        "prompt_prefix_hash": None,
    }

    # Extract cached tokens from response metadata
    if response and hasattr(response, 'metadata') and response.metadata:
        usage = response.metadata.get("usage")
        if usage:
            cached_tokens = 0

            # Handle object format (OpenAI SDK)
            if hasattr(usage, 'prompt_tokens_details'):
                prompt_tokens_details = usage.prompt_tokens_details
                if hasattr(prompt_tokens_details, 'cached_tokens'):
                    cached_tokens = prompt_tokens_details.cached_tokens or 0
            # Handle dict format
            elif isinstance(usage, dict):
                prompt_tokens_details = usage.get("prompt_tokens_details", {})
                if isinstance(prompt_tokens_details, dict):
                    cached_tokens = prompt_tokens_details.get("cached_tokens", 0)

            if cached_tokens > 0:
                result["cached_prompt_tokens"] = cached_tokens
                # Azure charges 50% less for cached tokens
                result["cached_token_savings"] = cached_tokens * 0.000003  # ~$0.003/1K saved

    # Compute prefix hash for stable_prefix category grouping
    if system_prompt:
        prefix = system_prompt[:prefix_length] if len(system_prompt) > prefix_length else system_prompt
        result["prompt_prefix_hash"] = hashlib.md5(prefix.encode()).hexdigest()[:16]

    return result


def extract_token_breakdown_from_messages(
    messages: List[Dict] = None,
    system_prompt: str = None,
    user_message: str = None,
    chat_history: Any = None,
    conversation_memory: Any = None,
) -> Dict[str, int]:
    """
    Extract token breakdown from various message formats.
    
    Universal function - works with:
    - OpenAI/Azure message format: [{"role": "system", "content": "..."}, ...]
    - Anthropic message format: Similar structure
    - LangChain messages: Convertible to dict format
    - Semantic Kernel ChatHistory: Has .messages attribute
    - Custom ConversationMemory: Has .chat_history or .get_context_for_prompt()
    
    Auto-populates token fields for comprehensive tracking:
    - system_prompt_tokens
    - user_message_tokens
    - chat_history_tokens
    - chat_history_count (number of messages in history)
    - conversation_context_tokens
    
    Args:
        messages: Full messages array (OpenAI/Azure/Anthropic format)
        system_prompt: System prompt text (alternative to messages)
        user_message: User message text (alternative to messages)
        chat_history: ChatHistory object (Semantic Kernel format)
        conversation_memory: ConversationMemory object (custom format)
        
    Returns:
        Dict with all token breakdown fields
        
    Example:
        # From OpenAI messages
        breakdown = extract_token_breakdown_from_messages(
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "Help me"}
            ]
        )
        # Returns: {
        #   "system_prompt_tokens": 3,
        #   "user_message_tokens": 2,  # Last user message
        #   "chat_history_tokens": 5,  # Previous messages
        #   "chat_history_count": 2,   # 1 user + 1 assistant
        #   "conversation_context_tokens": 0
        # }
    """
    
    breakdown = {
        'system_prompt_tokens': 0,
        'user_message_tokens': 0,
        'chat_history_tokens': 0,
        'chat_history_count': 0,
        'conversation_context_tokens': 0,
    }
    
    # METHOD 1: Extract from messages array (OpenAI/Azure/Anthropic)
    if messages:
        user_messages = []
        assistant_messages = []
        
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            tokens = estimate_tokens(content) if content else 0
            
            if role == 'system':
                breakdown['system_prompt_tokens'] += tokens
            
            elif role == 'user':
                user_messages.append((content, tokens))
            
            elif role in ['assistant', 'function', 'tool']:
                assistant_messages.append((content, tokens))
                breakdown['chat_history_tokens'] += tokens
        
        # Last user message is current, rest go to history
        if user_messages:
            # Current user message (last one)
            breakdown['user_message_tokens'] = user_messages[-1][1]
            
            # Previous user messages go to history
            for content, tokens in user_messages[:-1]:
                breakdown['chat_history_tokens'] += tokens
        
        # Calculate chat_history_count = previous user messages + all assistant messages
        breakdown['chat_history_count'] = (len(user_messages) - 1) + len(assistant_messages)
    

    # METHOD 2: Extract from individual strings
    else:
        if system_prompt:
            breakdown['system_prompt_tokens'] = estimate_tokens(system_prompt)
        
        if user_message:
            breakdown['user_message_tokens'] = estimate_tokens(user_message)
        
        # METHOD 3: Extract from Semantic Kernel ChatHistory
        if chat_history:
            try:
                if hasattr(chat_history, 'messages'):
                    history_tokens = 0
                    history_count = 0
                    for msg in chat_history.messages:
                        # Skip system messages (already counted)
                        if hasattr(msg, 'role') and msg.role != 'system':
                            content = str(msg.content) if hasattr(msg, 'content') else ''
                            history_tokens += estimate_tokens(content)
                            history_count += 1
                    breakdown['chat_history_tokens'] = history_tokens
                    breakdown['chat_history_count'] = history_count
            except Exception:
                pass  # Silent fail - not critical

        # METHOD 4: Extract from ConversationMemory
        if conversation_memory:
            try:
                # Try chat_history attribute
                if hasattr(conversation_memory, 'chat_history'):
                    history = conversation_memory.chat_history
                    if hasattr(history, 'messages'):
                        history_tokens = 0
                        history_count = 0
                        for msg in history.messages:
                            if hasattr(msg, 'role') and msg.role != 'system':
                                content = str(msg.content) if hasattr(msg, 'content') else ''
                                history_tokens += estimate_tokens(content)
                                history_count += 1
                        breakdown['chat_history_tokens'] = history_tokens
                        breakdown['chat_history_count'] = history_count
                
                # Try get_context_for_prompt method
                if hasattr(conversation_memory, 'get_context_for_prompt'):
                    context_text = conversation_memory.get_context_for_prompt()
                    if context_text and context_text != "No prior context.":
                        breakdown['conversation_context_tokens'] = estimate_tokens(context_text)
            except Exception:
                pass  # Silent fail - not critical
    
    return breakdown

def extract_model_parameters(
    client: Any = None,
    execution_settings: Any = None,
    temperature: float = None,
    max_tokens: int = None,
    top_p: float = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Extract model parameters from various sources.
    
    Universal function - works with:
    - OpenAI/Azure OpenAI clients
    - Anthropic clients
    - Semantic Kernel execution_settings
    - LangChain model configs
    - Direct parameter values (highest priority)
    
    Extraction priority:
    1. Explicit parameters (temperature, max_tokens, top_p passed directly)
    2. execution_settings object (Semantic Kernel)
    3. client object (OpenAI/Anthropic style)
    4. kwargs (catch-all for other frameworks)
    
    Args:
        client: LLM client object (OpenAI, Azure, Anthropic, etc.)
        execution_settings: Semantic Kernel execution settings
        temperature: Explicit temperature value (0.0-2.0)
        max_tokens: Explicit max tokens value
        top_p: Explicit top_p value (0.0-1.0)
        **kwargs: Additional parameters from other frameworks
        
    Returns:
        Dict with temperature, max_tokens, top_p (None if not found)
        
    Example:
        # From Semantic Kernel
        params = extract_model_parameters(
            execution_settings=sk_settings
        )
        
        # From OpenAI client
        params = extract_model_parameters(
            client=openai_client,
            temperature=0.7  # Override client default
        )
        
        # Direct values
        params = extract_model_parameters(
            temperature=0.7,
            max_tokens=1000
        )
    """
    params = {
        'temperature': temperature,
        'max_tokens': max_tokens,
        'top_p': top_p,
    }
    
    # PRIORITY 1: Check kwargs for framework-specific parameters
    # LangChain, LlamaIndex, and other frameworks may pass model config in kwargs
    if 'model_kwargs' in kwargs:
        model_kwargs = kwargs['model_kwargs']
        if isinstance(model_kwargs, dict):
            params['temperature'] = params['temperature'] or model_kwargs.get('temperature')
            params['max_tokens'] = params['max_tokens'] or model_kwargs.get('max_tokens')
            params['top_p'] = params['top_p'] or model_kwargs.get('top_p')
    
    # PRIORITY 2: Extract from execution_settings (Semantic Kernel)
    if execution_settings and not all(params.values()):
        try:
            # Semantic Kernel style
            if hasattr(execution_settings, 'temperature'):
                params['temperature'] = params['temperature'] or execution_settings.temperature
            if hasattr(execution_settings, 'max_tokens'):
                params['max_tokens'] = params['max_tokens'] or execution_settings.max_tokens
            if hasattr(execution_settings, 'top_p'):
                params['top_p'] = params['top_p'] or execution_settings.top_p
        except Exception:
            pass  # Silent fail - not critical
    
    # PRIORITY 3: Extract from client (OpenAI/Azure/Anthropic)
    if client and not all(params.values()):
        try:
            # OpenAI/Azure OpenAI style (stored in client)
            if hasattr(client, 'temperature'):
                params['temperature'] = params['temperature'] or client.temperature
            if hasattr(client, 'max_tokens'):
                params['max_tokens'] = params['max_tokens'] or client.max_tokens
            if hasattr(client, 'top_p'):
                params['top_p'] = params['top_p'] or client.top_p
            
            # Anthropic style (may be in client.default_request_params)
            if hasattr(client, 'default_request_params'):
                defaults = client.default_request_params
                params['temperature'] = params['temperature'] or defaults.get('temperature')
                params['max_tokens'] = params['max_tokens'] or defaults.get('max_tokens')
                params['top_p'] = params['top_p'] or defaults.get('top_p')
        except Exception:
            pass  # Silent fail - not critical
    
    return params

# =============================================================================
# FIRE-AND-FORGET JUDGE EVALUATION
# =============================================================================
# Runs judge evaluation in the background without blocking the main response.
# Quality scores still get tracked to Observatory, just asynchronously.

import asyncio
from typing import Callable, Optional

# Store for pending judge tasks (for cleanup/debugging if needed)
_pending_judge_tasks: set = set()

async def _run_judge_and_track(
    judge_instance,
    operation: str,
    prompt: str,
    response: str,
    llm_client: Any,
    conversation_id: Optional[str] = None,
    turn_number: Optional[int] = None,
    callback: Optional[Callable] = None,
):
    """Internal: Run judge evaluation and optionally call a callback with results."""
    try:
        quality_eval = await judge_instance.maybe_evaluate(
            operation=operation,
            prompt=prompt,
            response=response,
            llm_client=llm_client,
            conversation_id=conversation_id,
            turn_number=turn_number,
        )
        if callback and quality_eval:
            callback(quality_eval)
        return quality_eval
    except Exception as e:
        logger.warning(f"⚠️ Background judge evaluation failed for {operation}: {e}")
        return None


def fire_and_forget_judge(
    operation: str,
    prompt: str,
    response: str,
    llm_client: Any,
    conversation_id: Optional[str] = None,
    turn_number: Optional[int] = None,
    callback: Optional[Callable] = None,
) -> None:
    """
    Fire-and-forget judge evaluation - runs in background without blocking.

    Usage:
        # Instead of:
        # quality_eval = await judge.maybe_evaluate(operation, prompt, response, llm_client)

        # Do this:
        fire_and_forget_judge(
            operation=operation,
            prompt=full_prompt,
            response=result_str,
            llm_client=self.kernel,
            conversation_id=self.memory.conversation_id if self.memory else None,
            turn_number=self.memory.turn_number if self.memory else None,
        )
        # Response returns immediately to user, judge runs in background

    Args:
        operation: The operation name (e.g., "improve_bullet", "generate_sql")
        prompt: The full prompt sent to the LLM
        response: The LLM's response
        llm_client: The LLM client/kernel for the judge to use
        conversation_id: Optional conversation ID for tracking
        turn_number: Optional turn number for tracking
        callback: Optional callback function that receives the quality_eval result
    """
    try:
        # Get or create event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop - create task won't work, fall back to sync
            logger.debug(f"No event loop for fire-and-forget judge on {operation}")
            return

        # Create the background task
        task = loop.create_task(
            _run_judge_and_track(
                judge_instance=judge,
                operation=operation,
                prompt=prompt,
                response=response,
                llm_client=llm_client,
                conversation_id=conversation_id,
                turn_number=turn_number,
                callback=callback,
            )
        )

        # Track the task (for debugging/cleanup)
        _pending_judge_tasks.add(task)
        task.add_done_callback(lambda t: _pending_judge_tasks.discard(t))

        logger.debug(f"🔥 Fire-and-forget judge started for {operation}")

    except Exception as e:
        logger.warning(f"⚠️ Failed to start fire-and-forget judge for {operation}: {e}")


async def wait_for_pending_judges(timeout: float = 30.0) -> int:
    """
    Wait for all pending judge evaluations to complete.
    Useful at end of test scenarios or application shutdown.

    Returns:
        Number of tasks that were pending
    """
    if not _pending_judge_tasks:
        return 0

    pending_count = len(_pending_judge_tasks)
    logger.info(f"⏳ Waiting for {pending_count} pending judge evaluations...")

    try:
        await asyncio.wait_for(
            asyncio.gather(*_pending_judge_tasks, return_exceptions=True),
            timeout=timeout
        )
        logger.info(f"✅ All {pending_count} judge evaluations completed")
    except asyncio.TimeoutError:
        logger.warning(f"⚠️ Timeout waiting for judge evaluations ({len(_pending_judge_tasks)} still pending)")

    return pending_count


# =============================================================================
# MAIN WRAPPER: track_llm_call()
# =============================================================================

def track_llm_call(
    # TIER 1 - Core metrics (always include)
    model_name: str = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    latency_ms: float = 0,
    
    # Context
    agent_name: str = None,
    agent_role: str = None,
    operation: str = None,
    
    # Status
    success: bool = True,
    error: str = None,
    
    # TIER 2 - Prompt content
    prompt: str = None,
    response_text: str = None,
    prompt_normalized: str = None,
    system_prompt: str = None,
    user_message: str = None,
    messages: List[Dict[str, str]] = None,
    
    # TIER 2 - Optimization tracking
    routing_decision: RoutingDecision = None,
    cache_metadata: CacheMetadata = None,
    quality_evaluation: QualityEvaluation = None,
    prompt_breakdown: PromptBreakdown = None,
    prompt_metadata: PromptMetadata = None,
    
    # TIER 3 - A/B Testing
    prompt_variant_id: str = None,
    test_dataset_id: str = None,
    
    # Conversation linking
    conversation_id: str = None,
    turn_number: int = None,
    parent_call_id: str = None,
    user_id: str = None,
    
    # Model configuration
    temperature: float = None,
    max_tokens: int = None,
    top_p: float = None,
    model_config: ModelConfig = None,
    
    # Token breakdown (top-level)
    system_prompt_tokens: int = None,
    user_message_tokens: int = None,
    chat_history_tokens: int = None,
    chat_history_count: int = None,
    conversation_context_tokens: int = None,
    tool_definitions_tokens: int = None,
    
    # Tool/function calling
    tool_calls_made: List[Dict[str, Any]] = None,
    tool_call_count: int = None,
    tool_execution_time_ms: float = None,
    
    # Streaming
    time_to_first_token_ms: float = None,
    streaming_metrics: StreamingMetrics = None,
    
    # Error details
    error_type: str = None,
    error_code: str = None,
    retry_count: int = None,
    error_details: ErrorDetails = None,
    
    # Cached tokens
    cached_prompt_tokens: int = None,
    cached_token_savings: float = None,
    
    # Observability
    trace_id: str = None,
    request_id: str = None,
    environment: str = None,
    prompt_prefix_hash: str = None,
    
    # Experiment tracking
    experiment_id: str = None,
    control_group: bool = None,
    experiment_metadata: ExperimentMetadata = None,
    
    # Custom metadata
    metadata: dict = None,
) -> LLMCall:
    """
    Track an LLM call with auto-filled defaults for universal use.
    
    This is your main interface for tracking LLM calls. It automatically:
    - Extracts token breakdown from messages
    - Extracts model parameters from various sources
    - Populates prompt_breakdown for analysis
    - Cleans metadata before storage
    
    Supports all 139 fields across 3 tiers for comprehensive tracking.
    
    Args:
        model_name: Model used (defaults to DEFAULT_MODEL if not provided)
        prompt_tokens: Input token count
        completion_tokens: Output token count
        latency_ms: Response time in milliseconds
        
        [... all other 139 parameters ...]
        
    Returns:
        LLMCall object from Observatory
        
    Example:
        # Minimal usage
        track_llm_call(
            model_name="gpt-4o-mini",
            prompt_tokens=100,
            completion_tokens=50,
            latency_ms=500,
            operation="generate_sql"
        )
        
        # With auto-extraction
        track_llm_call(
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"}
            ],
            completion_tokens=50,
            latency_ms=500,
            operation="chat"
        )
        # Auto-extracts: system_prompt_tokens, user_message_tokens, etc.
    """
    
    # ═══════════════════════════════════════════════════════════════
    # AUTO-EXTRACT: Token breakdown
    # ═══════════════════════════════════════════════════════════════
    if not system_prompt_tokens and not user_message_tokens and not chat_history_tokens:
        token_breakdown = extract_token_breakdown_from_messages(
            messages=messages,
            system_prompt=system_prompt,
            user_message=user_message,
            chat_history=metadata.get('chat_history') if metadata else None,
            conversation_memory=metadata.get('conversation_memory') if metadata else None,
        )
        
        system_prompt_tokens = system_prompt_tokens or token_breakdown['system_prompt_tokens']
        user_message_tokens = user_message_tokens or token_breakdown['user_message_tokens']
        chat_history_tokens = chat_history_tokens or token_breakdown['chat_history_tokens']
        chat_history_count = chat_history_count or token_breakdown['chat_history_count']
        conversation_context_tokens = conversation_context_tokens or token_breakdown['conversation_context_tokens']
    
    # ═══════════════════════════════════════════════════════════════
    # AUTO-EXTRACT: Model parameters
    # ═══════════════════════════════════════════════════════════════
    if temperature is None or max_tokens is None or top_p is None:
        model_params = extract_model_parameters(
            client=metadata.get('client') if metadata else None,
            execution_settings=metadata.get('execution_settings') if metadata else None,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        
        temperature = temperature if temperature is not None else model_params['temperature']
        max_tokens = max_tokens if max_tokens is not None else model_params['max_tokens']
        top_p = top_p if top_p is not None else model_params['top_p']
    
    # ═══════════════════════════════════════════════════════════════
    # AUTO-CREATE: prompt_breakdown for analysis
    # ═══════════════════════════════════════════════════════════════
    if (system_prompt or user_message) and not prompt_breakdown:
        prompt_breakdown = create_prompt_breakdown(
            system_prompt=system_prompt,
            user_message=user_message,
            system_prompt_tokens=system_prompt_tokens,
            user_message_tokens=user_message_tokens,
            chat_history_tokens=chat_history_tokens,
            chat_history_count=chat_history_count,
            conversation_context_tokens=conversation_context_tokens,
            tool_definitions_tokens=tool_definitions_tokens,
            response_text=response_text,
        )
    
    # ═══════════════════════════════════════════════════════════════
    # CONVERT: agent_role string to enum
    # ═══════════════════════════════════════════════════════════════
    role_enum = None
    if agent_role:
        try:
            role_enum = AgentRole(agent_role)
        except ValueError:
            # Not a valid enum - store in metadata instead
            if metadata is None:
                metadata = {}
            metadata['agent_role_str'] = agent_role
    
    # ═══════════════════════════════════════════════════════════════
    # CLEAN: Remove non-serializable objects from metadata
    # ═══════════════════════════════════════════════════════════════
    if metadata:
        metadata.pop('conversation_memory', None)
        metadata.pop('execution_settings', None)
        metadata.pop('client', None)
        metadata.pop('chat_history', None)
    else:
        metadata = {} 

    # AUTO-TAG: Add current phase to every call
    metadata['phase'] = CURRENT_PHASE
    metadata['phase_description'] = PHASE_DESCRIPTIONS[CURRENT_PHASE]
    
    # ═══════════════════════════════════════════════════════════════
    # CALL: SDK track_llm_call with all parameters
    # ═══════════════════════════════════════════════════════════════
    return _sdk_track_llm_call(
        # CORE: Observatory instance
        observatory=obs,
        
        # TIER 1: Core LLM metrics (always required)
        model_name=model_name or DEFAULT_MODEL,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=max(latency_ms, 0.001),  # Ensure non-zero
        provider=DEFAULT_PROVIDER,
        
        # TIER 1: Context & identification
        agent_name=agent_name,
        agent_role=role_enum,
        operation=operation,
        
        # TIER 1: Status
        success=success,
        error=error,
        
        # TIER 2: Prompt content
        prompt=prompt,
        response_text=response_text,
        prompt_normalized=prompt_normalized,
        system_prompt=system_prompt,
        user_message=user_message,
        messages=messages,
        
        # TIER 2: Optimization tracking (SDK components)
        routing_decision=routing_decision,
        cache_metadata=cache_metadata,
        quality_evaluation=quality_evaluation,
        prompt_breakdown=prompt_breakdown,
        prompt_metadata=prompt_metadata,
        
        # TIER 3: A/B Testing
        prompt_variant_id=prompt_variant_id,
        test_dataset_id=test_dataset_id,
        
        # Conversation linking
        conversation_id=conversation_id,
        turn_number=turn_number,
        parent_call_id=parent_call_id,
        user_id=user_id,
        
        # Model configuration
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        model_config=model_config,
        
        # Token breakdown (auto-extracted)
        system_prompt_tokens=system_prompt_tokens,
        user_message_tokens=user_message_tokens,
        chat_history_tokens=chat_history_tokens,
        chat_history_count=chat_history_count,
        conversation_context_tokens=conversation_context_tokens,
        tool_definitions_tokens=tool_definitions_tokens,
        
        # Tool/function calling
        tool_calls_made=tool_calls_made,
        tool_call_count=tool_call_count,
        tool_execution_time_ms=tool_execution_time_ms,
        
        # Streaming metrics
        time_to_first_token_ms=time_to_first_token_ms,
        streaming_metrics=streaming_metrics,
        
        # Error tracking
        error_type=error_type,
        error_code=error_code,
        retry_count=retry_count,
        error_details=error_details,
        
        # Caching & token optimization
        cached_prompt_tokens=cached_prompt_tokens,
        cached_token_savings=cached_token_savings,
        
        # Observability & tracing
        trace_id=trace_id,
        request_id=request_id,
        environment=environment,
        prompt_prefix_hash=prompt_prefix_hash,
        
        # Experiment tracking
        experiment_id=experiment_id,
        control_group=control_group,
        experiment_metadata=experiment_metadata,
        
        # Custom metadata
        metadata=metadata,
    )

# =============================================================================
# TRACKED LLM CALL - CONTEXT MANAGER ABSTRACTION
# =============================================================================
# Encapsulates the 10-step optimization pattern into a clean async context manager.
# Reduces ~400 lines of boilerplate per LLM call method to ~15 lines.
#
# Usage:
#     async with TrackedLLMCall(
#         operation="quick_score_job",
#         agent_name="ResumeMatching",
#         agent_role="analyst",
#         complexity=0.4,
#         cache_key={"job_id": job_id, "resume_hash": resume_hash},
#         memory=self.memory,
#         metadata={"job_title": job.get('title')}
#     ) as ctx:
#         result = await ctx.call_llm(
#             chat_completion=self.chat_completion,
#             system_prompt=QUICK_SCORE_PROMPT,
#             user_message=f"Resume:\n{resume_text}\n\nJob:\n{job}",
#             temperature=0.3,
#             max_tokens=1500,
#         )
#         return self._parse_quick_score(result, job)

import hashlib
import time
import asyncio
from dataclasses import dataclass, field
from typing import Callable, Union

@dataclass
class TrackedLLMCallResult:
    """Result container from TrackedLLMCall context manager."""
    response_text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    latency_ms: float = 0
    model_used: str = ""
    cache_hit: bool = False
    cache_type: str = None  # "exact" or "semantic"
    routing_decision: RoutingDecision = None
    quality_evaluation: QualityEvaluation = None
    optimizations_applied: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


class TrackedLLMCall:
    """
    Async context manager that encapsulates the 10-step LLM optimization pattern.

    Steps handled automatically:
    1. Cache check (exact match via CacheManager)
    2. Semantic cache check (similarity match via SemanticCache)
    3. Prompt optimization (compression based on operation complexity)
    4. Model routing (complexity-based model selection)
    5. Prefix cache tracking (Azure/Anthropic prefix caching)
    6. LLM call execution (via call_llm method)
    7. Streaming detection (flag high-latency calls)
    8. Cache response storage (save for future hits)
    9. Quality evaluation (LLM-as-judge scoring)
    10. Observatory tracking (full metric capture)

    Attributes:
        operation: Operation name for tracking (e.g., "quick_score_job")
        agent_name: Name of the calling agent/plugin
        agent_role: Role enum or string (e.g., "analyst", "coordinator")
        complexity: Task complexity score (0.0-1.0) for routing
        cache_key: Dict of values to generate cache key from
        cache_ttl: Cache TTL in seconds (default from CacheManager)
        memory: ConversationMemory instance for context extraction
        conversation_id: ID linking related calls together
        turn_number: Turn number in conversation
        metadata: Additional metadata to track
        skip_cache: Set True to bypass cache check
        skip_quality_eval: Set True to bypass quality evaluation
        skip_routing: Set True to use default model always
    """

    def __init__(
        self,
        operation: str,
        agent_name: str = None,
        agent_role: str = None,
        complexity: float = 0.5,
        phase: str = None,  # "baseline" or "optimized" for comparison tracking
        cache_key: dict = None,
        cache_ttl: int = None,
        memory: Any = None,
        conversation_id: str = None,
        turn_number: int = None,
        metadata: dict = None,
        skip_cache: bool = False,
        skip_quality_eval: bool = False,
        skip_routing: bool = False,
    ):
        self.operation = operation
        self.agent_name = agent_name
        self.agent_role = agent_role
        self.complexity = complexity
        self.phase = phase
        self.cache_key = cache_key or {}
        self.cache_ttl = cache_ttl
        self.memory = memory
        self.conversation_id = conversation_id
        self.turn_number = turn_number
        self.metadata = metadata or {}
        self.skip_cache = skip_cache
        self.skip_quality_eval = skip_quality_eval
        self.skip_routing = skip_routing

        # Internal state
        self._start_time: float = None
        self._result: TrackedLLMCallResult = None
        self._system_prompt: str = None
        self._user_message: str = None
        self._full_prompt: str = None
        self._optimized_prompt: str = None
        self._execution_settings: Any = None
        self._cache_key_hash: str = None
        self._routing_decision: RoutingDecision = None
        self._prefix_tracked: bool = False
        self._streaming_flagged: bool = False
        self._error: Exception = None
        self._quality_task: asyncio.Task = None

    async def __aenter__(self) -> 'TrackedLLMCall':
        """Enter context - start timing and check caches."""
        self._start_time = time.perf_counter()
        self._result = TrackedLLMCallResult()
        self._result.optimizations_applied = {
            "cache_checked": False,
            "semantic_cache_checked": False,
            "prompt_optimized": False,
            "model_routed": False,
            "prefix_tracked": False,
            "streaming_flagged": False,
            "quality_evaluated": False,
        }

        # Generate cache key hash
        if self.cache_key:
            key_str = f"{self.operation}:{sorted(self.cache_key.items())}"
            self._cache_key_hash = hashlib.sha256(key_str.encode()).hexdigest()[:16]

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit context - track the call to Observatory."""
        latency_ms = (time.perf_counter() - self._start_time) * 1000
        self._result.latency_ms = latency_ms

        # Wait for quality evaluation if running
        if self._quality_task and not self._quality_task.done():
            try:
                await asyncio.wait_for(self._quality_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.debug("Quality evaluation timed out")
            except Exception as e:
                logger.debug(f"Quality evaluation failed: {e}")

        # Track the call regardless of success/failure
        try:
            await self._track_call(
                success=(exc_type is None),
                error=str(exc_val) if exc_val else None,
            )
        except Exception as track_error:
            logger.warning(f"Failed to track LLM call: {track_error}")

        # Don't suppress exceptions
        return False

    def _generate_cache_key(self, prompt: str) -> str:
        """Generate cache key from prompt and cache_key dict."""
        if self._cache_key_hash:
            return f"{self.operation}:{self._cache_key_hash}:{hashlib.sha256(prompt.encode()).hexdigest()[:16]}"
        return f"{self.operation}:{hashlib.sha256(prompt.encode()).hexdigest()[:32]}"

    async def _check_exact_cache(self, prompt: str) -> Optional[str]:
        """Step 1: Check exact match cache."""
        if self.skip_cache or not cache.enabled:
            return None

        self._result.optimizations_applied["cache_checked"] = True

        try:
            cache_result = await cache.get(
                operation=self.operation,
                prompt=prompt,
                cache_key=self._generate_cache_key(prompt),
            )

            if cache_result and cache_result.hit:
                self._result.cache_hit = True
                self._result.cache_type = "exact"
                self._result.metadata["cache_key"] = cache_result.key
                return cache_result.response
        except Exception as e:
            logger.debug(f"Cache check failed: {e}")

        return None

    async def _check_semantic_cache(self, prompt: str) -> Optional[str]:
        """Step 2: Check semantic similarity cache."""
        if self.skip_cache or not semantic_cache or not semantic_cache.enabled:
            return None

        self._result.optimizations_applied["semantic_cache_checked"] = True

        try:
            sem_result = await semantic_cache.get(
                operation=self.operation,
                prompt=prompt,
            )

            if sem_result and sem_result.hit:
                self._result.cache_hit = True
                self._result.cache_type = "semantic"
                self._result.metadata["semantic_similarity"] = sem_result.similarity
                return sem_result.response
        except Exception as e:
            logger.debug(f"Semantic cache check failed: {e}")

        return None

    def _optimize_prompt(self, system_prompt: str, user_message: str) -> tuple:
        """Step 3: Optimize prompt based on operation complexity."""
        if not prompt_optimizer.enabled:
            return system_prompt, user_message, None

        self._result.optimizations_applied["prompt_optimized"] = True

        try:
            result = prompt_optimizer.get_optimized_prompt(
                operation=self.operation,
                default_prompt=system_prompt,
            )

            optimized_system = result.get("prompt", system_prompt)
            max_tokens = result.get("max_tokens")

            self._result.metadata["prompt_variant"] = result.get("variant", "default")
            self._result.metadata["prompt_savings"] = result.get("savings", {})

            return optimized_system, user_message, max_tokens
        except Exception as e:
            logger.debug(f"Prompt optimization failed: {e}")
            return system_prompt, user_message, None

    def _route_model(self, estimated_tokens: int = None) -> str:
        """Step 4: Route to optimal model based on complexity."""
        if self.skip_routing or not router.enabled:
            self._result.model_used = DEFAULT_MODEL
            return DEFAULT_MODEL

        self._result.optimizations_applied["model_routed"] = True

        try:
            self._routing_decision = router.select(
                operation=self.operation,
                complexity=self.complexity,
                estimated_tokens=estimated_tokens,
            )

            self._result.model_used = self._routing_decision.selected_model
            self._result.routing_decision = self._routing_decision
            self._result.metadata["routing_reason"] = self._routing_decision.reason

            return self._routing_decision.selected_model
        except Exception as e:
            logger.debug(f"Model routing failed: {e}")
            self._result.model_used = DEFAULT_MODEL
            return DEFAULT_MODEL

    def _track_prefix(self, prompt: str):
        """Step 5: Track prefix for Azure/Anthropic prefix caching."""
        if not prefix_cache.enabled:
            return

        self._result.optimizations_applied["prefix_tracked"] = True
        self._prefix_tracked = True

        try:
            prefix_cache.track(
                operation=self.operation,
                prompt=prompt,
            )
        except Exception as e:
            logger.debug(f"Prefix tracking failed: {e}")

    def _flag_streaming(self, latency_ms: float, output_tokens: int):
        """Step 7: Flag for streaming if high latency or large output."""
        if not streaming_detector.enabled:
            return

        try:
            should_stream = streaming_detector.should_stream(
                operation=self.operation,
                latency_ms=latency_ms,
                output_tokens=output_tokens,
            )

            if should_stream:
                self._streaming_flagged = True
                self._result.optimizations_applied["streaming_flagged"] = True
                self._result.metadata["streaming_recommended"] = True
        except Exception as e:
            logger.debug(f"Streaming detection failed: {e}")

    async def _cache_response(self, prompt: str, response: str):
        """Step 8: Cache the response for future use."""
        if self.skip_cache or not cache.enabled:
            return

        try:
            await cache.set(
                operation=self.operation,
                prompt=prompt,
                response=response,
                cache_key=self._generate_cache_key(prompt),
                ttl=self.cache_ttl,
            )

            # Also cache in semantic cache if available
            if semantic_cache and semantic_cache.enabled:
                await semantic_cache.set(
                    operation=self.operation,
                    prompt=prompt,
                    response=response,
                )
        except Exception as e:
            logger.debug(f"Cache storage failed: {e}")

    async def _evaluate_quality(self, prompt: str, response: str) -> Optional[QualityEvaluation]:
        """Step 9: Evaluate response quality with LLM-as-judge."""
        if self.skip_quality_eval or not judge.enabled:
            return None

        self._result.optimizations_applied["quality_evaluated"] = True

        try:
            evaluation = await judge.maybe_evaluate(
                operation=self.operation,
                prompt=prompt,
                response=response,
                client=None,  # Judge uses its own client
            )

            if evaluation:
                self._result.quality_evaluation = evaluation
                self._result.metadata["quality_score"] = evaluation.overall_score

            return evaluation
        except Exception as e:
            logger.debug(f"Quality evaluation failed: {e}")
            return None

    async def _track_call(self, success: bool, error: str = None):
        """Step 10: Track the call to Observatory."""
        # Build metadata
        tracking_metadata = {
            **self.metadata,
            **self._result.metadata,
            "optimizations_applied": self._result.optimizations_applied,
        }

        # Add phase to metadata for comparison tracking
        if self.phase:
            tracking_metadata["phase"] = self.phase

        # Extract conversation context if memory provided
        chat_history_count = None
        if self.memory:
            try:
                if hasattr(self.memory, 'chat_history') and hasattr(self.memory.chat_history, 'messages'):
                    chat_history_count = len(self.memory.chat_history.messages)
            except Exception:
                pass

        # Track with full schema
        track_llm_call(
            model_name=self._result.model_used or DEFAULT_MODEL,
            prompt_tokens=self._result.prompt_tokens,
            completion_tokens=self._result.completion_tokens,
            latency_ms=self._result.latency_ms,

            agent_name=self.agent_name,
            agent_role=self.agent_role,
            operation=self.operation,

            success=success,
            error=error,

            prompt=self._full_prompt,
            response_text=self._result.response_text,
            system_prompt=self._system_prompt,
            user_message=self._user_message,

            routing_decision=self._result.routing_decision,
            cache_metadata=create_cache_metadata(
                cache_hit=self._result.cache_hit,
                cache_key=self._generate_cache_key(self._full_prompt) if self._full_prompt else None,
                cache_type=self._result.cache_type,
            ) if self._result.cache_hit else None,
            quality_evaluation=self._result.quality_evaluation,

            conversation_id=self.conversation_id,
            turn_number=self.turn_number,
            chat_history_count=chat_history_count,

            temperature=getattr(self._execution_settings, 'temperature', None) if self._execution_settings else None,
            max_tokens=getattr(self._execution_settings, 'max_tokens', None) if self._execution_settings else None,

            metadata=tracking_metadata,

            # Phase for baseline/optimized comparison
            environment=self.phase,

            **classify_error(self._error, self.operation) if self._error else {},
        )

    async def call_llm(
        self,
        chat_completion: Any,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        execution_settings: Any = None,
        chat_history: Any = None,
    ) -> str:
        """
        Execute the LLM call with all optimizations applied.

        This is the main method you call inside the context manager.
        It handles steps 1-9 of the optimization pattern automatically.

        Args:
            chat_completion: Semantic Kernel chat completion service
            system_prompt: System prompt text
            user_message: User message text
            temperature: LLM temperature (default 0.7)
            max_tokens: Max output tokens (default 1000)
            execution_settings: Optional pre-configured execution settings
            chat_history: Optional ChatHistory for multi-turn context

        Returns:
            Response text from the LLM (or cached response)

        Example:
            async with TrackedLLMCall(operation="analyze") as ctx:
                result = await ctx.call_llm(
                    chat_completion=self.chat_completion,
                    system_prompt="You are an analyst...",
                    user_message="Analyze this data...",
                    temperature=0.3,
                )
        """
        from semantic_kernel.contents import ChatHistory
        from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings

        self._system_prompt = system_prompt
        self._user_message = user_message
        self._full_prompt = f"{system_prompt}\n\n{user_message}"

        # ═══════════════════════════════════════════════════════════════
        # STEP 1: Check exact match cache
        # ═══════════════════════════════════════════════════════════════
        cached = await self._check_exact_cache(self._full_prompt)
        if cached:
            self._result.response_text = cached
            return cached

        # ═══════════════════════════════════════════════════════════════
        # STEP 2: Check semantic cache
        # ═══════════════════════════════════════════════════════════════
        semantic_cached = await self._check_semantic_cache(self._full_prompt)
        if semantic_cached:
            self._result.response_text = semantic_cached
            return semantic_cached

        # ═══════════════════════════════════════════════════════════════
        # STEP 3: Optimize prompt (compression)
        # ═══════════════════════════════════════════════════════════════
        optimized_system, optimized_user, suggested_max_tokens = self._optimize_prompt(
            system_prompt, user_message
        )
        self._optimized_prompt = f"{optimized_system}\n\n{optimized_user}"

        # Use suggested max_tokens if available and lower
        if suggested_max_tokens and suggested_max_tokens < max_tokens:
            max_tokens = suggested_max_tokens

        # ═══════════════════════════════════════════════════════════════
        # STEP 4: Route to optimal model
        # ═══════════════════════════════════════════════════════════════
        estimated_tokens = estimate_tokens(self._optimized_prompt)
        selected_model = self._route_model(estimated_tokens)

        # ═══════════════════════════════════════════════════════════════
        # STEP 5: Track prefix for prefix caching
        # ═══════════════════════════════════════════════════════════════
        self._track_prefix(optimized_system)

        # ═══════════════════════════════════════════════════════════════
        # STEP 6: Execute LLM call
        # ═══════════════════════════════════════════════════════════════
        # Build chat history
        history = chat_history if chat_history else ChatHistory()
        history.add_system_message(optimized_system)
        history.add_user_message(optimized_user)

        # Build execution settings
        if execution_settings:
            self._execution_settings = execution_settings
        else:
            self._execution_settings = AzureChatPromptExecutionSettings(
                temperature=temperature,
                max_tokens=max_tokens,
            )

        call_start = time.perf_counter()

        try:
            response = await chat_completion.get_chat_message_content(
                chat_history=history,
                settings=self._execution_settings,
            )

            call_latency = (time.perf_counter() - call_start) * 1000
            response_text = str(response.content) if response else ""

            # Extract token usage from response metadata if available
            if hasattr(response, 'metadata') and response.metadata:
                usage = response.metadata.get('usage', {})
                self._result.prompt_tokens = usage.get('prompt_tokens', estimate_tokens(self._optimized_prompt))
                self._result.completion_tokens = usage.get('completion_tokens', estimate_tokens(response_text))
            else:
                self._result.prompt_tokens = estimate_tokens(self._optimized_prompt)
                self._result.completion_tokens = estimate_tokens(response_text)

            self._result.total_tokens = self._result.prompt_tokens + self._result.completion_tokens
            self._result.response_text = response_text

        except Exception as e:
            self._error = e
            raise

        # ═══════════════════════════════════════════════════════════════
        # STEP 7: Flag for streaming if needed
        # ═══════════════════════════════════════════════════════════════
        self._flag_streaming(call_latency, self._result.completion_tokens)

        # ═══════════════════════════════════════════════════════════════
        # STEP 8: Cache the response
        # ═══════════════════════════════════════════════════════════════
        await self._cache_response(self._full_prompt, response_text)

        # ═══════════════════════════════════════════════════════════════
        # STEP 9: Evaluate quality (async, tracked for completion)
        # ═══════════════════════════════════════════════════════════════
        # Run quality evaluation - tracked so __aexit__ can await completion
        self._quality_task = asyncio.create_task(
            self._evaluate_quality(self._full_prompt, response_text)
        )

        return response_text

    @property
    def result(self) -> TrackedLLMCallResult:
        """Access the result container with all metrics."""
        return self._result


# =============================================================================
# SESSION HELPERS
# =============================================================================

def start_session(
    operation_type: str = None,
    **metadata
) -> Any:
    """
    Start a new Observatory session for tracking related LLM calls.
    
    Sessions group multiple LLM calls together (e.g., a multi-turn conversation,
    a complex workflow with multiple agent interactions).
    
    Universal wrapper - works for any project type.
    
    Args:
        operation_type: Type of operation (e.g., "chat", "workflow", "analysis")
        **metadata: Additional session metadata (user_id, conversation_id, etc.)
        
    Returns:
        Session object from Observatory
        
    Example:
        # Start a session
        session = start_session(
            operation_type="job_search_workflow",
            user_id="user_123",
            conversation_id="conv_456"
        )
        
        # Track calls within session
        track_llm_call(..., metadata={"session_id": session.id})
        
        # End session
        end_session(session, success=True)
    """
    return obs.start_session(operation_type=operation_type, **metadata)


def end_session(
    session: Any,
    success: bool = True,
    error: str = None,
    **metadata
) -> None:
    """
    End an Observatory session.
    
    Marks the session as complete and records success/failure status.
    
    Universal wrapper - works for any project type.
    
    Args:
        session: Session object from start_session()
        success: Whether the session completed successfully
        error: Error message if session failed
        **metadata: Additional metadata to store with session
        
    Example:
        session = start_session("chat")
        
        try:
            # Your application logic
            track_llm_call(...)
            end_session(session, success=True)
        except Exception as e:
            end_session(session, success=False, error=str(e))
    """
    return obs.end_session(session, success=success, error=error)

# =============================================================================
# EXPORTS
# =============================================================================

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Observatory instance & components
    'obs',              # Main Observatory instance
    'judge',            # LLM Judge for quality evaluation
    'cache',            # Exact match cache (CacheManager) - in-memory
    'persistent_cache', # Cross-session cache (PersistentCacheManager) - SQLite
    'prefix_cache',     # Prefix cache detector (Azure/Anthropic)
    'semantic_cache',   # Semantic similarity cache (SemanticCache)
    'router',           # Model router for intelligent selection
    'prompts',          # Prompt manager for A/B testing
    'prompt_optimizer', # Prompt compression and token efficiency
    'optimization_tracker', # Phase-based optimization tracking

    # Context manager abstraction (10-step pattern)
    'TrackedLLMCall',       # Main context manager for tracked LLM calls
    'TrackedLLMCallResult', # Result container with metrics
    
    # Execution optimization detectors
    'batch_detector',     
    'parallel_detector',  
    'streaming_detector', 
    'sequential_detector',
    'context_growth_detector',
    'token_efficiency_detector',

    # Universal execution components 
    'batch_processor',     
    'parallel_executor',    
    
    # Main interface
    'track_llm_call',   # Main wrapper with auto-extraction
    
    # Helper functions
    'classify_error',
    'extract_azure_cache_metrics',
    'extract_token_breakdown_from_messages',
    'extract_model_parameters',
    'fire_and_forget_judge',
    'wait_for_pending_judges',
    
    # Session management
    'start_session',
    'end_session',
    
    # Configuration constants
    'PROJECT_NAME',
    'DEFAULT_MODEL',
    'DEFAULT_PROVIDER',
    'CURRENT_PHASE',
    'OBSERVATORY_DB_PATH',
    'PROMPT_VARIANTS',        
    'OPERATION_COMPLEXITY',   
    
    # Data models (re-exported from SDK for convenience)
    'LLMCall',
    'RoutingDecision',
    'CacheMetadata',
    'QualityEvaluation',
    'PromptBreakdown',
    'PromptMetadata',
    'ModelConfig',
    'StreamingMetrics',
    'ExperimentMetadata',
    'ErrorDetails',
    'SemanticCacheResult',

    # Cache classes (for direct instantiation if needed)
    'PersistentCacheManager',
    
    # Helper functions (re-exported from SDK for convenience)
    'create_routing_decision',
    'create_cache_metadata',
    'create_quality_evaluation',
    'create_prompt_metadata',
    'create_prompt_breakdown',
    'create_semantic_cache_metadata',
    'estimate_tokens',
    
    # Enums (re-exported from SDK for convenience)
    'ModelProvider',
    'AgentRole',
]