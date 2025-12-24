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
"""

import os
import re
import hashlib
from typing import Optional, Dict, List, Any
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# IMPORT FROM OBSERVATORY SDK
# =============================================================================

from observatory import (
    # Core
    Observatory,
    ModelProvider,
    AgentRole,
    
    # SDK Components
    LLMJudge,
    CacheManager,
    ModelRouter,
    PromptManager,
    
    # Convenience functions (rename to avoid conflict with our wrapper)
    track_llm_call as _sdk_track_llm_call,
    create_routing_decision,
    create_cache_metadata,
    create_quality_evaluation,
    create_prompt_metadata,
    create_prompt_breakdown,
    estimate_tokens,
    
    # Models for type hints (existing)
    RoutingDecision,
    CacheMetadata,
    QualityEvaluation,
    PromptBreakdown,
    PromptMetadata,
    
    # Additional models for complete schema
    ModelConfig,
    StreamingMetrics,
    ExperimentMetadata,
    ErrorDetails,

    # Semantic Cache
    SemanticCache,
    SemanticCacheResult,
    create_semantic_cache_metadata,
)

# =============================================================================
# PHASE CONFIGURATION
# =============================================================================

# Set via environment variable or .env file:
#   OBSERVATORY_PHASE=baseline   (Phase 1: Track only, no optimizations)
#   OBSERVATORY_PHASE=optimized  (Phase 2: Routing, caching, quality enabled)
#
# Or run with: OBSERVATORY_PHASE=optimized python your_app.py

CURRENT_PHASE = os.getenv("OBSERVATORY_PHASE", "baseline")

if CURRENT_PHASE not in ("baseline", "optimized"):
    print(f"⚠️ Invalid OBSERVATORY_PHASE '{CURRENT_PHASE}', defaulting to 'baseline'")
    CURRENT_PHASE = "baseline"

# =============================================================================
# PROJECT CONFIGURATION - CAREER COPILOT
# =============================================================================

PROJECT_NAME = "Career Copilot"

# Model configuration - Azure OpenAI
DEFAULT_PROVIDER = ModelProvider.AZURE
DEFAULT_MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")

# Database path - adjust to your Observatory location
OBSERVATORY_DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "ai-agent-observatory", "observatory.db")
)
if 'DATABASE_URL' not in os.environ:
    os.environ['DATABASE_URL'] = f"sqlite:///{OBSERVATORY_DB_PATH}"

print(f"📦 Observatory Config: {PROJECT_NAME}")
print(f"   Database: {OBSERVATORY_DB_PATH}")
print(f"   Model: {DEFAULT_MODEL}")
print(f"   Phase: {CURRENT_PHASE}")

# =============================================================================
# INITIALIZE OBSERVATORY
# =============================================================================

obs = Observatory(
    project_name=PROJECT_NAME,
    enabled=True,
)

# =============================================================================
# CONFIGURE LLM JUDGE - CAREER COPILOT DOMAIN
# =============================================================================

judge = LLMJudge(
    observatory=obs,
    
    # Operations worth evaluating (high-value career advice outputs)
    operations={
        "improve_bullet",
        "generate_change_report",
        "deep_analyze_job",
        "deep_analyze_with_guidance",
        "critique_match",
        "streamlit_chat",
        "cli_chat_message",
        "generate_sql",
        "explain_recent_match",  
        "refine_analysis", 
    },
    
    # Operations to skip (low-value or simple)
    skip_operations={
        "job_search",
        "save_jobs",
        "list_resumes",
        "quick_score_job",
        "generate_refinements",  # Low-value planning operation
    },
    
    # Evaluate 100% of judge-worthy calls
    sample_rate=1.0,
    
    # Career advice domain criteria (must sum to 1.0)
    criteria={
        "relevance": 0.25,      # How relevant is the career advice?
        "accuracy": 0.25,       # Is the information correct?
        "helpfulness": 0.25,    # Does it help the user's job search?
        "professionalism": 0.15, # Is it professionally appropriate?
        "clarity": 0.10,        # Is it clear and well-structured?
    },
    
    # Career Copilot domain context
    domain_context="career advice, resume optimization, and job matching",
    
    # Model for judging (same as main)
    judge_model=DEFAULT_MODEL,
    
    # Track judge calls in Observatory
    track_judge_calls=True,
)

# =============================================================================
# CONFIGURE CACHE MANAGER - CAREER COPILOT OPERATIONS
# =============================================================================

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
    default_ttl=3600,       # 1 hour default
    max_entries=1000,       # Max cache size
    normalize_prompts=True, # Normalize for better cache hits
)

# =============================================================================
# CONFIGURE SEMANTIC CACHE - VECTOR-BASED SIMILARITY MATCHING
# =============================================================================
# Unlike CacheManager (exact hash match), SemanticCache uses embeddings to find
# semantically similar prompts. "Find Python jobs" ≈ "Search for Python positions"
#
# pip install chromadb  (required dependency)

semantic_cache = SemanticCache(
    observatory=obs,
    
    operations={
        # ═══════════════════════════════════════════════════════════════
        # HIGH VALUE - SQL Generation
        # ═══════════════════════════════════════════════════════════════
        # Users ask similar questions in different ways:
        #   "show me jobs from Deloitte" ≈ "find Deloitte jobs" ≈ "Deloitte positions"
        # 
        # Analysis showed ~20% duplicate patterns in generate_sql
        "generate_sql": {
            "ttl": 86400,       # 24 hours (SQL patterns are stable)
            "threshold": 0.95,  # 95% - high precision (exact SQL matters)
            "cluster_id": "sql_generation",
        },
        
        # ═══════════════════════════════════════════════════════════════
        # HIGH VALUE - Quick Scoring
        # ═══════════════════════════════════════════════════════════════
        # Same resume scored against many similar jobs:
        #   "Python Developer at Google" ≈ "Python Engineer at Meta"
        #
        # High volume operation - significant savings potential
        "quick_score_job": {
            "ttl": 3600,        # 1 hour (job relevance can change)
            "threshold": 0.90,  # 90% - jobs in same domain match well
            "cluster_id": "resume_matching",
        },
        
        # ═══════════════════════════════════════════════════════════════
        # MEDIUM VALUE - Deep Analysis
        # ═══════════════════════════════════════════════════════════════
        # Expensive operation (~2000+ tokens per call)
        # Worth caching aggressively with lower threshold
        "deep_analyze_job": {
            "ttl": 3600,        # 1 hour
            "threshold": 0.88,  # 88% - lower threshold for expensive ops
            "cluster_id": "deep_analysis",
        },
    },
    
    # Defaults for any operation not explicitly configured
    default_ttl=3600,
    default_threshold=0.92,
    
    # Master switch - set to False to disable without removing config
    enabled=False,
)

# =============================================================================
# CONFIGURE MODEL ROUTER - CAREER COPILOT ROUTING RULES
# =============================================================================

router = ModelRouter(
    observatory=obs,
    default_model=DEFAULT_MODEL,
    fallback_model="gpt-4o-mini",
    
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

# =============================================================================
# CONFIGURE PROMPT MANAGER (A/B TESTING)
# =============================================================================

prompts = PromptManager(observatory=obs)

# Example: Register system prompt with variants for Career Copilot
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


# =============================================================================
# HELPER FUNCTIONS: Error Classification & Cache Key Generation
# =============================================================================

def classify_error(error: Exception, operation: str = None) -> dict:
    """
    Classify errors for tracking. Returns dict with error_type, error_code, error_category.
    
    Use with ** unpacking in track_llm_call:
        track_llm_call(
            ...
            **classify_error(e, operation="generate_sql"),
            retry_count=0,
            ...
        )
    
    Categories:
        - database_schema: Column/table not found errors
        - database: Other database errors
        - network: Timeout errors
        - throttling: Rate limit errors
        - data_missing: Not found errors
        - input_error: Validation errors
        - response_format: JSON parse errors
        - unclassified: Unknown errors
    
    Args:
        error: The exception that was caught
        operation: Optional operation name for context
        
    Returns:
        Dict with error_type, error_code, error_category
    """
    error_str = str(error).lower()
    error_type = type(error).__name__
    
    # Database schema errors (e.g., "no such column: salary")
    if "no such column" in error_str or "no such table" in error_str:
        return {
            "error_type": error_type,
            "error_code": "SCHEMA_ERROR",
            "error_category": "database_schema"
        }
    
    # Timeout errors
    elif "timeout" in error_str or "timed out" in error_str:
        return {
            "error_type": error_type,
            "error_code": "TIMEOUT",
            "error_category": "network"
        }
    
    # Rate limiting
    elif "rate limit" in error_str or "429" in error_str:
        return {
            "error_type": error_type,
            "error_code": "RATE_LIMIT",
            "error_category": "throttling"
        }
    
    # Not found errors
    elif re.search(r"not found|no .* found", error_str):
        return {
            "error_type": error_type,
            "error_code": "NOT_FOUND",
            "error_category": "data_missing"
        }
    
    # Validation errors
    elif "invalid" in error_str or "validation" in error_str:
        return {
            "error_type": error_type,
            "error_code": "VALIDATION",
            "error_category": "input_error"
        }
    
    # JSON parse errors
    elif "json" in error_str or "parse" in error_str or "decode" in error_str:
        return {
            "error_type": error_type,
            "error_code": "PARSE_ERROR",
            "error_category": "response_format"
        }
    
    # SQLite specific (catch-all for other DB errors)
    elif "sqlite" in error_type.lower() or "database" in error_str:
        return {
            "error_type": error_type,
            "error_code": "DB_ERROR",
            "error_category": "database"
        }
    
    # Connection errors
    elif "connection" in error_str or "connect" in error_str:
        return {
            "error_type": error_type,
            "error_code": "CONNECTION_ERROR",
            "error_category": "network"
        }
    
    # Authentication errors
    elif "auth" in error_str or "unauthorized" in error_str or "401" in error_str:
        return {
            "error_type": error_type,
            "error_code": "AUTH_ERROR",
            "error_category": "authentication"
        }
    
    # Default - unclassified
    else:
        return {
            "error_type": error_type,
            "error_code": "UNKNOWN",
            "error_category": "unclassified"
        }


def generate_cache_key(operation: str, *key_parts) -> str:
    """
    Generate a cache key from operation + identifying content.
    
    Creates a deterministic hash that can be used to:
    1. Identify duplicate calls (same key = potential cache hit)
    2. Track cache performance
    3. Enable semantic caching when activated
    
    Usage:
        # For resume-job matching
        cache_key = generate_cache_key("quick_score_job", resume_text[:500], job.get('id'))
        
        # For SQL generation
        cache_key = generate_cache_key("generate_sql", question)
        
        # For deep analysis
        cache_key = generate_cache_key("deep_analyze_job", resume_text[:500], job.get('id'))
    
    Args:
        operation: The operation name (e.g., "quick_score_job", "generate_sql")
        *key_parts: Variable arguments that uniquely identify this call
                   (e.g., resume text, job ID, user query)
    
    Returns:
        16-character hex hash string
    """
    # Combine operation with key parts, truncating each part to avoid huge keys
    combined = f"{operation}:" + ":".join(str(p)[:500] for p in key_parts if p)
    return hashlib.md5(combined.encode()).hexdigest()[:16]


def calculate_complexity_score(user_message: str, tool_call_count: int = 0) -> float:
    """
    Calculate query complexity score (0.0 - 1.0) for model routing decisions.
    
    Higher scores indicate more complex queries that may benefit from 
    premium models. Lower scores suggest simpler queries suitable for 
    cheaper models.
    
    Used by:
    - Model routing: Route simple queries to gpt-4o-mini, complex to gpt-4o
    - Cost optimization: Identify over-provisioned calls
    - Dashboard Story 3: Routing analysis
    
    Usage:
        complexity = calculate_complexity_score(user_message, tool_count=2)
        routing_decision = create_routing_decision(
            chosen_model="gpt-4o-mini",
            complexity_score=complexity,
            ...
        )
    
    Args:
        user_message: The user's input message
        tool_count: Number of tools/functions called (higher = more complex)
    
    Returns:
        Float between 0.0 (simple) and 1.0 (complex)
    """
    if not user_message:
        return 0.0
    
    score = 0.0
    message_lower = user_message.lower()
    
    # Length factor (longer messages tend to be more complex)
    score += min(0.3, len(user_message) / 1000)
    
    # Simple patterns reduce complexity
    simple_patterns = ['what is', 'who is', 'when', 'where', 'how many', 'list', 'show me', 'find']
    if any(p in message_lower for p in simple_patterns):
        score -= 0.1
    
    # Complex patterns increase complexity
    complex_patterns = ['analyze', 'compare', 'evaluate', 'explain why', 'recommend', 
                        'improve', 'critique', 'summarize', 'synthesize', 'create']
    if any(p in message_lower for p in complex_patterns):
        score += 0.2
    
    # Multi-step requests are more complex
    multi_step_patterns = ['then', 'after that', 'also', 'and then', 'finally']
    if any(p in message_lower for p in multi_step_patterns):
        score += 0.15
    
    # Tool usage suggests complexity
    score += min(0.3, tool_call_count * 0.1)
    
    # Clamp to valid range
    return max(0.0, min(1.0, score))


def calculate_prefix_hash(system_prompt: str, static_content: str = "") -> str:
    """
    Hash the static prefix portion of a prompt for prefix caching detection.
    
    Many LLM calls share identical prefixes (system prompt + context) but differ
    only in the variable portion (e.g., different job descriptions). This hash
    identifies shared prefixes to detect caching opportunities.
    
    Used by:
    - Prefix cache detection: Find calls with identical prefixes
    - Cost optimization: Recommend prompt prefix caching
    - Dashboard Story 2: Cache opportunity analysis
    
    Usage:
        # In quick_score_job - resume is static, job varies
        prefix_hash = calculate_prefix_hash(SYSTEM_PROMPT, resume_text)
        
        track_llm_call(
            prompt_prefix_hash=prefix_hash,
            ...
        )
    
    Args:
        system_prompt: The system prompt text
        static_content: Other static content (e.g., resume text that doesn't change)
    
    Returns:
        16-character hex hash string
    """
    prefix = f"{system_prompt}\n{static_content}"
    return hashlib.md5(prefix.encode()).hexdigest()[:16]

# =============================================================================
# HELPER FUNCTIONS: Token Breakdown & Model Parameters Extraction
# =============================================================================

def extract_token_breakdown_from_messages(
    messages: List[Dict] = None,
    system_prompt: str = None,
    user_message: str = None,
    chat_history: Any = None,
    conversation_memory: Any = None,
) -> Dict[str, int]:
    """
    Extract token breakdown from various message formats.
    
    Auto-populates system_prompt_tokens, user_message_tokens, chat_history_tokens,
    and conversation_context_tokens for comprehensive token tracking.
    
    Args:
        messages: Full messages array (OpenAI/SK format)
        system_prompt: System prompt text
        user_message: User message text
        chat_history: Semantic Kernel ChatHistory object
        conversation_memory: ConversationMemory object
        
    Returns:
        Dict with all token breakdown fields
    """
    breakdown = {
        'system_prompt_tokens': 0,
        'user_message_tokens': 0,
        'chat_history_tokens': 0,
        'conversation_context_tokens': 0,
    }
    
    # Method 1: Extract from messages array (OpenAI/Azure format)
    if messages:
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            tokens = estimate_tokens(content) if content else 0
            
            if role == 'system':
                breakdown['system_prompt_tokens'] += tokens
            elif role == 'user':
                breakdown['user_message_tokens'] += tokens
            elif role in ['assistant', 'function']:
                breakdown['chat_history_tokens'] += tokens
    
    # Method 2: Extract from individual strings
    else:
        if system_prompt:
            breakdown['system_prompt_tokens'] = estimate_tokens(system_prompt)
        
        if user_message:
            breakdown['user_message_tokens'] = estimate_tokens(user_message)
        
        # Method 3: Extract from Semantic Kernel ChatHistory
        if chat_history:
            try:
                if hasattr(chat_history, 'messages'):
                    history_tokens = 0
                    for msg in chat_history.messages:
                        # Skip system messages (already counted)
                        if hasattr(msg, 'role') and msg.role != 'system':
                            content = str(msg.content) if hasattr(msg, 'content') else ''
                            history_tokens += estimate_tokens(content)
                    breakdown['chat_history_tokens'] = history_tokens
            except Exception:
                pass  # Silent fail

        # Also try from conversation_memory if it has chat_history stored
        if not breakdown['chat_history_tokens'] and conversation_memory:
            try:
                if hasattr(conversation_memory, 'chat_history'):
                    history = conversation_memory.chat_history
                    if hasattr(history, 'messages'):
                        history_tokens = 0
                        for msg in history.messages:
                            if hasattr(msg, 'role') and msg.role != 'system':
                                content = str(msg.content) if hasattr(msg, 'content') else ''
                                history_tokens += estimate_tokens(content)
                        breakdown['chat_history_tokens'] = history_tokens
            except Exception:
                pass  # Silent fail
    
    # Method 4: Extract from ConversationMemory context
    if conversation_memory:
        try:
            if hasattr(conversation_memory, 'get_context_for_prompt'):
                context_text = conversation_memory.get_context_for_prompt()
                if context_text and context_text != "No prior context.":
                    breakdown['conversation_context_tokens'] = estimate_tokens(context_text)
        except Exception:
            pass  # Silent fail
    
    return breakdown


def extract_model_parameters(
    client: Any = None,
    execution_settings: Any = None,
    temperature: float = None,
    max_tokens: int = None,
    top_p: float = None,
) -> Dict[str, Any]:
    """
    Extract model parameters from various sources.
    
    Tries to extract from execution_settings (Semantic Kernel) or client object.
    
    Args:
        client: OpenAI/Azure client object
        execution_settings: Semantic Kernel execution settings
        temperature: Explicit temperature value
        max_tokens: Explicit max tokens value
        top_p: Explicit top_p value
        
    Returns:
        Dict with temperature, max_tokens, top_p
    """
    params = {
        'temperature': temperature,
        'max_tokens': max_tokens,
        'top_p': top_p,
    }
    
    # Try to extract from execution_settings (Semantic Kernel)
    if execution_settings:
        try:
            if hasattr(execution_settings, 'temperature'):
                params['temperature'] = params['temperature'] or execution_settings.temperature
            if hasattr(execution_settings, 'max_tokens'):
                params['max_tokens'] = params['max_tokens'] or execution_settings.max_tokens
            if hasattr(execution_settings, 'top_p'):
                params['top_p'] = params['top_p'] or execution_settings.top_p
        except Exception:
            pass  # Silent fail
    
    # Try to extract from client (OpenAI/Azure)
    if client and not all(params.values()):
        try:
            if hasattr(client, 'temperature'):
                params['temperature'] = params['temperature'] or client.temperature
            if hasattr(client, 'max_tokens'):
                params['max_tokens'] = params['max_tokens'] or client.max_tokens
            if hasattr(client, 'top_p'):
                params['top_p'] = params['top_p'] or client.top_p
        except Exception:
            pass  # Silent fail
    
    return params

# =============================================================================
# WRAPPER: track_llm_call (COMPLETE 139 FIELD SUPPORT)
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
    
    # NEW: CONVERSATION LINKING
    conversation_id: str = None,
    turn_number: int = None,
    parent_call_id: str = None,
    user_id: str = None,
    
    # NEW: MODEL CONFIGURATION
    temperature: float = None,
    max_tokens: int = None,
    top_p: float = None,
    model_config: ModelConfig = None,
    
    # NEW: TOKEN BREAKDOWN (Top-level for fast queries)
    system_prompt_tokens: int = None,
    user_message_tokens: int = None,
    chat_history_tokens: int = None,
    conversation_context_tokens: int = None,
    tool_definitions_tokens: int = None,
    
    # NEW: TOOL/FUNCTION CALLING
    tool_calls_made: List[Dict[str, Any]] = None,
    tool_call_count: int = None,
    tool_execution_time_ms: float = None,
    
    # NEW: STREAMING
    time_to_first_token_ms: float = None,
    streaming_metrics: StreamingMetrics = None,
    
    # NEW: ERROR DETAILS
    error_type: str = None,
    error_code: str = None,
    error_category: str = None,  # Added for classify_error() support
    retry_count: int = None,
    error_details: ErrorDetails = None,
    
    # NEW: CACHED TOKENS
    cached_prompt_tokens: int = None,
    cached_token_savings: float = None,
    
    # NEW: OBSERVABILITY
    trace_id: str = None,
    request_id: str = None,
    environment: str = None,

    # NEW: PREFIX CACHE DETECTION
    prompt_prefix_hash: str = None,
    
    # NEW: EXPERIMENT TRACKING
    experiment_id: str = None,
    control_group: bool = None,
    experiment_metadata: ExperimentMetadata = None,
    
    # CUSTOM METADATA
    metadata: dict = None,
):
    """
    Track an LLM call with auto-filled defaults for Career Copilot.
    
    Supports all 139 fields across 3 tiers plus new conversation linking,
    model config, tool tracking, streaming, error details, experiments, and observability.
    
    Args:
        # TIER 1 - Core (always include)
        model_name: Model used (defaults to DEFAULT_MODEL)
        prompt_tokens: Input token count
        completion_tokens: Output token count
        latency_ms: Response time in ms
        agent_name: Name of agent/plugin
        agent_role: Role (analyst, reviewer, writer, retriever, planner, formatter, fixer, orchestrator, custom)
        operation: Operation name
        success: Whether call succeeded
        error: Error message if failed
        
        # PROMPT CONTENT (Tier 2)
        prompt: Combined prompt text
        response_text: Response from model
        prompt_normalized: Normalized prompt for cache key generation
        system_prompt: System prompt (tracked separately)
        user_message: User message (tracked separately)
        messages: Full conversation as [{role, content}, ...]
        
        # OPTIMIZATION TRACKING (Tier 2-3)
        routing_decision: Routing metadata
        cache_metadata: Cache metadata
        quality_evaluation: Quality evaluation
        prompt_breakdown: Prompt component breakdown
        prompt_metadata: Prompt template metadata
        prompt_variant_id: A/B test variant ID
        test_dataset_id: Test dataset ID
        
        # NEW: CONVERSATION LINKING
        conversation_id: Conversation identifier (links multi-turn chats)
        turn_number: Turn number in conversation (1, 2, 3...)
        parent_call_id: Parent call ID (for retries/branches)
        user_id: User identifier
        
        # NEW: MODEL CONFIGURATION
        temperature: Model temperature setting
        max_tokens: Max tokens limit
        top_p: Top-p sampling parameter
        llm_config: Full ModelConfig object with all settings
        
        # NEW: TOKEN BREAKDOWN (Top-level for fast queries)
        system_prompt_tokens: System prompt token count
        user_message_tokens: User message token count
        chat_history_tokens: Chat history token count
        conversation_context_tokens: Conversation memory/state tokens
        tool_definitions_tokens: Function calling schema tokens
        
        # NEW: TOOL/FUNCTION CALLING
        tool_calls_made: List of tool calls with details
        tool_call_count: Number of tools called
        tool_execution_time_ms: Total tool execution time
        
        # NEW: STREAMING
        time_to_first_token_ms: Time to first token (TTFT)
        streaming_metrics: Full StreamingMetrics object
        
        # NEW: ERROR DETAILS
        error_type: Error classification (RATE_LIMIT, TIMEOUT, etc.)
        error_code: Provider error code (429, 500, etc.)
        error_category: Error category from classify_error()
        retry_count: Number of retries attempted
        error_details: Full ErrorDetails object
        
        # NEW: CACHED TOKENS
        cached_prompt_tokens: Tokens served from cache
        cached_token_savings: Cost saved via caching
        
        # NEW: OBSERVABILITY
        trace_id: OpenTelemetry trace ID
        request_id: Provider request ID
        environment: Deployment environment (dev/staging/prod)
        
        # NEW: EXPERIMENT TRACKING
        experiment_id: A/B test experiment ID
        control_group: Is this control group?
        experiment_metadata: Full ExperimentMetadata object
        
        # CUSTOM METADATA
        metadata: Additional metadata dict
    
    Returns:
        LLMCall object
    """
    # ⭐ AUTO-EXTRACT: Token breakdown and model parameters
    # This ensures ALL calls get Tier 2 fields populated automatically
    
    # Extract token breakdown if not explicitly provided
    if not system_prompt_tokens and not user_message_tokens and not chat_history_tokens:
        token_breakdown = extract_token_breakdown_from_messages(
            messages=messages,
            system_prompt=system_prompt,
            user_message=user_message,
            chat_history=None,  # We can add support for chat_history param if needed
            conversation_memory=metadata.get('conversation_memory') if metadata else None
        )
        
        # Use extracted values
        system_prompt_tokens = system_prompt_tokens or token_breakdown['system_prompt_tokens']
        user_message_tokens = user_message_tokens or token_breakdown['user_message_tokens']
        chat_history_tokens = chat_history_tokens or token_breakdown['chat_history_tokens']
        conversation_context_tokens = conversation_context_tokens or token_breakdown['conversation_context_tokens']
    
    # Extract model parameters if not explicitly provided
    if temperature is None or max_tokens is None or top_p is None:
        model_params = extract_model_parameters(
            execution_settings=metadata.get('execution_settings') if metadata else None,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        
        temperature = temperature if temperature is not None else model_params['temperature']
        max_tokens = max_tokens if max_tokens is not None else model_params['max_tokens']
        top_p = top_p if top_p is not None else model_params['top_p']
    
    # ⭐ NEW: Auto-create prompt_breakdown if system_prompt or user_message provided
    # This ensures these fields get extracted to columns for Stories 2 & 6
    if (system_prompt or user_message) and not prompt_breakdown:
        prompt_breakdown = create_prompt_breakdown(
            system_prompt=system_prompt,
            user_message=user_message,
            system_prompt_tokens=system_prompt_tokens,
            user_message_tokens=user_message_tokens,
            chat_history_tokens=chat_history_tokens,
            response_text=response_text,
        )
    
    # Convert string agent_role to AgentRole enum if provided
    role_enum = None
    if agent_role:
        try:
            role_enum = AgentRole(agent_role)
        except ValueError:
            # If not a valid enum value, store in metadata instead
            if metadata is None:
                metadata = {}
            metadata['agent_role_str'] = agent_role
    
    # ⭐ CLEAN METADATA: Remove non-serializable objects before saving
    if metadata:
        metadata.pop('conversation_memory', None)
        metadata.pop('execution_settings', None)
    
    return _sdk_track_llm_call(
        observatory=obs,
        model_name=model_name or DEFAULT_MODEL,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=max(latency_ms, 0.001),
        provider=DEFAULT_PROVIDER,
        agent_name=agent_name,
        agent_role=role_enum,
        operation=operation,
        success=success,
        error=error,
        prompt=prompt,
        response_text=response_text,
        prompt_normalized=prompt_normalized,
        system_prompt=system_prompt,
        user_message=user_message,
        messages=messages,
        routing_decision=routing_decision,
        cache_metadata=cache_metadata,
        quality_evaluation=quality_evaluation,
        prompt_breakdown=prompt_breakdown,
        prompt_metadata=prompt_metadata,
        prompt_variant_id=prompt_variant_id,
        test_dataset_id=test_dataset_id,
        
        # NEW: Conversation linking
        conversation_id=conversation_id,
        turn_number=turn_number,
        parent_call_id=parent_call_id,
        user_id=user_id,
        
        # NEW: Model configuration
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        model_config=model_config,
        
        # NEW: Token breakdown
        system_prompt_tokens=system_prompt_tokens,
        user_message_tokens=user_message_tokens,
        chat_history_tokens=chat_history_tokens,
        conversation_context_tokens=conversation_context_tokens,
        tool_definitions_tokens=tool_definitions_tokens,
        
        # NEW: Tool tracking
        tool_calls_made=tool_calls_made,
        tool_call_count=tool_call_count,
        tool_execution_time_ms=tool_execution_time_ms,
        
        # NEW: Streaming
        time_to_first_token_ms=time_to_first_token_ms,
        streaming_metrics=streaming_metrics,
        
        # NEW: Error details
        error_type=error_type,
        error_code=error_code,
        retry_count=retry_count,
        error_details=error_details,
        
        # NEW: Cached tokens
        cached_prompt_tokens=cached_prompt_tokens,
        cached_token_savings=cached_token_savings,
        
        # NEW: Observability
        trace_id=trace_id,
        request_id=request_id,
        environment=environment,
        
        # NEW: Prefix cache detection
        prompt_prefix_hash=prompt_prefix_hash,
        
        # NEW: Experiment tracking
        experiment_id=experiment_id,
        control_group=control_group,
        experiment_metadata=experiment_metadata,
        
        metadata=metadata,
    )



# =============================================================================
# SESSION HELPERS
# =============================================================================

def start_session(operation_type: str = None, **metadata):
    """Start a tracking session."""
    return obs.start_session(operation_type, **metadata)


def end_session(session, success: bool = True, error: str = None):
    """End a tracking session."""
    return obs.end_session(session, success=success, error=error)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configured instances
    'obs',
    'judge',
    'cache',
    'router',
    'prompts',
    'semantic_cache',
    
    # Config values
    'PROJECT_NAME',
    'DEFAULT_MODEL',
    'DEFAULT_PROVIDER',
    'CURRENT_PHASE',
    
    # Functions (SDK-matching names)
    'track_llm_call',
    'start_session',
    'end_session',
    
    # Helper functions for error classification and cache keys
    'classify_error',
    'generate_cache_key',
    'calculate_complexity_score',
    'calculate_prefix_hash',
    
    # Re-exported for convenience
    'create_routing_decision',
    'create_cache_metadata',
    'create_quality_evaluation',
    'create_prompt_metadata',
    'create_prompt_breakdown',
    'estimate_tokens',
    
    # Types for type hints (existing)
    'RoutingDecision',
    'CacheMetadata',
    'QualityEvaluation',
    'PromptBreakdown',
    'PromptMetadata',
    'AgentRole',
    
    # Additional types
    'ModelConfig',
    'StreamingMetrics',
    'ExperimentMetadata',
    'ErrorDetails',

    # Semantic cache helpers
    'SemanticCacheResult',
    'create_semantic_cache_metadata',
]