"""
Observatory Configuration - Career Copilot
COMPREHENSIVE: All Tier 2 metrics with full model support

Supports ALL new Observatory fields:
- PromptBreakdown (auto-extracted from messages)
- PromptMetadata (template versioning)
- QualityEvaluation (all new fields)
- RoutingDecision (routing_strategy field)
- CacheMetadata (all new fields)
"""

import os
import sys
import hashlib
from typing import Optional, List, Dict, Any

# =============================================================================
# MODEL CONFIGURATION (from .env)
# =============================================================================

DEFAULT_MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
DEFAULT_PROVIDER = "AZURE"  # or "OPENAI" if using OpenAI directly

print(f"📦 Observatory using model: {DEFAULT_MODEL}")

# =============================================================================
# OBSERVATORY IMPORTS
# =============================================================================

try:
    from observatory import Observatory, ModelProvider, AgentRole
    from observatory.models import (
        RoutingDecision, 
        CacheMetadata, 
        QualityEvaluation,
        PromptBreakdown,
        PromptMetadata
    )
    FULL_MODELS_AVAILABLE = True
    print("✅ Observatory package imported successfully (full models)")
except ImportError as e:
    print(f"⚠️ Full Observatory import failed: {e}")
    # Try importing without new models for backward compatibility
    try:
        from observatory import Observatory, ModelProvider, AgentRole
        from observatory.models import RoutingDecision, CacheMetadata, QualityEvaluation
        PromptBreakdown = None
        PromptMetadata = None
        FULL_MODELS_AVAILABLE = False
        print("⚠️ Observatory imported without PromptBreakdown/PromptMetadata (older version)")
    except ImportError:
        print("❌ ERROR: Cannot import Observatory package")
        sys.exit(1)

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

OBSERVATORY_DB_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "ai-agent-observatory",
    "observatory.db"
)
OBSERVATORY_DB_PATH = os.path.abspath(OBSERVATORY_DB_PATH)

if not os.path.exists(os.path.dirname(OBSERVATORY_DB_PATH)):
    print(f"⚠️ WARNING: Observatory folder not found at expected location")
    print(f"   Expected: {os.path.dirname(OBSERVATORY_DB_PATH)}")

os.environ['DATABASE_URL'] = f"sqlite:///{OBSERVATORY_DB_PATH}"

# =============================================================================
# INITIALIZE OBSERVATORY
# =============================================================================

obs = Observatory(
    project_name="Career Copilot",
    enabled=True
)

print(f"✅ Observatory initialized for Career Copilot")
print(f"   Database: {OBSERVATORY_DB_PATH}")
print(f"   Project: Career Copilot")
print(f"   Model: {DEFAULT_MODEL}")


# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

def start_tracking_session(operation_type: str, metadata: dict = None):
    """Start a tracking session for an operation."""
    if metadata:
        return obs.start_session(operation_type, **metadata)
    else:
        return obs.start_session(operation_type)


def end_tracking_session(session, success: bool = True, error: str = None):
    """End a tracking session."""
    if session:
        if not success:
            session.success = False
        if error:
            session.error = error
    obs.end_session(session)


# =============================================================================
# PROMPT BREAKDOWN HELPERS
# =============================================================================

def estimate_tokens(text: str) -> int:
    """Estimate tokens (rough: 4 chars ≈ 1 token)."""
    if not text:
        return 0
    return len(text) // 4


def create_prompt_breakdown(
    system_prompt: str = None,
    chat_history: list = None,
    user_message: str = None,
    response_text: str = None
) -> Optional['PromptBreakdown']:
    """
    Create a PromptBreakdown object for tracking prompt components.
    
    Args:
        system_prompt: The system prompt text
        chat_history: List of {"role": "user/assistant", "content": "..."} dicts
        user_message: The current user message
        response_text: The LLM response text
    
    Returns:
        PromptBreakdown object or None if not available
    """
    if PromptBreakdown is None:
        return None
    
    def estimate_history_tokens(history: list) -> int:
        if not history:
            return 0
        total = 0
        for msg in history:
            content = msg.get('content', '')
            total += estimate_tokens(content)
        return total
    
    return PromptBreakdown(
        system_prompt=system_prompt[:2000] if system_prompt else None,
        system_prompt_tokens=estimate_tokens(system_prompt),
        chat_history=chat_history,
        chat_history_tokens=estimate_history_tokens(chat_history),
        chat_history_count=len(chat_history) if chat_history else 0,
        user_message=user_message[:1000] if user_message else None,
        user_message_tokens=estimate_tokens(user_message),
        response_text=response_text[:2000] if response_text else None,
    )


def extract_prompt_breakdown_from_messages(messages: list, response_text: str = None) -> Optional['PromptBreakdown']:
    """
    Extract PromptBreakdown from a standard messages list.
    
    Args:
        messages: List of {"role": "system/user/assistant", "content": "..."} dicts
        response_text: The LLM response text
    
    Returns:
        PromptBreakdown object
    """
    if PromptBreakdown is None:
        return None
    
    system_prompt = None
    chat_history = []
    user_message = None
    
    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')
        
        if role == 'system':
            system_prompt = content
        elif role == 'user':
            # Last user message is the current one
            if user_message:
                chat_history.append({"role": "user", "content": user_message})
            user_message = content
        elif role == 'assistant':
            chat_history.append({"role": "assistant", "content": content})
    
    return create_prompt_breakdown(
        system_prompt=system_prompt,
        chat_history=chat_history,
        user_message=user_message,
        response_text=response_text
    )


# =============================================================================
# PROMPT METADATA HELPERS
# =============================================================================

def create_prompt_metadata(
    template_id: str = None,
    version: str = None,
    compressible_sections: List[str] = None,
    optimization_flags: Dict[str, bool] = None,
    config_version: str = None
) -> Optional['PromptMetadata']:
    """
    Create a PromptMetadata object for template versioning.
    
    Args:
        template_id: Identifier for the prompt template (e.g., "career_copilot_system")
        version: Version string (e.g., "1.0.0")
        compressible_sections: List of section names that could be compressed
        optimization_flags: Dict[str, bool] of optimization flags (e.g., {"fast_mode": True})
        config_version: Overall config version
    
    Returns:
        PromptMetadata object or None if not available
    """
    if PromptMetadata is None:
        return None
    
    return PromptMetadata(
        prompt_template_id=template_id,
        prompt_version=version,
        compressible_sections=compressible_sections,
        optimization_flags=optimization_flags,
        config_version=config_version,
    )


# =============================================================================
# QUALITY EVALUATION HELPERS
# =============================================================================

def create_quality_evaluation(
    score: float,
    reasoning: str = None,
    hallucination: bool = False,
    factual_error: bool = False,
    failure_reason: str = None,
    improvement_suggestion: str = None,
    hallucination_details: str = None,
    evidence_cited: bool = None,
    confidence: float = 0.85,
    judge_model: str = None,
    criteria_scores: Dict[str, float] = None,
    error_category: str = None
) -> QualityEvaluation:
    """
    Create a QualityEvaluation object with ALL fields.
    
    Args:
        score: Quality score 0-10
        reasoning: Explanation for the score
        hallucination: Whether hallucination was detected
        factual_error: Whether factual error was detected
        failure_reason: Category (HALLUCINATION, FACTUAL_ERROR, LOW_QUALITY, VERY_LOW_QUALITY)
        improvement_suggestion: How to improve the response
        hallucination_details: Specific details about what was hallucinated
        evidence_cited: Whether the response cited sources/evidence
        confidence: Judge's confidence in evaluation (0-1)
        judge_model: Model used for judging
        criteria_scores: Dict of individual criteria scores
        error_category: Error category if any
    
    Returns:
        QualityEvaluation object
    """
    # Determine failure_reason if not provided
    if failure_reason is None:
        if hallucination:
            failure_reason = "HALLUCINATION"
        elif factual_error:
            failure_reason = "FACTUAL_ERROR"
        elif score < 3:
            failure_reason = "VERY_LOW_QUALITY"
        elif score < 5:
            failure_reason = "LOW_QUALITY"
    
    return QualityEvaluation(
        judge_score=score,
        judge_model=judge_model,
        reasoning=reasoning,
        hallucination_flag=hallucination,
        confidence_score=confidence,
        criteria_scores=criteria_scores,
        error_category=error_category,
        # Extended fields
        failure_reason=failure_reason,
        improvement_suggestion=improvement_suggestion,
        hallucination_details=hallucination_details,
        factual_error=factual_error,
        evidence_cited=evidence_cited,
    )


# =============================================================================
# ROUTING DECISION HELPERS
# =============================================================================

def create_routing_decision(
    chosen_model: str,
    alternative_models: List[str] = None,
    reasoning: str = None,
    complexity_score: float = None,
    rule_triggered: str = None,
    estimated_cost_savings: float = None,
    routing_strategy: str = None,
    model_scores: Dict[str, float] = None
) -> RoutingDecision:
    """
    Create a RoutingDecision object with ALL fields.
    
    Args:
        chosen_model: The model that was selected
        alternative_models: Other models that were considered
        reasoning: Why this model was chosen
        complexity_score: Estimated task complexity (0-1)
        rule_triggered: Which routing rule was triggered
        estimated_cost_savings: Estimated savings vs default model
        routing_strategy: Strategy used ("cost_optimized", "quality_first", "balanced")
        model_scores: Dict of model -> suitability scores
    
    Returns:
        RoutingDecision object
    """
    return RoutingDecision(
        chosen_model=chosen_model,
        alternative_models=alternative_models or [],
        reasoning=reasoning,
        complexity_score=complexity_score,
        rule_triggered=rule_triggered,
        estimated_cost_savings=estimated_cost_savings,
        routing_strategy=routing_strategy,
        model_scores=model_scores,
    )


# =============================================================================
# CACHE METADATA HELPERS
# =============================================================================

def create_cache_metadata(
    cache_hit: bool,
    cache_key: str = None,
    cache_cluster_id: str = None,
    similarity_score: float = None,
    normalization_strategy: str = None,
    eviction_info: str = None,
    # New fields
    cache_key_candidates: List[str] = None,
    dynamic_fields: List[str] = None,
    content_hash: str = None,
    ttl_seconds: int = None
) -> CacheMetadata:
    """
    Create a CacheMetadata object with ALL fields.
    
    Args:
        cache_hit: Whether this was a cache hit
        cache_key: The cache key used
        cache_cluster_id: Cluster ID for semantic caching
        similarity_score: Similarity to cached entry (0-1)
        normalization_strategy: How the prompt was normalized
        eviction_info: Info about cache eviction
        cache_key_candidates: Alternative keys considered
        dynamic_fields: Fields that were extracted/replaced
        content_hash: Hash of the content for deduplication
        ttl_seconds: Time-to-live for this cache entry
    
    Returns:
        CacheMetadata object
    """
    return CacheMetadata(
        cache_hit=cache_hit,
        cache_key=cache_key,
        cache_cluster_id=cache_cluster_id,
        similarity_score=similarity_score,
        normalization_strategy=normalization_strategy,
        eviction_info=eviction_info,
        cache_key_candidates=cache_key_candidates,
        dynamic_fields=dynamic_fields,
        content_hash=content_hash,
        ttl_seconds=ttl_seconds,
    )


def compute_content_hash(content: str) -> str:
    """Compute a hash for content deduplication."""
    if not content:
        return None
    return hashlib.md5(content.encode()).hexdigest()[:16]


# =============================================================================
# MAIN TRACKING FUNCTION
# =============================================================================

def track_llm_call(
    model_name: str = None,  # Defaults to AZURE_OPENAI_DEPLOYMENT_NAME
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    latency_ms: float = 0,
    agent_name: str = None,
    operation: str = None,
    metadata: dict = None,
    prompt: str = None,
    response_text: str = None,
    # Tier 2: Routing & Cache
    routing_decision: RoutingDecision = None,
    cache_metadata: CacheMetadata = None,
    # Tier 2: Quality Evaluation
    quality_evaluation: QualityEvaluation = None,
    # Tier 2: Prompt Analysis
    prompt_breakdown: 'PromptBreakdown' = None,
    prompt_metadata: 'PromptMetadata' = None,
    # Auto-extraction options
    messages: list = None,  # Auto-extract breakdown from messages
    system_prompt: str = None,  # For manual breakdown
    user_message: str = None,  # For manual breakdown
    chat_history: list = None,  # For manual breakdown
    # Flags
    required_retry: bool = False,
    success: bool = True,
    error: str = None,
):
    """
    Record an LLM call in Observatory with full Tier 2 metrics.
    
    Model name defaults to AZURE_OPENAI_DEPLOYMENT_NAME from .env
    
    Args:
        model_name: Name of the model (default: from env var)
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        latency_ms: Latency in milliseconds
        agent_name: Name of the agent/plugin
        operation: Operation type (e.g., "streamlit_chat", "improve_bullet")
        metadata: Additional metadata dict
        prompt: Full prompt text
        response_text: LLM response text
        routing_decision: RoutingDecision object
        cache_metadata: CacheMetadata object
        quality_evaluation: QualityEvaluation object
        prompt_breakdown: PromptBreakdown object (or auto-extracted)
        prompt_metadata: PromptMetadata object
        messages: Messages list to auto-extract breakdown from
        system_prompt: System prompt for manual breakdown
        user_message: User message for manual breakdown
        chat_history: Chat history for manual breakdown
        required_retry: Whether this call needed a retry
        success: Whether the call succeeded
        error: Error message if failed
    
    Example:
        track_llm_call(
            prompt_tokens=1500,
            completion_tokens=300,
            latency_ms=2500,
            operation="streamlit_chat",
            messages=chat_history,
            response_text="Here's my advice...",
            prompt_metadata=prompt_meta,
            quality_evaluation=quality_eval
        )
    """
    # Use default model from env if not specified
    if model_name is None:
        model_name = DEFAULT_MODEL
    
    # Initialize metadata
    enhanced_metadata = metadata.copy() if metadata else {}
    
    # Add prompt length metrics
    if prompt:
        enhanced_metadata['prompt_length_chars'] = len(prompt)
    if response_text:
        enhanced_metadata['response_length_chars'] = len(response_text)
    if required_retry:
        enhanced_metadata['required_retry'] = True
    
    # Auto-extract prompt breakdown if messages provided
    if prompt_breakdown is None and messages:
        prompt_breakdown = extract_prompt_breakdown_from_messages(messages, response_text)
    
    # Manual prompt breakdown if components provided
    if prompt_breakdown is None and (system_prompt or user_message or chat_history):
        prompt_breakdown = create_prompt_breakdown(
            system_prompt=system_prompt,
            chat_history=chat_history,
            user_message=user_message,
            response_text=response_text
        )
    
    # Determine provider
    provider = ModelProvider.AZURE if DEFAULT_PROVIDER == "AZURE" else ModelProvider.OPENAI
    
    # Build record_call kwargs
    call_kwargs = {
        'provider': provider,
        'model_name': model_name,
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'latency_ms': latency_ms,
        'agent_name': agent_name,
        'operation': operation,
        'metadata': enhanced_metadata,
        'prompt': prompt,
        'response_text': response_text,
        'routing_decision': routing_decision,
        'cache_metadata': cache_metadata,
        'quality_evaluation': quality_evaluation,
        'success': success,
        'error': error,
    }
    
    # Add new fields if supported
    if prompt_breakdown is not None:
        call_kwargs['prompt_breakdown'] = prompt_breakdown
    if prompt_metadata is not None:
        call_kwargs['prompt_metadata'] = prompt_metadata
    
    # Record the call
    session_started = False
    try:
        obs.record_call(**call_kwargs)
    except ValueError as e:
        if "No active session" in str(e):
            # No session exists - create a temporary one
            temp_session = obs.start_session("llm_call")
            session_started = True
            obs.record_call(**call_kwargs)
            obs.end_session(temp_session)
        else:
            raise


# =============================================================================
# AGENT ROLE MAPPING
# =============================================================================

AGENT_ROLE_MAP = {
    "ResumeMatching": AgentRole.ANALYST,
    "JobPlugin": AgentRole.ANALYST,
    "DatabaseQueryPlugin": AgentRole.ANALYST,
    "ResumeTailoring": AgentRole.FIXER,
    "SelfImprovingMatch": AgentRole.REVIEWER,
    "ResumePreprocessor": AgentRole.ANALYST,
    "JobPreprocessor": AgentRole.ANALYST,
}


def get_agent_role(plugin_name: str) -> AgentRole:
    """Get the appropriate AgentRole for a plugin."""
    return AGENT_ROLE_MAP.get(plugin_name, AgentRole.ANALYST)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Config
    'DEFAULT_MODEL',
    'DEFAULT_PROVIDER',
    'FULL_MODELS_AVAILABLE',
    # Observatory
    'obs',
    'Observatory',
    'ModelProvider',
    'AgentRole',
    # Models
    'RoutingDecision',
    'CacheMetadata',
    'QualityEvaluation',
    'PromptBreakdown',
    'PromptMetadata',
    # Session functions
    'start_tracking_session',
    'end_tracking_session',
    'track_llm_call',
    # Helper functions
    'create_prompt_breakdown',
    'extract_prompt_breakdown_from_messages',
    'create_prompt_metadata',
    'create_quality_evaluation',
    'create_routing_decision',
    'create_cache_metadata',
    'compute_content_hash',
    'estimate_tokens',
    'get_agent_role',
]