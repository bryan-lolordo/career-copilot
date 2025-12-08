"""
Observatory Configuration - Career Copilot
UPDATED: Discovery Mode - Tier 2 metrics, no auto-routing defaults
"""

import os
import sys

try:
    from observatory import Observatory, ModelProvider, AgentRole
    from observatory.models import RoutingDecision, CacheMetadata, QualityEvaluation
    print("✅ Observatory package imported successfully")
except ImportError as e:
    print(f"❌ ERROR: Cannot import Observatory package")
    print(f"   {e}")
    sys.exit(1)

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

obs = Observatory(
    project_name="Career Copilot",
    enabled=True
)

print(f"✅ Observatory initialized for Career Copilot")
print(f"   Database: {OBSERVATORY_DB_PATH}")
print(f"   Project: Career Copilot")


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


def track_llm_call(
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: float,
    agent_name: str = None,
    operation: str = None,
    metadata: dict = None,
    prompt: str = None,
    response_text: str = None,
    routing_decision = None,
    cache_metadata = None,
    quality_evaluation = None,
    required_retry: bool = False  # NEW - Track if this call needed retry
):
    """
    Record an LLM call in Observatory.
    Creates a temporary session if none exists.
    
    DISCOVERY MODE: Automatically adds Tier 2 metrics to metadata
    """
    # Initialize metadata
    enhanced_metadata = metadata or {}
    
    # TIER 2 METRICS - Auto-calculate from available data
    if prompt:
        enhanced_metadata['prompt_length_chars'] = len(prompt)
    
    if response_text:
        enhanced_metadata['response_length_chars'] = len(response_text)
    
    # Track retry flag
    if required_retry:
        enhanced_metadata['required_retry'] = True
    
    # Check if there's an active session
    session_started = False
    try:
        # Try to record with existing session
        obs.record_call(
            provider=ModelProvider.AZURE,
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            agent_name=agent_name,
            operation=operation,
            metadata=enhanced_metadata,
            prompt=prompt,
            response_text=response_text,
            routing_decision=routing_decision,  # None in discovery mode
            cache_metadata=cache_metadata,      # None in discovery mode
            quality_evaluation=quality_evaluation
        )
    except ValueError as e:
        if "No active session" in str(e):
            # No session exists - create a temporary one
            temp_session = obs.start_session("llm_call")
            session_started = True
            
            # Record the call
            obs.record_call(
                provider=ModelProvider.AZURE,
                model_name=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                agent_name=agent_name,
                operation=operation,
                metadata=enhanced_metadata,
                prompt=prompt,
                response_text=response_text,
                routing_decision=routing_decision,
                cache_metadata=cache_metadata,
                quality_evaluation=quality_evaluation
            )
            
            # End the temporary session
            obs.end_session(temp_session)
        else:
            raise


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