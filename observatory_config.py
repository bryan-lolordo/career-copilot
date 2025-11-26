"""
Observatory Configuration - Career Copilot
===========================================

Centralized Observatory setup for tracking metrics across all Career Copilot modules.
Import this file in any module that needs Observatory tracking.

Usage:
    from observatory_config import obs, track_llm_call, start_tracking_session, end_tracking_session

Setup Instructions:
1. Install Observatory in Career Copilot:
   cd career_copilot
   pip install -e ../ai-agent-observatory
   
2. Verify imports work:
   python -c "from observatory import Observatory; print('✅ Observatory imported successfully')"
"""

import os
import sys

# ============================================================================
# IMPORT OBSERVATORY FROM YOUR PROJECT
# ============================================================================

try:
    from observatory import Observatory, ModelProvider, AgentRole
    print("✅ Observatory package imported successfully")
except ImportError as e:
    print(f"❌ ERROR: Cannot import Observatory package")
    print(f"   {e}")
    print(f"\n📋 Setup Instructions:")
    print(f"   1. Navigate to your Career Copilot folder:")
    print(f"      cd /path/to/career_copilot")
    print(f"   2. Install Observatory as editable package:")
    print(f"      pip install -e ../ai-agent-observatory")
    print(f"   3. Verify it works:")
    print(f"      python -c 'from observatory import Observatory'")
    sys.exit(1)


# ============================================================================
# OBSERVATORY INITIALIZATION - Uses Your Centralized Database
# ============================================================================

# Point to YOUR centralized Observatory database
# This path assumes career_copilot and ai-agent-observatory are sibling folders:
#
# Desktop/
# ├── ai-agent-observatory/
# │   └── observatory.db          <- Your centralized database
# └── career_copilot/
#     └── observatory_config.py   <- This file
#

OBSERVATORY_DB_PATH = os.path.join(
    os.path.dirname(__file__),  # career_copilot/
    "..",                        # Desktop/
    "ai-agent-observatory",      # ai-agent-observatory/
    "observatory.db"             # The database file
)
OBSERVATORY_DB_PATH = os.path.abspath(OBSERVATORY_DB_PATH)

# Verify database exists
if not os.path.exists(os.path.dirname(OBSERVATORY_DB_PATH)):
    print(f"⚠️ WARNING: Observatory folder not found at expected location")
    print(f"   Expected: {os.path.dirname(OBSERVATORY_DB_PATH)}")
    print(f"   Current working directory: {os.getcwd()}")
    print(f"\n   Adjust OBSERVATORY_DB_PATH in observatory_config.py to point to your database")

# Set database URL for Observatory
os.environ['DATABASE_URL'] = f"sqlite:///{OBSERVATORY_DB_PATH}"

# Initialize Observatory
obs = Observatory(
    project_name="Career Copilot",
    enabled=True  # Set to False to disable all tracking
)

print(f"✅ Observatory initialized for Career Copilot")
print(f"   Database: {OBSERVATORY_DB_PATH}")
print(f"   Project: Career Copilot")
print(f"   All metrics will be stored in your centralized Observatory database")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def start_tracking_session(operation_type: str, metadata: dict = None):
    """
    Start a tracking session for an operation.
    
    Args:
        operation_type: Type of operation (e.g., "resume_matching", "chat_message", "code_review")
        metadata: Optional metadata to store with the session
    
    Returns:
        Session object (pass this to end_tracking_session)
    
    Example:
        session = start_tracking_session("chat_message", {"user_id": "123"})
    """
    if metadata:
        return obs.start_session(operation_type, **metadata)
    else:
        return obs.start_session(operation_type)


def end_tracking_session(session, success: bool = True, error: str = None):
    """
    End a tracking session.
    
    Args:
        session: Session object from start_tracking_session
        success: Whether the operation succeeded
        error: Error message if operation failed
    
    Example:
        end_tracking_session(session, success=True)
        # or
        end_tracking_session(session, success=False, error="API timeout")
    """
    # Update session object before ending (Observatory.end_session doesn't accept these params)
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
    # NEW: Enhanced fields for improved tracking
    prompt: str = None,
    response_text: str = None,
    routing_decision = None,
    cache_metadata = None,
    quality_evaluation = None
):
    """
    Record an LLM call in Observatory with enhanced tracking.
    
    Args:
        model_name: Name of the model (e.g., "gpt-4", "gpt-35-turbo")
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens
        latency_ms: Time taken in milliseconds
        agent_name: Name of the agent making the call (optional)
        operation: Type of operation (optional)
        metadata: Additional metadata (optional)
        
        # Enhanced fields (optional - auto-created if not provided)
        prompt: Raw prompt text sent to LLM
        response_text: Response text from LLM
        routing_decision: RoutingDecision object (auto-created with defaults if None)
        cache_metadata: CacheMetadata object (auto-created with defaults if None)
        quality_evaluation: QualityEvaluation object (remains None if not provided)
    
    Note:
        The enhanced collector automatically creates default routing and cache metadata
        if not provided, so these fields are truly optional.
    """
    obs.record_call(
        provider=ModelProvider.AZURE,  # Career Copilot uses Azure OpenAI
        model_name=model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        agent_name=agent_name,
        operation=operation,
        metadata=metadata or {},
        # Pass enhanced fields (will use defaults if None)
        prompt=prompt,
        response_text=response_text,
        routing_decision=routing_decision,
        cache_metadata=cache_metadata,
        quality_evaluation=quality_evaluation
    )


# ============================================================================
# AGENT ROLE MAPPING
# ============================================================================

# Map Career Copilot plugin names to Observatory agent roles
# Available roles: ANALYST, REVIEWER, FIXER, ORCHESTRATOR, CUSTOM
AGENT_ROLE_MAP = {
    "ResumeMatching": AgentRole.ANALYST,      # Analyzes resume-job matches
    "JobPlugin": AgentRole.ANALYST,           # Searches and retrieves jobs
    "DatabaseQueryPlugin": AgentRole.ANALYST, # Queries database
    "ResumeTailoring": AgentRole.FIXER,       # Fixes/improves resumes
    "SelfImprovingMatch": AgentRole.REVIEWER, # Reviews and improves matches
    "ResumePreprocessor": AgentRole.ANALYST,  # Analyzes resume content
    "JobPreprocessor": AgentRole.ANALYST,     # Analyzes job content
}


def get_agent_role(plugin_name: str) -> AgentRole:
    """Get the appropriate AgentRole for a plugin."""
    return AGENT_ROLE_MAP.get(plugin_name, AgentRole.ANALYST)