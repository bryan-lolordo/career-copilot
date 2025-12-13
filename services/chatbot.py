# services/chatbot.py
"""
Streamlit Chatbot Service - Career Copilot
COMPREHENSIVE: All Observatory Tier 1, 2, 3 metrics

Captures:
- Tier 1: Core metrics (tokens, latency, cost)
- Tier 2: PromptBreakdown, PromptMetadata, QualityEvaluation
- Tier 3: RoutingDecision, CacheMetadata, A/B Testing support
"""

import asyncio
import json
import logging
import os
import re
import time

from agents.semantic_kernel_setup import (
    create_kernel_with_plugins,
    create_execution_settings,
    create_chat_history_with_system_prompt,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_VERSION,
    extract_messages_from_history
)
from services.conversation_memory import ConversationMemory, get_memory_manager

# Observatory Integration - Complete imports
from observatory_config import (
    obs,
    start_session,
    end_session,
    track_llm_call,
    create_prompt_metadata,
    create_prompt_breakdown,
    create_routing_decision,
    create_cache_metadata,
    judge,
    DEFAULT_MODEL,
    PromptMetadata,
    ModelConfig,
    StreamingMetrics,
    ExperimentMetadata,
    ErrorDetails,
)

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# ============================================================================
# PROMPT METADATA CONFIGURATION
# ============================================================================

# Create PromptMetadata for system prompt tracking
# Bump version when you change SYSTEM_PROMPT in semantic_kernel_setup.py
PROMPT_META = create_prompt_metadata(
    template_id="career_copilot_streamlit",
    version=SYSTEM_PROMPT_VERSION,
    compressible_sections=["AVAILABLE TOOLS", "CRITICAL DECISION RULES"],
    optimization_flags={"auto_function_calling": True},
    config_version="1.0"
) if PromptMetadata else None


# ============================================================================
# GLOBAL KERNEL INITIALIZATION
# ============================================================================
memory_manager = get_memory_manager()
memory = memory_manager.get_session("streamlit_default")

kernel, chat_completion, db_service, memory = create_kernel_with_plugins(memory)
execution_settings = create_execution_settings()
history = create_chat_history_with_system_prompt()

context = memory.context

# START GLOBAL OBSERVATORY SESSION (for the lifetime of this Streamlit app)
obs_session = obs.start_session("streamlit_app_session")
logger.info(f"Observatory session started: {obs_session.id}")

logger.info("Chatbot service initialized with full Observatory tracking")
logger.info(f"System prompt version: {SYSTEM_PROMPT_VERSION}")


# ============================================================================
# MAIN CHAT FUNCTION
# ============================================================================
async def chat_with_kernel(message: str) -> tuple[str, str]:
    """
    Send a message to the chatbot and get a reply.

    Args:
        message: User's message

    Returns:
        Tuple of (response_text, plugin_used)
    """
    logger.info(f"Processing Streamlit message: '{message[:50]}...'")
    
    # ADD THESE THREE LINES:
    memory.turn_number += 1
    memory.conversation_id = obs_session.id

    start_time = time.time()

    try:
        # Add user message to chat history
        history.add_user_message(message)
        memory.chat_history = history
        logger.debug(f"Message added to history: {time.time() - start_time:.2f}s")

        # Send request to Semantic Kernel / Azure OpenAI
        llm_start_time = time.time()
        response = await chat_completion.get_chat_message_content(
            chat_history=history,
            settings=execution_settings,
            kernel=kernel,
        )
        latency_ms = (time.time() - llm_start_time) * 1000
        logger.info(f"LLM response received: {latency_ms:.0f}ms")

        # Detect which plugin was used (if any)
        plugin_used = detect_plugin_used(response)
        if plugin_used:
            logger.info(f"Plugin triggered: {plugin_used}")

        # Extract tool/function calls if available
        tool_calls = []
        tool_count = 0
        if hasattr(response, 'metadata') and response.metadata:
            function_result = response.metadata.get('function_result')
            if function_result:
                tool_calls.append({
                    "name": function_result.get('name'),
                    "arguments": function_result.get('arguments')
                })
                tool_count = 1

        # Extract token usage
        prompt_tokens, completion_tokens = extract_token_usage(response)
        
        # Get response text
        response_text = str(response)
        
        # Extract messages for prompt breakdown (BEFORE adding assistant response)
        messages_for_breakdown = extract_messages_from_history(history)
        
        # Create prompt breakdown for Tier 2
        prompt_breakdown = create_prompt_breakdown_from_messages(messages_for_breakdown)
        
        # LLM Judge evaluation (50% sampling)
        quality_eval = await judge.maybe_evaluate(
            operation="streamlit_chat",
            prompt=message,
            response=response_text,
            llm_client=kernel, 
        )
        
        # Tier 3: Routing decision (placeholder - ready for optimization)
        routing_decision = create_routing_decision(
            chosen_model=DEFAULT_MODEL,  # Will be filled by observatory_config from env
            alternative_models=["gpt-4o", "gpt-4o-mini"],
            reasoning="Chat interaction - using default model",
            complexity_score=0.5
        ) if create_routing_decision else None
        
        # Tier 3: Cache metadata (placeholder - ready for optimization)
        cache_metadata = create_cache_metadata(
            cache_hit=False,
            cache_key=None,
            cache_cluster_id="streamlit_chat"
        ) if create_cache_metadata else None
        
        # Track in Observatory - COMPLETE with ALL tiers
        track_llm_call(
            # Core metrics (Tier 1) - model auto-detected from env
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            agent_name=plugin_used if plugin_used else "ChatAgent",
            agent_role="orchestrator",  # Added: agent role
            operation="streamlit_chat",
            success=True,  # Added: explicit success
            
            # Prompt analysis (Tier 2) - auto-extracts breakdown from messages
            messages=messages_for_breakdown,
            response_text=response_text,
            prompt_metadata=PROMPT_META,  # Track system prompt version
            prompt_breakdown=prompt_breakdown,  # Added: token breakdown
            
            # Quality evaluation (Tier 2) - may be None if not sampled
            quality_evaluation=quality_eval,
            
            # Optimization tracking (Tier 3)
            routing_decision=routing_decision,  # Added: routing
            cache_metadata=cache_metadata,  # Added: cache
            
            # A/B Testing support (Tier 3)
            prompt_variant_id=None,  # Added: ready for A/B tests
            test_dataset_id=None,  # Added: ready for test runs
            
            # NEW: Conversation linking
            conversation_id=obs_session.id if hasattr(obs_session, 'id') else "streamlit_session",
            turn_number=len(history.messages),
            user_id=None,  # Could be added if user authentication exists
            parent_call_id=None,
            
            # NEW: Model configuration (from execution_settings)
            temperature=0.7,
            max_tokens=800,
            top_p=None,
            
            # NEW: Separate prompt components (for fast top-level queries)
            system_prompt=SYSTEM_PROMPT,
            user_message=message,
            
            # NEW: Token breakdown (top-level for fast queries without JSON parsing)
            system_prompt_tokens=prompt_breakdown.system_prompt_tokens if prompt_breakdown else None,
            user_message_tokens=prompt_breakdown.user_message_tokens if prompt_breakdown else None,
            chat_history_tokens=prompt_breakdown.chat_history_tokens if prompt_breakdown else None,
            conversation_context_tokens=None,
            tool_definitions_tokens=None,
            
            # NEW: Tool/function calling
            tool_calls_made=tool_calls if tool_calls else None,
            tool_call_count=tool_count,
            tool_execution_time_ms=None,
            
            # NEW: Observability
            trace_id=obs_session.id if hasattr(obs_session, 'id') else None,
            request_id=None,
            environment=os.getenv("ENVIRONMENT", "development"),
            
            # Additional metadata
            metadata={
                "plugin_used": plugin_used,
                "message_length": len(message),
                "response_length": len(response_text),
                "conversation_turn": len(history.messages),
                "system_prompt_version": SYSTEM_PROMPT_VERSION,
                "judged": quality_eval is not None
            }
        )
        
        total_tokens = prompt_tokens + completion_tokens
        logger.info(f"Tokens: {prompt_tokens} prompt + {completion_tokens} completion = {total_tokens} total")
        
        # Clean up any stray HTML tags from LLM response
        response_text = re.sub(r'</?div[^>]*>', '', response_text)
        response_text = re.sub(r'</?p[^>]*>', '', response_text)

        # Debug context
        logger.debug(f"Context - Resume={context.active_resume_id}, Job={context.active_job_id}, LastAction={context.last_action}")

        # Add assistant response to history
        history.add_message(response)
        memory.chat_history = history
        
        total_time = time.time() - start_time
        logger.info(f"Chat message complete: {total_time:.2f}s total")
        
        return response_text, plugin_used
        
    except Exception as e:
        logger.error(f"Error in chat_with_kernel: {e}", exc_info=True)
        
        # Track failed call - COMPLETE
        track_llm_call(
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=(time.time() - start_time) * 1000,
            agent_name="ChatAgent",
            agent_role="orchestrator",
            operation="streamlit_chat",
            success=False,  # Failed
            error=str(e),
            
            # Prompt tracking
            prompt_metadata=PROMPT_META,
            prompt_breakdown=None,
            system_prompt=SYSTEM_PROMPT,
            user_message=message,
            
            # Optimization tracking
            quality_evaluation=None,
            routing_decision=None,
            cache_metadata=None,
            prompt_variant_id=None,
            test_dataset_id=None,
            
            # NEW: Conversation linking
            conversation_id=obs_session.id if hasattr(obs_session, 'id') else "streamlit_session",
            turn_number=len(history.messages),
            user_id=None,
            parent_call_id=None,
            
            # NEW: Model configuration
            temperature=0.7,
            max_tokens=800,
            top_p=None,
            
            # NEW: Error details
            error_type=type(e).__name__,
            error_code=None,
            retry_count=0,
            
            # NEW: Observability
            trace_id=obs_session.id if hasattr(obs_session, 'id') else None,
            request_id=None,
            environment=os.getenv("ENVIRONMENT", "development"),
            
            metadata={
                "message_length": len(message),
                "error_type": type(e).__name__
            }
        )
        
        raise


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def detect_plugin_used(response) -> str:
    """Detect which plugin was triggered from the response."""
    plugin_used = None
    
    if hasattr(response, "metadata") and response.metadata:
        if "function_call" in response.metadata:
            plugin_used = response.metadata["function_call"].get("name")

    if not plugin_used and hasattr(response, "items"):
        for item in getattr(response, "items", []):
            if hasattr(item, "function_call") and item.function_call:
                plugin_used = item.function_call.name
                break
    
    return plugin_used


def extract_token_usage(response) -> tuple[int, int]:
    """Extract token usage from response metadata."""
    prompt_tokens = 0
    completion_tokens = 0
    
    if hasattr(response, 'metadata') and response.metadata:
        usage = response.metadata.get('usage')
        if usage:
            prompt_tokens = getattr(usage, 'prompt_tokens', 0)
            completion_tokens = getattr(usage, 'completion_tokens', 0)
    
    return prompt_tokens, completion_tokens


def create_prompt_breakdown_from_messages(messages: list) -> dict:
    """
    Create prompt breakdown from messages list.
    
    Args:
        messages: List of {"role": "...", "content": "..."} dicts
    
    Returns:
        PromptBreakdown object or None
    """
    if not create_prompt_breakdown:
        return None
    
    system_prompt = None
    system_tokens = 0
    user_message = None
    user_tokens = 0
    chat_history = []
    chat_history_tokens = 0
    
    for msg in messages:
        role = msg.get("role", "").lower()
        content = msg.get("content", "")
        tokens = len(content) // 4
        
        if role == "system":
            system_prompt = content
            system_tokens = tokens
        elif role == "user":
            # Keep last user message
            user_message = content
            user_tokens = tokens
        else:
            chat_history.append(msg)
            chat_history_tokens += tokens
    
    return create_prompt_breakdown(
        system_prompt=system_prompt,
        system_prompt_tokens=system_tokens,
        user_message=user_message,
        user_message_tokens=user_tokens,
        chat_history=chat_history if chat_history else None,
        chat_history_tokens=chat_history_tokens if chat_history else None,
    )


# ============================================================================
# HELPER: Reset conversation history
# ============================================================================
def reset_chat_history():
    """Reset the conversation history and memory."""
    global history, memory
    logger.info("Resetting chat history and memory")
    
    history = create_chat_history_with_system_prompt()
    memory.chat_history = history
    # Reset memory context
    memory.context.awaiting_confirmation = False
    memory.context.pending_action = None
    memory.context.last_action = None
    memory.context.last_searched_jobs = None
    memory.context.available_resumes = None
    memory.context.selected_resume_for_matching = None
    memory.context.awaiting_resume_selection = False
    memory.context.awaiting_job_filter_selection = False


# ============================================================================
# HELPER: Get conversation history
# ============================================================================
def get_chat_history() -> list[dict]:
    """Get the current conversation history."""
    return extract_messages_from_history(history)


# ============================================================================
# CHATBOT CLASS
# ============================================================================

class CareerCopilotChatbot:
    """Wrapper class for Streamlit compatibility"""
    
    async def chat_async(self, message: str) -> str:
        response, _ = await chat_with_kernel(message)
        return response
    
    def chat(self, message: str) -> str:
        return asyncio.run(self.chat_async(message))
    
    def reset(self):
        reset_chat_history()
    
    def get_system_prompt_version(self) -> str:
        return SYSTEM_PROMPT_VERSION


# ============================================================================
# ACCESSOR FUNCTIONS
# ============================================================================

def get_kernel():
    """Get the global kernel instance for reuse across pages."""
    return kernel


def get_database_service():
    """Get the database service instance for reuse across pages."""
    return db_service


def get_prompt_metadata():
    """Get the current prompt metadata."""
    return PROMPT_META