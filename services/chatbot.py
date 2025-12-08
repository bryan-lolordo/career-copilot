# services/chatbot.py
"""
Streamlit Chatbot Service - Career Copilot
COMPREHENSIVE: All Observatory Tier 2 metrics

Captures:
- PromptBreakdown (auto-extracted from chat history)
- PromptMetadata (system prompt versioning)
- QualityEvaluation (from LLM Judge)
- Full metadata tracking
"""

import asyncio
import logging
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

# Observatory Integration
from observatory_config import (
    obs, 
    track_llm_call,
    create_prompt_metadata,
    PromptMetadata
)
from llm_judge import maybe_judge_response

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
    
    start_time = time.time()

    try:
        # Add user message to chat history
        history.add_user_message(message)
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

        # Extract token usage
        prompt_tokens, completion_tokens = extract_token_usage(response)
        
        # Get response text
        response_text = str(response)
        
        # Extract messages for prompt breakdown (BEFORE adding assistant response)
        messages_for_breakdown = extract_messages_from_history(history)
        
        # LLM Judge evaluation (50% sampling)
        quality_eval = await maybe_judge_response(
            kernel,
            "streamlit_chat",
            message,
            response_text,
            context={
                "plugin_used": plugin_used,
                "conversation_turn": len(history.messages)
            }
        )
        
        # Track in Observatory - SINGLE CALL with ALL data
        track_llm_call(
            # Core metrics (model auto-detected from env)
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            agent_name=plugin_used if plugin_used else "ChatAgent",
            operation="streamlit_chat",
            
            # Prompt analysis - auto-extracts breakdown from messages
            messages=messages_for_breakdown,
            response_text=response_text,
            prompt_metadata=PROMPT_META,  # Track system prompt version
            
            # Quality evaluation (may be None if not sampled)
            quality_evaluation=quality_eval,
            
            # Additional metadata
            metadata={
                "plugin_used": plugin_used,
                "message_length": len(message),
                "response_length": len(response_text),
                "conversation_turn": len(history.messages),
                "system_prompt_version": SYSTEM_PROMPT_VERSION,
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
        
        total_time = time.time() - start_time
        logger.info(f"Chat message complete: {total_time:.2f}s total")
        
        return response_text, plugin_used
        
    except Exception as e:
        logger.error(f"Error in chat_with_kernel: {e}", exc_info=True)
        
        # Track failed call
        track_llm_call(
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=(time.time() - start_time) * 1000,
            operation="streamlit_chat",
            success=False,
            error=str(e),
            metadata={"message_length": len(message)}
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


# ============================================================================
# HELPER: Reset conversation history
# ============================================================================
def reset_chat_history():
    """Reset the conversation history and memory."""
    global history, memory
    logger.info("Resetting chat history and memory")
    
    history = create_chat_history_with_system_prompt()
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