# services/chatbot.py
"""
Streamlit Chatbot Service

This module provides the chatbot interface for Streamlit.
All kernel setup is imported from agents.semantic_kernel_setup (the main source of truth).
"""

import asyncio
import json
import logging
import re
import time
from agents.semantic_kernel_setup import (
    create_kernel_with_plugins,
    create_execution_settings,
    create_chat_history_with_system_prompt,
    SYSTEM_PROMPT
)
from services.conversation_memory import ConversationMemory, get_memory_manager

# Observatory Integration
from observatory_config import start_tracking_session, end_tracking_session, track_llm_call
import os

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# ============================================================================
# GLOBAL KERNEL INITIALIZATION
# ============================================================================
# Initialize globally so it persists between Streamlit calls
# This avoids recreating the kernel on every message

# Get or create memory for this session
memory_manager = get_memory_manager()
memory = memory_manager.get_session("streamlit_default")

kernel, chat_completion, db_service, memory = create_kernel_with_plugins(memory)
execution_settings = create_execution_settings()
history = create_chat_history_with_system_prompt()

# Quick access to context for backwards compatibility
context = memory.context

logger.info("Chatbot service initialized with Observatory tracking")


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
        - response_text: The chatbot's reply
        - plugin_used: Name of plugin that was called (or None)
    """
    # Start tracking this chat message
    session = start_tracking_session(
        "streamlit_chat_message",
        metadata={"message_length": len(message)}
    )
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
        plugin_used = None
        if hasattr(response, "metadata") and response.metadata:
            if "function_call" in response.metadata:
                plugin_used = response.metadata["function_call"].get("name")

        if not plugin_used and hasattr(response, "items"):
            for item in getattr(response, "items", []):
                if hasattr(item, "function_call") and item.function_call:
                    plugin_used = item.function_call.name
                    break

        if plugin_used:
            logger.info(f"Plugin triggered: {plugin_used}")

        # Extract token usage
        prompt_tokens = 0
        completion_tokens = 0
        model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
        
        if hasattr(response, 'metadata') and response.metadata:
            usage = response.metadata.get('usage')
            if usage:
                prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                completion_tokens = getattr(usage, 'completion_tokens', 0)
        
        # Get response text (needed for tracking)
        response_text = str(response)
        
        # Track in Observatory with prompt and response
        track_llm_call(
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            agent_name=plugin_used if plugin_used else "ChatAgent",
            operation="streamlit_chat",
            metadata={
                "plugin_used": plugin_used,
                "message_length": len(message)
            },
            # NEW: Track prompt and response text
            prompt=message,
            response_text=response_text
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
        
        # End session successfully
        end_tracking_session(session, success=True)
        
        return response_text, plugin_used
        
    except Exception as e:
        # End session with error
        end_tracking_session(session, success=False, error=str(e))
        logger.error(f"Error in chat_with_kernel: {e}", exc_info=True)
        raise


# ============================================================================
# HELPER: Reset conversation history
# ============================================================================
def reset_chat_history():
    """
    Reset the conversation history and memory.
    Useful for starting a fresh conversation in Streamlit.
    """
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
    """
    Get the current conversation history.
    
    Returns:
        List of message dictionaries with 'role' and 'content'
    """
    messages = []
    for msg in history.messages:
        messages.append({
            "role": msg.role.value if hasattr(msg.role, 'value') else str(msg.role),
            "content": str(msg.content)
        })
    return messages


class CareerCopilotChatbot:
    """Wrapper class for Streamlit compatibility"""
    
    async def chat_async(self, message: str) -> str:
        response, _ = await chat_with_kernel(message)
        return response
    
    def chat(self, message: str) -> str:
        return asyncio.run(self.chat_async(message))
    
    def reset(self):
        reset_chat_history()


# ============================================================================
# HELPER: Get kernel and database service for other pages
# ============================================================================
def get_kernel():
    """Get the global kernel instance for reuse across pages."""
    return kernel

def get_database_service():
    """Get the database service instance for reuse across pages."""
    return db_service