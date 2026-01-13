# services/chatbot.py
"""
Streamlit Chatbot Service - Career Copilot
COMPREHENSIVE: Full Observatory integration with two-phase optimization system

BASELINE MODE: Detects optimization opportunities without changing behavior
OPTIMIZED MODE: Applies optimizations (caching, routing, compression)

Captures:
- Tier 1: Core metrics (tokens, latency, cost)
- Tier 2: PromptBreakdown, PromptMetadata, QualityEvaluation
- Tier 3: RoutingDecision, CacheMetadata, A/B Testing support
"""

import asyncio
import logging
import os
import re
import time
import uuid

from agents.semantic_kernel_setup import (
    create_kernel_with_plugins,
    create_execution_settings,
    create_chat_history_with_system_prompt,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_VERSION,
    extract_messages_from_history,
    get_tool_definitions_tokens,
)
from services.conversation_memory import ConversationMemory, get_memory_manager

# Observatory Integration - CORRECT imports (only what exists in observatory_config.py)
from observatory_config import (
    # Main tracking
    obs,
    track_llm_call,
    
    # Optimization components
    cache,
    semantic_cache,
    prefix_cache,
    router,
    prompt_optimizer,
    batch_detector,
    streaming_detector,
    judge,
    
    # Session management
    start_session,
    end_session,
    
    # Config constants
    DEFAULT_MODEL,
    CURRENT_PHASE,
    
    # Data models
    PromptMetadata,
    
    # Helper functions
    create_prompt_metadata,
    create_prompt_breakdown,
    create_routing_decision,
    create_cache_metadata,
    estimate_tokens,
    classify_error,
)

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
memory_manager = get_memory_manager()
memory = memory_manager.get_session("streamlit_default")

kernel, chat_completion, db_service, memory = create_kernel_with_plugins(memory)
execution_settings = create_execution_settings()
history = create_chat_history_with_system_prompt()

context = memory.context

# Calculate tool definition tokens once (reused across all calls)
tool_definitions_tokens = get_tool_definitions_tokens(kernel)

# START GLOBAL OBSERVATORY SESSION (for the lifetime of this Streamlit app)
obs_session = start_session("streamlit_app_session", metadata={
    "mode": "streamlit",
    "system_prompt_version": SYSTEM_PROMPT_VERSION,
    "phase": CURRENT_PHASE,
})
logger.info(f"Observatory session started: {obs_session.id if hasattr(obs_session, 'id') else 'streamlit_session'}")

logger.info("Chatbot service initialized with full Observatory tracking")
logger.info(f"System prompt version: {SYSTEM_PROMPT_VERSION}")
logger.info(f"Observatory phase: {CURRENT_PHASE.upper()}")


# ============================================================================
# MAIN CHAT FUNCTION WITH TWO-PHASE OPTIMIZATION
# ============================================================================
async def chat_with_kernel(message: str) -> tuple[str, str]:
    """
    Send a message to the chatbot and get a reply.
    
    Implements two-phase optimization system:
    - BASELINE: Detects opportunities (logs "💡 OPPORTUNITY")
    - OPTIMIZED: Applies optimizations (logs "✅ APPLIED")

    Args:
        message: User's message

    Returns:
        Tuple of (response_text, plugin_used)
    """
    logger.info(f"Processing Streamlit message: '{message[:50]}...'")
    
    # Update memory tracking
    memory.turn_number += 1
    memory.conversation_id = obs_session.id if hasattr(obs_session, 'id') else "streamlit_session"
    memory.request_id = f"{memory.conversation_id}_turn{memory.turn_number}"

    start_time = time.time()

    try:
        # Add user message to chat history
        history.add_user_message(message)
        memory.chat_history = history
        logger.debug(f"Message added to history: {time.time() - start_time:.2f}s")

        # ═══════════════════════════════════════════════════════════════
        # STEP 1: Check exact cache
        # ═══════════════════════════════════════════════════════════════
        operation = "streamlit_chat"
        cache_key_data = {"user_message": message, "turn": memory.turn_number}
        
        cached_response, cache_meta = cache.get(
            operation=operation,
            key_data=cache_key_data
        )
        
        if cached_response:  # None in baseline, actual response in optimized
            # Track cache hit
            track_llm_call(
                operation=operation,
                prompt_tokens=0,  # Cached - no LLM call
                completion_tokens=0,
                latency_ms=1.0,  # Minimal latency
                success=True,
                response_text=cached_response,
                cache_metadata=cache_meta,
                agent_name="ChatAgent",
                agent_role="orchestrator",
                conversation_id=memory.conversation_id,
                turn_number=memory.turn_number,
                request_id=memory.request_id,
                trace_id=memory.conversation_id,
                environment=os.getenv("ENVIRONMENT", "development"),
                metadata={
                    "phase": CURRENT_PHASE,  # ← CRITICAL
                    "cache_hit": True,
                    "system_prompt_version": SYSTEM_PROMPT_VERSION,
                }
            )
            
            # Add to history and return
            history.add_assistant_message(cached_response)
            memory.chat_history = history
            
            logger.info(f"✅ Cache hit! Skipped LLM call.")
            return cached_response, None
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 2: Check semantic cache (if available)
        # ═══════════════════════════════════════════════════════════════
        if semantic_cache:
            result = semantic_cache.get(operation=operation, prompt=message)
            if result.hit:  # False in baseline, True in optimized if similar match
                # Track semantic cache hit
                track_llm_call(
                    operation=operation,
                    prompt_tokens=0,  # Cached - no LLM call
                    completion_tokens=0,
                    latency_ms=1.0,
                    success=True,
                    response_text=result.response,
                    cache_metadata=create_cache_metadata(
                        cache_hit=True,
                        similarity_score=result.similarity
                    ),
                    agent_name="ChatAgent",
                    agent_role="orchestrator",
                    conversation_id=memory.conversation_id,
                    turn_number=memory.turn_number,
                    request_id=memory.request_id,
                    trace_id=memory.conversation_id,
                    environment=os.getenv("ENVIRONMENT", "development"),
                    metadata={
                        "phase": CURRENT_PHASE,  # ← CRITICAL
                        "semantic_cache_hit": True,
                        "similarity": result.similarity,
                        "system_prompt_version": SYSTEM_PROMPT_VERSION,
                    }
                )
                
                # Add to history and return
                history.add_assistant_message(result.response)
                memory.chat_history = history
                
                logger.info(f"✅ Semantic cache hit ({result.similarity:.1%} similar)! Skipped LLM call.")
                return result.response, None
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 3: Get optimized prompt and max_tokens
        # ═══════════════════════════════════════════════════════════════
        optimized_prompt, max_tokens_limit, prompt_meta = prompt_optimizer.get_optimized_prompt(
            operation=operation,
            default_prompt=SYSTEM_PROMPT
        )
        # Returns: default in baseline, compressed in optimized
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 4: Get routed model
        # ═══════════════════════════════════════════════════════════════
        routed_model, routing_meta = router.route(
            operation=operation,
            prompt_tokens=estimate_tokens(optimized_prompt),
            complexity=0.5  # Medium complexity for chat
        )
        # Returns: default model in baseline, routed model in optimized
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 5: Track prefix for prefix caching detection
        # ═══════════════════════════════════════════════════════════════
        prefix_cache.track_call(
            operation=operation,
            system_prompt=optimized_prompt,
            system_prompt_tokens=estimate_tokens(optimized_prompt),
        )
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 6: Make LLM call (use optimized values)
        # ═══════════════════════════════════════════════════════════════
        
        # Update execution settings with optimized values
        execution_settings.max_tokens = max_tokens_limit
        
        # Update system prompt in history if it was optimized
        if optimized_prompt != SYSTEM_PROMPT and len(history.messages) > 0:
            if history.messages[0].role.value.lower() in ['system', 'developer']:
                history.messages[0].content = optimized_prompt
        
        llm_start_time = time.time()
        
        try:
            # Make the actual LLM call
            response = await chat_completion.get_chat_message_content(
                chat_history=history,
                settings=execution_settings,
                kernel=kernel,
            )
            
            latency_ms = (time.time() - llm_start_time) * 1000
            response_text = str(response)
            success = True
            error = None
            error_type = None
            error_code = None
            
        except Exception as e:
            latency_ms = (time.time() - llm_start_time) * 1000
            response_text = f"Error: {str(e)}"
            success = False
            error = str(e)
            
            # Classify error using helper function
            error_info = classify_error(e, operation=operation)
            error_type = error_info['error_type']
            error_code = error_info['error_code']
            
            logger.error(f"LLM call failed: {error_type} - {error_code}")
            response = None
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 7: Extract token usage and metadata
        # ═══════════════════════════════════════════════════════════════
        prompt_tokens = 0
        completion_tokens = 0
        time_to_first_token_ms = None
        
        if success and response and hasattr(response, 'metadata') and response.metadata:
            usage = response.metadata.get('usage')
            if usage:
                prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                completion_tokens = getattr(usage, 'completion_tokens', 0)
            
            # Check for streaming metrics
            if response.metadata.get('is_streaming'):
                time_to_first_token_ms = response.metadata.get('time_to_first_token_ms')
        
        # Detect which plugin was used (if any)
        plugin_used = detect_plugin_used(response) if success else None
        if plugin_used:
            logger.info(f"Plugin triggered: {plugin_used}")
        
        # Extract tool/function calls if available
        tool_calls = []
        tool_count = 0
        if success and response and hasattr(response, 'metadata') and response.metadata:
            function_result = response.metadata.get('function_result')
            if function_result:
                tool_calls.append({
                    "name": function_result.get('name'),
                    "arguments": function_result.get('arguments')
                })
                tool_count = 1
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 8: Detect streaming candidates
        # ═══════════════════════════════════════════════════════════════
        streaming_candidate = streaming_detector.check_call(
            operation=operation,
            latency_ms=latency_ms,
            completion_tokens=completion_tokens,
        )
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 9: Cache the response (if successful)
        # ═══════════════════════════════════════════════════════════════
        if success:
            cache.set(
                operation=operation,
                key_data=cache_key_data,
                value=response_text
            )
            
            if semantic_cache:
                semantic_cache.set(
                    operation=operation,
                    prompt=message,
                    response=response_text
                )
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 10: Track with Observatory (CRITICAL - INCLUDE PHASE)
        # ═══════════════════════════════════════════════════════════════
        
        # Extract messages for prompt breakdown
        messages_for_breakdown = extract_messages_from_history(history)
        
        # Create prompt breakdown
        prompt_breakdown = create_prompt_breakdown_from_messages(messages_for_breakdown)
        
        # LLM Judge evaluation (if enabled and successful)
        quality_eval = None
        if success:
            quality_eval = await judge.maybe_evaluate(
                operation=operation,
                prompt=message,
                response=response_text,
                llm_client=kernel,
            )
        
        # Track in Observatory with all 139 fields
        track_llm_call(
            # TIER 1: Core metrics
            model_name=routed_model,  # Uses routed model from Step 4
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            agent_name=plugin_used if plugin_used else "ChatAgent",
            agent_role="orchestrator",
            operation=operation,
            success=success,
            error=error,
            
            # TIER 2: Prompt content
            messages=messages_for_breakdown,
            response_text=response_text,
            system_prompt=optimized_prompt,  # Track which prompt was used
            user_message=message,
            prompt_breakdown=prompt_breakdown,
            
            # TIER 2: Optimization tracking
            routing_decision=routing_meta,  # From Step 4
            cache_metadata=cache_meta,  # From Step 1
            quality_evaluation=quality_eval,
            prompt_metadata=prompt_meta,  # From Step 3
            
            # TIER 3: A/B Testing support
            prompt_variant_id=None,
            test_dataset_id=None,
            
            # Conversation linking
            conversation_id=memory.conversation_id,
            turn_number=memory.turn_number,
            user_id=None,
            parent_call_id=None,
            
            # Model configuration
            temperature=execution_settings.temperature,
            max_tokens=max_tokens_limit,  # From Step 3
            top_p=getattr(execution_settings, 'top_p', None),
            
            # Token breakdown (auto-extracted from messages)
            system_prompt_tokens=estimate_tokens(optimized_prompt),
            user_message_tokens=estimate_tokens(message),
            tool_definitions_tokens=tool_definitions_tokens,
            
            # Tool/function calling
            tool_calls_made=tool_calls if tool_calls else None,
            tool_call_count=tool_count,
            
            # Streaming
            time_to_first_token_ms=time_to_first_token_ms,
            
            # Error details (if failed)
            error_type=error_type,
            error_code=error_code,
            
            # Observability
            trace_id=memory.conversation_id,
            request_id=memory.request_id,
            environment=os.getenv("ENVIRONMENT", "development"),
            
            # Metadata - CRITICAL: Include phase
            metadata={
                "phase": CURRENT_PHASE,  # ← CRITICAL for phase tracking
                "plugin_used": plugin_used,
                "message_length": len(message),
                "response_length": len(response_text) if success else 0,
                "conversation_turn": len(history.messages),
                "system_prompt_version": SYSTEM_PROMPT_VERSION,
                "judged": quality_eval is not None,
                "streaming_candidate": streaming_candidate,
            }
        )
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 11: Track batch detection
        # ═══════════════════════════════════════════════════════════════
        batch_detector.track_call(
            operation=operation,
            call_id=memory.request_id,
            latency_ms=latency_ms,
        )
        
        # Log results
        if success:
            logger.info(f"LLM response: {latency_ms:.0f}ms, {prompt_tokens + completion_tokens} tokens")
            
            # Clean up any stray HTML tags from LLM response
            response_text = re.sub(r'</?div[^>]*>', '', response_text)
            response_text = re.sub(r'</?p[^>]*>', '', response_text)
            
            # Add assistant response to history
            history.add_message(response)
            memory.chat_history = history
        else:
            logger.error(f"LLM call failed after {latency_ms:.0f}ms")
        
        # Debug context
        logger.debug(f"Context - Resume={context.active_resume_id}, Job={context.active_job_id}, LastAction={context.last_action}")
        
        total_time = time.time() - start_time
        logger.info(f"Chat message complete: {total_time:.2f}s total")
        
        return response_text, plugin_used
        
    except Exception as e:
        logger.error(f"Error in chat_with_kernel: {e}", exc_info=True)
        
        # Classify error
        error_info = classify_error(e, operation="streamlit_chat")
        
        # Track failed call
        track_llm_call(
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=(time.time() - start_time) * 1000,
            agent_name="ChatAgent",
            agent_role="orchestrator",
            operation="streamlit_chat",
            success=False,
            error=str(e),
            
            # Prompt tracking
            system_prompt=SYSTEM_PROMPT,
            user_message=message,
            
            # Conversation linking
            conversation_id=memory.conversation_id,
            turn_number=memory.turn_number,
            request_id=memory.request_id,
            
            # Model configuration
            temperature=execution_settings.temperature,
            max_tokens=execution_settings.max_tokens,
            top_p=getattr(execution_settings, 'top_p', None),
            
            # Token breakdown
            tool_definitions_tokens=tool_definitions_tokens,
            
            # Error details
            error_type=error_info['error_type'],
            error_code=error_info['error_code'],
            retry_count=0,
            
            # Observability
            trace_id=memory.conversation_id,
            environment=os.getenv("ENVIRONMENT", "development"),
            
            # Metadata - CRITICAL: Include phase
            metadata={
                "phase": CURRENT_PHASE,  # ← CRITICAL
                "message_length": len(message),
                "error_type": error_info['error_type'],
                "system_prompt_version": SYSTEM_PROMPT_VERSION,
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
    
    # First pass: find system prompt
    for msg in messages:
        if msg.get("role", "").lower() == "system":
            system_prompt = msg.get("content", "")
            system_tokens = estimate_tokens(system_prompt)
            break
    
    # Second pass: separate user messages from others
    user_messages = []
    other_messages = []
    
    for msg in messages:
        role = msg.get("role", "").lower()
        if role == "user":
            user_messages.append(msg)
        elif role not in ["system"]:  # assistant, function, etc.
            other_messages.append(msg)
    
    # Last user message is current, all others are history
    if user_messages:
        last_user = user_messages[-1]
        user_message = last_user.get("content", "")
        user_tokens = estimate_tokens(user_message)
        
        # Previous user messages go to history
        for msg in user_messages[:-1]:
            chat_history.append(msg)
            chat_history_tokens += estimate_tokens(msg.get("content", ""))
    
    # All assistant/function messages go to history
    for msg in other_messages:
        chat_history.append(msg)
        chat_history_tokens += estimate_tokens(msg.get("content", ""))
    
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
# HELPER: Get optimization stats
# ============================================================================
def get_optimization_stats() -> dict:
    """
    Get current optimization statistics.
    
    Returns statistics from cache components if enabled.
    """
    stats = {
        "phase": CURRENT_PHASE,
        "enabled": cache.enabled or (semantic_cache and semantic_cache.enabled),
    }
    
    if cache.enabled:
        stats["exact_cache"] = {
            "hits": cache._hits,
            "misses": cache._misses,
            "hit_rate": cache._hits / max(cache._hits + cache._misses, 1),
        }
    
    if semantic_cache and semantic_cache.enabled:
        stats["semantic_cache"] = {
            "hits": semantic_cache._hits,
        }
    
    return stats


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
    
    def get_optimization_stats(self) -> dict:
        """Get optimization statistics."""
        return get_optimization_stats()
    
    def get_phase(self) -> str:
        """Get current Observatory phase."""
        return CURRENT_PHASE


# ============================================================================
# ACCESSOR FUNCTIONS
# ============================================================================

def get_kernel():
    """Get the global kernel instance for reuse across pages."""
    return kernel


def get_database_service():
    """Get the database service instance for reuse across pages."""
    return db_service
