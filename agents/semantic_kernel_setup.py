# agents/semantic_kernel_setup.py
"""
Semantic Kernel Setup - Single Source of Truth

This module is the MAIN configuration file for Career Copilot.
Both CLI and Streamlit chatbot import from here.

COMPREHENSIVE: Full Observatory integration with two-phase optimization system
- BASELINE MODE: Detects optimization opportunities without changing behavior
- OPTIMIZED MODE: Applies optimizations (caching, routing, compression)

To modify:
- System prompt → Edit SYSTEM_PROMPT below (and bump SYSTEM_PROMPT_VERSION!)
- Plugin configuration → Edit create_kernel_with_plugins()
- Execution settings → Edit create_execution_settings()
"""

import asyncio
import logging
import os
import time
from dotenv import load_dotenv

from semantic_kernel import Kernel
from semantic_kernel.utils.logging import setup_logging
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)

# Plugins
from agents.plugins.JobPlugin import JobPlugin
from agents.plugins.ResumeMatchingPlugin import ResumeMatchingPlugin
from agents.plugins.ResumePreprocessorPlugin import ResumePreprocessorPlugin
from agents.plugins.JobPreprocessorPlugin import JobPreprocessorPlugin
from agents.plugins.QueryDatabasePlugin import DatabaseQueryPlugin
from agents.plugins.ResumeTailoringPlugin import ResumeTailoringPlugin
from agents.plugins.SelfImprovingMatchPlugin import SelfImprovingMatchPlugin

# Services
from services.database_service import DatabaseService
from services.conversation_memory import ConversationMemory

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
    parallel_detector,
    streaming_detector,
    judge,
    
    # Session management
    start_session,
    end_session,
    
    # Config constants
    DEFAULT_MODEL,
    CURRENT_PHASE,
    PROMPT_VARIANTS,
    OPERATION_COMPLEXITY,
    
    # Data models
    PromptMetadata,
    PromptBreakdown,
    RoutingDecision,
    CacheMetadata,
    QualityEvaluation,
    ModelConfig,
    StreamingMetrics,
    
    # Helper functions
    create_prompt_breakdown,
    create_routing_decision,
    create_cache_metadata,
    estimate_tokens,
)

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# SYSTEM PROMPT VERSION - BUMP THIS WHEN YOU CHANGE THE PROMPT!
# ============================================================================
SYSTEM_PROMPT_VERSION = "1.2.0"  # Semantic versioning: MAJOR.MINOR.PATCH

# ============================================================================
# SYSTEM PROMPT - Single source of truth
# ============================================================================
SYSTEM_PROMPT = """
You are Career Copilot – an AI assistant that helps users with job searches and résumé analysis.

## 🎯 CONVERSATIONAL CAPABILITIES

You have conversation memory and can handle natural, multi-turn dialogues:

**Context Awareness:**
- Remember which job/resume is currently being discussed
- Track recent searches and actions
- Reference previous items without asking again

**Handle Follow-up Questions:**
- "Why?" or "Tell me more" → Provide detailed explanation of last action
- "Show me more" or "Next one" → Show additional results
- "This job" or "That position" → Use the currently active job
- "My resume" → Use the currently active resume
- "The previous one" → Reference recent items
- "Tell me about job #2" → Explain details of the 2nd job from last search

**Examples of Natural Conversations:**
User: "Search for Python jobs in Chicago"
You: [searches and shows results]
User: "Tell me about the second one"
You: [calls get_job_details with job_number=2]
User: "What about job 4?"
You: [calls get_job_details with job_number=4]
User: "Save all these jobs"
You: [calls save_searched_jobs with job_numbers="all"]

## 🔧 AVAILABLE TOOLS

### 🔍 JobPlugin

**find_jobs** - Searches for NEW jobs from external job boards
USE THIS when user wants to search for NEW jobs:
- "search for Python jobs"
- "find me Data Scientist positions"
- "look for Software Engineer roles"

**get_job_details** - Get details about a specific job from last search
USE THIS when user asks about a specific job:
- "tell me about job #2"
- "what's the description for the first one"
- "more details on job 3"

**save_searched_jobs** - Save jobs from the last search
USE THIS when user wants to save jobs:
- "save all" → saves all jobs from last search
- "save these jobs" → saves all jobs
- "save jobs 1,3,5" → saves specific jobs by number
- "save the jobs" → saves all

**get_saved_jobs** - Retrieves jobs that were previously saved

### 💾 DatabaseQueryPlugin

**query_database_with_ai** - Queries EXISTING saved jobs/resumes using natural language SQL
USE THIS when user asks about SAVED data:
- "show me saved jobs from Deloitte"
- "what jobs do I have from Company X"
- "find jobs created today"
- "show all remote positions I've saved"
- "how many resumes do I have"

**get_top_matches** - Retrieves top job matches sorted by MATCH SCORE
USE THIS when user asks about their BEST matches:
- "show my top matches"
- "what are my best matches"
- "show top 5 jobs for my resume"
This returns jobs sorted by match percentage (highest first).

**get_recent_saved_jobs** - Retrieves recently saved jobs sorted by SAVE DATE
USE THIS when user asks about RECENTLY saved jobs:
- "show my recent jobs"
- "what jobs did I save recently"
- "show last 10 saved jobs"
This returns jobs sorted by when they were added (newest first).

**get_database_stats** - Get statistics about the database

**get_database_schema** - Get database structure information

### 🎯 ResumeMatching - CONVERSATIONAL FLOW

**WHEN USER SAYS "match my resume", FOLLOW THIS FLOW:**

1. **list_resumes** - Show available resumes and ask which one
   Example: "You have 2 resumes. Which one? (say 'first', 'second', or a number)"

2. **select_resume_for_matching** - User picks a resume
   They say: "the first one", "resume 1", "my latest resume"
   You call: select_resume_for_matching with their selection

3. **select_job_filter_for_matching** - User picks job filter
   Options presented:
   - "All jobs in database (23 jobs)"
   - "Only unmatched jobs (15 jobs)"
   - "Filter by keyword (e.g., 'AI Analyst', 'Data Scientist')"
   
   They say: "all jobs", "unmatched only", "AI Analyst roles"
   You call: select_job_filter_for_matching with their choice

This multi-step flow gives users control over what gets matched.

**OTHER MATCHING FUNCTIONS:**

**explain_recent_match** - Explains a match WITHOUT re-running analysis (FAST)
USE THIS when user asks: "why did I get X%?", "tell me about match #2"
- "why did I get 87%?" → explain_recent_match with match_number=1
- "tell me about match #3" → explain_recent_match with match_number=3
- NEVER re-run full matching just to explain results!

**show_saved_matches** - Shows previously saved match results from database
USE THIS when: "show me my matches", "what are my top matches"

**find_best_job_matches** - DEPRECATED: Direct matching (use conversational flow instead)

**match_most_recent_resume** - DEPRECATED: Use conversational flow instead

### 🧹 ResumePreprocessorPlugin
Processes and cleans résumés for later matching.

### 🧹 JobPreprocessorPlugin
Processes job postings for better matching.

### ✏️ ResumeTailoring
Improve résumé content for specific job postings.

## 🎲 CRITICAL DECISION RULES

**NEW vs SAVED JOBS:**
- "search", "find new", "look for" jobs → use JobPlugin.find_jobs
- "saved", "existing", "show me jobs from X" → use DatabaseQueryPlugin.query_database_with_ai
- When in doubt, ask: "Do you want to search for NEW jobs or see SAVED jobs?"

**TOP MATCHES vs RECENT JOBS:**
- "show my top matches", "best matches" → use DatabaseQueryPlugin.get_top_matches (sorted by score)
- "show recent jobs", "what did I save" → use DatabaseQueryPlugin.get_recent_saved_jobs (sorted by date)
- If ambiguous, ask: "Do you want jobs by MATCH SCORE or by SAVE DATE?"

**JOB EXPLORATION FLOW:**
After find_jobs returns results, user can:
1. Ask about specific jobs: "tell me about job #2" → use get_job_details
2. Save all: "save all" → use save_searched_jobs with "all"
3. Save specific: "save 1,3,5" → use save_searched_jobs with "1,3,5"
4. Continue exploring: "what about job 4?" → use get_job_details again

**MATCHING FLOW:**
When user says "match my resume":
1. Call list_resumes
2. Wait for selection → call select_resume_for_matching
3. Wait for job filter → call select_job_filter_for_matching
4. Matching executes automatically

If they ask "why 87%?" later:
- Use explain_recent_match (retrieves stored results)
- NEVER call find_best_job_matches to re-match

## 📝 RESPONSE STYLE
- Be concise but informative
- Use markdown formatting for readability
- For job lists, use numbered format
- Always confirm actions: "I've saved 5 jobs to your database"
"""


# ============================================================================
# KERNEL SETUP FUNCTION
# ============================================================================

def create_kernel_with_plugins(memory: ConversationMemory = None):
    """
    Create and configure a kernel with all plugins registered.
    
    This is the single source of truth for kernel setup.
    Both CLI and Streamlit use this function.
    
    Args:
        memory: Optional existing ConversationMemory. If None, creates a new one.
    
    Returns:
        Tuple of (kernel, chat_completion, db_service, memory)
    """
    
    # Initialize kernel
    kernel = Kernel()
    
    # Add Azure OpenAI chat completion service
    chat_completion = AzureChatCompletion(
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        base_url=os.getenv("AZURE_OPENAI_BASE_URL"),
    )
    kernel.add_service(chat_completion)
    
    # Initialize services
    db_service = DatabaseService()
    
    # Create memory if not provided (for CLI use)
    if memory is None:
        memory = ConversationMemory(session_id="cli_session")
    
    # Register all plugins with memory where relevant
    kernel.add_plugin(JobPlugin(context=memory.context, memory=memory), plugin_name="JobPlugin")
    
    # Create matching plugin instance (reused across others)
    resume_matching_plugin = ResumeMatchingPlugin(kernel, db_service, memory)
    kernel.add_plugin(resume_matching_plugin, plugin_name="ResumeMatching")
    
    # Preprocessor plugins (no memory needed)
    kernel.add_plugin(ResumePreprocessorPlugin(), plugin_name="ResumePreprocessorPlugin")
    kernel.add_plugin(JobPreprocessorPlugin(), plugin_name="JobPreprocessorPlugin")
    
    # Database querying with memory awareness
    kernel.add_plugin(DatabaseQueryPlugin(kernel, memory), plugin_name="DatabaseQueryPlugin")
    
    # Resume tailoring with memory
    kernel.add_plugin(ResumeTailoringPlugin(kernel, memory), plugin_name="ResumeTailoring")
    
    # Self-improving match plugin (depends on matching plugin + memory)
    self_improving_plugin = SelfImprovingMatchPlugin(kernel, resume_matching_plugin, memory.context, memory=memory)
    kernel.add_plugin(self_improving_plugin, plugin_name="SelfImprovingMatch")
    
    return kernel, chat_completion, db_service, memory


def create_execution_settings() -> AzureChatPromptExecutionSettings:
    """
    Create execution settings with auto function calling enabled.
    
    Returns:
        Configured execution settings
    """
    execution_settings = AzureChatPromptExecutionSettings()
    execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
    execution_settings.max_tokens = 800
    execution_settings.temperature = 0.7
    return execution_settings


def create_chat_history_with_system_prompt() -> ChatHistory:
    """
    Create a new chat history with the system prompt already added.
    
    Returns:
        ChatHistory with system prompt
    """
    history = ChatHistory()
    history.add_system_message(SYSTEM_PROMPT)
    return history


# ============================================================================
# HELPER: Extract messages from ChatHistory for Observatory tracking
# ============================================================================
def extract_messages_from_history(chat_history) -> list:
    """
    Extract messages list from Semantic Kernel ChatHistory.
    
    Returns:
        List of {"role": "system/user/assistant", "content": "..."} dicts
    """
    messages = []
    for msg in chat_history.messages:
        role = msg.role.value if hasattr(msg.role, 'value') else str(msg.role)
        # Normalize role names
        if role.lower() in ['system', 'developer']:
            role = 'system'
        elif role.lower() in ['user', 'human']:
            role = 'user'
        elif role.lower() in ['assistant', 'ai', 'bot']:
            role = 'assistant'
        
        messages.append({
            "role": role,
            "content": str(msg.content)
        })
    return messages


# ============================================================================
# HELPER: Calculate tool definition tokens
# ============================================================================
def get_tool_definitions_tokens(kernel) -> int:
    """
    Calculate total tool definition tokens for Observatory tracking.
    
    Extracts FULL function schemas including all parameters, types, and constraints.
    This matches what Azure OpenAI actually receives for function calling.
    
    Args:
        kernel: Semantic Kernel instance with loaded plugins
        
    Returns:
        int: Total tokens consumed by tool/function definitions
    """
    import json
    
    try:
        total_tokens = 0
        function_count = 0
        
        # Iterate through all plugins in kernel
        for plugin_name in kernel.plugins:
            plugin = kernel.plugins[plugin_name]
            
            # Iterate through functions in each plugin
            for function_name in plugin.functions:
                function = plugin.functions[function_name]
                
                # Build COMPLETE schema (what Azure actually gets)
                schema = {
                    "type": "function",
                    "function": {
                        "name": f"{plugin_name}-{function_name}",
                        "description": function.description or "",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    }
                }
                
                # Extract ALL parameters from function metadata
                if hasattr(function, 'metadata') and function.metadata:
                    params = function.metadata.parameters
                    
                    for param in params:
                        # Get parameter type
                        param_type = "string"  # Default
                        if hasattr(param, 'type_'):
                            type_str = str(param.type_)
                            if 'int' in type_str.lower():
                                param_type = "integer"
                            elif 'float' in type_str.lower():
                                param_type = "number"
                            elif 'bool' in type_str.lower():
                                param_type = "boolean"
                            elif 'list' in type_str.lower() or 'array' in type_str.lower():
                                param_type = "array"
                            elif 'dict' in type_str.lower():
                                param_type = "object"
                        
                        # Add parameter to schema
                        schema["function"]["parameters"]["properties"][param.name] = {
                            "type": param_type,
                            "description": param.description or f"Parameter {param.name}",
                        }
                        
                        # Track if required
                        if param.is_required:
                            schema["function"]["parameters"]["required"].append(param.name)
                
                # Calculate actual tokens for FULL schema
                schema_json = json.dumps(schema, indent=2)
                function_tokens = estimate_tokens(schema_json)
                total_tokens += function_tokens
                function_count += 1
        
        logger.info(f"Tool definitions: {function_count} functions, {total_tokens} tokens")
        return total_tokens
        
    except Exception as e:
        logger.warning(f"Error extracting tool tokens: {e}. Using fallback estimate.")
        return 12000  # Conservative estimate for ~60 functions





# ============================================================================
# CLI MAIN FUNCTION WITH TWO-PHASE OPTIMIZATION
# ============================================================================

async def main():
    """
    Main CLI chat loop with comprehensive Observatory tracking.
    
    Implements two-phase optimization system:
    - BASELINE: Detects opportunities (logs "💡 OPPORTUNITY")
    - OPTIMIZED: Applies optimizations (logs "✅ APPLIED")
    
    This runs when you execute: python -m agents.semantic_kernel_setup
    """
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        force=True
    )
    
    # Set up Semantic Kernel logging (but keep it quieter)
    setup_logging()
    logging.getLogger("kernel").setLevel(logging.WARNING)
    
    # Create kernel with all plugins
    kernel, chat_completion, db_service, memory = create_kernel_with_plugins()
    
    # Create execution settings and chat history
    execution_settings = create_execution_settings()
    history = create_chat_history_with_system_prompt()
    
    # Calculate tool definition tokens once (reused across all calls)
    tool_definitions_tokens = get_tool_definitions_tokens(kernel)
    
    # Startup confirmation
    logger.info("Career Copilot CLI initialized successfully")
    logger.info(f"System prompt version: {SYSTEM_PROMPT_VERSION}")
    logger.info(f"Observatory phase: {CURRENT_PHASE.upper()}")
    print("\n🚀 Career Copilot initialized successfully.")
    print(f"📋 System prompt version: {SYSTEM_PROMPT_VERSION}")
    print(f"🎚️  Observatory phase: {CURRENT_PHASE.upper()}")
    print("Try saying: 'match my resume' or 'search for Python jobs'\n")

    # Start CLI session tracking
    session = start_session("cli_session", metadata={
        "mode": "interactive",
        "system_prompt_version": SYSTEM_PROMPT_VERSION,
        "phase": CURRENT_PHASE,
    })
    logger.info(f"Started Observatory tracking session: {session.id if hasattr(session, 'id') else 'cli_session'}")
    
    message_count = 0

    # 💬 Interactive chat loop
    try:
        while True:
            userInput = input("User > ").strip()
            if userInput.lower() == "exit":
                logger.info("User requested exit")
                print("👋 Goodbye!")
                break

            message_count += 1
            logger.info(f"Processing message #{message_count}: '{userInput[:50]}...'")

            # Link conversation to memory
            memory.conversation_id = session.id if hasattr(session, 'id') else "cli_session"
            memory.turn_number = message_count

            # Add user message to history
            history.add_user_message(userInput)
            
            # Generate request_id for this turn (for plugin linking)
            turn_request_id = f"{memory.conversation_id}_turn{message_count}"
            memory.request_id = turn_request_id
            
            # ═══════════════════════════════════════════════════════════════
            # STEP 1: Check exact cache
            # ═══════════════════════════════════════════════════════════════
            operation = "cli_chat_message"
            cache_key_data = {"user_message": userInput, "turn": message_count}
            
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
                    metadata={
                        "phase": CURRENT_PHASE,
                        "cache_hit": True,
                        "message_number": message_count,
                        "system_prompt_version": SYSTEM_PROMPT_VERSION,
                    }
                )
                
                print("Assistant >", cached_response)
                history.add_assistant_message(cached_response)
                
                logger.info(f"✅ Cache hit! Skipped LLM call.")
                continue  # Skip to next user message
            
            # ═══════════════════════════════════════════════════════════════
            # STEP 2: Check semantic cache (if available)
            # ═══════════════════════════════════════════════════════════════
            if semantic_cache:
                result = semantic_cache.get(operation=operation, prompt=userInput)
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
                        metadata={
                            "phase": CURRENT_PHASE,
                            "semantic_cache_hit": True,
                            "similarity": result.similarity,
                            "message_number": message_count,
                            "system_prompt_version": SYSTEM_PROMPT_VERSION,
                        }
                    )
                    
                    print("Assistant >", result.response)
                    history.add_assistant_message(result.response)
                    
                    logger.info(f"✅ Semantic cache hit ({result.similarity:.1%} similar)! Skipped LLM call.")
                    continue  # Skip to next user message
            
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
            
            start_time = time.time()
            
            try:
                # Make the actual LLM call
                result = await chat_completion.get_chat_message_content(
                    chat_history=history,
                    settings=execution_settings,
                    kernel=kernel,
                )
                
                latency_ms = (time.time() - start_time) * 1000
                response_text = str(result)
                success = True
                error = None
                error_type = None
                error_code = None
                error_category = None
                
            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000
                response_text = f"Error: {str(e)}"
                success = False
                error = str(e)
                
                # Classify error using helper function
                from observatory_config import classify_error
                error_info = classify_error(e, operation=operation)
                error_type = error_info['error_type']
                error_code = error_info['error_code']
                error_category = error_info['error_category']
                
                logger.error(f"LLM call failed: {error_type} - {error_code}")
                result = None
            
            # ═══════════════════════════════════════════════════════════════
            # STEP 7: Extract token usage and metadata
            # ═══════════════════════════════════════════════════════════════
            prompt_tokens = 0
            completion_tokens = 0
            time_to_first_token_ms = None
            
            if success and result and hasattr(result, 'metadata') and result.metadata:
                usage = result.metadata.get('usage')
                if usage:
                    prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                    completion_tokens = getattr(usage, 'completion_tokens', 0)
                
                # Check for streaming metrics
                if result.metadata.get('is_streaming'):
                    time_to_first_token_ms = result.metadata.get('time_to_first_token_ms')
            
            # Extract tool/function calls if available
            tool_calls = []
            tool_count = 0
            if success and result and hasattr(result, 'metadata') and result.metadata:
                function_result = result.metadata.get('function_result')
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
                        prompt=userInput,
                        response=response_text
                    )
            
            # ═══════════════════════════════════════════════════════════════
            # STEP 10: Track with Observatory (CRITICAL - INCLUDE PHASE)
            # ═══════════════════════════════════════════════════════════════
            
            # Extract messages for prompt breakdown
            messages_for_breakdown = extract_messages_from_history(history)
            
            # LLM Judge evaluation (if enabled and successful)
            quality_eval = None
            if success:
                quality_eval = await judge.maybe_evaluate(
                    operation=operation,
                    prompt=userInput,
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
                agent_name="ChatAgent",
                agent_role="orchestrator",
                operation=operation,
                success=success,
                error=error,
                
                # TIER 2: Prompt content
                messages=messages_for_breakdown,
                response_text=response_text,
                system_prompt=optimized_prompt,  # Track which prompt was used
                user_message=userInput,
                
                # TIER 2: Optimization tracking
                routing_decision=routing_meta,  # From Step 4
                cache_metadata=cache_meta,  # From Step 1
                quality_evaluation=quality_eval,
                prompt_metadata=prompt_meta,  # From Step 3
                
                # Conversation linking
                conversation_id=memory.conversation_id,
                turn_number=message_count,
                user_id=None,
                parent_call_id=None,
                
                # Model configuration
                temperature=execution_settings.temperature,
                max_tokens=max_tokens_limit,  # From Step 3
                top_p=getattr(execution_settings, 'top_p', None),
                
                # Token breakdown (auto-extracted from messages)
                system_prompt_tokens=estimate_tokens(optimized_prompt),
                user_message_tokens=estimate_tokens(userInput),
                tool_definitions_tokens=tool_definitions_tokens,
                
                # Tool/function calling
                tool_calls_made=tool_calls if tool_calls else None,
                tool_call_count=tool_count,
                
                # Streaming
                time_to_first_token_ms=time_to_first_token_ms,
                
                # Error details (if failed)
                error_type=error_type,
                error_code=error_code,
                error_category=error_category,
                
                # Observability
                trace_id=memory.conversation_id,
                request_id=turn_request_id,
                environment=os.getenv("ENVIRONMENT", "development"),
                
                # Metadata - CRITICAL: Include phase
                metadata={
                    "phase": CURRENT_PHASE,  # ← CRITICAL for phase tracking
                    "message_number": message_count,
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
                call_id=turn_request_id,
                latency_ms=latency_ms,
            )
            
            # Display response to user
            if success:
                logger.info(f"LLM response: {latency_ms:.0f}ms, {prompt_tokens + completion_tokens} tokens")
                print("Assistant >", response_text)
                history.add_message(result)
            else:
                logger.error(f"LLM call failed after {latency_ms:.0f}ms")
                print(f"Assistant > ❌ {response_text}")
        
        # Show final statistics
        if cache.enabled:
            print(f"\n📊 Cache Statistics:")
            print(f"   Hit Rate: {cache._hits / max(cache._hits + cache._misses, 1):.1%}")
            print(f"   Hits: {cache._hits}, Misses: {cache._misses}")
        
        if semantic_cache and semantic_cache.enabled:
            print(f"   Semantic Hits: {semantic_cache._hits}")
        
        # End session successfully
        end_session(session, success=True, metadata={
            "total_messages": message_count,
            "phase": CURRENT_PHASE,
        })
        logger.info(f"CLI session ended successfully. Total messages: {message_count}")
        
    except Exception as e:
        # End session with error
        end_session(session, success=False, error=str(e))
        logger.error(f"CLI session ended with error: {e}")
        raise


# Run the main function when executed directly
if __name__ == "__main__":
    asyncio.run(main())