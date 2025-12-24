# agents/semantic_kernel_setup.py
"""
Semantic Kernel Setup - Single Source of Truth

This module is the MAIN configuration file for Career Copilot.
Both CLI and Streamlit chatbot import from here.

COMPREHENSIVE: Full Observatory Tier 1, 2, 3 metrics support
- Tier 1: Core metrics (tokens, latency, cost)
- Tier 2: PromptBreakdown, PromptMetadata, QualityEvaluation
- Tier 3: RoutingDecision, CacheMetadata, A/B Testing support

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

# Observatory Integration - Complete imports
from observatory_config import (
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
    calculate_complexity_score,
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
# HELPER: Extract messages from ChatHistory for prompt breakdown
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
# HELPER: Create prompt breakdown from messages
# ============================================================================
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
            system_tokens = len(system_prompt) // 4
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
        user_tokens = len(user_message) // 4
        
        # Previous user messages go to history
        for msg in user_messages[:-1]:
            chat_history.append(msg)
            chat_history_tokens += len(msg.get("content", "")) // 4
    
    # All assistant/function messages go to history
    for msg in other_messages:
        chat_history.append(msg)
        chat_history_tokens += len(msg.get("content", "")) // 4
    
    return create_prompt_breakdown(
        system_prompt=system_prompt,
        system_prompt_tokens=system_tokens,
        user_message=user_message,
        user_message_tokens=user_tokens,
        chat_history=chat_history if chat_history else None,
        chat_history_tokens=chat_history_tokens if chat_history else None,
    )


# ============================================================================
# CLI PROMPT METADATA
# ============================================================================
CLI_PROMPT_META = create_prompt_metadata(
    template_id="career_copilot_cli",
    version=SYSTEM_PROMPT_VERSION,
    compressible_sections=["AVAILABLE TOOLS", "CRITICAL DECISION RULES"],
    optimization_flags={"auto_function_calling": True},
    config_version="1.0"
) if PromptMetadata else None


# ============================================================================
# CLI MAIN FUNCTION
# ============================================================================

async def main():
    """
    Main CLI chat loop.
    
    This runs when you execute: python -m agents.semantic_kernel_setup
    """
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        force=True  # Override any existing logging config
    )
    
    # Set up Semantic Kernel logging (but keep it quieter)
    setup_logging()
    logging.getLogger("kernel").setLevel(logging.WARNING)
    
    # Create kernel with all plugins
    kernel, chat_completion, db_service, memory = create_kernel_with_plugins()
    
    # Create execution settings and chat history
    execution_settings = create_execution_settings()
    history = create_chat_history_with_system_prompt()
    
    # ✅ Startup confirmation
    logger.info("Career Copilot CLI initialized successfully")
    logger.info(f"System prompt version: {SYSTEM_PROMPT_VERSION}")
    print("\n🚀 Career Copilot initialized successfully.")
    print(f"📋 System prompt version: {SYSTEM_PROMPT_VERSION}")
    print("Try saying: 'match my resume' or 'search for Python jobs'\n")

    # Start CLI session tracking
    session = start_session("cli_session", metadata={
        "mode": "interactive",
        "system_prompt_version": SYSTEM_PROMPT_VERSION
    })
    logger.info("Started Observatory tracking session")
    
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

            # ADD THESE TWO LINES:
            memory.conversation_id = session.id
            memory.turn_number = message_count

            # Add user message to history
            history.add_user_message(userInput)
            
            # Track LLM call timing
            start_time = time.time()
            
            # Let the AI handle the conversation and plugin calls
            result = await chat_completion.get_chat_message_content(
                chat_history=history,
                settings=execution_settings,
                kernel=kernel,
            )
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract token usage if available
            prompt_tokens = 0
            completion_tokens = 0
            
            if hasattr(result, 'metadata') and result.metadata:
                usage = result.metadata.get('usage')
                if usage:
                    prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                    completion_tokens = getattr(usage, 'completion_tokens', 0)
            
            # Extract tool/function calls if available
            tool_calls = []
            tool_count = 0
            if hasattr(result, 'metadata') and result.metadata:
                function_result = result.metadata.get('function_result')
                if function_result:
                    tool_calls.append({
                        "name": function_result.get('name'),
                        "arguments": function_result.get('arguments')
                    })
                    tool_count = 1
            
            # Get response text
            response_text = str(result)
            
            # Extract messages for prompt breakdown (BEFORE adding assistant response)
            messages_for_breakdown = extract_messages_from_history(history)
            
            # Create prompt breakdown for Tier 2
            prompt_breakdown = create_prompt_breakdown_from_messages(messages_for_breakdown)
            
            # LLM Judge evaluation (50% sampling)
            quality_eval = await judge.maybe_evaluate(
                operation="cli_chat_message",
                prompt=userInput,
                response=response_text,
                llm_client=kernel, 
            )
            
            # Tier 3: Routing decision (placeholder - ready for optimization)
            routing_decision = create_routing_decision(
                chosen_model=DEFAULT_MODEL,
                alternative_models=["gpt-4o", "gpt-4o-mini"],
                reasoning="CLI chat interaction - using default model",
                complexity_score=calculate_complexity_score(userInput, tool_call_count=tool_count)
            ) if create_routing_decision else None
            
            # Tier 3: Cache metadata (placeholder - ready for optimization)
            cache_metadata = create_cache_metadata(
                cache_hit=False,
                cache_key=None,
                cache_cluster_id="cli_chat"
            ) if create_cache_metadata else None
            
            # Track in Observatory - COMPLETE with ALL tiers
            track_llm_call(
                # Core metrics (Tier 1) - model auto-detected from env
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                agent_name="ChatAgent",
                agent_role="orchestrator",  # Added: agent role
                operation="cli_chat_message",
                success=True,  # Added: explicit success
                
                # Prompt analysis (Tier 2)
                messages=messages_for_breakdown,
                response_text=response_text,
                prompt_metadata=CLI_PROMPT_META,  # Track system prompt version
                prompt_breakdown=prompt_breakdown,  # Added: token breakdown
                
                # Quality evaluation (Tier 2)
                quality_evaluation=quality_eval,
                
                # Optimization tracking (Tier 3)
                routing_decision=routing_decision,  # Added: routing
                cache_metadata=cache_metadata,  # Added: cache
                
                # A/B Testing support (Tier 3)
                prompt_variant_id=None,  # Added: ready for A/B tests
                test_dataset_id=None,  # Added: ready for test runs
                
                # NEW: Conversation linking
                conversation_id=session.id if hasattr(session, 'id') else "cli_session",
                turn_number=message_count,
                user_id=None,  # Could be added if user authentication exists
                parent_call_id=None,  # Orchestrator is root of call tree
                
                # NEW: Model configuration (from execution_settings)
                temperature=0.7,
                max_tokens=800,
                top_p=None,
                
                # NEW: Separate prompt components (for fast top-level queries)
                system_prompt=SYSTEM_PROMPT,
                user_message=userInput,
                
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

                # NEW: Streaming
                time_to_first_token_ms=None,
                
                # NEW: Observability
                trace_id=session.id if hasattr(session, 'id') else None,
                request_id=session.id + f"_turn{message_count}",  # Plugins use this as parent_call_id
                environment=os.getenv("ENVIRONMENT", "development"),
                
                # Metadata
                metadata={
                    "message_number": message_count,
                    "system_prompt_version": SYSTEM_PROMPT_VERSION,
                    "judged": quality_eval is not None
                }
            )
            
            logger.info(f"LLM response received: {latency_ms:.0f}ms, {prompt_tokens + completion_tokens} tokens")
            
            print("Assistant >", response_text)
            history.add_message(result)
        
        # End session successfully
        end_session(session, success=True)
        logger.info(f"CLI session ended successfully. Total messages: {message_count}")
        
    except Exception as e:
        # End session with error
        end_session(session, success=False, error=str(e))
        logger.error(f"CLI session ended with error: {e}")
        raise


# Run the main function when executed directly
if __name__ == "__main__":
    asyncio.run(main())