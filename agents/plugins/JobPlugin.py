# agents/plugins/JobPlugin.py
"""
Job Plugin - Career Copilot
UPDATED: Complete Observatory Tier 1, 2, 3 metrics coverage
"""

from semantic_kernel.functions import kernel_function
from typing import Annotated
import json
import logging
import time
import os

from services.job_api import search_jobs
from services.db import save_jobs

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
)

# =============================================================================
# PROMPT VERSIONING (for API operations)
# =============================================================================
JOB_SEARCH_VERSION = "1.0.0"

JOB_SEARCH_META = create_prompt_metadata(
    template_id="job_plugin_search",
    version=JOB_SEARCH_VERSION,
    compressible_sections=[],
    optimization_flags={"api_call": True, "cacheable": True},
    config_version="1.0"
) if PromptMetadata else None

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("\033[96m[CareerCopilot Plugin] %(message)s\033[0m")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class JobPlugin:
    """
    Plugin to search for jobs using external APIs and save them to the database.
    """
    
    def __init__(self, context=None, memory=None):
        """
        Initialize the JobPlugin with optional shared conversation context.
        Args:
            context: Shared ConversationContext instance for memory.
            memory: ConversationMemory instance for tracking.
        """
        self.context = context
        self.memory = memory

        # Create execution settings once per plugin instance
        from agents.semantic_kernel_setup import create_execution_settings
        self.exec_settings = create_execution_settings()

    @kernel_function(
        name="find_jobs",
        description=(
            "Searches for NEW job postings from external job boards. "
            "Use this when the user wants to SEARCH or FIND NEW jobs: "
            "'search for jobs', 'find me Python developer positions', 'look for new opportunities'. "
            "DO NOT use this to query already saved jobs - use query_database_with_ai for that."
        )
    )
    async def find_jobs(
        self,
        query: Annotated[str, "The job title or keywords to search for (e.g., 'Software Engineer', 'Data Scientist')"],
        location: Annotated[str, "The location to search in (e.g., 'Chicago, IL', 'Remote')"] = "Chicago, IL",
        num_results: Annotated[int, "Number of job results to return (default 5)"] = 5
    ) -> Annotated[str, "JSON string containing job search results"]:
        """
        Search for jobs and store in context for exploration and later saving.
        """

            # DEBUG - ADD THESE LINES AT THE VERY TOP
        print(f"\n🔍 DEBUG JobPlugin.find_jobs:")
        print(f"   self.memory = {self.memory}")
        print(f"   type(self.memory) = {type(self.memory)}")
        if self.memory:
            print(f"   hasattr conversation_id? {hasattr(self.memory, 'conversation_id')}")
            print(f"   hasattr turn_number? {hasattr(self.memory, 'turn_number')}")
            print(f"   conversation_id = {getattr(self.memory, 'conversation_id', 'NO ATTRIBUTE')}")
            print(f"   turn_number = {getattr(self.memory, 'turn_number', 'NO ATTRIBUTE')}")
        print()

        logger.info(f"Searching for jobs: query='{query}', location='{location}', num_results={num_results}")
        
        start_time = time.time()

        try:
            # Fetch jobs from API
            jobs = search_jobs(query, location, num_results)
            
            latency_ms = (time.time() - start_time) * 1000
            logger.info(f"Retrieved {len(jobs)} job(s) for '{query}' in {location}")

            if not jobs:
                result = json.dumps({
                    "summary": f"No jobs found for '{query}' in {location}. Try different keywords or location.",
                    "jobs": []
                })
                
                # Tier 3: Cache metadata (ready for optimization)
                cache_metadata = create_cache_metadata(
                    cache_hit=False,
                    cache_key=f"job_search:{query.lower()}:{location.lower()}",
                    cache_cluster_id="job_search"
                ) if create_cache_metadata else None
                
                # Track API call - COMPLETE with all tiers
                track_llm_call(
                    # Core metrics (Tier 1)
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=latency_ms,
                    agent_name="JobPlugin",
                    agent_role="retriever",  # Added: agent role
                    operation="find_jobs",
                    success=True,  # Added: explicit success
                    
                    # Prompt content (Tier 2) - for API calls
                    prompt=f"Search: {query} in {location}",
                    response_text=result[:300],
                    prompt_metadata=JOB_SEARCH_META,
                    prompt_breakdown=None,  # No prompt breakdown for API calls
                    
                    # Quality (Tier 2) - N/A for API calls
                    quality_evaluation=None,
                    
                    # Optimization tracking (Tier 3)
                    routing_decision=None,  # No LLM routing for API calls
                    cache_metadata=cache_metadata,  # Added: cache
                    
                    # A/B Testing support (Tier 3)
                    prompt_variant_id=None,  # Added: ready for A/B tests
                    test_dataset_id=None,  # Added: ready for test runs
                    
                    # NEW: Conversation linking
                    conversation_id=self.memory.conversation_id if self.memory else None,
                    turn_number=self.memory.turn_number if self.memory else None,
                    
                    # NEW: Observability
                    environment=os.getenv("ENVIRONMENT", "development"),
                    
                    # Metadata
                    metadata={
                        "query": query,
                        "location": location,
                        "num_results_requested": num_results,
                        "jobs_found": 0,
                        "is_api_call": True,
                        "conversation_memory": self.memory,        
                        "execution_settings": self.exec_settings
                    }
                )
                
                return result

            # Store jobs in context for later reference and saving
            if self.context:
                self.context.last_searched_jobs = jobs
                self.context.last_search_query = query
                self.context.last_search_location = location
                self.context.last_action = "found_jobs"

            # Build summary
            job_summaries = []
            for i, job in enumerate(jobs, start=1):
                job_summaries.append(
                    f"{i}. {job.get('title', 'Unknown Title')} at {job.get('company', 'Unknown Company')} ({job.get('location', 'Unknown Location')})"
                )

            summary_text = "\n".join(job_summaries)

            result = json.dumps({
                "summary": f"Found {len(jobs)} '{query}' jobs in {location}:\n{summary_text}",
                "jobs": jobs
            })
            
            # Tier 3: Cache metadata (ready for optimization)
            cache_metadata = create_cache_metadata(
                cache_hit=False,
                cache_key=f"job_search:{query.lower()}:{location.lower()}",
                cache_cluster_id="job_search"
            ) if create_cache_metadata else None
            
            # Track API call - COMPLETE with all tiers
            track_llm_call(
                # Core metrics (Tier 1)
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=latency_ms,
                agent_name="JobPlugin",
                agent_role="retriever",  # Added: agent role
                operation="find_jobs",
                success=True,  # Added: explicit success
                
                # Prompt content (Tier 2)
                prompt=f"Search: {query} in {location}",
                response_text=result[:500],
                prompt_metadata=JOB_SEARCH_META,
                prompt_breakdown=None,
                
                # Quality (Tier 2)
                quality_evaluation=None,
                
                # Optimization tracking (Tier 3)
                routing_decision=None,
                cache_metadata=cache_metadata,  # Added: cache
                
                # A/B Testing support (Tier 3)
                prompt_variant_id=None,  # Added
                test_dataset_id=None,  # Added
                
                # NEW: Conversation linking
                conversation_id=self.memory.conversation_id if self.memory else None,
                turn_number=self.memory.turn_number if self.memory else None,
                    
                # NEW: Observability
                environment=os.getenv("ENVIRONMENT", "development"),
                
                # Metadata
                metadata={
                    "query": query,
                    "location": location,
                    "num_results_requested": num_results,
                    "jobs_found": len(jobs),
                    "is_api_call": True,
                    "conversation_memory": self.memory,        
                    "execution_settings": self.exec_settings
                }
            )
            
            print(f"📊 Tracked job search: {latency_ms:.0f}ms, {len(jobs)} jobs found")

            return result

        except Exception as e:
            logger.error(f"Error in JobPlugin.find_jobs: {e}", exc_info=True)
            
            # Track failed call - COMPLETE
            track_llm_call(
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=(time.time() - start_time) * 1000,
                agent_name="JobPlugin",
                agent_role="retriever",
                operation="find_jobs",
                success=False,  # Failed
                error=str(e),
                prompt=f"Search: {query} in {location}",
                prompt_metadata=JOB_SEARCH_META,
                prompt_breakdown=None,
                quality_evaluation=None,
                routing_decision=None,
                cache_metadata=None,
                prompt_variant_id=None,
                test_dataset_id=None,
                
                # NEW: Error details
                error_type=type(e).__name__,
                retry_count=0,

                # NEW: Conversation linking
                conversation_id=self.memory.conversation_id if self.memory else None,
                turn_number=self.memory.turn_number if self.memory else None,
                
                # NEW: Observability
                environment=os.getenv("ENVIRONMENT", "development"),
                
                metadata={
                    "query": query,
                    "location": location,
                    "is_api_call": True,
                    "error_type": type(e).__name__,
                    "conversation_memory": self.memory,
                    "execution_settings": self.exec_settings
                }
            )
            
            return json.dumps({
                "summary": f"❌ Job search failed: {str(e)}. Please try again or contact support.",
                "jobs": []
            })

    @kernel_function(
        name="get_job_details",
        description=(
            "Get detailed information about a specific job from the last search results. "
            "Use when user asks: 'tell me about job #2', 'what's the description for the first one', 'more details on job 3'"
        )
    )
    async def get_job_details(
        self,
        job_number: Annotated[int, "The job number from the search results (1-indexed)"]
    ) -> Annotated[str, "Detailed job information"]:
        """
        Retrieve details for a specific job from the last search.
        """
        start_time = time.time()
        
        if not self.context or not hasattr(self.context, 'last_searched_jobs'):
            return "❌ No recent job search found. Please search for jobs first."
        
        jobs = self.context.last_searched_jobs
        
        if job_number < 1 or job_number > len(jobs):
            return f"❌ Invalid job number. Please choose between 1 and {len(jobs)}."
        
        job = jobs[job_number - 1]
        
        # Store this job as the active one
        if hasattr(self.context, 'active_job_index'):
            self.context.active_job_index = job_number - 1
        
        details = f"**{job.get('title', 'Unknown Title')}** at **{job.get('company', 'Unknown Company')}**\n\n"
        details += f"📍 **Location:** {job.get('location', 'Unknown')}\n"
        details += f"🔗 **Apply:** {job.get('link', 'No link available')}\n\n"
        details += f"**Description:**\n{job.get('description', 'No description available')}"
        
        # Track retrieval operation - COMPLETE
        track_llm_call(
            # Core metrics (Tier 1)
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=(time.time() - start_time) * 1000,
            agent_name="JobPlugin",
            agent_role="retriever",  # Added
            operation="get_job_details",
            success=True,  # Added
            
            # Prompt content (Tier 2)
            prompt=f"Get details for job #{job_number}",
            response_text=details[:300],
            prompt_metadata=None,
            prompt_breakdown=None,
            
            # Quality (Tier 2)
            quality_evaluation=None,
            
            # Optimization tracking (Tier 3)
            routing_decision=None,
            cache_metadata=None,
            
            # A/B Testing support (Tier 3)
            prompt_variant_id=None,
            test_dataset_id=None,

            # NEW: Conversation linking
            conversation_id=self.memory.conversation_id if self.memory else None,
            turn_number=self.memory.turn_number if self.memory else None,    
            
            # NEW: Observability
            environment=os.getenv("ENVIRONMENT", "development"),
            
            # Metadata
            metadata={
                "job_number": job_number,
                "job_title": job.get('title'),
                "job_company": job.get('company'),
                "is_retrieval": True,
                "conversation_memory": self.memory,
                "execution_settings": self.exec_settings
            }
        )
        
        return details

    @kernel_function(
        name="save_searched_jobs",
        description=(
            "Saves jobs from the last search to the database. "
            "Use when user says: 'save all', 'save these jobs', 'save jobs 1,3,5', 'save the jobs'"
        )
    )
    async def save_searched_jobs(
        self,
        job_numbers: Annotated[str, "Comma-separated job numbers to save (e.g., '1,3,5') or 'all' to save everything"] = "all"
    ) -> Annotated[str, "Confirmation of saved jobs"]:
        """
        Save specific jobs or all jobs from the last search.
        """
        start_time = time.time()
        
        if not self.context or not hasattr(self.context, 'last_searched_jobs'):
            return "❌ No recent job search found. Please search for jobs first."
        
        jobs = self.context.last_searched_jobs
        query = getattr(self.context, 'last_search_query', '')
        location = getattr(self.context, 'last_search_location', '')
        
        # Determine which jobs to save
        if job_numbers.lower() == "all":
            jobs_to_save = jobs
            job_indices = list(range(1, len(jobs) + 1))
        else:
            try:
                indices = [int(n.strip()) for n in job_numbers.split(',')]
                jobs_to_save = [jobs[i-1] for i in indices if 1 <= i <= len(jobs)]
                job_indices = [i for i in indices if 1 <= i <= len(jobs)]
                
                if not jobs_to_save:
                    return f"❌ Invalid job numbers. Please choose from 1-{len(jobs)}."
            except (ValueError, IndexError) as e:
                return f"❌ Error parsing job numbers: {str(e)}. Use format: '1,3,5' or 'all'"
        
        # Save to database
        save_jobs(jobs_to_save, query, location)
        
        logger.info(f"Saved {len(jobs_to_save)} jobs to database")
        
        if job_numbers.lower() == "all":
            result = f"✅ Saved all {len(jobs_to_save)} '{query}' jobs in {location}."
        else:
            result = f"✅ Saved {len(jobs_to_save)} selected job(s): #{', #'.join(map(str, job_indices))}."
        
        # Track database write - COMPLETE
        track_llm_call(
            # Core metrics (Tier 1)
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=(time.time() - start_time) * 1000,
            agent_name="JobPlugin",
            agent_role="writer",  # Added
            operation="save_searched_jobs",
            success=True,  # Added
            
            # Prompt content (Tier 2)
            prompt=f"Save jobs: {job_numbers}",
            response_text=result,
            prompt_metadata=None,
            prompt_breakdown=None,
            
            # Quality (Tier 2)
            quality_evaluation=None,
            
            # Optimization tracking (Tier 3)
            routing_decision=None,
            cache_metadata=None,
            
            # A/B Testing support (Tier 3)
            prompt_variant_id=None,
            test_dataset_id=None,

            # NEW: Conversation linking
            conversation_id=self.memory.conversation_id if self.memory else None,
            turn_number=self.memory.turn_number if self.memory else None,
            
            # NEW: Observability
            environment=os.getenv("ENVIRONMENT", "development"),
            
            # Metadata
            metadata={
                "job_numbers_requested": job_numbers,
                "jobs_saved": len(jobs_to_save),
                "query": query,
                "location": location,
                "is_db_write": True,
                "conversation_memory": self.memory,
                "execution_settings": self.exec_settings
            }
        )
        
        print(f"📊 Tracked job save: {len(jobs_to_save)} jobs saved")
        
        return result

    @kernel_function(
        name="get_saved_jobs",
        description="Retrieves all jobs that have been saved to the database from previous searches"
    )
    async def get_saved_jobs(
        self,
        limit: Annotated[int, "Maximum number of jobs to return (default 10)"] = 10
    ) -> Annotated[str, "JSON string containing saved jobs"]:
        """
        Retrieve saved jobs from the database.
        """
        start_time = time.time()
        
        try:
            import sqlite3
            from services.db import DB_PATH
            
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, title, company, location, link, description
                FROM jobs
                ORDER BY id DESC
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                result = json.dumps({
                    "summary": "No saved jobs found in the database.",
                    "jobs": []
                })
                
                # Track database read - COMPLETE
                track_llm_call(
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=(time.time() - start_time) * 1000,
                    agent_name="JobPlugin",
                    agent_role="retriever",
                    operation="get_saved_jobs",
                    success=True,
                    prompt=f"Get saved jobs (limit={limit})",
                    response_text=result[:300],
                    prompt_metadata=None,
                    prompt_breakdown=None,
                    quality_evaluation=None,
                    routing_decision=None,
                    cache_metadata=None,
                    prompt_variant_id=None,
                    test_dataset_id=None,
                    
                    # NEW: Conversation linking
                    conversation_id=self.memory.conversation_id if self.memory else None,
                    turn_number=self.memory.turn_number if self.memory else None,

                    # NEW: Observability
                    environment=os.getenv("ENVIRONMENT", "development"),
                    
                    metadata={
                        "limit": limit,
                        "jobs_returned": 0,
                        "is_db_read": True,
                        "conversation_memory": self.memory,
                        "execution_settings": self.exec_settings
                    }
                )
                
                return result
            
            jobs = []
            for row in rows:
                jobs.append({
                    "id": row[0],
                    "title": row[1],
                    "company": row[2],
                    "location": row[3],
                    "link": row[4],
                    "description": row[5][:200] + "..." if row[5] and len(row[5]) > 200 else row[5]
                })
            
            result = json.dumps({
                "summary": f"Found {len(jobs)} saved jobs in the database.",
                "jobs": jobs
            })
            
            # Track database read - COMPLETE
            track_llm_call(
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=(time.time() - start_time) * 1000,
                agent_name="JobPlugin",
                agent_role="retriever",
                operation="get_saved_jobs",
                success=True,
                prompt=f"Get saved jobs (limit={limit})",
                response_text=result[:300],
                prompt_metadata=None,
                prompt_breakdown=None,
                quality_evaluation=None,
                routing_decision=None,
                cache_metadata=None,
                prompt_variant_id=None,
                test_dataset_id=None,
                
                # NEW: Conversation linking
                conversation_id=self.memory.conversation_id if self.memory else None,
                turn_number=self.memory.turn_number if self.memory else None,

                # NEW: Observability
                environment=os.getenv("ENVIRONMENT", "development"),
                
                metadata={
                    "limit": limit,
                    "jobs_returned": len(jobs),
                    "is_db_read": True,
                    "conversation_memory": self.memory,
                    "execution_settings": self.exec_settings
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving saved jobs: {e}", exc_info=True)
            
            # Track failed call - COMPLETE
            track_llm_call(
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=(time.time() - start_time) * 1000,
                agent_name="JobPlugin",
                agent_role="retriever",
                operation="get_saved_jobs",
                success=False,
                error=str(e),
                prompt=f"Get saved jobs (limit={limit})",
                prompt_metadata=None,
                prompt_breakdown=None,
                quality_evaluation=None,
                routing_decision=None,
                cache_metadata=None,
                prompt_variant_id=None,
                test_dataset_id=None,
                
                # NEW: Error details
                error_type=type(e).__name__,
                retry_count=0,

                # NEW: Conversation linking
                conversation_id=self.memory.conversation_id if self.memory else None,
                turn_number=self.memory.turn_number if self.memory else None,
                
                # NEW: Observability
                environment=os.getenv("ENVIRONMENT", "development"),
                
                metadata={
                    "limit": limit,
                    "is_db_read": True,
                    "error_type": type(e).__name__,
                    "conversation_memory": self.memory,
                    "execution_settings": self.exec_settings
                }
            )
            
            return json.dumps({
                "summary": f"❌ Error retrieving saved jobs: {str(e)}",
                "jobs": []
            })