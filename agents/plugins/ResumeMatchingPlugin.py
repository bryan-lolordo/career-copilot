# agents/plugins/ResumeMatchingPlugin.py
"""
Resume Matching Plugin - Career Copilot
UPDATED: Complete Observatory integration with full 10-step optimization pattern
"""

from semantic_kernel.functions import kernel_function
from typing import Annotated
from openai import AsyncAzureOpenAI
import json
import logging
import time
import os
import uuid
import hashlib

# ═════════════════════════════════════════════════════════════════════════
# OBSERVATORY INTEGRATION - STANDARDIZED IMPORT BLOCK FOR LLM-MAKING PLUGINS
# ═════════════════════════════════════════════════════════════════════════
from observatory_config import (
    # Main tracking
    track_llm_call,

    # Optimization components (10-step pattern - import ALL for consistency)
    cache,
    persistent_cache,  # SQLite-backed cache for cross-session persistence
    # semantic_cache,  # Disabled for resume matching
    prefix_cache,
    router,
    prompt_optimizer,
    batch_detector,
    parallel_detector,
    streaming_detector,
    sequential_detector,        
    context_growth_detector,    
    token_efficiency_detector,  
    judge,
    batch_processor,       
    parallel_executor,
    
    # Config constants
    DEFAULT_MODEL,
    CURRENT_PHASE,
    
    # Data models
    PromptMetadata,
    
    # Helper functions
    create_prompt_metadata,
    create_prompt_breakdown,
    create_routing_decision,
    # create_cache_metadata,  # Not needed without semantic cache
    estimate_tokens,
    classify_error,
    extract_azure_cache_metrics,
    fire_and_forget_judge,  # Non-blocking judge evaluation
)

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# PROMPT VERSIONING
# =============================================================================
QUICK_SCORE_PROMPT_VERSION = "1.0.0"
DEEP_ANALYSIS_PROMPT_VERSION = "1.0.0"

class ResumeMatchingPlugin:
    def __init__(self, kernel, chat_completion, database_service, memory=None):
        """
        Args:
            kernel: Your Semantic Kernel instance
            database_service: Your database access layer to fetch resumes and jobs
            memory: ConversationMemory instance for context tracking
        """
        self.kernel = kernel
        self.chat_completion = chat_completion
        self.db = database_service
        self.memory = memory

        # Create execution settings once per plugin instance
        from agents.semantic_kernel_setup import create_execution_settings
        self.exec_settings = create_execution_settings()
    
    @kernel_function(
        name="list_resumes",
        description="Lists all available resumes in the database so the user can choose which one to match"
    )
    async def list_resumes(self) -> Annotated[str, "Formatted list of available resumes"]:
        """Lists all resumes from the database."""
        start_time = time.time()
        
        resumes = self.db.list_all_resumes()

        if not resumes:
            # Track empty result
            track_llm_call(
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=(time.time() - start_time) * 1000,
                agent_name="ResumeMatching",
                agent_role="retriever",
                operation="list_resumes",
                success=True,
                prompt="List all resumes",
                response_text="No resumes found",
                conversation_id=self.memory.conversation_id if self.memory else None,
                turn_number=self.memory.turn_number if self.memory else None,
                parent_call_id=self.memory.request_id if self.memory else None,
                request_id=str(uuid.uuid4()),
                trace_id=self.memory.conversation_id if self.memory else None,
                environment=os.getenv("ENVIRONMENT", "development"),
                metadata={
                    "phase": CURRENT_PHASE,
                    "result_count": 0,
                    "is_db_read": True,
                }
            )
            return "No resumes found in the database. Please upload a resume first."
        
        response = "📄 Available resumes:\n\n"
        for i, resume in enumerate(resumes, 1):
            response += f"{i}. **{resume['name']}** (ID: {resume['id']})\n"
        
        # Store resumes in context for "the first one" references
        if self.memory:
            self.memory.context.available_resumes = resumes
            self.memory.context.awaiting_resume_selection = True
        
        response += "\nWhich resume would you like to match?"
        
        # Track successful retrieval
        track_llm_call(
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=(time.time() - start_time) * 1000,
            agent_name="ResumeMatching",
            agent_role="retriever",
            operation="list_resumes",
            success=True,
            prompt="List all resumes",
            response_text=response[:500],
            conversation_id=self.memory.conversation_id if self.memory else None,
            turn_number=self.memory.turn_number if self.memory else None,
            parent_call_id=self.memory.request_id if self.memory else None,
            request_id=str(uuid.uuid4()),
            trace_id=self.memory.conversation_id if self.memory else None,
            environment=os.getenv("ENVIRONMENT", "development"),
            metadata={
                "phase": CURRENT_PHASE,
                "result_count": len(resumes),
                "is_db_read": True,
            }
        )
        
        return response
    
    @kernel_function(
        name="select_resume_for_matching",
        description=(
            "Selects a resume for matching when user specifies which one. "
            "Use when user says: 'the first one', 'resume 1', 'my latest resume', 'the second resume'"
        )
    )
    async def select_resume_for_matching(
        self,
        selection: Annotated[str, "User's resume selection (e.g., '1', 'first', 'latest', 'most recent', 'resume 2')"]
    ) -> Annotated[str, "Confirmation and next step"]:
        """
        Store selected resume and ask about job filtering.
        """
        start_time = time.time()
        selection_lower = selection.lower().strip()
        
        # Parse selection
        if selection_lower in ["1", "first", "first one", "the first"]:
            resume_index = 0
        elif selection_lower in ["2", "second", "second one", "the second"]:
            resume_index = 1
        elif selection_lower in ["3", "third", "third one", "the third"]:
            resume_index = 2
        elif selection_lower in ["latest", "most recent", "newest", "last"]:
            resume_index = 0  # Most recent is first in the list
        else:
            # Try to parse as number
            try:
                resume_index = int(selection.strip()) - 1
            except:
                # Track parse error
                track_llm_call(
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=(time.time() - start_time) * 1000,
                    agent_name="ResumeMatching",
                    agent_role="retriever",
                    operation="select_resume_for_matching",
                    success=False,
                    error=f"Invalid selection: {selection}",
                    error_type="ValidationError",
                    retry_count=0,
                    conversation_id=self.memory.conversation_id if self.memory else None,
                    turn_number=self.memory.turn_number if self.memory else None,
                    parent_call_id=self.memory.request_id if self.memory else None,
                    request_id=str(uuid.uuid4()),
                    trace_id=self.memory.conversation_id if self.memory else None,
                    environment=os.getenv("ENVIRONMENT", "development"),
                    metadata={
                        "phase": CURRENT_PHASE,
                        "selection": selection,
                    }
                )
                return "❌ I didn't understand that selection. Please say 'first', 'second', or a number like '1' or '2'."
        
        # Get resumes from context
        if not self.memory or not hasattr(self.memory.context, 'available_resumes'):
            # Fallback: get resumes again
            resumes = self.db.list_all_resumes()
            if self.memory:
                self.memory.context.available_resumes = resumes
        else:
            resumes = self.memory.context.available_resumes
        
        if not resumes or resume_index < 0 or resume_index >= len(resumes):
            # Track invalid selection
            track_llm_call(
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=(time.time() - start_time) * 1000,
                agent_name="ResumeMatching",
                agent_role="retriever",
                operation="select_resume_for_matching",
                success=False,
                error="Invalid resume index",
                retry_count=0,
                prompt=f"Select resume: {selection}",
                conversation_id=self.memory.conversation_id if self.memory else None,
                turn_number=self.memory.turn_number if self.memory else None,
                parent_call_id=self.memory.request_id if self.memory else None,
                request_id=str(uuid.uuid4()),
                trace_id=self.memory.conversation_id if self.memory else None,
                environment=os.getenv("ENVIRONMENT", "development"),
                metadata={
                    "phase": CURRENT_PHASE,
                    "selection": selection,
                    "resume_index": resume_index,
                }
            )
            return f"❌ Invalid selection. Please choose between 1 and {len(resumes) if resumes else 0}."
        
        selected_resume = resumes[resume_index]
        
        # Store selection in context
        if self.memory:
            self.memory.context.selected_resume_for_matching = selected_resume
            self.memory.context.awaiting_resume_selection = False
            self.memory.context.awaiting_job_filter_selection = True
            self.memory.set_current_focus(resume_id=selected_resume['id'])
        
        # Get job counts for context
        import sqlite3
        from services.db import DB_PATH
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM jobs")
        total_jobs = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*) FROM jobs j
            WHERE NOT EXISTS (
                SELECT 1 FROM resume_job_matches m
                WHERE m.resume_id = ? AND m.job_id = j.id
            )
        """, (selected_resume['id'],))
        unmatched_jobs = cursor.fetchone()[0]
        
        conn.close()
        
        response = f"✅ Selected: **{selected_resume['name']}**\n\n"
        response += f"Which jobs would you like to match?\n\n"
        response += f"1️⃣ All jobs in database ({total_jobs} jobs)\n"
        response += f"2️⃣ Only unmatched jobs ({unmatched_jobs} jobs)\n"
        response += f"3️⃣ Filter by keyword (e.g., 'AI Analyst', 'Data Scientist')\n\n"
        response += f"What would you like?"
        
        # Track successful selection
        track_llm_call(
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=(time.time() - start_time) * 1000,
            agent_name="ResumeMatching",
            agent_role="retriever",
            operation="select_resume_for_matching",
            success=True,
            prompt=f"Select resume: {selection}",
            response_text=response[:500],
            conversation_id=self.memory.conversation_id if self.memory else None,
            turn_number=self.memory.turn_number if self.memory else None,
            parent_call_id=self.memory.request_id if self.memory else None,
            request_id=str(uuid.uuid4()),
            trace_id=self.memory.conversation_id if self.memory else None,
            environment=os.getenv("ENVIRONMENT", "development"),
            metadata={
                "phase": CURRENT_PHASE,
                "selection": selection,
                "resume_id": selected_resume['id'],
                "resume_name": selected_resume['name'],
                "total_jobs": total_jobs,
                "unmatched_jobs": unmatched_jobs,
                "is_db_read": True,
            }
        )
        
        return response
    
    @kernel_function(
        name="select_job_filter_for_matching",
        description=(
            "Selects which jobs to match against when user specifies. "
            "Use when user says: 'all jobs', 'only unmatched', 'AI Analyst roles', 'jobs with Python'"
        )
    )
    async def select_job_filter_for_matching(
        self,
        filter_choice: Annotated[str, "User's job filter choice (e.g., 'all', 'unmatched', 'AI Analyst', 'Data Scientist')"]
    ) -> Annotated[str, "Start matching with selected filter"]:
        """
        Apply job filter and start matching.
        """
        start_time = time.time()
        
        if not self.memory or not hasattr(self.memory.context, 'selected_resume_for_matching'):
            track_llm_call(
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=(time.time() - start_time) * 1000,
                agent_name="ResumeMatching",
                agent_role="retriever",
                operation="select_job_filter_for_matching",
                success=False,
                error="No resume selected",
                retry_count=0,
                prompt=f"Filter: {filter_choice}",
                conversation_id=self.memory.conversation_id if self.memory else None,
                turn_number=self.memory.turn_number if self.memory else None,
                parent_call_id=self.memory.request_id if self.memory else None,
                request_id=str(uuid.uuid4()),
                trace_id=self.memory.conversation_id if self.memory else None,
                environment=os.getenv("ENVIRONMENT", "development"),
                metadata={
                    "phase": CURRENT_PHASE,
                    "filter_choice": filter_choice,
                }
            )
            return "❌ Please select a resume first. Say 'match my resume' to start."
        
        resume = self.memory.context.selected_resume_for_matching
        resume_id = resume['id']
        filter_lower = filter_choice.lower().strip()
        
        import sqlite3
        from services.db import DB_PATH
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Determine filter
        if filter_lower in ["all", "all jobs", "1", "everything", "every job"]:
            cursor.execute("SELECT id FROM jobs")
            job_filter = "all jobs"
        elif filter_lower in ["unmatched", "only unmatched", "2", "new jobs", "jobs i haven't matched"]:
            cursor.execute("""
                SELECT j.id FROM jobs j
                WHERE NOT EXISTS (
                    SELECT 1 FROM resume_job_matches m
                    WHERE m.resume_id = ? AND m.job_id = j.id
                )
            """, (resume_id,))
            job_filter = "unmatched jobs"
        else:
            # Keyword filter
            keyword = filter_choice.strip()
            cursor.execute("""
                SELECT id FROM jobs
                WHERE title LIKE ? OR description LIKE ? OR company LIKE ?
            """, (f'%{keyword}%', f'%{keyword}%', f'%{keyword}%'))
            job_filter = f"jobs matching '{keyword}'"
        
        job_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        if not job_ids:
            track_llm_call(
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=(time.time() - start_time) * 1000,
                agent_name="ResumeMatching",
                agent_role="retriever",
                operation="select_job_filter_for_matching",
                success=True,
                prompt=f"Filter: {filter_choice}",
                response_text="No jobs found",
                conversation_id=self.memory.conversation_id if self.memory else None,
                turn_number=self.memory.turn_number if self.memory else None,
                parent_call_id=self.memory.request_id if self.memory else None,
                request_id=str(uuid.uuid4()),
                trace_id=self.memory.conversation_id if self.memory else None,
                environment=os.getenv("ENVIRONMENT", "development"),
                metadata={
                    "phase": CURRENT_PHASE,
                    "filter_choice": filter_choice,
                    "job_filter": job_filter,
                    "result_count": 0,
                }
            )
            return f"❌ No jobs found for filter: {job_filter}. Try a different filter or add more jobs."
        
        # Store filter in context
        if self.memory:
            self.memory.context.selected_job_ids_for_matching = job_ids
            self.memory.context.awaiting_job_filter_selection = False
        
        # Track filter selection
        track_llm_call(
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=(time.time() - start_time) * 1000,
            agent_name="ResumeMatching",
            agent_role="retriever",
            operation="select_job_filter_for_matching",
            success=True,
            prompt=f"Filter: {filter_choice}",
            response_text=f"Starting match for {len(job_ids)} jobs",
            conversation_id=self.memory.conversation_id if self.memory else None,
            turn_number=self.memory.turn_number if self.memory else None,
            parent_call_id=self.memory.request_id if self.memory else None,
            request_id=str(uuid.uuid4()),
            trace_id=self.memory.conversation_id if self.memory else None,
            environment=os.getenv("ENVIRONMENT", "development"),
            metadata={
                "phase": CURRENT_PHASE,
                "filter_choice": filter_choice,
                "job_filter": job_filter,
                "result_count": len(job_ids),
                "resume_id": resume_id,
                "resume_name": resume['name'],
                "is_db_read": True,
            }
        )
        
        # Start matching
        response = f"🚀 Starting match for **{resume['name']}** against {len(job_ids)} {job_filter}...\n\n"
        response += f"This will take about {len(job_ids) * 2} seconds.\n\n"
        
        # Call the actual matching function
        match_result = await self._execute_filtered_matching(resume_id, job_ids)
        
        # Clear context
        if self.memory:
            if hasattr(self.memory.context, 'selected_resume_for_matching'):
                delattr(self.memory.context, 'selected_resume_for_matching')
            if hasattr(self.memory.context, 'selected_job_ids_for_matching'):
                delattr(self.memory.context, 'selected_job_ids_for_matching')
        
        return response + match_result
    
    async def _execute_filtered_matching(self, resume_id: int, job_ids: list) -> str:
        """Execute matching for a specific resume against filtered jobs."""
        logger.info(f"Starting resume matching: Resume #{resume_id} vs {len(job_ids)} jobs")
        workflow_start_time = time.time()
        
        try:
            # Get resume
            resume = self.db.get_resume_by_id(resume_id)
            if not resume:
                return f"❌ Error: Resume with ID {resume_id} not found."
            
            resume_text = resume.get('content', '')
            resume_name = resume.get('name', 'Unknown Resume')
            
            # Get jobs
            jobs = []
            for job_id in job_ids:
                job = self.db.get_job_by_id(job_id)
                if job:
                    jobs.append(job)
            
            if not jobs:
                return "❌ No jobs found for matching."
            
            logger.info(f"Phase 1: Quick scoring {len(jobs)} jobs...")
            quick_score_start = time.time()

            # PHASE 1: Quick scoring
            # BASELINE: Sequential execution (detect opportunities)
            # OPTIMIZED: Batch + parallel execution (apply optimization)
            phase1_metrics = None

            if CURRENT_PHASE == "optimized":
                # ═══════════════════════════════════════════════════════════════
                # OPTIMIZED: Batch + parallel execution
                # ═══════════════════════════════════════════════════════════════
                batches = batch_processor.create_batches(
                    items=jobs,
                    batch_size=3,
                    operation="quick_score_job"
                )

                logger.info(f"⚡ OPTIMIZED: Created {len(batches)} batches from {len(jobs)} jobs")

                # Define batch processing function
                async def score_batch(batch, batch_num):
                    """Score all jobs in a batch sequentially."""
                    results = []
                    for i, job in enumerate(batch):
                        job_idx = (batch_num * 3) + i + 1
                        logger.debug(f"Quick scoring job {job_idx}/{len(jobs)}: {job.get('title', 'Unknown')}")

                        job_start_time = time.time()
                        scored = await self._quick_score_job_match(resume_text, job)
                        results.append(scored)

                        job_latency = (time.time() - job_start_time) * 1000
                        logger.debug(f"  └─ Score: {scored['score']}/100, Latency: {job_latency:.0f}ms")

                    return results

                # Execute batches in parallel (max 3 concurrent batches)
                batch_results, phase1_metrics = await parallel_executor.execute(
                    batches=batches,
                    process_func=score_batch,
                    max_concurrent=3,
                    operation="quick_score_job",
                    return_metrics=True
                )

                # Flatten results
                scored_jobs = []
                for batch_result in batch_results:
                    if batch_result:
                        scored_jobs.extend(batch_result)

            else:
                # ═══════════════════════════════════════════════════════════════
                # BASELINE: Sequential execution (detect opportunities)
                # ═══════════════════════════════════════════════════════════════
                logger.info(f"📊 BASELINE: Sequential scoring of {len(jobs)} jobs")
                scored_jobs = []
                for i, job in enumerate(jobs):
                    logger.debug(f"Quick scoring job {i+1}/{len(jobs)}: {job.get('title', 'Unknown')}")

                    job_start_time = time.time()
                    scored = await self._quick_score_job_match(resume_text, job)
                    scored_jobs.append(scored)

                    job_latency = (time.time() - job_start_time) * 1000
                    logger.debug(f"  └─ Score: {scored['score']}/100, Latency: {job_latency:.0f}ms")

            quick_score_duration = time.time() - quick_score_start

            if CURRENT_PHASE == "optimized":
                logger.info(f"Phase 1 complete: {len(scored_jobs)} jobs scored in {quick_score_duration:.1f}s (batch+parallel)")
                if phase1_metrics and phase1_metrics.time_saved_ms > 0:
                    logger.info(f"   ⚡ Time saved: {phase1_metrics.time_saved_ms:.0f}ms ({phase1_metrics.estimated_sequential_ms:.0f}ms seq → {phase1_metrics.total_elapsed_ms:.0f}ms parallel)")
            else:
                logger.info(f"Phase 1 complete: {len(scored_jobs)} jobs scored in {quick_score_duration:.1f}s (sequential)")

            # ═══════════════════════════════════════════════════════════════
            # WORKFLOW-LEVEL OPTIMIZATION DETECTION
            # ═══════════════════════════════════════════════════════════════
            if len(jobs) >= 2:
                # Batch opportunity detection
                batch_opportunity = batch_detector.analyze_workflow(
                    operation="quick_score_job",
                    call_count=len(jobs),
                    total_duration_ms=quick_score_duration * 1000,
                    metadata={
                        "resume_id": resume_id,
                        "phase": "quick_score",
                        "jobs_processed": len(jobs),
                    }
                )
                
                if batch_opportunity:
                    logger.info(f"💡 BATCH OPPORTUNITY DETECTED:")
                    logger.info(f"   {len(jobs)} sequential calls → ~{(len(jobs)+2)//3} batches recommended")
                    logger.info(f"   Estimated savings: 60% cost reduction")
                
                # Parallel opportunity detection
                parallel_opportunity = parallel_detector.analyze_workflow(
                    operation="quick_score_job",
                    call_count=len(jobs),
                    total_duration_ms=quick_score_duration * 1000,
                    are_independent=True,  # These jobs don't depend on each other
                    metadata={
                        "resume_id": resume_id,
                        "phase": "quick_score",
                    }
                )
                
                if parallel_opportunity:
                    logger.info(f"⚡ PARALLEL OPPORTUNITY DETECTED:")
                    logger.info(f"   {quick_score_duration:.1f}s sequential → ~{quick_score_duration/3:.1f}s with 3 concurrent")
                    logger.info(f"   Estimated time savings: 66%")
                
            
            # Sort by score
            scored_jobs.sort(key=lambda x: x['score'], reverse=True)
            top_jobs = scored_jobs[:3]  # Top 3 for deep analysis
            
            logger.info(f"Phase 2: Deep analysis on top {len(top_jobs)} jobs...")
            deep_analysis_start = time.time()

            # PHASE 2: Deep analysis
            # BASELINE: Sequential execution (detect opportunities)
            # OPTIMIZED: Parallel execution (apply optimization)
            phase2_metrics = None

            if CURRENT_PHASE == "optimized":
                # ═══════════════════════════════════════════════════════════════
                # OPTIMIZED: Parallel execution
                # ═══════════════════════════════════════════════════════════════
                logger.info(f"⚡ OPTIMIZED: Parallel analysis of {len(top_jobs)} jobs")

                async def analyze_job(batch, batch_idx):
                    """Deep analyze a single job from batch."""
                    job_data = batch[0]  # Each batch contains one job
                    logger.debug(f"Deep analysis {batch_idx + 1}/{len(top_jobs)}: {job_data['title']}")

                    job_start_time = time.time()
                    full_job = self.db.get_job_by_id(job_data['job_id'])
                    detailed = await self._deep_analyze_job_match(resume_text, full_job, job_data['score'])

                    job_latency = (time.time() - job_start_time) * 1000
                    logger.debug(f"  └─ Latency: {job_latency:.0f}ms")

                    return detailed

                # Each job is its own "batch" of size 1
                job_batches = [[job] for job in top_jobs]

                analysis_results, phase2_metrics = await parallel_executor.execute(
                    batches=job_batches,
                    process_func=analyze_job,
                    max_concurrent=3,
                    operation="deep_analyze_job",
                    return_metrics=True
                )

                detailed_matches = [result for result in analysis_results if result]

            else:
                # ═══════════════════════════════════════════════════════════════
                # BASELINE: Sequential execution (detect opportunities)
                # ═══════════════════════════════════════════════════════════════
                logger.info(f"📊 BASELINE: Sequential analysis of {len(top_jobs)} jobs")
                detailed_matches = []
                for i, job_data in enumerate(top_jobs):
                    logger.debug(f"Deep analysis {i + 1}/{len(top_jobs)}: {job_data['title']}")

                    job_start_time = time.time()
                    full_job = self.db.get_job_by_id(job_data['job_id'])
                    detailed = await self._deep_analyze_job_match(resume_text, full_job, job_data['score'])
                    detailed_matches.append(detailed)

                    job_latency = (time.time() - job_start_time) * 1000
                    logger.debug(f"  └─ Latency: {job_latency:.0f}ms")

            deep_analysis_duration = time.time() - deep_analysis_start

            if CURRENT_PHASE == "optimized":
                logger.info(f"Phase 2 complete: {len(detailed_matches)} jobs analyzed in {deep_analysis_duration:.1f}s (parallel)")
                if phase2_metrics and phase2_metrics.time_saved_ms > 0:
                    logger.info(f"   ⚡ Time saved: {phase2_metrics.time_saved_ms:.0f}ms ({phase2_metrics.estimated_sequential_ms:.0f}ms seq → {phase2_metrics.total_elapsed_ms:.0f}ms parallel)")
            else:
                logger.info(f"Phase 2 complete: {len(detailed_matches)} jobs analyzed in {deep_analysis_duration:.1f}s (sequential)")
            
            # Save to database
            for match in detailed_matches:
                self.db.save_match(
                    resume_id=resume_id,
                    job_id=match['job_id'],
                    score=match['score'],
                    reason=match['reason'],
                    detailed_analysis=match.get('detailed_analysis')
                )
            
            # Calculate totals
            total_duration = time.time() - workflow_start_time
            logger.info(f"✅ Matching complete: {total_duration:.1f}s total ({quick_score_duration:.1f}s quick + {deep_analysis_duration:.1f}s deep)")
            
            # Format response
            response = f"✅ **Matching Complete!**\n\n"
            response += f"📊 **Summary:**\n"
            response += f"- Resume: {resume_name}\n"
            response += f"- Jobs Analyzed: {len(jobs)}\n"
            response += f"- Time Taken: {total_duration:.1f}s\n\n"
            response += f"🎯 **Top Matches:**\n\n"
            
            for i, match in enumerate(detailed_matches, 1):
                response += f"{i}. **{match['title']}** at {match['company']}\n"
                response += f"   Score: {match['score']}/100\n"
                if isinstance(match.get('reason'), list):
                    response += f"   Key Points:\n"
                    for reason in match['reason'][:2]:
                        response += f"   • {reason}\n"
                response += "\n"
            
            return response
            
        except Exception as e:
            logger.error(f"Error in matching workflow: {e}", exc_info=True)
            raise

    @kernel_function(
        name="explain_recent_match",
        description=(
            "Explains a specific match from recent matching results without re-running analysis. "
            "Use when user asks: 'why did I get X%?', 'tell me about match #2', 'why was I a match?'. "
            "DO NOT re-run matching - just retrieve and explain the stored results."
        )
    )
    async def explain_recent_match(
        self,
        match_number: Annotated[int, "Which match to explain (1 for first, 2 for second, etc.)"] = 1
    ) -> Annotated[str, "Detailed explanation of the specified match"]:
        """
        Retrieves and explains a specific match from memory without re-running analysis.
        """
        start_time = time.time()
        
        if not self.memory:
            track_llm_call(
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=(time.time() - start_time) * 1000,
                agent_name="ResumeMatching",
                agent_role="retriever",
                operation="explain_recent_match",
                success=False,
                error="No memory available",
                retry_count=0,
                prompt=f"Explain match #{match_number}",
                conversation_id=self.memory.conversation_id if self.memory else None,
                turn_number=self.memory.turn_number if self.memory else None,
                parent_call_id=self.memory.request_id if self.memory else None,
                request_id=str(uuid.uuid4()),
                trace_id=self.memory.conversation_id if self.memory else None,
                environment=os.getenv("ENVIRONMENT", "development"),
                metadata={
                    "phase": CURRENT_PHASE,
                    "match_number": match_number,
                }
            )
            return "❌ No memory available to retrieve match results."
        
        recent_matches = self.memory.get_recent_matches(limit=10)
        
        if not recent_matches:
            track_llm_call(
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=(time.time() - start_time) * 1000,
                agent_name="ResumeMatching",
                agent_role="retriever",
                operation="explain_recent_match",
                success=False,
                error="No recent matches found",
                retry_count=0,
                prompt=f"Explain match #{match_number}",
                conversation_id=self.memory.conversation_id if self.memory else None,
                turn_number=self.memory.turn_number if self.memory else None,
                parent_call_id=self.memory.request_id if self.memory else None,
                request_id=str(uuid.uuid4()),
                trace_id=self.memory.conversation_id if self.memory else None,
                environment=os.getenv("ENVIRONMENT", "development"),
                metadata={
                    "phase": CURRENT_PHASE,
                    "match_number": match_number,
                }
            )
            return "❌ No recent match results found. Please run 'match my resume' first."
        
        if match_number < 1 or match_number > len(recent_matches):
            track_llm_call(
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=(time.time() - start_time) * 1000,
                agent_name="ResumeMatching",
                agent_role="retriever",
                operation="explain_recent_match",
                success=False,
                error=f"Invalid match number: {match_number}",
                retry_count=0,
                prompt=f"Explain match #{match_number}",
                conversation_id=self.memory.conversation_id if self.memory else None,
                turn_number=self.memory.turn_number if self.memory else None,
                parent_call_id=self.memory.request_id if self.memory else None,
                request_id=str(uuid.uuid4()),
                trace_id=self.memory.conversation_id if self.memory else None,
                environment=os.getenv("ENVIRONMENT", "development"),
                metadata={
                    "phase": CURRENT_PHASE,
                    "match_number": match_number,
                    "available_matches": len(recent_matches),
                }
            )
            return f"❌ Invalid match number. Please choose between 1 and {len(recent_matches)}."
        
        match = recent_matches[match_number - 1]
        
        response = f"## Match #{match_number}: {match.get('title', 'Unknown')} at {match.get('company', 'Unknown')}\n\n"
        response += f"**Score:** {match.get('score', 'N/A')}/100\n\n"
        response += f"**📍 Location:** {match.get('location', 'Unknown')}\n\n"
        response += f"**💡 Why This Match:**\n{match.get('reason', 'No explanation available')}\n\n"
        
        if match.get('matched_skills'):
            response += f"**✅ Your Matching Skills:**\n"
            for skill in match['matched_skills']:
                response += f"  • {skill}\n"
            response += "\n"
        
        if match.get('missing_skills'):
            response += f"**⚠️  Skills You're Missing:**\n"
            for skill in match['missing_skills']:
                response += f"  • {skill}\n"
            response += "\n"
        
        if match.get('key_strengths'):
            response += f"**💪 Your Key Strengths for This Role:**\n"
            for strength in match['key_strengths']:
                response += f"  • {strength}\n"
            response += "\n"
        
        if match.get('recommendation'):
            response += f"**📋 Recommendation:**\n{match['recommendation']}\n\n"
        
        response += f"**🔗 Apply:** {match.get('link', 'No link available')}"
        
        if self.memory:
            self.memory.set_current_focus(job_id=match.get('job_id'))
        
        # Track successful retrieval
        track_llm_call(
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=(time.time() - start_time) * 1000,
            agent_name="ResumeMatching",
            agent_role="retriever",
            operation="explain_recent_match",
            success=True,
            prompt=f"Explain match #{match_number}",
            response_text=response[:500],
            conversation_id=self.memory.conversation_id if self.memory else None,
            turn_number=self.memory.turn_number if self.memory else None,
            parent_call_id=self.memory.request_id if self.memory else None,
            request_id=str(uuid.uuid4()),
            trace_id=self.memory.conversation_id if self.memory else None,
            environment=os.getenv("ENVIRONMENT", "development"),
            metadata={
                "phase": CURRENT_PHASE,
                "match_number": match_number,
                "job_id": match.get('job_id'),
                "job_title": match.get('title'),
                "score": match.get('score'),
            }
        )
        
        return response
    
    @kernel_function(
        name="show_saved_matches",
        description=(
            "Shows previously saved match results from the database WITHOUT re-running matching. "
            "Use when user asks: 'show me my matches', 'what are my top matches', 'show my match results'. "
            "DO NOT use find_best_job_matches - that re-runs expensive analysis. This just retrieves saved data."
        )
    )
    async def show_saved_matches(
        self,
        resume_id: Annotated[str, "The resume ID to show matches for. Use 'most_recent' for latest."] = "most_recent",
        limit: Annotated[int, "Number of matches to show (default 5)"] = 5
    ) -> Annotated[str, "List of saved matches with scores"]:
        """
        Retrieves saved match results from the database without re-running analysis.
        """
        start_time = time.time()
        
        if resume_id == "most_recent":
            resume = self.db.get_most_recent_resume()
            if not resume:
                track_llm_call(
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=(time.time() - start_time) * 1000,
                    agent_name="ResumeMatching",
                    agent_role="retriever",
                    operation="show_saved_matches",
                    success=False,
                    error="No resumes found",
                    error_type="NotFoundError",
                    retry_count=0,
                    conversation_id=self.memory.conversation_id if self.memory else None,
                    turn_number=self.memory.turn_number if self.memory else None,
                    parent_call_id=self.memory.request_id if self.memory else None,
                    request_id=str(uuid.uuid4()),
                    trace_id=self.memory.conversation_id if self.memory else None,
                    environment=os.getenv("ENVIRONMENT", "development"),
                    metadata={
                        "phase": CURRENT_PHASE,
                        "resume_id": resume_id,
                    }
                )
                return "❌ No resumes found. Please upload a resume first."
            resume_id = str(resume['id'])
            resume_name = resume['name']
        else:
            resume = self.db.get_resume_by_id(resume_id)
            if not resume:
                track_llm_call(
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=(time.time() - start_time) * 1000,
                    agent_name="ResumeMatching",
                    agent_role="retriever",
                    operation="show_saved_matches",
                    success=False,
                    error=f"Resume not found: {resume_id}",
                    error_type="NotFoundError",
                    retry_count=0,
                    conversation_id=self.memory.conversation_id if self.memory else None,
                    turn_number=self.memory.turn_number if self.memory else None,
                    parent_call_id=self.memory.request_id if self.memory else None,
                    request_id=str(uuid.uuid4()),
                    trace_id=self.memory.conversation_id if self.memory else None,
                    environment=os.getenv("ENVIRONMENT", "development"),
                    metadata={
                        "phase": CURRENT_PHASE,
                        "resume_id": resume_id,
                    }
                )
                return f"❌ Resume with ID {resume_id} not found."
            resume_name = resume['name']
        
        import sqlite3
        from services.db import DB_PATH
        
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            query = """
                SELECT 
                    jm.score, jm.reason, jm.detailed_analysis,
                    j.id, j.title, j.company, j.location, j.link
                FROM resume_job_matches jm
                JOIN jobs j ON jm.job_id = j.id
                WHERE jm.resume_id = ?
                ORDER BY jm.score DESC
                LIMIT ?
            """
            
            cursor.execute(query, (int(resume_id), limit))
            matches = cursor.fetchall()
            conn.close()
            
            if not matches:
                track_llm_call(
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=(time.time() - start_time) * 1000,
                    agent_name="ResumeMatching",
                    agent_role="retriever",
                    operation="show_saved_matches",
                    success=True,
                    prompt=f"Show matches for resume_id={resume_id}",
                    response_text="No saved matches found",
                    conversation_id=self.memory.conversation_id if self.memory else None,
                    turn_number=self.memory.turn_number if self.memory else None,
                    parent_call_id=self.memory.request_id if self.memory else None,
                    request_id=str(uuid.uuid4()),
                    trace_id=self.memory.conversation_id if self.memory else None,
                    environment=os.getenv("ENVIRONMENT", "development"),
                    metadata={
                        "phase": CURRENT_PHASE,
                        "resume_id": resume_id,
                        "resume_name": resume_name,
                        "result_count": 0,
                    }
                )
                return f"❌ No saved matches found for '{resume_name}'.\n\nRun matching first: 'match my resume'"
            
            # Store in memory
            if self.memory:
                self.memory.set_current_focus(resume_id=int(resume_id))
                for match in matches:
                    score, reason, detailed_json, job_id, title, company, location, link = match
                    match_data = {
                        'job_id': job_id, 'title': title, 'company': company,
                        'location': location, 'link': link, 'score': score, 'reason': reason
                    }
                    
                    if detailed_json:
                        try:
                            detailed = json.loads(detailed_json)
                            match_data.update({
                                'matched_skills': detailed.get('matched_skills', []),
                                'missing_skills': detailed.get('missing_skills', []),
                                'key_strengths': detailed.get('strengths', []),
                                'recommendation': detailed.get('improvement_suggestions', [])
                            })
                        except:
                            pass
                    
                    self.memory.add_match_result(match_data)
            
            # Format response
            response = f"🎯 Top {len(matches)} Matches for '{resume_name}':\n\n"
            
            for i, match in enumerate(matches, 1):
                score, reason, _, job_id, title, company, location, link = match
                response += f"{i}. **{title}** at {company} - {score}% match\n"
                response += f"   📍 {location}\n"
                response += f"   🔗 {link}\n\n"
            
            response += f"\n💬 Try: 'explain match #1' or 'tell me about match #{len(matches)}'"
            
            # Track successful retrieval
            track_llm_call(
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=(time.time() - start_time) * 1000,
                agent_name="ResumeMatching",
                agent_role="retriever",
                operation="show_saved_matches",
                success=True,
                prompt=f"Show matches for resume_id={resume_id}",
                response_text=response[:500],
                conversation_id=self.memory.conversation_id if self.memory else None,
                turn_number=self.memory.turn_number if self.memory else None,
                parent_call_id=self.memory.request_id if self.memory else None,
                request_id=str(uuid.uuid4()),
                trace_id=self.memory.conversation_id if self.memory else None,
                environment=os.getenv("ENVIRONMENT", "development"),
                metadata={
                    "phase": CURRENT_PHASE,
                    "resume_id": resume_id,
                    "resume_name": resume_name,
                    "result_count": len(matches),
                    "is_db_read": True,
                }
            )
            
            return response
            
        except Exception as e:
            error_info = classify_error(e, operation="show_saved_matches")
            track_llm_call(
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=(time.time() - start_time) * 1000,
                agent_name="ResumeMatching",
                agent_role="retriever",
                operation="show_saved_matches",
                success=False,
                error=str(e),
                error_type=error_info['error_type'],
                error_code=error_info['error_code'],
                retry_count=0,
                prompt=f"Show matches for resume_id={resume_id}",
                conversation_id=self.memory.conversation_id if self.memory else None,
                turn_number=self.memory.turn_number if self.memory else None,
                parent_call_id=self.memory.request_id if self.memory else None,
                request_id=str(uuid.uuid4()),
                trace_id=self.memory.conversation_id if self.memory else None,
                environment=os.getenv("ENVIRONMENT", "development"),
                metadata={
                    "phase": CURRENT_PHASE,
                    "resume_id": resume_id,
                }
            )
            return f"❌ Error retrieving matches: {str(e)}"


    @kernel_function(
        name="find_best_job_matches",
        description="DEPRECATED: Use the new multi-step matching flow instead. This function is expensive and should only be called if explicitly requested."
    )
    async def find_best_job_matches(
        self,
        resume_id: Annotated[str, "The ID of the resume to match against jobs"],
        top_n: Annotated[int, "Number of top matches to return (default 5)"] = 5,
        save_to_db: Annotated[bool, "Whether to save match results to database"] = True
    ) -> Annotated[str, "A formatted summary of the top matching jobs with scores and reasons"]:
        """
        DEPRECATED: Direct matching function. Use the conversational flow instead.
        """
        resume = self.db.get_resume_by_id(resume_id)
        if not resume:
            return f"❌ Error: Resume with ID {resume_id} not found. Use 'list resumes' to see available resumes."
        
        jobs = self.db.get_all_jobs()
        if not jobs:
            return "❌ No jobs found in the database. Please add some jobs first."
        
        # Convert to int list for filtering
        job_ids = [job['id'] for job in jobs]
        
        return await self._execute_filtered_matching(int(resume_id), job_ids)
    
    @kernel_function(
        name="match_most_recent_resume",
        description="DEPRECATED: Use list_resumes and select_resume_for_matching for better control."
    )
    async def match_most_recent_resume(
        self,
        top_n: Annotated[int, "Number of top matches to return (default 5)"] = 5
    ) -> Annotated[str, "Top job matches for the most recent resume"]:
        """
        DEPRECATED: Convenience function to match the most recent resume.
        """
        resume = self.db.get_most_recent_resume()
        if not resume:
            return "❌ No resumes found. Please upload a resume first."
        
        resume_id = str(resume['id'])
        return await self.find_best_job_matches(resume_id=resume_id, top_n=top_n)
    
    # =========================================================================
    # _quick_score_job_match - WITH FULL 10-STEP OPTIMIZATION PATTERN
    # =========================================================================
    async def _quick_score_job_match(self, resume_text: str, job: dict) -> dict:
        """
        Quick scoring method - provides a fast initial score for all jobs.
        Implements full 10-step optimization pattern.
        """
        # Build prompt with separate system and user components for tracking
        system_prompt = """You are an expert resume matcher. Score how well this resume matches the job.

Analyze:
1. Skills alignment - Does the candidate have the required technical skills?
2. Experience level - Does years/level of experience match requirements?
3. Role responsibilities - Do past roles align with job duties?
4. Education/certifications - Does background meet requirements?

Provide:
- Overall score (0-100, be discriminating - use full range)
- Score breakdown for each category
- 2-4 concise bullet points explaining the match quality (what aligns, what's missing, key strengths/gaps)

CRITICAL: Return ONLY valid JSON. No markdown, no code blocks, just raw JSON.

JSON format:
{
  "score": 85,
  "confidence": 0.75,
  "confidence_reasoning": "High confidence on skills match due to explicit mentions, but uncertain about exact experience level and education details",
  "uncertainty_factors": [
    "Resume doesn't specify total years of experience",
    "Master's degree requirement unclear from resume"
  ],
  "score_breakdown": {
    "skills_match": 90,
    "experience_match": 85,
    "requirements_match": 80,
    "education_match": 85
  },
  "reason_bullets": [
    "Strong technical skills match with 5 years Python and AWS experience",
    "Background in AI consulting aligns with role responsibilities",
    "Missing advanced ML frameworks (TensorFlow, PyTorch) mentioned in posting",
    "Master's degree requirement not clearly met"
  ]
}"""

        user_message = f"""Resume:
{resume_text[:2000]}

Job:
Title: {job.get('title', 'N/A')}
Company: {job.get('company', 'N/A')}
Description: {job.get('description', 'N/A')[:1500]}"""

        try:
            logger.debug(f"Quick scoring job: {job.get('title', 'Unknown')}")
            
            # ═══════════════════════════════════════════════════════════════
            # INITIALIZE ALL VARIABLES (prevents undefined variable errors)
            # ═══════════════════════════════════════════════════════════════
            cache_meta = None
            routing_meta = None
            prompt_meta = None
            quality_eval = None
            streaming_candidate = False
            prompt_breakdown = None
            result_str = None
            latency_ms = 0
            prompt_tokens = 0
            completion_tokens = 0
            optimized_prompt = system_prompt  # Default to system_prompt
            max_tokens_limit = 1500  # Default - increased for full JSON response
            routed_model = DEFAULT_MODEL  # Default
            
            # ═══════════════════════════════════════════════════════════════
            # STEP 1: Check persistent cache first (survives restarts), then in-memory
            # ═══════════════════════════════════════════════════════════════
            operation = "quick_score_job"
            cache_key_data = {
                "job_id": job.get('id'),
                "resume_text_hash": hashlib.md5(resume_text[:2000].encode()).hexdigest()
            }

            # Try persistent cache first (SQLite - survives restarts)
            cached_result, cache_meta = persistent_cache.get(
                operation=operation,
                key_data=cache_key_data
            )
            cache_source = "persistent"

            # Fall back to in-memory cache if no persistent hit
            if not cached_result:
                cached_result, cache_meta = cache.get(
                    operation=operation,
                    key_data=cache_key_data
                )
                cache_source = "memory"

            # Ensure cache_meta is None if empty dict
            if isinstance(cache_meta, dict) and not cache_meta:
                cache_meta = None

            if cached_result:
                logger.debug(f"✅ Cache hit ({cache_source}) for job {job.get('id')}")
                result_str = cached_result
                latency_ms = 1.0
                prompt_tokens = 0
                completion_tokens = 0
                
                # Track cache hit
                track_llm_call(
                    operation=operation,
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=1.0,
                    success=True,
                    response_text=result_str,
                    cache_metadata=cache_meta,
                    agent_name="ResumeMatching",
                    agent_role="analyst",
                    conversation_id=self.memory.conversation_id if self.memory else None,
                    turn_number=self.memory.turn_number if self.memory else None,
                    parent_call_id=self.memory.request_id if self.memory else None,
                    request_id=str(uuid.uuid4()),
                    trace_id=self.memory.conversation_id if self.memory else None,
                    environment=os.getenv("ENVIRONMENT", "development"),
                    metadata={
                        "phase": CURRENT_PHASE,
                        "cache_hit": True,
                        "cache_source": cache_source,
                        "job_id": job.get('id'),
                        "job_title": job.get('title', 'Unknown'),
                    }
                )

                # Skip to result parsing
            
            else:
                # If no cache hit, proceed with LLM call
                if not cached_result:
                    # ═══════════════════════════════════════════════════════════════
                    # STEP 3: Get optimized prompt and max_tokens
                    # ═══════════════════════════════════════════════════════════════
                    optimized_prompt, max_tokens_limit, prompt_meta = prompt_optimizer.get_optimized_prompt(
                        operation=operation,
                        default_prompt=system_prompt
                    )
                    # Passthrough: if optimizer returns None, use the original system_prompt
                    if optimized_prompt is None:
                        optimized_prompt = system_prompt
                    if max_tokens_limit is None:
                        max_tokens_limit = 1500  # Default for quick scoring

                    # ═══════════════════════════════════════════════════════════════
                    # STEP 4: Get routed model
                    # ═══════════════════════════════════════════════════════════════
                    routed_model, routing_meta = router.select(
                        operation=operation,
                        prompt=optimized_prompt + user_message,
                        estimated_tokens=estimate_tokens(optimized_prompt + user_message),
                        complexity=0.4  # Quick scoring is lower complexity
                    )
                    
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
                    # Redefine full_prompt with optimized prompt for caching
                    full_prompt = f"{optimized_prompt}\n\n{user_message}"

                    # Make LLM call using chat completion
                    from semantic_kernel.contents import ChatHistory
                    from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
                        AzureChatPromptExecutionSettings,
                    )

                    quick_score_history = ChatHistory()
                    quick_score_history.add_system_message(optimized_prompt)
                    quick_score_history.add_user_message(user_message)

                    # DEBUG LOGGING: See what's being sent to the LLM
                    logger.info(f"[QUICK_SCORE] === LLM REQUEST for job: {job.get('title', 'Unknown')} ===")
                    logger.info(f"[QUICK_SCORE] System prompt (first 300 chars):\n{optimized_prompt[:300]}...")
                    logger.info(f"[QUICK_SCORE] User message (first 200 chars):\n{user_message[:200]}...")
                    logger.info(f"[QUICK_SCORE] History has {len(quick_score_history.messages)} messages:")
                    for i, msg in enumerate(quick_score_history.messages):
                        role = msg.role.value if hasattr(msg.role, 'value') else str(msg.role)
                        content_preview = str(msg.content)[:150].replace('\n', ' ')
                        logger.info(f"[QUICK_SCORE]   [{i}] {role}: {content_preview}...")

                    # Create ISOLATED execution settings WITHOUT function calling
                    # This prevents the kernel's global system prompt from interfering
                    quick_score_settings = AzureChatPromptExecutionSettings()
                    quick_score_settings.max_tokens = 1500
                    quick_score_settings.temperature = 0.3
                    # NO function_choice_behavior - we want pure completion, not tool calling

                    llm_start_time = time.time()
                    result = await self.chat_completion.get_chat_message_content(
                        chat_history=quick_score_history,
                        settings=quick_score_settings,  # Use isolated settings WITHOUT function calling
                        # NOTE: NOT passing kernel to prevent function calling from triggering
                    )
                    latency_ms = (time.time() - llm_start_time) * 1000

                    result_str = str(result)

                    # Extract Azure cache metrics (stable_prefix category)
                    azure_cache_metrics = extract_azure_cache_metrics(result, optimized_prompt)
                    if azure_cache_metrics.get("cached_prompt_tokens", 0) > 0:
                        logger.info(f"[QUICK_SCORE] ✅ Azure cache hit! {azure_cache_metrics['cached_prompt_tokens']:,} tokens cached")

                    # DEBUG LOGGING: See what we got back
                    logger.info(f"[QUICK_SCORE] === LLM RESPONSE ===")
                    logger.info(f"[QUICK_SCORE] Raw response (first 500 chars):\n{result_str[:500]}")

                    # Extract token usage from metadata
                    if hasattr(result, 'metadata') and result.metadata:
                        usage = result.metadata.get('usage')
                        if usage:
                            if hasattr(usage, 'prompt_tokens'):
                                prompt_tokens = usage.prompt_tokens or 0
                                completion_tokens = usage.completion_tokens or 0
                            elif isinstance(usage, dict):
                                prompt_tokens = usage.get('prompt_tokens', 0)
                                completion_tokens = usage.get('completion_tokens', 0)

                    # Fallback to estimation if not available
                    if not prompt_tokens:
                        prompt_tokens = estimate_tokens(optimized_prompt + user_message)
                    if not completion_tokens:
                        completion_tokens = estimate_tokens(result_str)

                    # ═══════════════════════════════════════════════════════════════
                    # STEP 7: Detect streaming candidates
                    # ═══════════════════════════════════════════════════════════════
                    streaming_candidate = streaming_detector.check_call(
                        operation=operation,
                        latency_ms=latency_ms,
                        completion_tokens=completion_tokens,
                    )

                    # ═══════════════════════════════════════════════════════════════
                    # STEP 8: Cache the response (both persistent and in-memory)
                    # ═══════════════════════════════════════════════════════════════
                    # Store in persistent cache (SQLite - survives restarts)
                    persistent_cache.set(
                        operation=operation,
                        key_data=cache_key_data,
                        value=result_str
                    )
                    # Also store in memory cache (faster for same-session lookups)
                    cache.set(
                        operation=operation,
                        key_data=cache_key_data,
                        value=result_str
                    )

                    # ═══════════════════════════════════════════════════════════════
                    # STEP 9: Create prompt breakdown and evaluate quality (quick_score)
                    # ═══════════════════════════════════════════════════════════════
                    prompt_breakdown = create_prompt_breakdown(
                        system_prompt=optimized_prompt,
                        system_prompt_tokens=estimate_tokens(optimized_prompt),
                        user_message=user_message,
                        user_message_tokens=estimate_tokens(user_message),
                    )

                    # LLM Judge evaluation (fire-and-forget - doesn't block response)
                    fire_and_forget_judge(
                        operation=operation,
                        prompt=full_prompt[:5000],
                        response=result_str[:5000],
                        llm_client=self.kernel,
                        conversation_id=self.memory.conversation_id if self.memory else None,
                        turn_number=self.memory.turn_number if self.memory else None,
                    )
                    quality_eval = None  # Judge runs in background

                    # ═══════════════════════════════════════════════════════════════
                    # STEP 10: Track with Observatory (CRITICAL - INCLUDE PHASE)
                    # ═══════════════════════════════════════════════════════════════
                    track_llm_call(
                        # Core metrics
                        model_name=routed_model,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        latency_ms=latency_ms,
                        agent_name="ResumeMatching",
                        agent_role="analyst",
                        operation=operation,
                        success=True,
                        
                        # Prompt content
                        system_prompt=optimized_prompt,
                        user_message=user_message,
                        response_text=result_str,
                        prompt_breakdown=prompt_breakdown,
                        
                        # Optimization tracking
                        routing_decision=routing_meta,
                        cache_metadata=None, 
                        quality_evaluation=quality_eval,
                        prompt_metadata=None,
                        
                        # Model configuration
                        temperature=0.3,  # Factual scoring
                        max_tokens=max_tokens_limit,
                        
                        # Token breakdown
                        system_prompt_tokens=estimate_tokens(optimized_prompt),
                        user_message_tokens=estimate_tokens(user_message),
                        
                        # Conversation linking
                        conversation_id=self.memory.conversation_id if self.memory else None,
                        turn_number=self.memory.turn_number if self.memory else None,
                        parent_call_id=self.memory.request_id if self.memory else None,
                        request_id=str(uuid.uuid4()),
                        
                        # Observability
                        trace_id=self.memory.conversation_id if self.memory else None,
                        environment=os.getenv("ENVIRONMENT", "development"),

                        # Azure prompt cache metrics (stable_prefix category)
                        **azure_cache_metrics,

                        # Metadata - CRITICAL: Include phase
                        metadata={
                            "phase": CURRENT_PHASE,
                            "job_id": job.get('id'),
                            "job_title": job.get('title', 'Unknown'),
                            "judged": quality_eval is not None,
                            "streaming_candidate": bool(streaming_candidate),
                        }
                    )

                    logger.debug(f"Quick score LLM call: {latency_ms:.0f}ms, {prompt_tokens + completion_tokens} tokens")

                    # ═══════════════════════════════════════════════════════════════
                    # STEP 11: Track batch detection (individual call)
                    # ═══════════════════════════════════════════════════════════════
                    # Generate unique call ID for linking
                    call_id = str(uuid.uuid4())

                    batch_detector.track_call(
                        operation=operation,
                        call_id=call_id,
                        latency_ms=latency_ms,
                        agent_name="ResumeMatching",
                    )
                    
                    # Check token efficiency (should be low prompt/completion ratio for scoring)
                    if completion_tokens > 0:
                        token_efficiency_detector.check_call(
                            operation=operation,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            call_id=call_id,
                            agent_name="ResumeMatching",
                        )
            
            # ═══════════════════════════════════════════════════════════════
            # RESULT PARSING (common path for all branches above)
            # ═══════════════════════════════════════════════════════════════
            
            # Parse response
            if '```json' in result_str:
                result_str = result_str.split('```json')[1].split('```')[0].strip()
            elif '```' in result_str:
                result_str = result_str.split('```')[1].split('```')[0].strip()
            
            start_idx = result_str.find('{')
            end_idx = result_str.rfind('}')
            if start_idx != -1 and end_idx != -1:
                result_str = result_str[start_idx:end_idx+1]
            
            match_data = json.loads(result_str)
            
            return {
                'job_id': job.get('id'),
                'title': job.get('title', 'Unknown Title'),
                'company': job.get('company', 'Unknown Company'),
                'location': job.get('location', 'Unknown Location'),
                'link': job.get('link', ''),
                'score': int(match_data.get('score', 0)),
                'confidence': float(match_data.get('confidence', 0.5)),
                'confidence_reasoning': match_data.get('confidence_reasoning', ''),
                'uncertainty_factors': match_data.get('uncertainty_factors', []),
                'score_breakdown': match_data.get('score_breakdown', {}),
                'reason': match_data.get('reason_bullets', ['No explanation provided.'])
            }
            
        except Exception as e:
            logger.error(f"Error in quick scoring for job {job.get('title', 'Unknown')}: {e}")
            
            # Classify error
            error_info = classify_error(e, operation="quick_score_job")
            
            # Track error
            track_llm_call(
                model_name=DEFAULT_MODEL,
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=0,
                agent_name="ResumeMatching",
                agent_role="analyst",
                operation="quick_score_job",
                success=False,
                error=str(e),
                error_type=error_info['error_type'],
                error_code=error_info['error_code'],
                retry_count=0,
                conversation_id=self.memory.conversation_id if self.memory else None,
                turn_number=self.memory.turn_number if self.memory else None,
                parent_call_id=self.memory.request_id if self.memory else None,
                request_id=str(uuid.uuid4()),
                trace_id=self.memory.conversation_id if self.memory else None,
                environment=os.getenv("ENVIRONMENT", "development"),
                metadata={
                    "phase": CURRENT_PHASE,
                    "job_id": job.get('id'),
                    "job_title": job.get('title', 'Unknown'),
                    "error_type": error_info['error_type'],
                }
            )
            
            return {
                'job_id': job.get('id'),
                'title': job.get('title', 'Unknown Title'),
                'company': job.get('company', 'Unknown Company'),
                'location': job.get('location', 'Unknown Location'),
                'link': job.get('link', ''),
                'score': 50,
                'confidence': 0.3,
                'reason': ["Error in scoring"]
            }
    
    # =========================================================================
    # _deep_analyze_job_match - WITH FULL 10-STEP OPTIMIZATION PATTERN
    # =========================================================================
    async def _deep_analyze_job_match(self, resume_text: str, job: dict, original_score: int) -> dict:
        """
        Deep analysis method - provides line-by-line semantic matching with exact text highlights.
        This is SLOWER and only used for top matches.
        Implements full 10-step optimization pattern.
        """
        system_prompt = """You are an expert resume matcher. Perform semantic analysis to find connections between job requirements and resume content.

🎨 CRITICAL INSTRUCTIONS FOR HIGHLIGHT TEXT:

YOU MUST COPY EXACT TEXT FROM THE DOCUMENTS. DO NOT WRITE SUMMARIES.

RULE: job_requirement and job_highlight_text must be IDENTICAL.
RULE: resume_bullet and resume_highlight_text must be IDENTICAL.

CRITICAL: Return ONLY valid JSON. No markdown, no code blocks.

Format:
{
  "overall_score": 85,
  "confidence": 0.82,
  "confidence_reasoning": "High confidence due to explicit skill matches, but some uncertainty about experience depth",
  "uncertainty_factors": [
    "Resume mentions cloud experience but doesn't specify years",
    "Job requires 'senior level' but resume doesn't state seniority explicitly"
  ],
  "score_breakdown": {
    "skills_match": 90,
    "experience_match": 80,
    "requirements_match": 85,
    "education_match": 75
  },
  "matched_bullets": [
    {
      "job_requirement": "...",
      "job_highlight_text": "...",
      "resume_bullet": "...",
      "resume_highlight_text": "...",
      "match_strength": "strong/moderate/weak",
      "explanation": "..."
    }
  ],
  "matched_skills": ["skill1", "skill2"],
  "missing_skills": ["skill3", "skill4"],
  "strengths": ["strength1", "strength2"],
  "gaps": ["gap1", "gap2"],
  "improvement_suggestions": ["tip 1", "tip 2"],
  "summary": "Overall assessment"
}"""

        user_message = f"""**RESUME:**
{resume_text[:4000]}

**JOB:**
Title: {job.get('title', 'N/A')}
Company: {job.get('company', 'N/A')}
{job.get('description', 'N/A')[:3500]}

Return 10 matched bullets with EXACT TEXT from both documents."""

        try:
            logger.debug(f"Deep analyzing job: {job.get('title', 'Unknown')}")
            
            # ═══════════════════════════════════════════════════════════════
            # INITIALIZE ALL VARIABLES (prevents undefined variable errors)
            # ═══════════════════════════════════════════════════════════════
            cache_meta = None
            routing_meta = None
            prompt_meta = None
            quality_eval = None
            streaming_candidate = False
            prompt_breakdown = None
            result_str = None
            latency_ms = 0
            prompt_tokens = 0
            completion_tokens = 0
            optimized_prompt = system_prompt  # Default
            max_tokens_limit = 2500  # Default - increased for detailed analysis JSON
            routed_model = DEFAULT_MODEL  # Default

            # ═══════════════════════════════════════════════════════════════
            # STEP 1: Check persistent cache first (survives restarts), then in-memory
            # ═══════════════════════════════════════════════════════════════
            operation = "deep_analyze_job"
            cache_key_data = {
                "job_id": job.get('id'),
                "resume_text_hash": hashlib.md5(resume_text[:4000].encode()).hexdigest()
            }

            # Try persistent cache first (SQLite - survives restarts)
            cached_result, cache_meta = persistent_cache.get(
                operation=operation,
                key_data=cache_key_data
            )
            cache_source = "persistent"

            # Fall back to in-memory cache if no persistent hit
            if not cached_result:
                cached_result, cache_meta = cache.get(
                    operation=operation,
                    key_data=cache_key_data
                )
                cache_source = "memory"

            # Ensure cache_meta is None if empty dict
            if isinstance(cache_meta, dict) and not cache_meta:
                cache_meta = None

            if cached_result:
                logger.debug(f"✅ Cache hit ({cache_source}) for deep analysis of job {job.get('id')}")
                result_str = cached_result
                latency_ms = 1.0
                prompt_tokens = 0
                completion_tokens = 0
                
                # Track cache hit
                track_llm_call(
                    operation=operation,
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=1.0,
                    success=True,
                    response_text=result_str,
                    cache_metadata=cache_meta,
                    agent_name="ResumeMatching",
                    agent_role="analyst",
                    conversation_id=self.memory.conversation_id if self.memory else None,
                    turn_number=self.memory.turn_number if self.memory else None,
                    parent_call_id=self.memory.request_id if self.memory else None,
                    request_id=str(uuid.uuid4()),
                    trace_id=self.memory.conversation_id if self.memory else None,
                    environment=os.getenv("ENVIRONMENT", "development"),
                    metadata={
                        "phase": CURRENT_PHASE,
                        "cache_hit": True,
                        "cache_source": cache_source,
                        "job_id": job.get('id'),
                        "job_title": job.get('title', 'Unknown'),
                        "original_score": original_score,
                    }
                )

                # Skip to result parsing

            else:
                # If no cache hit, proceed with LLM call
                if not cached_result:
                    # ═══════════════════════════════════════════════════════════════
                    # STEP 3: Get optimized prompt and max_tokens (deep_analyze)
                    # ═══════════════════════════════════════════════════════════════
                    optimized_prompt, max_tokens_limit, prompt_meta = prompt_optimizer.get_optimized_prompt(
                        operation=operation,
                        default_prompt=system_prompt
                    )
                    # Passthrough: if optimizer returns None, use the original system_prompt
                    if optimized_prompt is None:
                        optimized_prompt = system_prompt
                    if max_tokens_limit is None:
                        max_tokens_limit = 2500  # Default for deep analysis

                    # ═══════════════════════════════════════════════════════════════
                    # STEP 4: Get routed model
                    # ═══════════════════════════════════════════════════════════════
                    routed_model, routing_meta = router.select(
                        operation=operation,
                        prompt=optimized_prompt + user_message,
                        estimated_tokens=estimate_tokens(optimized_prompt + user_message),
                        complexity=0.7  # Deep analysis is higher complexity
                    )
                    
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
                    # Redefine full_prompt with optimized prompt for caching
                    full_prompt = f"{optimized_prompt}\n\n{user_message}"

                    # Make LLM call using chat completion
                    from semantic_kernel.contents import ChatHistory
                    from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
                        AzureChatPromptExecutionSettings,
                    )

                    deep_analysis_history = ChatHistory()
                    deep_analysis_history.add_system_message(optimized_prompt)
                    deep_analysis_history.add_user_message(user_message)

                    # DEBUG LOGGING: See what's being sent to the LLM
                    logger.info(f"[DEEP_ANALYSIS] === LLM REQUEST for job: {job.get('title', 'Unknown')} ===")
                    logger.info(f"[DEEP_ANALYSIS] System prompt (first 300 chars):\n{optimized_prompt[:300]}...")
                    logger.info(f"[DEEP_ANALYSIS] User message (first 200 chars):\n{user_message[:200]}...")
                    logger.info(f"[DEEP_ANALYSIS] History has {len(deep_analysis_history.messages)} messages:")
                    for i, msg in enumerate(deep_analysis_history.messages):
                        role = msg.role.value if hasattr(msg.role, 'value') else str(msg.role)
                        content_preview = str(msg.content)[:150].replace('\n', ' ')
                        logger.info(f"[DEEP_ANALYSIS]   [{i}] {role}: {content_preview}...")

                    # Create ISOLATED execution settings WITHOUT function calling
                    # This prevents the kernel's global system prompt from interfering
                    deep_analysis_settings = AzureChatPromptExecutionSettings()
                    deep_analysis_settings.max_tokens = 2500
                    deep_analysis_settings.temperature = 0.5
                    # NO function_choice_behavior - we want pure completion, not tool calling

                    llm_start_time = time.time()
                    result = await self.chat_completion.get_chat_message_content(
                        chat_history=deep_analysis_history,
                        settings=deep_analysis_settings,  # Use isolated settings WITHOUT function calling
                        # NOTE: NOT passing kernel to prevent function calling from triggering
                    )
                    latency_ms = (time.time() - llm_start_time) * 1000

                    result_str = str(result)

                    # Extract Azure cache metrics (stable_prefix category)
                    azure_cache_metrics = extract_azure_cache_metrics(result, optimized_prompt)
                    if azure_cache_metrics.get("cached_prompt_tokens", 0) > 0:
                        logger.info(f"[DEEP_ANALYSIS] ✅ Azure cache hit! {azure_cache_metrics['cached_prompt_tokens']:,} tokens cached")

                    # DEBUG LOGGING: See what we got back
                    logger.info(f"[DEEP_ANALYSIS] === LLM RESPONSE ===")
                    logger.info(f"[DEEP_ANALYSIS] Raw response (first 500 chars):\n{result_str[:500]}")

                    # Extract token usage from metadata
                    if hasattr(result, 'metadata') and result.metadata:
                        usage = result.metadata.get('usage')
                        if usage:
                            if hasattr(usage, 'prompt_tokens'):
                                prompt_tokens = usage.prompt_tokens or 0
                                completion_tokens = usage.completion_tokens or 0
                            elif isinstance(usage, dict):
                                prompt_tokens = usage.get('prompt_tokens', 0)
                                completion_tokens = usage.get('completion_tokens', 0)

                    # Fallback to estimation if not available
                    if not prompt_tokens:
                        prompt_tokens = estimate_tokens(optimized_prompt + user_message)
                    if not completion_tokens:
                        completion_tokens = estimate_tokens(result_str)

                    # ═══════════════════════════════════════════════════════════════
                    # STEP 7: Detect streaming candidates (deep_analyze)
                    # ═══════════════════════════════════════════════════════════════
                    streaming_candidate = streaming_detector.check_call(
                        operation=operation,
                        latency_ms=latency_ms,
                        completion_tokens=completion_tokens,
                    )

                    # ═══════════════════════════════════════════════════════════════
                    # STEP 8: Cache the response (both persistent and in-memory)
                    # ═══════════════════════════════════════════════════════════════
                    # Store in persistent cache (SQLite - survives restarts)
                    persistent_cache.set(
                        operation=operation,
                        key_data=cache_key_data,
                        value=result_str
                    )
                    # Also store in memory cache (faster for same-session lookups)
                    cache.set(
                        operation=operation,
                        key_data=cache_key_data,
                        value=result_str
                    )

                    # ═══════════════════════════════════════════════════════════════
                    # STEP 9: Create prompt breakdown and evaluate quality (deep_analyze)
                    # ═══════════════════════════════════════════════════════════════
                    prompt_breakdown = create_prompt_breakdown(
                        system_prompt=optimized_prompt,
                        system_prompt_tokens=estimate_tokens(optimized_prompt),
                        user_message=user_message,
                        user_message_tokens=estimate_tokens(user_message),
                    )

                    # LLM Judge evaluation (fire-and-forget - doesn't block response)
                    fire_and_forget_judge(
                        operation=operation,
                        prompt=full_prompt[:5000],
                        response=result_str[:5000],
                        llm_client=self.kernel,
                        conversation_id=self.memory.conversation_id if self.memory else None,
                        turn_number=self.memory.turn_number if self.memory else None,
                    )
                    quality_eval = None  # Judge runs in background

                    # ═══════════════════════════════════════════════════════════════
                    # STEP 10: Track with Observatory (CRITICAL - INCLUDE PHASE)
                    # ═══════════════════════════════════════════════════════════════
                    track_llm_call(
                        # Core metrics
                        model_name=routed_model,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        latency_ms=latency_ms,
                        agent_name="ResumeMatching",
                        agent_role="analyst",
                        operation=operation,
                        success=True,
                        
                        # Prompt content
                        system_prompt=optimized_prompt,
                        user_message=user_message,
                        response_text=result_str,
                        prompt_breakdown=prompt_breakdown,
                        
                        # Optimization tracking
                        routing_decision=routing_meta,
                        cache_metadata=None, 
                        quality_evaluation=quality_eval,
                        prompt_metadata=None,
                        
                        # Model configuration
                        temperature=0.5,  # Balanced analysis
                        max_tokens=max_tokens_limit,
                        
                        # Token breakdown
                        system_prompt_tokens=estimate_tokens(optimized_prompt),
                        user_message_tokens=estimate_tokens(user_message),
                        
                        # Conversation linking
                        conversation_id=self.memory.conversation_id if self.memory else None,
                        turn_number=self.memory.turn_number if self.memory else None,
                        parent_call_id=self.memory.request_id if self.memory else None,
                        request_id=str(uuid.uuid4()),
                        
                        # Observability
                        trace_id=self.memory.conversation_id if self.memory else None,
                        environment=os.getenv("ENVIRONMENT", "development"),

                        # Azure prompt cache metrics (stable_prefix category)
                        **azure_cache_metrics,

                        # Metadata - CRITICAL: Include phase
                        metadata={
                            "phase": CURRENT_PHASE,
                            "job_id": job.get('id'),
                            "job_title": job.get('title', 'Unknown'),
                            "original_score": original_score,
                            "judged": quality_eval is not None,
                            "streaming_candidate": bool(streaming_candidate),
                        }
                    )

                    logger.debug(f"Deep analysis LLM call: {latency_ms:.0f}ms, {prompt_tokens + completion_tokens} tokens")

                    # ═══════════════════════════════════════════════════════════════
                    # STEP 11: Track batch detection (individual call)
                    # ═══════════════════════════════════════════════════════════════
                    # Generate unique call ID for linking
                    call_id = str(uuid.uuid4())

                    batch_detector.track_call(
                        operation=operation,
                        call_id=call_id,
                        latency_ms=latency_ms,
                        agent_name="ResumeMatching",
                    )
                    
                    # Check token efficiency (deep analysis should have good prompt/completion ratio)
                    if completion_tokens > 0:
                        token_efficiency_detector.check_call(
                            operation=operation,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            call_id=call_id,
                            agent_name="ResumeMatching",
                        )
            
            # ═══════════════════════════════════════════════════════════════
            # RESULT PARSING (common path for all branches above)
            # ═══════════════════════════════════════════════════════════════

            # Parse response
            if '```json' in result_str:
                result_str = result_str.split('```json')[1].split('```')[0].strip()
            elif '```' in result_str:
                result_str = result_str.split('```')[1].split('```')[0].strip()
            
            start_idx = result_str.find('{')
            end_idx = result_str.rfind('}')
            if start_idx != -1 and end_idx != -1:
                result_str = result_str[start_idx:end_idx+1]
            
            match_data = json.loads(result_str)
            
            return {
                'job_id': job.get('id'),
                'title': job.get('title', 'Unknown Title'),
                'company': job.get('company', 'Unknown Company'),
                'location': job.get('location', 'Unknown Location'),
                'link': job.get('link', ''),
                'description': job.get('description', ''),
                'score': original_score if original_score > 0 else int(match_data.get('overall_score', 0)),
                'confidence': float(match_data.get('confidence', 0.5)),
                'confidence_reasoning': match_data.get('confidence_reasoning', ''),
                'uncertainty_factors': match_data.get('uncertainty_factors', []),
                'reason': match_data.get('summary', 'No summary provided.'),
                'score_breakdown': match_data.get('score_breakdown', {}),
                'matched_bullets': match_data.get('matched_bullets', []),
                'matched_skills': match_data.get('matched_skills', []),
                'missing_skills': match_data.get('missing_skills', []),
                'key_strengths': match_data.get('strengths', []),
                'gaps': match_data.get('gaps', []),
                'recommendation': match_data.get('improvement_suggestions', []),
                'detailed_analysis': json.dumps(match_data)
            }
            
        except Exception as e:
            logger.error(f"Deep analysis error for '{job.get('title', 'Unknown')}': {e}", exc_info=True)
            
            # Classify error
            error_info = classify_error(e, operation="deep_analyze_job")
            
            # Track error
            track_llm_call(
                model_name=DEFAULT_MODEL,
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=0,
                agent_name="ResumeMatching",
                agent_role="analyst",
                operation="deep_analyze_job",
                success=False,
                error=str(e),
                error_type=error_info['error_type'],
                error_code=error_info['error_code'],
                retry_count=0,
                conversation_id=self.memory.conversation_id if self.memory else None,
                turn_number=self.memory.turn_number if self.memory else None,
                parent_call_id=self.memory.request_id if self.memory else None,
                request_id=str(uuid.uuid4()),
                trace_id=self.memory.conversation_id if self.memory else None,
                environment=os.getenv("ENVIRONMENT", "development"),
                metadata={
                    "phase": CURRENT_PHASE,
                    "job_id": job.get('id'),
                    "job_title": job.get('title', 'Unknown'),
                    "error_type": error_info['error_type'],
                }
            )
            
            return {
                'job_id': job.get('id'),
                'title': job.get('title', 'Unknown Title'),
                'company': job.get('company', 'Unknown Company'),
                'location': job.get('location', 'Unknown Location'),
                'link': job.get('link', ''),
                'score': original_score,
                'confidence': 0.4,
                'reason': f"Match score: {original_score}/100 (detailed analysis unavailable)",
                'matched_skills': [],
                'missing_skills': [],
                'key_strengths': [],
                'gaps': [],
                'recommendation': '',
                'detailed_analysis': None
            }