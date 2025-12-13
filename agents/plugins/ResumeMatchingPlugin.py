# agents/plugins/ResumeMatchingPlugin.py
"""
Resume Matching Plugin - Career Copilot
UPDATED: Complete Observatory Tier 1, 2, 3 metrics coverage
"""

from semantic_kernel.functions import kernel_function
from typing import Annotated
import json
import logging
import time
import os

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
)

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# PROMPT VERSIONING
# =============================================================================
QUICK_SCORE_PROMPT_VERSION = "1.0.0"
DEEP_ANALYSIS_PROMPT_VERSION = "1.0.0"

# Create PromptMetadata for resume matching operations
QUICK_SCORE_META = create_prompt_metadata(
    template_id="resume_matching_quick_score",
    version=QUICK_SCORE_PROMPT_VERSION,
    compressible_sections=["Resume", "Job Description"],
    optimization_flags={"scoring_task": True},
    config_version="1.0"
) if PromptMetadata else None

DEEP_ANALYSIS_META = create_prompt_metadata(
    template_id="resume_matching_deep_analysis",
    version=DEEP_ANALYSIS_PROMPT_VERSION,
    compressible_sections=["Resume", "Job Description", "Instructions"],
    optimization_flags={"semantic_matching": True},
    config_version="1.0"
) if PromptMetadata else None


class ResumeMatchingPlugin:
    def __init__(self, kernel, database_service, memory=None):
        """
        Args:
            kernel: Your Semantic Kernel instance
            database_service: Your database access layer to fetch resumes and jobs
            memory: ConversationMemory instance for context tracking
        """
        self.kernel = kernel
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
                # NEW: Observability
                environment=os.getenv("ENVIRONMENT", "development"),
                
                metadata={"result_count": 0, "is_db_read": True, "conversation_memory": self.memory, "execution_settings": self.exec_settings}
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
            # NEW: Observability
            environment=os.getenv("ENVIRONMENT", "development"),
            
            metadata={"result_count": len(resumes), "is_db_read": True, "conversation_memory": self.memory, "execution_settings": self.exec_settings}
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
                
                # NEW: Error details
                retry_count=0,
                    prompt=f"Select resume: {selection}",
                    # NEW: Observability
                    environment=os.getenv("ENVIRONMENT", "development"),
                    
                    metadata={"selection": selection}
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
                
                # NEW: Error details
                retry_count=0,
                prompt=f"Select resume: {selection}",
                # NEW: Observability
                environment=os.getenv("ENVIRONMENT", "development"),
                
                metadata={"selection": selection, "resume_index": resume_index}
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
            # NEW: Observability
            environment=os.getenv("ENVIRONMENT", "development"),
            
            metadata={
                "selection": selection,
                "resume_id": selected_resume['id'],
                "resume_name": selected_resume['name'],
                "total_jobs": total_jobs,
                "unmatched_jobs": unmatched_jobs,
                "is_db_read": True,
                "conversation_memory": self.memory,
                "execution_settings": self.exec_settings
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
                
                # NEW: Error details
                retry_count=0,
                prompt=f"Filter: {filter_choice}",
                # NEW: Observability
                environment=os.getenv("ENVIRONMENT", "development"),
                
                metadata={"filter_choice": filter_choice, "conversation_memory": self.memory, "execution_settings": self.exec_settings}
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
                # NEW: Observability
                environment=os.getenv("ENVIRONMENT", "development"),
                
                metadata={"filter_choice": filter_choice, "job_filter": job_filter, "result_count": 0, "conversation_memory": self.memory, "execution_settings": self.exec_settings}
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
            # NEW: Observability
            environment=os.getenv("ENVIRONMENT", "development"),
            
            metadata={
                "filter_choice": filter_choice,
                "job_filter": job_filter,
                "result_count": len(job_ids),
                "resume_id": resume_id,
                "resume_name": resume['name'],
                "is_db_read": True,
                "conversation_memory": self.memory,
                "execution_settings": self.exec_settings
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
        # Start tracking the full matching session
        session = start_session(
            "resume_matching_workflow",
            # NEW: Observability
            environment=os.getenv("ENVIRONMENT", "development"),
            
            metadata={
                "resume_id": resume_id,
                "num_jobs": len(job_ids),
                "operation": "filtered_matching"
            }
        )
        logger.info(f"Starting resume matching: Resume #{resume_id} vs {len(job_ids)} jobs")
        workflow_start_time = time.time()
        
        try:
            # Get resume
            resume = self.db.get_resume_by_id(resume_id)
            if not resume:
                end_session(session, success=False, error="Resume not found")
                
                # NEW: Error details
                error_type=type(e).__name__ if 'e' in locals() else "UNKNOWN",
                retry_count=0,
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
                end_session(session, success=False, error="No jobs found")
                
                # NEW: Error details
                error_type=type(e).__name__ if 'e' in locals() else "UNKNOWN",
                retry_count=0,
                return "❌ No jobs found for matching."
            
            logger.info(f"Phase 1: Quick scoring {len(jobs)} jobs...")
            quick_score_start = time.time()
            
            # PHASE 1: Quick scoring for ALL jobs
            scored_jobs = []
            for i, job in enumerate(jobs, 1):
                logger.debug(f"Quick scoring job {i}/{len(jobs)}: {job.get('title', 'Unknown')}")
                job_start_time = time.time()
                
                scored = await self._quick_score_job_match(resume_text, job)
                scored_jobs.append(scored)
                
                job_latency = (time.time() - job_start_time) * 1000
                logger.debug(f"  └─ Score: {scored['score']}/100, Latency: {job_latency:.0f}ms")
            
            quick_score_duration = time.time() - quick_score_start
            logger.info(f"Phase 1 complete: {len(jobs)} jobs scored in {quick_score_duration:.1f}s")
            
            # Sort by score
            scored_jobs.sort(key=lambda x: x['score'], reverse=True)
            top_jobs = scored_jobs[:3]  # Top 3 for deep analysis
            
            logger.info(f"Phase 2: Deep analysis on top {len(top_jobs)} jobs...")
            deep_analysis_start = time.time()
            
            # PHASE 2: Deep analysis for top matches
            detailed_matches = []
            for i, job in enumerate(top_jobs, 1):
                logger.debug(f"Deep analysis {i}/{len(top_jobs)}: {job['title']}")
                job_start_time = time.time()
                
                # Get full job details
                full_job = self.db.get_job_by_id(job['job_id'])
                detailed = await self._deep_analyze_job_match(resume_text, full_job, job['score'])
                detailed_matches.append(detailed)
                
                job_latency = (time.time() - job_start_time) * 1000
                logger.debug(f"  └─ Latency: {job_latency:.0f}ms")
            
            deep_analysis_duration = time.time() - deep_analysis_start
            logger.info(f"Phase 2 complete: {len(top_jobs)} jobs analyzed in {deep_analysis_duration:.1f}s")
            
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
            
            # End session successfully
            end_session(session, success=True)
            
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
            end_session(session, success=False, error=str(e))
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
                
                # NEW: Error details
                retry_count=0,
                prompt=f"Explain match #{match_number}",
                # NEW: Observability
                environment=os.getenv("ENVIRONMENT", "development"),
                
                metadata={"match_number": match_number, "conversation_memory": self.memory, "execution_settings": self.exec_settings}
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
                
                # NEW: Error details
                retry_count=0,
                prompt=f"Explain match #{match_number}",
                # NEW: Observability
                environment=os.getenv("ENVIRONMENT", "development"),
                
                metadata={"match_number": match_number, "conversation_memory": self.memory, "execution_settings": self.exec_settings}
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
                
                # NEW: Error details
                retry_count=0,
                prompt=f"Explain match #{match_number}",
                # NEW: Observability
                environment=os.getenv("ENVIRONMENT", "development"),
                
                metadata={"match_number": match_number, "available_matches": len(recent_matches), "conversation_memory": self.memory, "execution_settings": self.exec_settings}
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
            # NEW: Observability
            environment=os.getenv("ENVIRONMENT", "development"),
            
            metadata={
                "match_number": match_number,
                "job_id": match.get('job_id'),
                "job_title": match.get('title'),
                "score": match.get('score'),
                "conversation_memory": self.memory,
                "execution_settings": self.exec_settings
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
                
                # NEW: Error details
                error_type=type(e).__name__ if 'e' in locals() else "UNKNOWN",
                retry_count=0,
                    prompt=f"Show matches for resume_id={resume_id}",
                    # NEW: Observability
                    environment=os.getenv("ENVIRONMENT", "development"),
                    
                    metadata={"resume_id": resume_id, "conversation_memory": self.memory, "execution_settings": self.exec_settings}
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
                
                # NEW: Error details
                error_type=type(e).__name__ if 'e' in locals() else "UNKNOWN",
                retry_count=0,
                    prompt=f"Show matches for resume_id={resume_id}",
                    # NEW: Observability
                    environment=os.getenv("ENVIRONMENT", "development"),
                    
                    metadata={"resume_id": resume_id, "conversation_memory": self.memory, "execution_settings": self.exec_settings}
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
                    # NEW: Observability
                    environment=os.getenv("ENVIRONMENT", "development"),
                    
                    metadata={"resume_id": resume_id, "resume_name": resume_name, "result_count": 0, "conversation_memory": self.memory, "execution_settings": self.exec_settings}
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
                # NEW: Observability
                environment=os.getenv("ENVIRONMENT", "development"),
                
                metadata={
                    "resume_id": resume_id,
                    "resume_name": resume_name,
                    "result_count": len(matches),
                    "is_db_read": True,
                    "conversation_memory": self.memory,
                    "execution_settings": self.exec_settings
                }
            )
            
            return response
            
        except Exception as e:
            track_llm_call(
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=(time.time() - start_time) * 1000,
                agent_name="ResumeMatching",
                agent_role="retriever",
                operation="show_saved_matches",
                success=False,
                error=str(e),
                
                # NEW: Error details
                error_type=type(e).__name__ if 'e' in locals() else "UNKNOWN",
                retry_count=0,
                prompt=f"Show matches for resume_id={resume_id}",
                # NEW: Observability
                environment=os.getenv("ENVIRONMENT", "development"),
                
                metadata={"resume_id": resume_id, "conversation_memory": self.memory, "execution_settings": self.exec_settings}
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
    
    async def _quick_score_job_match(self, resume_text: str, job: dict) -> dict:
        """
        Quick scoring method - provides a fast initial score for all jobs.
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
}

CONFIDENCE SCORING RULES:
- confidence: 0.9-1.0 = Very confident (clear, explicit evidence)
- confidence: 0.7-0.89 = Moderately confident (solid inference, minor ambiguity)
- confidence: 0.5-0.69 = Low confidence (significant assumptions made)
- confidence: <0.5 = Very uncertain (major gaps or contradictions)

List specific uncertainty_factors whenever confidence < 0.85"""

        user_message = f"""Resume:
{resume_text[:2000]}

Job:
Title: {job.get('title', 'N/A')}
Company: {job.get('company', 'N/A')}
Description: {job.get('description', 'N/A')[:1500]}"""

        prompt = f"{system_prompt}\n\n{user_message}"
        
        try:
            # Track this LLM call
            llm_start_time = time.time()
            
            result = await self.kernel.invoke_prompt(prompt)
            
            latency_ms = (time.time() - llm_start_time) * 1000
            result_str = str(result).strip()
            
            # Estimate tokens (rough approximation: 1 token ≈ 4 chars)
            prompt_tokens = len(prompt) // 4
            completion_tokens = len(result_str) // 4
            
            # Create prompt breakdown for Tier 2
            prompt_breakdown = create_prompt_breakdown(
                system_prompt=system_prompt,
                system_prompt_tokens=len(system_prompt) // 4,
                user_message=user_message,
                user_message_tokens=len(user_message) // 4,
            ) if create_prompt_breakdown else None
            
            # Tier 3: Routing decision
            routing_decision = create_routing_decision(
                chosen_model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini"),
                alternative_models=["gpt-4o", "gpt-4"],
                reasoning="Quick scoring - efficient model sufficient",
                complexity_score=0.4,
                estimated_cost_savings=0.026
            ) if create_routing_decision else None
            
            # Tier 3: Cache metadata
            cache_metadata = create_cache_metadata(
                cache_hit=False,
                cache_key=None,
                cache_cluster_id="resume_job_matching"
            ) if create_cache_metadata else None
            
            # Track in Observatory - COMPLETE with all tiers
            track_llm_call(
                # Core metrics (Tier 1)
                model_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini"),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                agent_name="ResumeMatching",
                agent_role="analyst",
                operation="quick_score_job",
                success=True,
                
                # Prompt content (Tier 2)
                system_prompt=system_prompt,
                user_message=user_message,
                prompt=prompt,
                response_text=result_str,
                prompt_metadata=QUICK_SCORE_META,
                prompt_breakdown=prompt_breakdown,
                
                # Optimization tracking (Tier 3)
                routing_decision=routing_decision,
                cache_metadata=cache_metadata,
                
                # A/B Testing (Tier 3)
                prompt_variant_id=None,
                test_dataset_id=None,
                
                # NEW: Conversation linking
                conversation_id=self.memory.conversation_id if self.memory else None,
                turn_number=self.memory.turn_number if self.memory else None,
                
                # NEW: Model configuration
                temperature=0.3,  # Factual scoring
                max_tokens=None,
                
                # NEW: Token breakdown (top-level)
                system_prompt_tokens=prompt_breakdown.system_prompt_tokens if prompt_breakdown else None,
                user_message_tokens=prompt_breakdown.user_message_tokens if prompt_breakdown else None,
                
                # NEW: Observability
                environment=os.getenv("ENVIRONMENT", "development"),
                
                # Metadata
                metadata={
                    "job_id": job.get('id'),
                    "job_title": job.get('title', 'Unknown'),
                    "conversation_memory": self.memory,
                    "execution_settings": self.exec_settings
                }
            )
            
            logger.debug(f"Quick score LLM call: {latency_ms:.0f}ms, ~{prompt_tokens + completion_tokens} tokens")
            
            # Parse response (rest of the existing code stays the same)
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
            
            # Track error
            track_llm_call(
                model_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini"),
                prompt_tokens=len(prompt) // 4 if 'prompt' in locals() else 0,
                completion_tokens=0,
                latency_ms=(time.time() - llm_start_time) * 1000 if 'llm_start_time' in locals() else 0,
                agent_name="ResumeMatching",
                agent_role="analyst",
                operation="quick_score_job",
                success=False,
                error=str(e),
                
                # NEW: Error details
                error_type=type(e).__name__ if 'e' in locals() else "UNKNOWN",
                retry_count=0,
                prompt_metadata=QUICK_SCORE_META,
                # NEW: Observability
                environment=os.getenv("ENVIRONMENT", "development"),
                
                metadata={"job_id": job.get('id'), "job_title": job.get('title', 'Unknown'), "conversation_memory": self.memory, "execution_settings": self.exec_settings}
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
    
    async def _deep_analyze_job_match(self, resume_text: str, job: dict, original_score: int) -> dict:
        """
        Deep analysis method - provides line-by-line semantic matching with exact text highlights.
        This is SLOWER and only used for top matches.
        """
        system_prompt = """You are an expert resume matcher. Perform semantic analysis to find connections between job requirements and resume content.

🎨 CRITICAL INSTRUCTIONS FOR HIGHLIGHT TEXT:

YOU MUST COPY EXACT TEXT FROM THE DOCUMENTS. DO NOT WRITE SUMMARIES.

RULE: job_requirement and job_highlight_text must be IDENTICAL.
RULE: resume_bullet and resume_highlight_text must be IDENTICAL.

STEP-BY-STEP PROCESS:
1. Read the job description below.
2. Find a COMPLETE sentence that states a requirement.
3. Copy that ENTIRE sentence word-for-word into BOTH fields (job_requirement and job_highlight_text).
4. Do the same for the resume (resume_bullet and resume_highlight_text).

✅ CORRECT EXAMPLE:
{
  "job_requirement": "Design and implement scalable data pipelines using Python, SQL, and cloud technologies.",
  "job_highlight_text": "Design and implement scalable data pipelines using Python, SQL, and cloud technologies.",
  "resume_bullet": "Built data pipelines in Python and SQL to process large datasets across AWS infrastructure.",
  "resume_highlight_text": "Built data pipelines in Python and SQL to process large datasets across AWS infrastructure.",
  "match_strength": "strong",
  "explanation": "The resume bullet clearly demonstrates experience designing and implementing data pipelines using Python and SQL, directly reflecting the job requirement."
}

❌ WRONG EXAMPLE:
{
  "job_requirement": "Data pipeline experience",
  "job_highlight_text": "Design and implement scalable data pipelines using Python, SQL, and cloud technologies.",
  "resume_bullet": "Created ETL processes for analytics.",
  "resume_highlight_text": "Built data pipelines in Python and SQL to process large datasets across AWS infrastructure."
}

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
}

CONFIDENCE SCORING RULES:
- confidence: 0.9-1.0 = Very confident (clear, explicit evidence)
- confidence: 0.7-0.89 = Moderately confident (solid inference, minor ambiguity)
- confidence: 0.5-0.69 = Low confidence (significant assumptions made)
- confidence: <0.5 = Very uncertain (major gaps or contradictions)

List specific uncertainty_factors whenever confidence < 0.85"""

        user_message = f"""**RESUME:**
{resume_text[:4000]}

**JOB:**
Title: {job.get('title', 'N/A')}
Company: {job.get('company', 'N/A')}
{job.get('description', 'N/A')[:3500]}

Return 10 matched bullets with EXACT TEXT from both documents."""

        prompt = f"{system_prompt}\n\n{user_message}"
        
        try:
            # Track this LLM call
            llm_start_time = time.time()
            
            result = await self.kernel.invoke_prompt(prompt)
            
            latency_ms = (time.time() - llm_start_time) * 1000
            result_str = str(result).strip()
            
            # Estimate tokens
            prompt_tokens = len(prompt) // 4
            completion_tokens = len(result_str) // 4
            
            # Create prompt breakdown for Tier 2
            prompt_breakdown = create_prompt_breakdown(
                system_prompt=system_prompt,
                system_prompt_tokens=len(system_prompt) // 4,
                user_message=user_message,
                user_message_tokens=len(user_message) // 4,
            ) if create_prompt_breakdown else None
            
            # LLM Judge evaluation (50% sampling)
            quality_eval = await judge.maybe_evaluate(
                operation="deep_analyze_job",
                prompt=prompt[:5000],
                response=result_str[:5000],
                llm_client=self.kernel, 
            )
            
            # Tier 3: Routing decision
            routing_decision = create_routing_decision(
                chosen_model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini"),
                alternative_models=["gpt-4o", "gpt-4"],
                reasoning="Deep analysis - complex reasoning task",
                complexity_score=0.7,
                estimated_cost_savings=0.010
            ) if create_routing_decision else None
            
            # Tier 3: Cache metadata
            cache_metadata = create_cache_metadata(
                cache_hit=False,
                cache_key=None,
                cache_cluster_id="resume_job_deep_analysis"
            ) if create_cache_metadata else None
            
            # Track in Observatory - COMPLETE with all tiers (single call)
            track_llm_call(
                # Core metrics (Tier 1)
                model_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini"),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                agent_name="ResumeMatching",
                agent_role="analyst",
                operation="deep_analyze_job",
                success=True,
                
                # Prompt content (Tier 2)
                system_prompt=system_prompt,
                user_message=user_message,
                prompt=prompt,
                response_text=result_str,
                prompt_metadata=DEEP_ANALYSIS_META,
                prompt_breakdown=prompt_breakdown,
                
                # Quality evaluation (Tier 2)
                quality_evaluation=quality_eval,
                
                # Optimization tracking (Tier 3)
                routing_decision=routing_decision,
                cache_metadata=cache_metadata,
                
                # A/B Testing (Tier 3)
                prompt_variant_id=None,
                test_dataset_id=None,
                
                # NEW: Conversation linking
                conversation_id=self.memory.conversation_id if self.memory else None,
                turn_number=self.memory.turn_number if self.memory else None,
                
                # NEW: Model configuration
                temperature=0.5,  # Balanced analysis
                max_tokens=None,
                
                # NEW: Token breakdown (top-level)
                system_prompt_tokens=prompt_breakdown.system_prompt_tokens if prompt_breakdown else None,
                user_message_tokens=prompt_breakdown.user_message_tokens if prompt_breakdown else None,
                
                # NEW: Observability
                environment=os.getenv("ENVIRONMENT", "development"),
                
                # Metadata
                metadata={
                    "job_id": job.get('id'),
                    "job_title": job.get('title', 'Unknown'),
                    "original_score": original_score,
                    "judged": quality_eval is not None,
                    "conversation_memory": self.memory,
                    "execution_settings": self.exec_settings
                }
            )
            
            logger.debug(f"Deep analysis LLM call: {latency_ms:.0f}ms, ~{prompt_tokens + completion_tokens} tokens")
            
            # Parse response (rest of the existing code stays the same)
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
            
            # Track error
            track_llm_call(
                model_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini"),
                prompt_tokens=len(prompt) // 4 if 'prompt' in locals() else 0,
                completion_tokens=0,
                latency_ms=(time.time() - llm_start_time) * 1000 if 'llm_start_time' in locals() else 0,
                agent_name="ResumeMatching",
                agent_role="analyst",
                operation="deep_analyze_job",
                success=False,
                error=str(e),
                
                # NEW: Error details
                error_type=type(e).__name__ if 'e' in locals() else "UNKNOWN",
                retry_count=0,
                prompt_metadata=DEEP_ANALYSIS_META,
                # NEW: Observability
                environment=os.getenv("ENVIRONMENT", "development"),
                
                metadata={"job_id": job.get('id'), "job_title": job.get('title', 'Unknown')}
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