# agents/plugins/ResumeTailoringPlugin.py
"""
Resume Tailoring Plugin - Career Copilot
UPDATED: Complete Observatory integration with full 10-step optimization pattern
"""

from semantic_kernel.functions import kernel_function
from typing import Annotated
import json
import logging
import os
import time
import uuid

# ═════════════════════════════════════════════════════════════════════════
# OBSERVATORY INTEGRATION - STANDARDIZED IMPORT BLOCK FOR LLM-MAKING PLUGINS
# ═════════════════════════════════════════════════════════════════════════
from observatory_config import (
    # Main tracking
    track_llm_call,
    
    # Optimization components (10-step pattern - import ALL for consistency)
    cache,
    semantic_cache,
    prefix_cache,
    router,
    prompt_optimizer,
    streaming_detector,
    batch_detector,
    parallel_detector,
    judge,
    
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

# =============================================================================
# PROMPT VERSIONING
# =============================================================================
IMPROVE_BULLET_PROMPT_VERSION = "1.1.0"
CHANGE_REPORT_PROMPT_VERSION = "1.0.0"

class ResumeTailoringPlugin:
    
    def __init__(self, kernel, memory=None):
        """
        Args:
            kernel: Your Semantic Kernel instance for AI operations
            memory: ConversationMemory instance for context tracking
        """
        self.kernel = kernel
        self.memory = memory

        # Create execution settings once per plugin instance
        from agents.semantic_kernel_setup import create_execution_settings
        self.exec_settings = create_execution_settings()
    
    @kernel_function(
        name="improve_resume_bullet",
        description="Generate improved versions of a resume bullet point tailored to a specific job"
    )
    async def improve_resume_bullet(
        self,
        resume_text: Annotated[str, "The full resume text for context"],
        job_description: Annotated[str, "The job description to tailor towards"],
        job_title: Annotated[str, "The job title"],
        company: Annotated[str, "The company name"],
        user_request: Annotated[str, "The user's specific request for improvement"],
        matched_skills: Annotated[str, "Comma-separated list of skills that match the job"] = "",
        missing_skills: Annotated[str, "Comma-separated list of skills missing from resume"] = "",
        gaps: Annotated[str, "Key gaps identified in the match analysis"] = ""
    ) -> Annotated[str, "JSON string with 3 improved bullet point suggestions"]:
        """
        Takes a user request and generates 3 improved resume bullet point variations
        that are tailored to the specific job requirements.
        
        Implements full 10-step optimization pattern.
        """
        
        # Build the prompt
        system_prompt = """You are an expert resume writer helping tailor a resume for a specific job.

Your Task:
Generate 3 improved resume bullet points that address the user's request. Each bullet should:
1. Be tailored to the job requirements above
2. Incorporate relevant skills from the "Missing Skills" list when appropriate
3. Use strong action verbs (Architected, Spearheaded, Implemented, etc.)
4. Include quantifiable metrics when possible (%, $, time saved, users served, etc.)
5. Be concise (1-2 lines maximum)
6. Sound natural and authentic to the candidate's experience

CRITICAL: Return ONLY valid JSON. No markdown, no code blocks, no explanations outside the JSON."""

        user_message = f"""**Context:**
- Job Title: {job_title} at {company}
- Matched Skills: {matched_skills if matched_skills else "Not specified"}
- Missing Skills: {missing_skills if missing_skills else "None identified"}
- Key Gaps: {gaps if gaps else "None identified"}

**Job Description (first 1500 chars):**
{job_description[:1500]}

**Current Resume (first 2000 chars):**
{resume_text[:2000]}

**User's Request:**
{user_request}

Required JSON format:
{{
  "suggestions": [
    {{
      "version": 1,
      "bullet": "Your first improved bullet point",
      "explanation": "Brief explanation of why this version is strong (mention which job requirements it addresses)"
    }},
    {{
      "version": 2,
      "bullet": "Your second alternative bullet point",
      "explanation": "Why this approach works"
    }},
    {{
      "version": 3,
      "bullet": "Your third variation",
      "explanation": "Reasoning for this version"
    }}
  ],
  "original_identified": "The original bullet point from the resume that you're improving, or 'New bullet point' if creating from scratch"
}}"""

        try:
            logger.info(f"🖊️  Generating resume improvements for: '{job_title}' at {company}")
            
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
            max_tokens_limit = 1500  # Default
            routed_model = DEFAULT_MODEL  # Default
            
            # ═══════════════════════════════════════════════════════════════
            # STEP 1: Check exact cache
            # ═══════════════════════════════════════════════════════════════
            operation = "improve_bullet"
            cache_key_data = {
                "user_request": user_request,
                "job_title": job_title,
                "company": company
            }
            
            cached_result, cache_meta = cache.get(
                operation=operation,
                key_data=cache_key_data
            )
            
            # Ensure cache_meta is None if empty dict
            if isinstance(cache_meta, dict) and not cache_meta:
                cache_meta = None
            
            if cached_result:
                logger.info(f"✅ Cache hit! Using cached improvement suggestions.")
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
                    agent_name="ResumeTailoring",
                    agent_role="writer",
                    conversation_id=self.memory.conversation_id if self.memory else None,
                    turn_number=self.memory.turn_number if self.memory else None,
                    parent_call_id=self.memory.request_id if self.memory else None,
                    request_id=str(uuid.uuid4()),
                    trace_id=self.memory.conversation_id if self.memory else None,
                    environment=os.getenv("ENVIRONMENT", "development"),
                    metadata={
                        "phase": CURRENT_PHASE,
                        "cache_hit": True,
                        "job_title": job_title,
                        "company": company,
                    }
                )
                
                # Skip to result parsing
            
            else:
                # ═══════════════════════════════════════════════════════════════
                # STEP 2: Check semantic cache (if available)
                # ═══════════════════════════════════════════════════════════════
                semantic_hit = False  # Track if semantic cache hit
                
                if semantic_cache:
                    result = await semantic_cache.get(operation=operation, prompt=user_request)
                    if result.hit:
                        logger.info(f"✅ Semantic cache hit ({result.similarity:.1%} similar)!")
                        result_str = result.response
                        latency_ms = 1.0
                        prompt_tokens = 0
                        completion_tokens = 0
                        semantic_hit = True
                        
                        # Track semantic cache hit
                        track_llm_call(
                            operation=operation,
                            prompt_tokens=0,
                            completion_tokens=0,
                            latency_ms=1.0,
                            success=True,
                            response_text=result_str,
                            cache_metadata=create_cache_metadata(
                                cache_hit=True,
                                similarity_score=result.similarity
                            ),
                            agent_name="ResumeTailoring",
                            agent_role="writer",
                            conversation_id=self.memory.conversation_id if self.memory else None,
                            turn_number=self.memory.turn_number if self.memory else None,
                            parent_call_id=self.memory.request_id if self.memory else None,
                            request_id=str(uuid.uuid4()),
                            trace_id=self.memory.conversation_id if self.memory else None,
                            environment=os.getenv("ENVIRONMENT", "development"),
                            metadata={
                                "phase": CURRENT_PHASE,
                                "semantic_cache_hit": True,
                                "similarity": result.similarity,
                                "job_title": job_title,
                            }
                        )
                        
                        # Skip to result parsing
                
                # If no cache hit, proceed with LLM call
                if not cached_result and not semantic_hit:
                    # ═══════════════════════════════════════════════════════════════
                    # STEP 3: Get optimized prompt and max_tokens
                    # ═══════════════════════════════════════════════════════════════
                    optimized_prompt, max_tokens_limit, prompt_meta = prompt_optimizer.get_optimized_prompt(
                        operation=operation,
                        default_prompt=system_prompt
                    )
                    
                    # ═══════════════════════════════════════════════════════════════
                    # STEP 4: Get routed model
                    # ═══════════════════════════════════════════════════════════════
                    routed_model, routing_meta = router.select(
                        operation=operation,
                        prompt=optimized_prompt + user_message,
                        estimated_tokens=estimate_tokens(optimized_prompt + user_message),
                        complexity=0.7  # Creative writing is medium-high complexity
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
                    full_prompt = f"{optimized_prompt}\n\n{user_message}"
                    
                    llm_start_time = time.time()
                    result = await self.kernel.invoke_prompt(full_prompt)
                    latency_ms = (time.time() - llm_start_time) * 1000
                    
                    result_str = str(result).strip()
                    prompt_tokens = estimate_tokens(full_prompt)
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
                    # STEP 8: Cache the response
                    # ═══════════════════════════════════════════════════════════════
                    cache.set(
                        operation=operation,
                        key_data=cache_key_data,
                        value=result_str
                    )
                    
                    if semantic_cache:
                        await semantic_cache.set(
                            operation=operation,
                            prompt=user_request,
                            response=result_str
                        )
                    
                    # ═══════════════════════════════════════════════════════════════
                    # STEP 9: Create prompt breakdown and evaluate quality
                    # ═══════════════════════════════════════════════════════════════
                    prompt_breakdown = create_prompt_breakdown(
                        system_prompt=optimized_prompt,
                        system_prompt_tokens=estimate_tokens(optimized_prompt),
                        user_message=user_message,
                        user_message_tokens=estimate_tokens(user_message),
                    )
                    
                    # LLM Judge evaluation
                    quality_eval = await judge.maybe_evaluate(
                        operation=operation,
                        prompt=full_prompt,
                        response=result_str,
                        llm_client=self.kernel,
                        conversation_id=self.memory.conversation_id if self.memory else None,
                        turn_number=self.memory.turn_number if self.memory else None,
                    )
                    
                    # ═══════════════════════════════════════════════════════════════
                    # STEP 10: Track with Observatory (CRITICAL - INCLUDE PHASE)
                    # ═══════════════════════════════════════════════════════════════
                    track_llm_call(
                        # Core metrics
                        model_name=routed_model,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        latency_ms=latency_ms,
                        agent_name="ResumeTailoring",
                        agent_role="writer",
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
                        temperature=0.7,  # Creative writing for resume improvement
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
                        
                        # Metadata - CRITICAL: Include phase
                        metadata={
                            "phase": CURRENT_PHASE,
                            "job_title": job_title,
                            "company": company,
                            "user_request": user_request[:100],
                            "has_matched_skills": bool(matched_skills),
                            "has_missing_skills": bool(missing_skills),
                            "judged": quality_eval is not None,
                            "streaming_candidate": streaming_candidate,
                        }
                    )
                    
                    logger.info(f"📊 Tracked resume tailoring: {latency_ms:.0f}ms, {prompt_tokens + completion_tokens} tokens")
            
            # ═══════════════════════════════════════════════════════════════
            # RESULT PARSING (common path for all branches above)
            # ═══════════════════════════════════════════════════════════════
            
            # Clean up response - extract JSON
            if '```json' in result_str:
                result_str = result_str.split('```json')[1].split('```')[0].strip()
            elif '```' in result_str:
                result_str = result_str.split('```')[1].split('```')[0].strip()
            
            # Extract JSON object
            start_idx = result_str.find('{')
            end_idx = result_str.rfind('}')
            if start_idx != -1 and end_idx != -1:
                result_str = result_str[start_idx:end_idx+1]
            
            # Parse and validate JSON
            suggestions_data = json.loads(result_str)
            
            # Validate structure
            if 'suggestions' not in suggestions_data:
                raise ValueError("Response missing 'suggestions' field")
            
            # Store suggestions in memory for "apply #2" commands
            if self.memory:
                self.memory.set_tailoring_suggestions(suggestions_data)
                self.memory.update_context(last_action="resume_tailoring")
            
            # Return as JSON string
            return json.dumps(suggestions_data, indent=2)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.debug(f"Raw response: {result_str[:500] if 'result_str' in locals() else 'N/A'}")
            
            # Classify error
            error_info = classify_error(e, operation="improve_bullet")
            
            # Track failed call
            track_llm_call(
                prompt_tokens=estimate_tokens(system_prompt + user_message) if 'system_prompt' in locals() else 0,
                completion_tokens=estimate_tokens(result_str) if 'result_str' in locals() else 0,
                latency_ms=latency_ms if 'latency_ms' in locals() else 0,
                agent_name="ResumeTailoring",
                agent_role="writer",
                operation="improve_bullet",
                success=False,
                error=f"JSON parsing error: {str(e)}",
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
                    "job_title": job_title,
                    "company": company,
                    "error_type": error_info['error_type'],
                }
            )
            
            return json.dumps({
                "error": "Failed to parse AI response",
                "suggestions": [],
                "original_identified": "Error"
            })

        except Exception as e:
            logger.error(f"Error generating suggestions: {type(e).__name__}: {e}")
            
            # Classify error
            error_info = classify_error(e, operation="improve_bullet")
            
            # Track failed call
            track_llm_call(
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=latency_ms if 'latency_ms' in locals() else 0,
                agent_name="ResumeTailoring",
                agent_role="writer",
                operation="improve_bullet",
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
                    "job_title": job_title,
                    "company": company,
                    "error_type": error_info['error_type'],
                }
            )
            
            return json.dumps({
                "error": str(e),
                "suggestions": [],
                "original_identified": "Error"
            })
    
    @kernel_function(
        name="generate_change_report",
        description="Generate a formatted report of all approved resume changes"
    )
    async def generate_change_report(
        self,
        approved_changes: Annotated[str, "JSON string of approved changes"],
        resume_name: Annotated[str, "Name of the resume"],
        job_title: Annotated[str, "Job title being tailored for"],
        company: Annotated[str, "Company name"]
    ) -> Annotated[str, "Formatted markdown report of all changes"]:
        """
        Takes a list of approved changes and generates a clean, copy-paste ready report.
        No LLM call - just formatting.
        """
        start_time = time.time()
        
        try:
            changes = json.loads(approved_changes)
            
            if not changes:
                return "No changes have been approved yet."
            
            # Build the report
            report = f"""# Resume Tailoring Report
## {resume_name}
**Tailored for:** {job_title} at {company}
**Date:** {import_datetime()}
**Total Changes:** {len(changes)}

---

"""
            
            for i, change in enumerate(changes, 1):
                report += f"""### Change {i}

**Original:**
{change.get('original', 'Not specified')}

**New Version:**
- {change.get('new', 'No text provided')}

**Why this improves your resume:**
{change.get('explanation', 'No explanation provided')}

---

"""
            
            report += f"""
## Quick Copy Section
Copy each improved bullet below and paste into your resume:

"""
            
            for i, change in enumerate(changes, 1):
                report += f"{i}. {change.get('new', '')}\n\n"
            
            # Track report generation (no LLM, just formatting)
            track_llm_call(
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=(time.time() - start_time) * 1000,
                agent_name="ResumeTailoring",
                agent_role="formatter",
                operation="generate_change_report",
                success=True,
                prompt=f"Generate report for {len(changes)} changes",
                response_text=report[:500],
                conversation_id=self.memory.conversation_id if self.memory else None,
                turn_number=self.memory.turn_number if self.memory else None,
                parent_call_id=self.memory.request_id if self.memory else None,
                request_id=str(uuid.uuid4()),
                trace_id=self.memory.conversation_id if self.memory else None,
                environment=os.getenv("ENVIRONMENT", "development"),
                metadata={
                    "phase": CURRENT_PHASE,
                    "resume_name": resume_name,
                    "job_title": job_title,
                    "company": company,
                    "num_changes": len(changes),
                    "is_formatting_only": True,
                }
            )
            
            return report
            
        except Exception as e:
            # Classify error
            error_info = classify_error(e, operation="generate_change_report")
            
            # Track error
            track_llm_call(
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=(time.time() - start_time) * 1000,
                agent_name="ResumeTailoring",
                agent_role="formatter",
                operation="generate_change_report",
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
                    "resume_name": resume_name,
                    "error_type": error_info['error_type'],
                }
            )
            return f"Error generating report: {str(e)}"


def import_datetime():
    """Helper to get current datetime."""
    from datetime import datetime
    return datetime.now().strftime("%B %d, %Y at %I:%M %p")