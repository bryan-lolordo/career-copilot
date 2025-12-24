# agents/plugins/ResumeTailoringPlugin.py
"""
Resume Tailoring Plugin - Career Copilot
UPDATED: Complete Observatory Tier 1, 2, 3 metrics coverage
"""

from semantic_kernel.functions import kernel_function
from typing import Annotated
import json
import logging
import os
import time

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
    classify_error,
    generate_cache_key,
)

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# PROMPT VERSIONING
# =============================================================================
IMPROVE_BULLET_PROMPT_VERSION = "1.1.0"
CHANGE_REPORT_PROMPT_VERSION = "1.0.0"

# Create PromptMetadata for resume tailoring operations
IMPROVE_BULLET_META = create_prompt_metadata(
    template_id="resume_tailoring_improve_bullet",
    version=IMPROVE_BULLET_PROMPT_VERSION,
    compressible_sections=["Context", "Your Task"],
    optimization_flags={"creative_task": True},
    config_version="1.0"
) if PromptMetadata else None

CHANGE_REPORT_META = create_prompt_metadata(
    template_id="resume_tailoring_change_report",
    version=CHANGE_REPORT_PROMPT_VERSION,
    compressible_sections=[],
    optimization_flags={"formatting_task": True},
    config_version="1.0"
) if PromptMetadata else None


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

        full_prompt = f"{system_prompt}\n\n{user_message}"

        try:
            # Track LLM call
            llm_start_time = time.time()
            
            result = await self.kernel.invoke_prompt(full_prompt)
            
            latency_ms = (time.time() - llm_start_time) * 1000
            result_str = str(result).strip()
            
            # Estimate tokens
            prompt_tokens = len(full_prompt) // 4
            completion_tokens = len(result_str) // 4
            
            # Create prompt breakdown for Tier 2
            prompt_breakdown = create_prompt_breakdown(
                system_prompt=system_prompt,
                system_prompt_tokens=len(system_prompt) // 4,
                user_message=user_message,
                user_message_tokens=len(user_message) // 4,
            ) if create_prompt_breakdown else None
            
            # LLM Judge evaluation (50% sampling) - get this BEFORE tracking
            quality_eval = await judge.maybe_evaluate(
                operation="improve_bullet",  
                prompt=full_prompt,
                response=result_str,
                llm_client=self.kernel, 
                conversation_id=self.memory.conversation_id if self.memory else None,
                turn_number=self.memory.turn_number if self.memory else None, 
            )
            # Tier 3: Routing decision (placeholder - ready for optimization)
            routing_decision = create_routing_decision(
                chosen_model=DEFAULT_MODEL,  # Will be filled by observatory_config
                alternative_models=["gpt-4o", "gpt-4o-mini"],
                reasoning="Creative task - using default model",
                complexity_score=0.7
            ) if create_routing_decision else None
            
            # Tier 3: Cache metadata (placeholder - ready for optimization)
            cache_metadata = create_cache_metadata(
                cache_hit=False,
                cache_key=generate_cache_key("improve_bullet", user_request, job_title),  
                cache_cluster_id="resume_tailoring"
            ) if create_cache_metadata else None
            
            # Track in Observatory - COMPLETE with all tiers
            track_llm_call(
                # Core metrics (Tier 1)
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                agent_name="ResumeTailoring",
                agent_role="writer",  # Added: agent role
                operation="improve_bullet",
                success=True,  # Added: explicit success
                
                # Prompt analysis (Tier 2)
                system_prompt=system_prompt,
                user_message=user_message,
                response_text=result_str,
                prompt_metadata=IMPROVE_BULLET_META,
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
                conversation_id=self.memory.conversation_id if self.memory else None,
                turn_number=self.memory.turn_number if self.memory else None,
                parent_call_id=self.memory.request_id if self.memory else None,

                # NEW: Model configuration
                temperature=0.7,  # Creative writing for resume improvement
                max_tokens=None,  # Keep None to see inefficiencies
                
                # NEW: Token breakdown (top-level)
                system_prompt_tokens=prompt_breakdown.system_prompt_tokens if prompt_breakdown else None,
                user_message_tokens=prompt_breakdown.user_message_tokens if prompt_breakdown else None,
                
                # NEW: Observability
                environment=os.getenv("ENVIRONMENT", "development"),
                
                # Metadata
                metadata={
                    "job_title": job_title,
                    "company": company,
                    "user_request": user_request[:100],
                    "has_matched_skills": bool(matched_skills),
                    "has_missing_skills": bool(missing_skills),
                    "judged": quality_eval is not None,
                    "conversation_memory": self.memory,
                    "execution_settings": self.exec_settings
                }
            )
            
            print(f"📊 Tracked resume tailoring: {latency_ms:.0f}ms, ~{prompt_tokens + completion_tokens} tokens")
            
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
            print(f"❌ JSON parsing error: {e}")
            print(f"Raw response: {result_str[:500]}")
            
            # Track failed call
            track_llm_call(
                prompt_tokens=len(full_prompt) // 4 if 'full_prompt' in locals() else 0,
                completion_tokens=0,
                latency_ms=(time.time() - llm_start_time) * 1000 if 'llm_start_time' in locals() else 0,
                agent_name="ResumeTailoring",
                agent_role="writer",
                operation="improve_bullet",
                success=False,
                error=f"JSON parsing error: {str(e)}",
                prompt_metadata=IMPROVE_BULLET_META,

                # NEW: Conversation linking
                conversation_id=self.memory.conversation_id if self.memory else None,
                turn_number=self.memory.turn_number if self.memory else None,
                parent_call_id=self.memory.request_id if self.memory else None,

                # NEW: Observability
                environment=os.getenv("ENVIRONMENT", "development"),
                
                metadata={"job_title": job_title, "company": company, "conversation_memory": self.memory, "execution_settings": self.exec_settings}
            )
            
            return json.dumps({
                "error": "Failed to parse AI response",
                "suggestions": [],
                "original_identified": "Error"
            })
        
        except Exception as e:
            print(f"❌ Error generating suggestions: {type(e).__name__}: {e}")
            
            # Track failed call
            track_llm_call(
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=(time.time() - llm_start_time) * 1000 if 'llm_start_time' in locals() else 0,
                agent_name="ResumeTailoring",
                agent_role="writer",
                operation="improve_bullet",
                success=False,
                error=str(e),
                prompt_metadata=IMPROVE_BULLET_META,

                # ERROR CLASSIFICATION
                **classify_error(e, operation="improve_bullet"),
                retry_count=0,

                # Conversation linking
                conversation_id=self.memory.conversation_id if self.memory else None,
                turn_number=self.memory.turn_number if self.memory else None,
                parent_call_id=self.memory.request_id if self.memory else None,

                # Observability
                environment=os.getenv("ENVIRONMENT", "development"),
                
                metadata={"job_title": job_title, "company": company, "conversation_memory": self.memory, "execution_settings": self.exec_settings}
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
                prompt_metadata=CHANGE_REPORT_META,
                routing_decision=None,
                cache_metadata=None,
                quality_evaluation=None,
                prompt_variant_id=None,
                test_dataset_id=None,

                # NEW: Conversation linking
                conversation_id=self.memory.conversation_id if self.memory else None,
                turn_number=self.memory.turn_number if self.memory else None,
                parent_call_id=self.memory.request_id if self.memory else None,

                # NEW: Observability
                environment=os.getenv("ENVIRONMENT", "development"),
                
                metadata={
                    "resume_name": resume_name,
                    "job_title": job_title,
                    "company": company,
                    "num_changes": len(changes),
                    "is_formatting_only": True,
                    "conversation_memory": self.memory,
                    "execution_settings": self.exec_settings
                }
            )
            
            return report
            
        except Exception as e:
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

                # ERROR CLASSIFICATION
                **classify_error(e, operation="generate_change_report"),
                retry_count=0,

                # Conversation linking
                conversation_id=self.memory.conversation_id if self.memory else None,
                turn_number=self.memory.turn_number if self.memory else None,
                parent_call_id=self.memory.request_id if self.memory else None,
                
                # Observability
                environment=os.getenv("ENVIRONMENT", "development"),
                
                metadata={"resume_name": resume_name, "conversation_memory": self.memory, "execution_settings": self.exec_settings}
            )
            return f"Error generating report: {str(e)}"


def import_datetime():
    """Helper to get current datetime."""
    from datetime import datetime
    return datetime.now().strftime("%B %d, %Y at %I:%M %p")