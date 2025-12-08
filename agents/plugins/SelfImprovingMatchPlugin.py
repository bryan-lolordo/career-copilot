# agents/plugins/SelfImprovingMatchPlugin.py
"""
Self-Improving Match Plugin - Career Copilot
UPDATED: Complete Observatory Tier 2 metrics coverage

AI reviews and improves its own matching analysis through iterative refinement.
"""

from semantic_kernel.functions import kernel_function
from typing import Annotated
import json
import time

# Observatory Integration - Updated imports
from observatory_config import (
    track_llm_call,
    create_prompt_metadata,
    PromptMetadata
)
from llm_judge import maybe_judge_response

# =============================================================================
# PROMPT VERSIONING
# =============================================================================
DEEP_ANALYZE_GUIDANCE_VERSION = "1.1.0"
CRITIQUE_MATCH_VERSION = "1.0.0"
GENERATE_REFINEMENTS_VERSION = "1.0.0"
REFINE_ANALYSIS_VERSION = "1.0.0"

# Create PromptMetadata for self-improving operations
DEEP_ANALYZE_GUIDANCE_META = create_prompt_metadata(
    template_id="self_improving_deep_analyze_guidance",
    version=DEEP_ANALYZE_GUIDANCE_VERSION,
    compressible_sections=["RETURN FORMAT", "PREVIOUS ANALYSIS"],
    optimization_flags={"iterative_refinement": True},
    config_version="1.0"
) if PromptMetadata else None

CRITIQUE_MATCH_META = create_prompt_metadata(
    template_id="self_improving_critique_match",
    version=CRITIQUE_MATCH_VERSION,
    compressible_sections=["Review This Match For"],
    optimization_flags={"quality_control": True},
    config_version="1.0"
) if PromptMetadata else None

GENERATE_REFINEMENTS_META = create_prompt_metadata(
    template_id="self_improving_generate_refinements",
    version=GENERATE_REFINEMENTS_VERSION,
    compressible_sections=["Current Analysis Issues"],
    optimization_flags={"guidance_generation": True},
    config_version="1.0"
) if PromptMetadata else None

REFINE_ANALYSIS_META = create_prompt_metadata(
    template_id="self_improving_refine_analysis",
    version=REFINE_ANALYSIS_VERSION,
    compressible_sections=["PREVIOUS ANALYSIS", "YOUR TASK"],
    optimization_flags={"refinement_mode": True},
    config_version="1.0"
) if PromptMetadata else None


class SelfImprovingMatchPlugin:
    
    def __init__(self, kernel, matching_plugin, context=None):
        """
        Args:
            kernel: Semantic Kernel instance
            matching_plugin: The ResumeMatchingPlugin instance
            context: Shared ConversationContext instance
        """
        self.kernel = kernel
        self.matching_plugin = matching_plugin
        self.context = context

    @kernel_function(
        name="self_improve_single_match",
        description="Self-improve matching for a single job-resume pair"
    )
    async def self_improve_single_match(
        self,
        resume_id: Annotated[str, "Resume ID"],
        job_id: Annotated[str, "Job ID"],
        max_iterations: Annotated[int, "Maximum refinement iterations"] = 2
    ) -> Annotated[str, "JSON with final match and improvement log"]:
        """
        Self-improve a single job match with AI quality control.
        
        Process:
        1. Deep analyze this specific job-resume pair
        2. AI critic reviews the analysis
        3. If issues found, refine and re-analyze
        4. Return updated match
        """
        
        print(f"\n🤖 Self-Improving Single Match: Resume {resume_id} + Job {job_id}")
        
        db_service = self.matching_plugin.db
        
        # Get resume and job data
        resume = db_service.get_resume_by_id(int(resume_id))
        job = db_service.get_job_by_id(int(job_id))
        
        if not resume or not job:
            return json.dumps({'error': 'Resume or job not found'})
        
        iteration = 0
        refinement_log = []
        current_analysis = None
        best_analysis = None
        best_score = 0
        quality_score = 0
        refinement_guidance = []
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\n📊 === Iteration {iteration}/{max_iterations} ===")
            
            # STEP 1: Deep analyze with accumulated refinement guidance
            print(f"   ▶️ Analyzing match...")
            
            if refinement_guidance:
                guidance_text = "\n".join([f"- {item}" for item in refinement_guidance])
                print(f"   📝 Applying {len(refinement_guidance)} refinements...")
            else:
                guidance_text = ""
            
            analysis = await self._deep_analyze_with_guidance(
                resume_text=resume['text'],
                job=job,
                guidance=guidance_text,
                previous_analysis=current_analysis if iteration > 1 else None
            )
            
            if (analysis['score'] == 0 and 
                not analysis.get('matched_skills') and 
                not analysis.get('_parsing_failed') and
                current_analysis and 
                current_analysis['score'] > 0):
                print(f"   ⚠️ Analysis completely failed, keeping previous analysis")
                analysis = current_analysis
            elif analysis.get('_parsing_failed'):
                print(f"   ⚠️ Partial parsing - extracted score: {analysis['score']}")
            
            current_analysis = analysis
            print(f"   ✅ Score: {analysis['score']}/100")
            
            # Track best analysis so far
            if analysis['score'] > best_score:
                best_analysis = current_analysis.copy()
                best_score = analysis['score']
                print(f"   🏆 New best score: {best_score}/100")
            
            # STEP 2: AI Critic reviews this single match
            print(f"   🔍 AI Critic reviewing...")
            
            critique = await self._critique_single_match(analysis, resume, job)
            
            try:
                critique_data = json.loads(critique)
            except:
                print("   ⚠️ Critique parsing failed")
                break
            
            quality_score = critique_data.get('overall_quality', 0)
            issues = critique_data.get('issues', [])
            
            print(f"   📊 Quality: {quality_score}/100")
            print(f"   ⚠️ Issues: {len(issues)}")
            
            # Log iteration
            iteration_log = {
                'iteration': iteration,
                'score': analysis['score'],
                'quality_score': quality_score,
                'issues': issues,
                'strengths': critique_data.get('strengths', []),
                'weaknesses': critique_data.get('weaknesses', [])
            }
            
            if '_refinement_stats' in analysis:
                iteration_log['refinement_stats'] = analysis['_refinement_stats']
            
            refinement_log.append(iteration_log)
            
            # STEP 3: Check if acceptable
            if quality_score >= 85 and len(issues) == 0:
                print(f"   ✅ Quality acceptable!")
                break
            
            if iteration >= max_iterations:
                print(f"   ⏰ Max iterations reached")
                break
            
            # STEP 4: Generate refinements for NEXT iteration
            print(f"   🔧 Generating refinements...")
            
            refinements = await self._generate_refinements_for_single_match(
                critique_data,
                analysis,
                resume,
                job
            )
            
            try:
                refinements_data = json.loads(refinements)
                adjustments = refinements_data.get('adjustments', [])
                focus_areas = refinements_data.get('focus_areas', [])
                
                for adj in adjustments:
                    guidance_item = f"{adj.get('area', 'General')}: {adj.get('change', 'N/A')}"
                    refinement_guidance.append(guidance_item)
                
                for focus in focus_areas:
                    refinement_guidance.append(f"Focus: {focus}")
                
                print(f"   📝 Added {len(adjustments)} adjustments for next iteration")
                
                refinement_log[-1]['refinements'] = adjustments
                
            except Exception as e:
                print(f"   ⚠️ Refinement parsing failed: {e}")
                break
    
        # FINAL: Save updated match using BEST analysis
        print(f"\n💾 Saving updated match...")
        print(f"   🏆 Using best analysis with score: {best_score}/100")
        
        if best_analysis:
            detailed_analysis_dict = {
                'score_breakdown': best_analysis.get('score_breakdown', {}),
                'matched_skills': best_analysis.get('matched_skills', []),
                'missing_skills': best_analysis.get('missing_skills', []),
                'matched_bullets': best_analysis.get('matched_bullets', []),
                'strengths': best_analysis.get('strengths', []),
                'gaps': best_analysis.get('gaps', []),
                'improvement_suggestions': best_analysis.get('improvement_suggestions', []),
                'summary': best_analysis.get('summary', best_analysis.get('reason', '')),
                'confidence': best_analysis.get('confidence', 0.85),
                'confidence_reasoning': best_analysis.get('confidence_reasoning', f'Analysis refined with quality score: {quality_score}/100'),
                'uncertainty_factors': best_analysis.get('uncertainty_factors', [])
            }
            
            db_service.save_match(
                resume_id=int(resume_id),
                job_id=int(job_id),
                score=best_analysis['score'],
                reason=best_analysis.get('reason', 'Improved analysis'),
                confidence=best_analysis.get('confidence', 0.85),
                detailed_analysis=json.dumps(detailed_analysis_dict)
            )
            
            print(f"   ✅ Match updated in database")
        
        return json.dumps({
            'final_score': best_score,
            'iterations': len(refinement_log),
            'final_quality': quality_score,
            'refinement_log': refinement_log,
            'job_title': job.get('title', 'Unknown'),
            'resume_name': resume.get('name', 'Unknown')
        }, indent=2)

    # =========================================================================
    # DEEP ANALYZE WITH GUIDANCE
    # =========================================================================
    async def _deep_analyze_with_guidance(
        self,
        resume_text: str,
        job: dict,
        guidance: str = "",
        previous_analysis: dict = None
    ) -> dict:
        """Deep analysis method with optional guidance and refinement mode."""
        
        existing_score = previous_analysis.get('score', 0) if previous_analysis else 0
        
        if previous_analysis and guidance:
            # Refinement mode
            return await self._refine_existing_analysis(resume_text, job, previous_analysis, guidance)
        else:
            # Initial analysis mode
            return await self._full_deep_analyze(resume_text, job, guidance)

    # =========================================================================
    # FULL DEEP ANALYZE (initial)
    # =========================================================================
    async def _full_deep_analyze(self, resume_text: str, job: dict, guidance: str = "") -> dict:
        """Full deep analysis without refinement mode."""
        
        guidance_section = ""
        if guidance:
            guidance_section = f"\n\n🔧 IMPORTANT - Apply these refinements:\n{guidance}"
        
        system_prompt = f"""You are an expert resume matcher. Perform semantic analysis.{guidance_section}

YOU MUST COPY EXACT TEXT FROM DOCUMENTS.

CRITICAL: Return ONLY valid JSON."""

        user_message = f"""**RESUME:**
{resume_text[:4000]}

**JOB:**
Title: {job.get('title', 'N/A')}
Company: {job.get('company', 'N/A')}
{job.get('description', 'N/A')[:3500]}

Format:
{{
  "overall_score": 85,
  "confidence": 0.82,
  "confidence_reasoning": "explanation",
  "uncertainty_factors": ["factor1"],
  "score_breakdown": {{
    "skills_match": 90,
    "experience_match": 80,
    "requirements_match": 85,
    "education_match": 75
  }},
  "matched_bullets": [
    {{
      "job_requirement": "exact text",
      "job_highlight_text": "exact text",
      "resume_bullet": "exact text",
      "resume_highlight_text": "exact text",
      "match_strength": "strong/moderate/weak",
      "explanation": "why"
    }}
  ],
  "matched_skills": ["skill1"],
  "missing_skills": ["skill2"],
  "strengths": ["strength1"],
  "gaps": ["gap1"],
  "improvement_suggestions": ["suggestion1"],
  "summary": "assessment"
}}"""

        full_prompt = f"{system_prompt}\n\n{user_message}"

        llm_start_time = time.time()
        result = await self.kernel.invoke_prompt(full_prompt)
        latency_ms = (time.time() - llm_start_time) * 1000
        result_str = str(result).strip()
        
        prompt_tokens = len(full_prompt) // 4
        completion_tokens = len(result_str) // 4
        
        # LLM Judge evaluation (50% sampling)
        quality_eval = await maybe_judge_response(
            self.kernel,
            "deep_analyze_with_guidance",
            full_prompt[:5000],
            result_str[:5000],
            context={
                "job_title": job.get('title', 'Unknown'),
                "mode": "initial_analysis"
            }
        )
        
        # Track in Observatory - SINGLE CALL with all data
        track_llm_call(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            agent_name="SelfImprovingMatch",
            operation="deep_analyze_with_guidance",
            
            # Prompt analysis
            system_prompt=system_prompt,
            user_message=user_message[:5000],
            response_text=result_str[:5000],
            prompt_metadata=DEEP_ANALYZE_GUIDANCE_META,
            
            # Quality evaluation
            quality_evaluation=quality_eval,
            
            # Metadata
            metadata={
                "job_id": job.get('id'),
                "job_title": job.get('title', 'Unknown'),
                "iteration_mode": "initial",
                "has_guidance": bool(guidance),
                "judged": quality_eval is not None
            }
        )

        print(f"   📊 Tracked deep analysis: {latency_ms:.0f}ms, ~{prompt_tokens + completion_tokens} tokens")
        
        # Parse JSON
        if '```json' in result_str:
            result_str = result_str.split('```json')[1].split('```')[0].strip()
        elif '```' in result_str:
            result_str = result_str.split('```')[1].split('```')[0].strip()
        
        start_idx = result_str.find('{')
        end_idx = result_str.rfind('}')
        if start_idx != -1 and end_idx != -1:
            result_str = result_str[start_idx:end_idx+1]
        
        try:
            parsed = json.loads(result_str)
            
            return {
                'score': int(parsed.get('overall_score', parsed.get('score', 0))),
                'confidence': float(parsed.get('confidence', 0.5)),
                'confidence_reasoning': parsed.get('confidence_reasoning', ''),
                'uncertainty_factors': parsed.get('uncertainty_factors', []),
                'reason': parsed.get('summary', ''),
                'matched_skills': parsed.get('matched_skills', []),
                'missing_skills': parsed.get('missing_skills', []),
                'matched_bullets': parsed.get('matched_bullets', []),
                'strengths': parsed.get('strengths', []),
                'gaps': parsed.get('gaps', []),
                'improvement_suggestions': parsed.get('improvement_suggestions', []),
                'summary': parsed.get('summary', ''),
                'score_breakdown': parsed.get('score_breakdown', {})
            }
        
        except Exception as e:
            return {
                'score': 0,
                'confidence': 0.3,
                'reason': f"Analysis failed: {str(e)}",
                'matched_skills': [],
                'missing_skills': [],
                'matched_bullets': [],
                'strengths': [],
                'gaps': [],
                'improvement_suggestions': [],
                'summary': 'Error',
                'score_breakdown': {},
                '_parsing_failed': True
            }

    # =========================================================================
    # REFINE EXISTING ANALYSIS
    # =========================================================================
    async def _refine_existing_analysis(self, resume_text: str, job: dict, previous_analysis: dict, guidance: str) -> dict:
        """Refinement mode - improve existing analysis."""
        
        existing_score = previous_analysis.get('score', 0)
        
        system_prompt = """You are refining a previous analysis based on feedback.

YOUR TASK:
1. Address each guidance point
2. Keep good matches
3. Improve or remove weak matches
4. Add missing connections
5. Adjust score appropriately

COPY EXACT TEXT in highlight fields.

CRITICAL: Return ONLY valid JSON."""

        user_message = f"""**PREVIOUS ANALYSIS:**
Score: {existing_score}/100
Matched Skills: {', '.join(previous_analysis.get('matched_skills', [])[:10])}
Missing Skills: {', '.join(previous_analysis.get('missing_skills', [])[:5])}
Matched Bullets: {len(previous_analysis.get('matched_bullets', []))}

**REFINEMENT GUIDANCE:**
{guidance}

**RETURN FORMAT:**
{{
  "overall_score": 85,
  "confidence": 0.82,
  "confidence_reasoning": "explanation",
  "uncertainty_factors": ["factor1"],
  "score_breakdown": {{
    "skills_match": 90,
    "experience_match": 80,
    "requirements_match": 85,
    "education_match": 75
  }},
  "matched_bullets": [
    {{
      "job_requirement": "exact text",
      "job_highlight_text": "exact text",
      "resume_bullet": "exact text",
      "resume_highlight_text": "exact text",
      "match_strength": "strong/moderate/weak",
      "explanation": "detailed",
      "refinement_note": "what changed"
    }}
  ],
  "matched_skills": ["skill1"],
  "missing_skills": ["skill2"],
  "strengths": ["strength1"],
  "gaps": ["gap1"],
  "improvement_suggestions": ["suggestion1"],
  "summary": "Brief summary of refinements",
  "bullets_kept": 5,
  "bullets_improved": 2,
  "bullets_added": 1
}}

**RESUME:**
{resume_text[:3000]}

**JOB:**
Title: {job.get('title', 'N/A')}
Company: {job.get('company', 'N/A')}
{job.get('description', 'N/A')[:2500]}"""

        full_prompt = f"{system_prompt}\n\n{user_message}"

        llm_start_time = time.time()
        result = await self.kernel.invoke_prompt(full_prompt)
        latency_ms = (time.time() - llm_start_time) * 1000
        result_str = str(result).strip()
        
        prompt_tokens = len(full_prompt) // 4
        completion_tokens = len(result_str) // 4
        
        # LLM Judge evaluation (50% sampling)
        quality_eval = await maybe_judge_response(
            self.kernel,
            "deep_analyze_with_guidance",
            full_prompt[:5000],
            result_str[:5000],
            context={
                "job_title": job.get('title', 'Unknown'),
                "mode": "refinement",
                "previous_score": existing_score
            }
        )
        
        # Track in Observatory - SINGLE CALL with all data
        track_llm_call(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            agent_name="SelfImprovingMatch",
            operation="refine_analysis",
            
            # Prompt analysis
            system_prompt=system_prompt,
            user_message=user_message[:5000],
            response_text=result_str[:5000],
            prompt_metadata=REFINE_ANALYSIS_META,
            
            # Quality evaluation
            quality_evaluation=quality_eval,
            
            # Metadata
            metadata={
                "job_id": job.get('id'),
                "job_title": job.get('title', 'Unknown'),
                "previous_score": existing_score,
                "iteration_mode": "refinement",
                "judged": quality_eval is not None
            }
        )
        
        print(f"   📊 Tracked refinement: {latency_ms:.0f}ms, ~{prompt_tokens + completion_tokens} tokens")
        
        # Parse JSON
        if '```json' in result_str:
            result_str = result_str.split('```json')[1].split('```')[0].strip()
        elif '```' in result_str:
            result_str = result_str.split('```')[1].split('```')[0].strip()
        
        start_idx = result_str.find('{')
        end_idx = result_str.rfind('}')
        if start_idx != -1 and end_idx != -1:
            result_str = result_str[start_idx:end_idx+1]
        
        try:
            parsed = json.loads(result_str)
            score = parsed.get('overall_score', parsed.get('score', existing_score))
            
            return {
                'score': int(score),
                'confidence': float(parsed.get('confidence', previous_analysis.get('confidence', 0.5))),
                'confidence_reasoning': parsed.get('confidence_reasoning', ''),
                'uncertainty_factors': parsed.get('uncertainty_factors', []),
                'reason': parsed.get('summary', previous_analysis.get('reason', '')),
                'matched_skills': parsed.get('matched_skills', previous_analysis.get('matched_skills', [])),
                'missing_skills': parsed.get('missing_skills', previous_analysis.get('missing_skills', [])),
                'matched_bullets': parsed.get('matched_bullets', previous_analysis.get('matched_bullets', [])),
                'strengths': parsed.get('strengths', previous_analysis.get('strengths', [])),
                'gaps': parsed.get('gaps', previous_analysis.get('gaps', [])),
                'improvement_suggestions': parsed.get('improvement_suggestions', previous_analysis.get('improvement_suggestions', [])),
                'summary': parsed.get('summary', previous_analysis.get('summary', '')),
                'score_breakdown': parsed.get('score_breakdown', previous_analysis.get('score_breakdown', {})),
                '_refinement_stats': {
                    'bullets_kept': parsed.get('bullets_kept', 0),
                    'bullets_improved': parsed.get('bullets_improved', 0),
                    'bullets_added': parsed.get('bullets_added', 0)
                }
            }
        
        except Exception as e:
            print(f"   ❌ Refinement parsing failed: {e}")
            return previous_analysis

    # =========================================================================
    # AI CRITIC - Reviews single match quality
    # =========================================================================
    async def _critique_single_match(self, analysis: dict, resume: dict, job: dict) -> str:
        """AI reviews a single match analysis."""
        
        matched_bullets = analysis.get('matched_bullets', [])
        
        system_prompt = """You are a quality control expert reviewing a single job-resume match analysis.

Review This Match For:
1. Is the score accurate given the matches?
2. Are strong/moderate/weak classifications correct?
3. Any key requirements missed?
4. Is reasoning specific enough?

CRITICAL: Return ONLY valid JSON. No markdown, no explanations."""

        user_message = f"""**Job:** {job.get('title', 'Unknown')} at {job.get('company', 'Unknown')}
**Match Score:** {analysis.get('score', 0)}/100

**Matched Skills:** {', '.join(analysis.get('matched_skills', [])[:10])}
**Missing Skills:** {', '.join(analysis.get('missing_skills', [])[:5])}

**Matched Bullets:** {len(matched_bullets)} matches found
- Strong: {len([b for b in matched_bullets if b.get('match_strength') == 'strong'])}
- Moderate: {len([b for b in matched_bullets if b.get('match_strength') == 'moderate'])}
- Weak: {len([b for b in matched_bullets if b.get('match_strength') == 'weak'])}

**JSON Format:**
{{
  "overall_quality": 75,
  "issues": [
    {{
      "issue": "Score seems too high given weak matches",
      "severity": "high"
    }}
  ],
  "strengths": ["What's good about this analysis"],
  "weaknesses": ["What needs improvement"],
  "recommendations": ["Specific improvements"]
}}"""

        full_prompt = f"{system_prompt}\n\n{user_message}"

        llm_start_time = time.time()
        result = await self.kernel.invoke_prompt(full_prompt)
        latency_ms = (time.time() - llm_start_time) * 1000
        result_str = str(result).strip()
        
        prompt_tokens = len(full_prompt) // 4
        completion_tokens = len(result_str) // 4
        
        # LLM Judge evaluation (50% sampling)
        quality_eval = await maybe_judge_response(
            self.kernel,
            "critique_match",
            full_prompt,
            result_str,
            context={
                "job_title": job.get('title', 'Unknown'),
                "match_score": analysis.get('score', 0)
            }
        )
        
        # Track in Observatory - SINGLE CALL with all data
        track_llm_call(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            agent_name="SelfImprovingMatch",
            operation="critique_match",
            
            # Prompt analysis
            system_prompt=system_prompt,
            user_message=user_message,
            response_text=result_str,
            prompt_metadata=CRITIQUE_MATCH_META,
            
            # Quality evaluation
            quality_evaluation=quality_eval,
            
            # Metadata
            metadata={
                "job_id": job.get('id'),
                "job_title": job.get('title', 'Unknown'),
                "match_score": analysis.get('score', 0),
                "num_matched_bullets": len(matched_bullets),
                "judged": quality_eval is not None
            }
        )
        
        print(f"   📊 Tracked critique: {latency_ms:.0f}ms, ~{prompt_tokens + completion_tokens} tokens")
        
        # Clean JSON
        if '```json' in result_str:
            result_str = result_str.split('```json')[1].split('```')[0].strip()
        
        start_idx = result_str.find('{')
        end_idx = result_str.rfind('}')
        if start_idx != -1 and end_idx != -1:
            result_str = result_str[start_idx:end_idx+1]
        
        return result_str

    # =========================================================================
    # GENERATE REFINEMENTS (no judge - low value operation)
    # =========================================================================
    async def _generate_refinements_for_single_match(
        self,
        critique: dict,
        analysis: dict,
        resume: dict,
        job: dict
    ) -> str:
        """Generate refinements for this specific match."""
        
        system_prompt = """You are improving a job-resume match analysis.

Your Task: Generate specific guidance to improve THIS match analysis.

CRITICAL: Return ONLY valid JSON."""

        user_message = f"""**Current Analysis Issues:**
{json.dumps(critique.get('weaknesses', []), indent=2)}

**Recommendations:**
{json.dumps(critique.get('recommendations', []), indent=2)}

**JSON Format:**
{{
  "adjustments": [
    {{
      "area": "scoring",
      "change": "Lower score by 10 points due to experience gap",
      "reason": "Candidate has 3 years, job requires 5"
    }}
  ],
  "focus_areas": [
    "Re-evaluate years of experience match",
    "Check if leadership requirement is met"
  ]
}}"""

        full_prompt = f"{system_prompt}\n\n{user_message}"

        llm_start_time = time.time()
        result = await self.kernel.invoke_prompt(full_prompt)
        latency_ms = (time.time() - llm_start_time) * 1000
        result_str = str(result).strip()
        
        prompt_tokens = len(full_prompt) // 4
        completion_tokens = len(result_str) // 4
        
        # Track in Observatory - NO JUDGE (low value operation)
        track_llm_call(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            agent_name="SelfImprovingMatch",
            operation="generate_refinements",
            
            # Prompt analysis
            system_prompt=system_prompt,
            user_message=user_message,
            response_text=result_str,
            prompt_metadata=GENERATE_REFINEMENTS_META,
            
            # Metadata
            metadata={
                "job_id": job.get('id'),
                "job_title": job.get('title', 'Unknown'),
                "match_score": analysis.get('score', 0),
                "num_weaknesses": len(critique.get('weaknesses', []))
            }
        )
        
        print(f"   📊 Tracked refinement generation: {latency_ms:.0f}ms, ~{prompt_tokens + completion_tokens} tokens")
        
        # Clean JSON
        if '```json' in result_str:
            result_str = result_str.split('```json')[1].split('```')[0].strip()
        
        start_idx = result_str.find('{')
        end_idx = result_str.rfind('}')
        if start_idx != -1 and end_idx != -1:
            result_str = result_str[start_idx:end_idx+1]
        
        return result_str