"""
LLM-as-a-Judge Quality Evaluation System - Career Copilot

Evaluates LLM response quality with 50% sampling on high-value operations.

Captures ALL QualityEvaluation fields:
- judge_score, judge_model, reasoning
- hallucination_flag, hallucination_details
- factual_error, evidence_cited
- failure_reason, improvement_suggestion
- confidence, criteria_scores
"""

import os
import json
import random
from typing import Optional, Dict
from observatory_config import (
    create_quality_evaluation, 
    QualityEvaluation,
    DEFAULT_MODEL
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Judge model - uses same model as main app from .env
JUDGE_MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")

# Operations to evaluate (high-value only)
JUDGE_OPERATIONS = {
    # Resume improvement
    "improve_bullet",
    "generate_change_report",
    
    # Expensive matching analysis
    "deep_analyze_job",
    "deep_analyze_with_guidance",
    "critique_match",
    
    # Chat interactions 
    "streamlit_chat",
    "cli_chat_message",
}

# Operations to skip (low value for quality evaluation)
SKIP_OPERATIONS = {
    "generate_sql",      # Just SQL generation
    "job_search",        # External API call
    "save_jobs",         # Database operation
    "list_resumes",      # Database operation
    "quick_score_job",   # Fast approximate scoring
}

# Sampling rate: evaluate 50% of judge-worthy calls
SAMPLE_RATE = 0.50

# Evaluation criteria and weights
EVALUATION_CRITERIA = {
    "relevance": 0.25,      # How relevant is the response to the query
    "accuracy": 0.25,       # Factual accuracy
    "helpfulness": 0.25,    # How actionable/useful
    "clarity": 0.15,        # Clear and well-structured
    "professionalism": 0.10 # Appropriate tone
}


# ============================================================================
# MAIN JUDGE FUNCTION (ASYNC)
# ============================================================================

async def maybe_judge_response(
    kernel,
    operation: str,
    prompt: str,
    response: str,
    enable_sampling: bool = True,
    context: Dict = None
) -> Optional[QualityEvaluation]:
    """
    Evaluate LLM response quality with sampling.
    
    Args:
        kernel: Semantic Kernel instance for making LLM calls
        operation: Operation name (e.g., "improve_bullet", "streamlit_chat")
        prompt: Original prompt sent to LLM
        response: LLM's response text
        enable_sampling: If False, judge every call (for testing)
        context: Optional context dict for more accurate evaluation
    
    Returns:
        QualityEvaluation object if judged, None if skipped
    """
    
    # Skip if not a judge-worthy operation
    if operation not in JUDGE_OPERATIONS:
        return None
    
    # Sample only X% of calls (unless sampling disabled)
    if enable_sampling and random.random() > SAMPLE_RATE:
        return None
    
    # Evaluate the response
    try:
        judge_prompt = create_judge_prompt(operation, prompt, response, context)
        result = await kernel.invoke_prompt(judge_prompt)
        result_str = str(result).strip()
        
        # Parse JSON response
        evaluation_data = parse_judge_response(result_str)
        
        # Extract criteria scores
        criteria_scores = extract_criteria_scores(evaluation_data)
        
        # Create QualityEvaluation with ALL fields
        quality = create_quality_evaluation(
            score=evaluation_data.get('score', 0),
            reasoning=evaluation_data.get('reasoning', ''),
            hallucination=evaluation_data.get('hallucination', False),
            factual_error=evaluation_data.get('factual_error', False),
            failure_reason=get_failure_reason(evaluation_data),
            improvement_suggestion=get_improvement_suggestion(evaluation_data),
            hallucination_details=evaluation_data.get('hallucination_details'),
            evidence_cited=evaluation_data.get('evidence_cited'),
            confidence=evaluation_data.get('confidence', 0.85),
            judge_model=JUDGE_MODEL,
            criteria_scores=criteria_scores
        )
        
        # Log evaluation
        log_evaluation(operation, quality)
        
        return quality
        
    except Exception as e:
        print(f"⚠️ Judge failed for {operation}: {e}")
        return None


# ============================================================================
# SYNCHRONOUS JUDGE (for non-async contexts)
# ============================================================================

def judge_response_sync(
    client,  # OpenAI/Azure client
    operation: str,
    prompt: str,
    response: str,
    enable_sampling: bool = True,
    context: Dict = None
) -> Optional[QualityEvaluation]:
    """
    Synchronous version of judge for non-async contexts.
    """
    # Skip if not a judge-worthy operation
    if operation not in JUDGE_OPERATIONS:
        return None
    
    # Sample only X% of calls
    if enable_sampling and random.random() > SAMPLE_RATE:
        return None
    
    try:
        judge_prompt = create_judge_prompt(operation, prompt, response, context)
        
        result = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": judge_prompt}],
            max_tokens=600,
            temperature=0.3
        )
        
        result_str = result.choices[0].message.content.strip()
        evaluation_data = parse_judge_response(result_str)
        criteria_scores = extract_criteria_scores(evaluation_data)
        
        quality = create_quality_evaluation(
            score=evaluation_data.get('score', 0),
            reasoning=evaluation_data.get('reasoning', ''),
            hallucination=evaluation_data.get('hallucination', False),
            factual_error=evaluation_data.get('factual_error', False),
            failure_reason=get_failure_reason(evaluation_data),
            improvement_suggestion=get_improvement_suggestion(evaluation_data),
            hallucination_details=evaluation_data.get('hallucination_details'),
            evidence_cited=evaluation_data.get('evidence_cited'),
            confidence=evaluation_data.get('confidence', 0.85),
            judge_model=JUDGE_MODEL,
            criteria_scores=criteria_scores
        )
        
        log_evaluation(operation, quality)
        return quality
        
    except Exception as e:
        print(f"⚠️ Judge failed for {operation}: {e}")
        return None


# ============================================================================
# JUDGE PROMPT TEMPLATES
# ============================================================================

def create_judge_prompt(operation: str, prompt: str, response: str, context: Dict = None) -> str:
    """Create evaluation prompt based on operation type."""
    
    # Truncate to avoid token limits
    prompt_preview = prompt[:1000] if len(prompt) > 1000 else prompt
    response_preview = response[:1500] if len(response) > 1500 else response
    
    # Operation-specific instructions
    task_context = get_task_context(operation)
    
    # Build context section if provided
    context_section = ""
    if context:
        context_section = f"\nADDITIONAL CONTEXT:\n{json.dumps(context, indent=2)[:500]}\n"
    
    # Main judge prompt with ALL fields
    judge_prompt = f"""You are an expert evaluator of AI career assistant responses.

TASK CONTEXT:
{task_context}

ORIGINAL USER REQUEST:
{prompt_preview}

AI RESPONSE:
{response_preview}
{context_section}

EVALUATION CRITERIA (rate each 0-10):
1. **Relevance** (25%): Does the response address the user's actual request?
2. **Accuracy** (25%): Is the information factually correct?
3. **Helpfulness** (25%): Is the response actionable and useful?
4. **Clarity** (15%): Is it well-structured and easy to understand?
5. **Professionalism** (10%): Is the tone appropriate for career advice?

CRITICAL CHECKS:
- **Hallucination**: Did the AI invent facts, companies, statistics, or specific details?
- **Factual Error**: Is there incorrect information about careers, skills, or industry standards?
- **Evidence**: Did the response cite sources or provide evidence for claims?

SCORING GUIDE:
- 9-10: Excellent - highly accurate, helpful, no issues
- 7-8: Good - mostly accurate, minor issues
- 5-6: Acceptable - some inaccuracies or vague advice
- 3-4: Poor - significant issues or unhelpful
- 0-2: Very poor - incorrect or harmful

Return ONLY valid JSON (no markdown, no code blocks):

{{
  "score": <number 0-10>,
  "reasoning": "<one sentence explaining the overall score>",
  "criteria_scores": {{
    "relevance": <0-10>,
    "accuracy": <0-10>,
    "helpfulness": <0-10>,
    "clarity": <0-10>,
    "professionalism": <0-10>
  }},
  "hallucination": <true or false>,
  "hallucination_details": "<specific made-up content, or null>",
  "factual_error": <true or false>,
  "factual_error_details": "<what was incorrect, or null>",
  "evidence_cited": <true or false>,
  "confidence": <0.0-1.0 your confidence in this evaluation>,
  "suggestions": ["<improvement 1>", "<improvement 2>"]
}}"""

    return judge_prompt


def get_task_context(operation: str) -> str:
    """Get operation-specific evaluation context."""
    
    contexts = {
        "improve_bullet": """
This is resume improvement advice. Evaluate for:
- Professional quality and appropriateness
- Actionable and specific suggestions
- Proper resume writing best practices (action verbs, quantified achievements)
- No generic or vague recommendations
""",
        "generate_change_report": """
This is a resume change report. Evaluate for:
- Clear before/after comparison
- Justified improvements
- Preserved key information
- Professional formatting
""",
        "deep_analyze_job": """
This is job-resume matching analysis. Evaluate for:
- Accurate skill assessment
- Realistic match score justification
- Identification of actual gaps vs hallucinated requirements
- Helpful and constructive feedback
""",
        "deep_analyze_with_guidance": """
This is guided job matching with self-improvement. Evaluate for:
- Application of previous feedback
- Improved accuracy over iterations
- Balanced assessment of strengths and gaps
""",
        "critique_match": """
This is a self-critique of matching analysis. Evaluate for:
- Honest assessment of prior analysis
- Identification of missed points
- Constructive suggestions for improvement
""",
        "streamlit_chat": """
This is conversational career advice via Streamlit. Evaluate for:
- Helpful and relevant information
- Professional and encouraging tone
- Factually accurate career guidance
- No making up specific companies or positions
""",
        "cli_chat_message": """
This is conversational career advice via CLI. Evaluate for:
- Helpful and relevant information
- Professional and encouraging tone
- Factually accurate career guidance
- No making up specific companies or positions
"""
    }
    
    return contexts.get(operation, """
This is AI-generated career assistance. Evaluate for:
- Accuracy and relevance
- Helpful and actionable content
- No hallucinated facts or recommendations
""")


# ============================================================================
# RESPONSE PARSING
# ============================================================================

def parse_judge_response(result_str: str) -> dict:
    """Parse judge response, handling various formats."""
    
    # Clean up markdown code blocks if present
    if '```json' in result_str:
        result_str = result_str.split('```json')[1].split('```')[0].strip()
    elif '```' in result_str:
        parts = result_str.split('```')
        if len(parts) >= 2:
            result_str = parts[1].strip()
    
    # Extract JSON object
    start_idx = result_str.find('{')
    end_idx = result_str.rfind('}')
    
    if start_idx == -1 or end_idx == -1:
        raise ValueError("No JSON object found in response")
    
    json_str = result_str[start_idx:end_idx+1]
    
    # Parse JSON
    evaluation = json.loads(json_str)
    
    # Validate required fields
    if 'score' not in evaluation:
        raise ValueError("Missing 'score' field in evaluation")
    
    # Ensure score is valid
    score = evaluation['score']
    if not isinstance(score, (int, float)) or score < 0 or score > 10:
        raise ValueError(f"Invalid score: {score}")
    
    return evaluation


def extract_criteria_scores(evaluation_data: dict) -> Optional[Dict[str, float]]:
    """Extract individual criteria scores from evaluation."""
    criteria = evaluation_data.get('criteria_scores', {})
    if not criteria:
        return None
    
    # Ensure all scores are floats
    return {k: float(v) for k, v in criteria.items() if isinstance(v, (int, float))}


def get_failure_reason(evaluation_data: dict) -> Optional[str]:
    """Determine failure reason category from evaluation."""
    
    if evaluation_data.get('hallucination', False):
        return "HALLUCINATION"
    elif evaluation_data.get('factual_error', False):
        return "FACTUAL_ERROR"
    elif evaluation_data.get('score', 10) < 3:
        return "VERY_LOW_QUALITY"
    elif evaluation_data.get('score', 10) < 5:
        return "LOW_QUALITY"
    else:
        return None


def get_improvement_suggestion(evaluation_data: dict) -> Optional[str]:
    """Extract improvement suggestion from evaluation."""
    suggestions = evaluation_data.get('suggestions', [])
    if suggestions:
        # Join multiple suggestions
        return " | ".join(str(s) for s in suggestions[:2])
    return None


def log_evaluation(operation: str, quality: QualityEvaluation):
    """Log evaluation result."""
    log_msg = f"📊 Judged {operation}: Score {quality.judge_score}/10"
    if quality.hallucination_flag:
        log_msg += " 🚨 HALLUCINATION"
    if getattr(quality, 'factual_error', False):
        log_msg += " ❌ FACTUAL_ERROR"
    if quality.criteria_scores:
        log_msg += f" [criteria: {quality.criteria_scores}]"
    print(log_msg)


# ============================================================================
# STATISTICS & REPORTING
# ============================================================================

def should_judge_operation(operation: str) -> bool:
    """Check if operation should be judged."""
    return operation in JUDGE_OPERATIONS


def get_judge_stats() -> dict:
    """Get judge configuration stats."""
    return {
        "sample_rate": SAMPLE_RATE,
        "judge_operations": sorted(JUDGE_OPERATIONS),
        "skip_operations": sorted(SKIP_OPERATIONS),
        "judge_model": JUDGE_MODEL,
        "evaluation_criteria": EVALUATION_CRITERIA,
    }


def add_judge_operation(operation: str):
    """Add an operation to the judge list."""
    JUDGE_OPERATIONS.add(operation)


def remove_judge_operation(operation: str):
    """Remove an operation from the judge list."""
    JUDGE_OPERATIONS.discard(operation)


def set_sample_rate(rate: float):
    """Set the sampling rate (0.0 to 1.0)."""
    global SAMPLE_RATE
    SAMPLE_RATE = max(0.0, min(1.0, rate))


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'JUDGE_MODEL',
    'JUDGE_OPERATIONS',
    'SKIP_OPERATIONS',
    'SAMPLE_RATE',
    'EVALUATION_CRITERIA',
    'maybe_judge_response',
    'judge_response_sync',
    'should_judge_operation',
    'get_judge_stats',
    'add_judge_operation',
    'remove_judge_operation',
    'set_sample_rate',
]