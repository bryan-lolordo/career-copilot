#!/usr/bin/env python3
"""
Observatory Comprehensive Test Data Generator

Generates synthetic but realistic LLM call data to populate ALL Observatory fields
and trigger all 7 Story conditions.

Usage:
    python generate_test_data.py --sessions 5 --calls-per-session 5
    python generate_test_data.py --help
"""

import argparse
import json
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add parent directory to path to import observatory
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from observatory.collector import ObservatoryCollector
from observatory.storage import Storage

# =============================================================================
# CONFIGURATION - Realistic Career Copilot Patterns
# =============================================================================

# All 6 agents with their operations (including LLMJudge)
AGENTS_AND_OPERATIONS = {
    "JobPlugin": [
        "search_jobs",
        "get_job_details", 
        "save_job",
        "get_saved_jobs",
        "remove_saved_job"
    ],
    "DatabaseQuery": [
        "query_database",
        "get_resume_by_id",
        "get_job_by_id",
        "get_all_resumes",
        "get_all_jobs",
        "delete_resume",
        "delete_job"
    ],
    "ResumeMatching": [
        "match_resume_to_job",
        "match_resume_to_jobs",
        "explain_match_score"
    ],
    "ResumeTailoring": [
        "tailor_resume",
        "suggest_improvements",
        "generate_cover_letter"
    ],
    "SelfImprovingMatch": [
        "analyze_match_performance",
        "suggest_improvements",
        "update_matching_criteria"
    ],
    "LLMJudge": [
        "evaluate_quality",
        "check_hallucination",
        "score_response"
    ]
}

# Realistic token ranges per operation type
OPERATION_PATTERNS = {
    # Job operations - moderate, API-based
    "search_jobs": {"prompt": (400, 800), "completion": (200, 600), "latency": (1.0, 3.0)},
    "get_job_details": {"prompt": (300, 500), "completion": (400, 900), "latency": (0.8, 2.0)},
    "save_job": {"prompt": (200, 400), "completion": (50, 150), "latency": (0.5, 1.2)},
    "get_saved_jobs": {"prompt": (250, 400), "completion": (300, 700), "latency": (0.7, 1.5)},
    "remove_saved_job": {"prompt": (200, 350), "completion": (50, 100), "latency": (0.4, 1.0)},
    
    # Database queries - lightweight
    "query_database": {"prompt": (250, 400), "completion": (100, 300), "latency": (0.5, 1.5)},
    "get_resume_by_id": {"prompt": (200, 350), "completion": (800, 1500), "latency": (0.6, 1.3)},
    "get_job_by_id": {"prompt": (200, 350), "completion": (400, 800), "latency": (0.5, 1.2)},
    "get_all_resumes": {"prompt": (250, 400), "completion": (200, 500), "latency": (0.7, 1.8)},
    "get_all_jobs": {"prompt": (250, 400), "completion": (300, 700), "latency": (0.8, 2.0)},
    "delete_resume": {"prompt": (200, 300), "completion": (50, 100), "latency": (0.4, 1.0)},
    "delete_job": {"prompt": (200, 300), "completion": (50, 100), "latency": (0.4, 1.0)},
    
    # Resume matching - heavy, complex
    "match_resume_to_job": {"prompt": (1500, 2500), "completion": (500, 1200), "latency": (3.0, 8.0)},
    "match_resume_to_jobs": {"prompt": (2000, 4000), "completion": (1000, 2500), "latency": (5.0, 15.0)},
    "explain_match_score": {"prompt": (1200, 2000), "completion": (600, 1500), "latency": (2.5, 6.0)},
    
    # Resume tailoring - very heavy
    "tailor_resume": {"prompt": (2500, 5000), "completion": (1500, 3500), "latency": (5.0, 15.0)},
    "suggest_improvements": {"prompt": (2000, 3500), "completion": (1000, 2500), "latency": (4.0, 10.0)},
    "generate_cover_letter": {"prompt": (1800, 3000), "completion": (1200, 2800), "latency": (4.5, 12.0)},
    
    # Self-improving - moderate
    "analyze_match_performance": {"prompt": (1000, 2000), "completion": (600, 1500), "latency": (2.0, 5.0)},
    "update_matching_criteria": {"prompt": (800, 1500), "completion": (400, 1000), "latency": (1.5, 4.0)},
    
    # LLMJudge - lightweight, fast
    "evaluate_quality": {"prompt": (800, 1200), "completion": (100, 200), "latency": (0.8, 2.0)},
    "check_hallucination": {"prompt": (700, 1100), "completion": (80, 180), "latency": (0.7, 1.8)},
    "score_response": {"prompt": (900, 1300), "completion": (90, 190), "latency": (0.9, 2.2)},
}

# Model configurations
MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]
MODEL_COSTS = {
    "gpt-4o-mini": {"input": 0.150 / 1_000_000, "output": 0.600 / 1_000_000},  # Per token
    "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    "gpt-4-turbo": {"input": 10.00 / 1_000_000, "output": 30.00 / 1_000_000},
}

# Sample prompts and completions for realism
SAMPLE_SYSTEM_PROMPTS = [
    "You are an AI career assistant helping users find jobs and optimize their resumes.",
    "You are a helpful AI agent specialized in job searching and resume matching.",
    "You are an expert career coach AI that helps professionals advance their careers.",
]

SAMPLE_USER_MESSAGES = [
    "Find me AI engineer jobs in Chicago",
    "Match my resume to this job posting",
    "Help me improve my resume for this position",
    "Show me my top job matches",
    "Explain why I matched well with this job",
]

# =============================================================================
# DATA GENERATOR
# =============================================================================

class TestDataGenerator:
    """Generates comprehensive test data for Observatory."""
    
    def __init__(self, db_path: str = None):
        self.storage = Storage(db_path) if db_path else Storage()
        self.collector = ObservatoryCollector(
            storage=self.storage,
            project_name="Career Copilot - Test Data"
        )
        
    def generate_session(self, session_num: int, calls_per_session: int = 5) -> str:
        """Generate a single test session with multiple calls."""
        
        session_id = str(uuid.uuid4())
        start_time = datetime.now() - timedelta(hours=random.randint(1, 48))
        
        print(f"\n📦 Generating Session {session_num}: {session_id[:8]}...")
        print(f"   Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Calls to generate: {calls_per_session}")
        
        # Track session metadata manually (since we're generating synthetic data)
        total_tokens = 0
        total_cost = 0
        
        for call_num in range(1, calls_per_session + 1):
            call_data = self.generate_call(
                session_id=session_id,
                turn_number=call_num,
                call_num=call_num,
                timestamp=start_time + timedelta(seconds=call_num * 10)
            )
            
            total_tokens += call_data["total_tokens"]
            total_cost += call_data["cost_usd"]
        
        end_time = start_time + timedelta(seconds=calls_per_session * 10 + 30)
        
        # Create session record
        self.storage.create_session(
            session_id=session_id,
            project_name="Career Copilot - Test Data",
            user_id=f"test_user_{random.randint(1, 5)}",
            start_time=start_time,
            end_time=end_time,
            total_calls=calls_per_session,
            total_tokens=total_tokens,
            total_cost=total_cost,
            metadata_json=json.dumps({
                "test_session": True,
                "session_number": session_num,
                "generated_at": datetime.now().isoformat()
            })
        )
        
        print(f"   ✅ Session complete: {calls_per_session} calls, {total_tokens:,} tokens, ${total_cost:.4f}")
        
        return session_id
    
    def generate_call(
        self, 
        session_id: str, 
        turn_number: int,
        call_num: int,
        timestamp: datetime
    ) -> Dict[str, Any]:
        """Generate a single LLM call with all fields populated."""
        
        # Select random agent and operation
        agent_name = random.choice(list(AGENTS_AND_OPERATIONS.keys()))
        operation = random.choice(AGENTS_AND_OPERATIONS[agent_name])
        
        # Get pattern for this operation
        pattern = OPERATION_PATTERNS.get(operation, {
            "prompt": (500, 1500),
            "completion": (200, 800),
            "latency": (1.0, 4.0)
        })
        
        # Generate tokens
        prompt_tokens = random.randint(*pattern["prompt"])
        completion_tokens = random.randint(*pattern["completion"])
        total_tokens = prompt_tokens + completion_tokens
        
        # Generate latency (with some outliers for Story 1)
        if random.random() < 0.1:  # 10% high latency outliers
            latency_seconds = random.uniform(8.0, 20.0)
        else:
            latency_seconds = random.uniform(*pattern["latency"])
        
        # Select model (mostly gpt-4o-mini, some gpt-4o)
        if random.random() < 0.85:
            model_name = "gpt-4o-mini"
        elif random.random() < 0.95:
            model_name = "gpt-4o"
        else:
            model_name = "gpt-4-turbo"
        
        # Calculate cost
        costs = MODEL_COSTS[model_name]
        cost_usd = (prompt_tokens * costs["input"]) + (completion_tokens * costs["output"])
        
        # Generate prompt content
        system_prompt = random.choice(SAMPLE_SYSTEM_PROMPTS)
        user_message = random.choice(SAMPLE_USER_MESSAGES)
        prompt_text = f"{system_prompt}\n\nUser: {user_message}"
        completion_text = f"[Generated response for {operation}]"
        
        # Caching (Story 2: Zero Cache Hits)
        cache_hit = random.random() < 0.15  # 15% cache hit rate (low for testing)
        cache_key = f"cache_{agent_name}_{operation}_{random.randint(1, 20)}" if random.random() < 0.5 else None
        
        # Routing (Story 3: Model Routing)
        routing_decision = None
        routing_reason = None
        original_model = None
        
        if random.random() < 0.3:  # 30% have routing decisions
            routing_decision = model_name
            original_model = random.choice(["gpt-4o", "gpt-4-turbo"])
            routing_reason = f"Routed to {model_name} based on complexity analysis"
        
        # Quality metrics (Story 4: Quality Issues)
        quality_score = None
        hallucination_flag = False
        error_flag = False
        error_message = None
        
        if agent_name == "LLMJudge" or random.random() < 0.3:  # Judge calls or 30% of others
            quality_score = random.uniform(6.5, 10.0)
            hallucination_flag = quality_score < 7.5 and random.random() < 0.1
        
        if random.random() < 0.05:  # 5% error rate
            error_flag = True
            error_message = random.choice([
                "Rate limit exceeded",
                "Timeout error",
                "Invalid API response",
                "Context length exceeded"
            ])
        
        # Token imbalance (Story 5)
        # Some operations naturally have high prompt:completion ratios
        
        # Temperature and other parameters
        temperature = random.choice([0.0, 0.3, 0.7, 1.0])
        max_tokens = random.choice([500, 1000, 2000, 4000])
        top_p = random.choice([0.9, 0.95, 1.0])
        
        # Custom metadata
        custom_metadata = {
            "test_generated": True,
            "call_number": call_num,
            "agent": agent_name,
            "operation": operation,
        }
        
        # Create the call record
        call_data = {
            "session_id": session_id,
            "conversation_id": session_id,
            "turn_number": turn_number,
            "timestamp": timestamp,
            "project_name": "Career Copilot - Test Data",
            "agent_name": agent_name,
            "operation_name": operation,
            "model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stop_sequences": None,
            "is_streaming": False,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "latency_seconds": latency_seconds,
            "cost_usd": cost_usd,
            "prompt_text": prompt_text,
            "completion_text": completion_text,
            "system_prompt": system_prompt,
            "user_message": user_message,
            "raw_request_json": json.dumps({
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            }),
            "raw_response_json": json.dumps({
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "model": model_name,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
            }),
            "cache_hit": cache_hit,
            "cache_key": cache_key,
            "routing_decision": routing_decision,
            "routing_reason": routing_reason,
            "original_model": original_model,
            "quality_score": quality_score,
            "hallucination_flag": hallucination_flag,
            "error_flag": error_flag,
            "error_message": error_message,
            "custom_metadata_json": json.dumps(custom_metadata)
        }
        
        # Save to database
        self.storage.create_llm_call(**call_data)
        
        print(f"   📊 Call {call_num}: {agent_name}.{operation} | "
              f"{total_tokens:,} tokens | {latency_seconds:.2f}s | ${cost_usd:.4f}")
        
        return call_data
    
    def generate_all(self, num_sessions: int = 5, calls_per_session: int = 5):
        """Generate multiple test sessions."""
        print(f"\n{'='*70}")
        print(f"🎲 Observatory Test Data Generator")
        print(f"{'='*70}")
        print(f"Generating: {num_sessions} sessions × {calls_per_session} calls = {num_sessions * calls_per_session} total calls")
        print(f"Database: {self.storage.db_path}")
        
        for session_num in range(1, num_sessions + 1):
            self.generate_session(session_num, calls_per_session)
        
        print(f"\n{'='*70}")
        print(f"✅ Generation Complete!")
        print(f"{'='*70}")
        
        # Print summary
        total_sessions = self.storage.session.query(
            self.storage.Session
        ).count()
        total_calls = self.storage.session.query(
            self.storage.LLMCall
        ).count()
        
        print(f"\n📊 Database Summary:")
        print(f"   Total Sessions: {total_sessions}")
        print(f"   Total LLM Calls: {total_calls}")
        print(f"\n🎯 Ready to view in Observatory Dashboard!")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive test data for Observatory"
    )
    parser.add_argument(
        "--sessions", "-s",
        type=int,
        default=5,
        help="Number of sessions to generate (default: 5)"
    )
    parser.add_argument(
        "--calls-per-session", "-c",
        type=int,
        default=5,
        help="Number of calls per session (default: 5)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to observatory database (default: uses config)"
    )
    
    args = parser.parse_args()
    
    generator = TestDataGenerator(db_path=args.db_path)
    generator.generate_all(
        num_sessions=args.sessions,
        calls_per_session=args.calls_per_session
    )


if __name__ == "__main__":
    main()