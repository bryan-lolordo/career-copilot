#!/usr/bin/env python3
"""
Observatory Integration Test
Generates test data for all dashboard pages to verify mapping.

Run: python test_observatory_integration.py
"""

import sys
import os

# Add project root to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from observatory_config import (
    track_llm_call,
    create_prompt_breakdown,
    create_prompt_metadata,
    create_quality_evaluation,
    create_routing_decision,
    create_cache_metadata,
    compute_content_hash,
    PromptBreakdown,
    PromptMetadata,
    QualityEvaluation,
    RoutingDecision,
    CacheMetadata
)

def test_basic_tracking():
    """Test 1: Basic LLM call - populates Overview page"""
    print("\n📊 Test 1: Basic LLM tracking (Overview page)")
    
    track_llm_call(
        prompt_tokens=150,
        completion_tokens=75,
        latency_ms=1234,
        agent_name="TestAgent",
        operation="basic_test",
        metadata={"test": "basic_tracking"}
    )
    print("   ✅ Basic call tracked")

def test_prompt_analysis():
    """Test 2: Prompt breakdown - populates Prompt Analysis page"""
    print("\n📝 Test 2: Prompt Analysis tracking")
    
    # Create prompt breakdown
    breakdown = create_prompt_breakdown(
        system_prompt="You are a helpful assistant for career advice.",
        user_message="What skills should I learn for AI jobs?",
        chat_history=[
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello! How can I help?"}
        ],
        response_text="Focus on Python, ML frameworks, and cloud platforms."
    )
    
    # Create prompt metadata (optimization_flags must be Dict[str, bool])
    metadata = create_prompt_metadata(
        template_id="career_advisor_v1",
        version="1.0.0",
        compressible_sections=["INSTRUCTIONS", "EXAMPLES"],
        optimization_flags={"uses_few_shot": True, "compressible": True},
        config_version="1.0"
    )
    
    track_llm_call(
        prompt_tokens=200,
        completion_tokens=100,
        latency_ms=890,
        agent_name="CareerAdvisor",
        operation="skill_recommendation",
        prompt_breakdown=breakdown,
        prompt_metadata=metadata,
        metadata={"test": "prompt_analysis"}
    )
    print("   ✅ Prompt breakdown tracked")
    print(f"      - System tokens: {breakdown.system_prompt_tokens if breakdown else 'N/A'}")
    print(f"      - History count: {breakdown.chat_history_count if breakdown else 'N/A'}")

def test_quality_evaluation():
    """Test 3: Quality evaluation - populates Quality page"""
    print("\n⭐ Test 3: Quality Evaluation tracking")
    
    quality = create_quality_evaluation(
        score=0.85,  # Changed from judge_score
        judge_model="gpt-4o-mini",
        reasoning="Response was accurate and helpful, minor formatting issues",
        hallucination=False,  # Changed from hallucination_flag
        hallucination_details=None,
        factual_error=False,
        evidence_cited=True,
        failure_reason=None,
        improvement_suggestion="Add more specific examples",  # Changed from list to string
        confidence=0.9,
        criteria_scores={
            "relevance": 0.9,
            "accuracy": 0.85,
            "helpfulness": 0.88,
            "clarity": 0.82,
            "professionalism": 0.90
        }
    )
    
    track_llm_call(
        prompt_tokens=300,
        completion_tokens=150,
        latency_ms=1567,
        agent_name="ResumeMatching",
        operation="deep_analyze_job",
        quality_evaluation=quality,
        metadata={"test": "quality_evaluation", "judged": True}
    )
    print("   ✅ Quality evaluation tracked")
    print(f"      - Judge score: {quality.judge_score if quality else 'N/A'}")
    print(f"      - Judge model: {quality.judge_model if quality else 'N/A'}")

def test_quality_with_hallucination():
    """Test 3b: Quality with hallucination flag"""
    print("\n⚠️  Test 3b: Quality with hallucination")
    
    quality = create_quality_evaluation(
        score=0.45,  # Changed from judge_score
        judge_model="gpt-4o-mini",
        reasoning="Response contained fabricated statistics",
        hallucination=True,  # Changed from hallucination_flag
        hallucination_details="Claimed '95% of AI jobs require PhD' - not supported by data",
        factual_error=True,
        evidence_cited=False,
        failure_reason="HALLUCINATION",
        improvement_suggestion="Remove unsupported claims",  # Changed from list to string
        confidence=0.7,
        criteria_scores={
            "relevance": 0.8,
            "accuracy": 0.3,
            "helpfulness": 0.5,
            "clarity": 0.7,
            "professionalism": 0.6
        }
    )
    
    track_llm_call(
        prompt_tokens=250,
        completion_tokens=180,
        latency_ms=2100,
        agent_name="ResumeTailoring",
        operation="improve_bullet",
        quality_evaluation=quality,
        metadata={"test": "hallucination_detected", "judged": True}
    )
    print("   ✅ Hallucination case tracked")

def test_routing_decision():
    """Test 4: Routing decision - populates Model Routing page"""
    print("\n🔀 Test 4: Model Routing tracking")
    
    routing = create_routing_decision(
        chosen_model="gpt-4o-mini",
        alternative_models=["gpt-4o", "gpt-4", "claude-3-sonnet"],
        reasoning="Simple task, efficient model sufficient",
        complexity_score=0.4,
        estimated_cost_savings=0.025,
        routing_strategy="complexity_based",
        model_scores={
            "gpt-4o-mini": 0.92,
            "gpt-4o": 0.95,
            "gpt-4": 0.88
        }
    )
    
    track_llm_call(
        prompt_tokens=180,
        completion_tokens=90,
        latency_ms=750,
        agent_name="DatabaseQuery",
        operation="generate_sql",
        routing_decision=routing,
        metadata={"test": "routing_decision"}
    )
    print("   ✅ Routing decision tracked")
    print(f"      - Chosen: {routing.chosen_model if routing else 'N/A'}")
    print(f"      - Savings: ${routing.estimated_cost_savings if routing else 'N/A'}")

def test_cache_metadata():
    """Test 5: Cache metadata - populates Cache Analysis page"""
    print("\n💾 Test 5: Cache Metadata tracking")
    
    prompt = "What are the top skills for AI engineering roles?"
    
    cache = create_cache_metadata(
        cache_hit=False,
        cache_key="skill_query_ai_engineering",
        cache_cluster_id="career_queries",
        similarity_score=0.0,
        cache_key_candidates=["ai_skills", "engineering_skills", "tech_skills"],
        dynamic_fields=["user_context", "timestamp"],
        content_hash=compute_content_hash(prompt),
        ttl_seconds=3600
    )
    
    track_llm_call(
        prompt_tokens=120,
        completion_tokens=200,
        latency_ms=1100,
        agent_name="CareerAdvisor",
        operation="skill_analysis",
        cache_metadata=cache,
        metadata={"test": "cache_miss"}
    )
    print("   ✅ Cache miss tracked")
    
    # Now simulate a cache hit
    cache_hit = create_cache_metadata(
        cache_hit=True,
        cache_key="skill_query_ai_engineering",
        cache_cluster_id="career_queries",
        similarity_score=0.95,
        cache_key_candidates=["ai_skills"],
        dynamic_fields=[],
        content_hash=compute_content_hash(prompt),
        ttl_seconds=3600
    )
    
    track_llm_call(
        prompt_tokens=120,
        completion_tokens=200,
        latency_ms=45,  # Much faster due to cache hit
        agent_name="CareerAdvisor",
        operation="skill_analysis",
        cache_metadata=cache_hit,
        metadata={"test": "cache_hit"}
    )
    print("   ✅ Cache hit tracked")
    print(f"      - Similarity: {cache_hit.similarity_score if cache_hit else 'N/A'}")

def test_full_integration():
    """Test 6: Full integration - all fields populated"""
    print("\n🎯 Test 6: Full Integration (all fields)")
    
    # All Tier 2 models populated
    breakdown = create_prompt_breakdown(
        system_prompt="You are an expert resume matcher...",
        user_message="Match my resume to this AI Engineer role at Google",
        chat_history=[
            {"role": "user", "content": "Match my resume"},
            {"role": "assistant", "content": "Which resume?"},
            {"role": "user", "content": "The first one"}
        ],
        response_text="Your resume scored 85/100 for this role..."
    )
    
    metadata = create_prompt_metadata(
        template_id="resume_matching_deep_analyze",
        version="1.0.0",
        compressible_sections=["EXAMPLES", "CONFIDENCE RULES"],
        optimization_flags={"detailed_analysis": True, "compressible": True},
        config_version="1.0"
    )
    
    quality = create_quality_evaluation(
        score=0.88,  # Changed from judge_score
        judge_model="gpt-4o-mini",
        reasoning="Comprehensive analysis with good skill matching",
        hallucination=False,  # Changed from hallucination_flag
        factual_error=False,
        evidence_cited=True,
        confidence=0.85,
        criteria_scores={
            "relevance": 0.92,
            "accuracy": 0.88,
            "helpfulness": 0.90,
            "clarity": 0.85,
            "professionalism": 0.87
        }
    )
    
    routing = create_routing_decision(
        chosen_model="gpt-4o-mini",
        alternative_models=["gpt-4o"],
        reasoning="Deep analysis task",
        complexity_score=0.7,
        estimated_cost_savings=0.015,
        routing_strategy="task_based",
        model_scores={"gpt-4o-mini": 0.88, "gpt-4o": 0.92}
    )
    
    cache = create_cache_metadata(
        cache_hit=False,
        cache_key="resume_job_match_123_456",
        cache_cluster_id="resume_matching",
        similarity_score=0.0,
        content_hash=compute_content_hash("resume+job"),
        ttl_seconds=7200
    )
    
    track_llm_call(
        prompt_tokens=500,
        completion_tokens=350,
        latency_ms=2500,
        agent_name="ResumeMatching",
        operation="deep_analyze_job",
        prompt_breakdown=breakdown,
        prompt_metadata=metadata,
        quality_evaluation=quality,
        routing_decision=routing,
        cache_metadata=cache,
        metadata={
            "test": "full_integration",
            "job_id": 123,
            "resume_id": 456,
            "judged": True
        }
    )
    print("   ✅ Full integration tracked")
    print("      - All 5 Tier 2 models populated")

def test_multiple_operations():
    """Test 7: Multiple operations for agent breakdown"""
    print("\n📈 Test 7: Multiple Operations (for charts)")
    
    operations = [
        ("ChatAgent", "streamlit_chat", 400, 200, 1500),
        ("ChatAgent", "streamlit_chat", 350, 180, 1400),
        ("ResumeMatching", "quick_score_job", 200, 100, 800),
        ("ResumeMatching", "quick_score_job", 220, 110, 850),
        ("ResumeMatching", "quick_score_job", 190, 95, 780),
        ("ResumeMatching", "deep_analyze_job", 450, 300, 2200),
        ("ResumeTailoring", "improve_bullet", 300, 250, 1800),
        ("DatabaseQuery", "generate_sql", 150, 50, 500),
        ("SelfImprovingMatch", "critique_match", 280, 200, 1600),
    ]
    
    for agent, op, prompt_tok, comp_tok, latency in operations:
        track_llm_call(
            prompt_tokens=prompt_tok,
            completion_tokens=comp_tok,
            latency_ms=latency,
            agent_name=agent,
            operation=op,
            metadata={"test": "multiple_operations"}
        )
    
    print(f"   ✅ Tracked {len(operations)} operations across multiple agents")

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("🧪 OBSERVATORY INTEGRATION TEST")
    print("=" * 60)
    
    # Check if models are available
    print("\n🔍 Checking model availability...")
    models_available = all([
        PromptBreakdown is not None,
        PromptMetadata is not None,
        QualityEvaluation is not None,
        RoutingDecision is not None,
        CacheMetadata is not None
    ])
    
    if models_available:
        print("   ✅ All Tier 2 models available")
    else:
        print("   ⚠️  Some models not available (check observatory_client)")
        print(f"      PromptBreakdown: {PromptBreakdown is not None}")
        print(f"      PromptMetadata: {PromptMetadata is not None}")
        print(f"      QualityEvaluation: {QualityEvaluation is not None}")
        print(f"      RoutingDecision: {RoutingDecision is not None}")
        print(f"      CacheMetadata: {CacheMetadata is not None}")
    
    # Run tests
    test_basic_tracking()
    test_prompt_analysis()
    test_quality_evaluation()
    test_quality_with_hallucination()
    test_routing_decision()
    test_cache_metadata()
    test_full_integration()
    test_multiple_operations()
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS COMPLETE")
    print("=" * 60)
    print("\nNow check your Observatory dashboard:")
    print("  📊 Overview     - Should show new calls, tokens, costs")
    print("  ⭐ Quality      - Should show judge scores, hallucination data")
    print("  📝 Prompts      - Should show breakdown, template versions")
    print("  🔀 Routing      - Should show model choices, savings")
    print("  💾 Cache        - Should show hits/misses, clusters")
    print("\nRun: streamlit run observatory_dashboard.py")

if __name__ == "__main__":
    run_all_tests()