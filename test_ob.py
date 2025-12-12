"""
Observatory Integration Test
Location: career-copilot/test_observatory_integration.py

Tests all plugins and verifies Observatory data fields are populated.

Run: python test_observatory_integration.py
"""

import asyncio
import os
import sys
import sqlite3
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# SETUP
# =============================================================================

print("=" * 60)
print("🧪 OBSERVATORY INTEGRATION TEST")
print("=" * 60)

# Test database path (separate from production)
TEST_DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "test_observatory.db")
)
os.environ['DATABASE_URL'] = f"sqlite:///{TEST_DB_PATH}"
os.environ['ENABLE_OBSERVATORY'] = 'true'

print(f"\n📁 Test database: {TEST_DB_PATH}")

# Clean up old test database
if os.path.exists(TEST_DB_PATH):
    os.remove(TEST_DB_PATH)
    print("🗑️  Removed old test database")


# =============================================================================
# IMPORT OBSERVATORY
# =============================================================================

print("\n🔌 Testing imports...")

try:
    from observatory_config import (
        obs,
        track_llm_call,
        start_session,
        end_session,
        create_prompt_breakdown,
        create_routing_decision,
        create_cache_metadata,
        create_prompt_metadata,
        judge,
        cache,
        router,
    )
    print("  ✅ observatory_config imports successful")
except ImportError as e:
    print(f"  ❌ observatory_config import failed: {e}")
    sys.exit(1)

try:
    from observatory import (
        Observatory,
        RoutingDecision,
        CacheMetadata,
        QualityEvaluation,
        PromptBreakdown,
        PromptMetadata,
    )
    print("  ✅ observatory SDK imports successful")
except ImportError as e:
    print(f"  ❌ observatory SDK import failed: {e}")
    sys.exit(1)


# =============================================================================
# MOCK DEPENDENCIES
# =============================================================================

def create_mock_kernel():
    """Create a mock Semantic Kernel that returns realistic responses."""
    kernel = MagicMock()
    
    async def mock_invoke_prompt(prompt):
        # Return different responses based on prompt content
        if "Score how well" in prompt or "quick" in prompt.lower():
            return MagicMock(__str__=lambda _: '''{
                "score": 85,
                "confidence": 0.78,
                "confidence_reasoning": "Good skills match",
                "uncertainty_factors": ["Experience level unclear"],
                "score_breakdown": {"skills_match": 90, "experience_match": 80},
                "reason_bullets": ["Strong Python skills", "AWS experience matches"]
            }''')
        elif "semantic analysis" in prompt or "deep" in prompt.lower():
            return MagicMock(__str__=lambda _: '''{
                "overall_score": 87,
                "confidence": 0.82,
                "confidence_reasoning": "Strong alignment",
                "uncertainty_factors": [],
                "score_breakdown": {"skills_match": 90, "experience_match": 85},
                "matched_bullets": [{"job_requirement": "Python", "resume_bullet": "5 years Python", "match_strength": "strong", "explanation": "Direct match"}],
                "matched_skills": ["Python", "AWS", "SQL"],
                "missing_skills": ["Kubernetes"],
                "strengths": ["Strong backend experience"],
                "gaps": ["No frontend experience"],
                "improvement_suggestions": ["Add K8s projects"],
                "summary": "Strong match for backend role"
            }''')
        elif "improved resume bullet" in prompt.lower() or "bullet point" in prompt.lower():
            return MagicMock(__str__=lambda _: '''{
                "suggestions": [
                    {"version": 1, "bullet": "Architected scalable microservices", "explanation": "Strong action verb"},
                    {"version": 2, "bullet": "Led team of 5 engineers", "explanation": "Shows leadership"},
                    {"version": 3, "bullet": "Reduced latency by 40%", "explanation": "Quantifiable impact"}
                ],
                "original_identified": "Built backend services"
            }''')
        elif "SQL" in prompt or "query" in prompt.lower():
            return MagicMock(__str__=lambda _: "SELECT * FROM jobs WHERE title LIKE '%Python%' LIMIT 10")
        else:
            return MagicMock(__str__=lambda _: '{"result": "ok"}')
    
    kernel.invoke_prompt = mock_invoke_prompt
    return kernel


def create_mock_db_service():
    """Create a mock database service."""
    db = MagicMock()
    
    db.list_all_resumes.return_value = [
        {"id": 1, "name": "John_Doe_Resume.pdf", "content": "Python developer with 5 years experience..."},
        {"id": 2, "name": "Jane_Smith_Resume.pdf", "content": "Data scientist specializing in ML..."},
    ]
    
    db.get_resume_by_id.return_value = {
        "id": 1,
        "name": "John_Doe_Resume.pdf",
        "content": "Python developer with 5 years experience in AWS, SQL, and microservices..."
    }
    
    db.get_most_recent_resume.return_value = db.get_resume_by_id.return_value
    
    db.get_all_jobs.return_value = [
        {"id": 1, "title": "Senior Python Developer", "company": "TechCorp", "location": "Chicago", "description": "Looking for Python expert...", "link": "https://example.com/job1"},
        {"id": 2, "title": "Data Engineer", "company": "DataCo", "location": "Remote", "description": "Build data pipelines...", "link": "https://example.com/job2"},
    ]
    
    db.get_job_by_id.return_value = db.get_all_jobs.return_value[0]
    
    db.save_match.return_value = True
    
    return db


def create_mock_memory():
    """Create a mock conversation memory."""
    memory = MagicMock()
    memory.context = MagicMock()
    memory.context.available_resumes = []
    memory.context.last_searched_jobs = []
    memory.get_recent_matches.return_value = [
        {"job_id": 1, "title": "Python Developer", "company": "TechCorp", "score": 85, "reason": "Good match"}
    ]
    return memory


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

async def test_track_llm_call_basic():
    """Test basic track_llm_call functionality."""
    print("\n📊 Testing basic track_llm_call...")
    
    track_llm_call(
        model_name="gpt-4o-mini",
        prompt_tokens=100,
        completion_tokens=50,
        latency_ms=500,
        agent_name="TestAgent",
        agent_role="analyst",
        operation="test_basic",
        success=True,
        prompt="Test prompt",
        response_text="Test response",
        metadata={"test": True}
    )
    print("  ✅ Basic track_llm_call completed")


async def test_track_llm_call_full():
    """Test track_llm_call with all Tier 1-3 fields."""
    print("\n📊 Testing full track_llm_call with all tiers...")
    
    # Create all tracking objects
    prompt_breakdown = create_prompt_breakdown(
        system_prompt="You are a helpful assistant.",
        system_prompt_tokens=10,
        user_message="Hello world",
        user_message_tokens=5,
    )
    
    routing_decision = create_routing_decision(
        chosen_model="gpt-4o-mini",
        alternative_models=["gpt-4o"],
        reasoning="Simple task",
        complexity_score=0.3,
        estimated_cost_savings=0.01
    )
    
    cache_metadata = create_cache_metadata(
        cache_hit=False,
        cache_key="test:key:123",
        cache_cluster_id="test_cluster"
    )
    
    prompt_metadata = create_prompt_metadata(
        template_id="test_template",
        version="1.0.0",
        compressible_sections=["context"],
        optimization_flags={"test": True}
    )
    
    track_llm_call(
        model_name="gpt-4o-mini",
        prompt_tokens=100,
        completion_tokens=50,
        latency_ms=500,
        agent_name="TestAgent",
        agent_role="analyst",
        operation="test_full",
        success=True,
        prompt="Full test prompt",
        response_text="Full test response",
        system_prompt="You are a test assistant.",
        user_message="Test user message",
        messages=[{"role": "system", "content": "test"}, {"role": "user", "content": "hello"}],
        routing_decision=routing_decision,
        cache_metadata=cache_metadata,
        prompt_breakdown=prompt_breakdown,
        prompt_metadata=prompt_metadata,
        prompt_variant_id="test_variant_a",
        test_dataset_id="test_dataset_001",
        metadata={"tier": "full", "test": True}
    )
    print("  ✅ Full track_llm_call completed")


async def test_resume_matching_plugin():
    """Test ResumeMatchingPlugin tracking."""
    print("\n📊 Testing ResumeMatchingPlugin...")
    
    try:
        from agents.plugins.ResumeMatchingPlugin import ResumeMatchingPlugin
        
        kernel = create_mock_kernel()
        db = create_mock_db_service()
        memory = create_mock_memory()
        
        plugin = ResumeMatchingPlugin(kernel, db, memory)
        
        # Test list_resumes
        result = await plugin.list_resumes()
        print(f"  ✅ list_resumes: {len(result)} chars")
        
        # Test select_resume_for_matching
        memory.context.available_resumes = db.list_all_resumes()
        result = await plugin.select_resume_for_matching("1")
        print(f"  ✅ select_resume_for_matching: {len(result)} chars")
        
        # Test quick score (internal method)
        job = db.get_all_jobs()[0]
        resume_text = db.get_resume_by_id(1)['content']
        result = await plugin._quick_score_job_match(resume_text, job)
        print(f"  ✅ _quick_score_job_match: score={result.get('score')}")
        
        # Test deep analyze (internal method)
        result = await plugin._deep_analyze_job_match(resume_text, job, 85)
        print(f"  ✅ _deep_analyze_job_match: score={result.get('score')}")
        
    except ImportError as e:
        print(f"  ⚠️ ResumeMatchingPlugin not importable: {e}")
    except Exception as e:
        print(f"  ❌ ResumeMatchingPlugin error: {e}")


async def test_resume_tailoring_plugin():
    """Test ResumeTailoringPlugin tracking."""
    print("\n📊 Testing ResumeTailoringPlugin...")
    
    try:
        from agents.plugins.ResumeTailoringPlugin import ResumeTailoringPlugin
        
        kernel = create_mock_kernel()
        memory = create_mock_memory()
        
        plugin = ResumeTailoringPlugin(kernel, memory)
        
        result = await plugin.improve_resume_bullet(
            resume_text="Python developer with 5 years experience...",
            job_description="Looking for senior Python developer...",
            job_title="Senior Python Developer",
            company="TechCorp",
            user_request="Make my experience sound more impactful"
        )
        print(f"  ✅ improve_resume_bullet: {len(result)} chars")
        
    except ImportError as e:
        print(f"  ⚠️ ResumeTailoringPlugin not importable: {e}")
    except Exception as e:
        print(f"  ❌ ResumeTailoringPlugin error: {e}")


async def test_query_database_plugin():
    """Test QueryDatabasePlugin tracking."""
    print("\n📊 Testing QueryDatabasePlugin...")
    
    try:
        from agents.plugins.QueryDatabasePlugin import DatabaseQueryPlugin
        
        kernel = create_mock_kernel()
        memory = create_mock_memory()
        
        plugin = DatabaseQueryPlugin(kernel, memory)
        
        result = await plugin.query_database_with_ai("Show me all Python jobs")
        print(f"  ✅ query_database_with_ai: {len(str(result))} chars")
        
    except ImportError as e:
        print(f"  ⚠️ QueryDatabasePlugin not importable: {e}")
    except Exception as e:
        print(f"  ❌ QueryDatabasePlugin error: {e}")


async def test_job_plugin():
    """Test JobPlugin tracking."""
    print("\n📊 Testing JobPlugin...")
    
    try:
        from agents.plugins.JobPlugin import JobPlugin
        
        context = MagicMock()
        context.last_searched_jobs = []
        
        plugin = JobPlugin(context=context)
        
        # Note: find_jobs makes real API calls, so we just test get_saved_jobs
        result = await plugin.get_saved_jobs(limit=5)
        print(f"  ✅ get_saved_jobs: {len(result)} chars")
        
    except ImportError as e:
        print(f"  ⚠️ JobPlugin not importable: {e}")
    except Exception as e:
        print(f"  ❌ JobPlugin error: {e}")


# =============================================================================
# VERIFY DATABASE
# =============================================================================

def verify_database():
    """Check what data was recorded in the Observatory database."""
    print("\n" + "=" * 60)
    print("🔍 VERIFYING DATABASE RECORDS")
    print("=" * 60)
    
    if not os.path.exists(TEST_DB_PATH):
        print("  ❌ Test database does not exist!")
        return
    
    conn = sqlite3.connect(TEST_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"\n📋 Tables found: {tables}")
    
    # Check llm_calls table
    if 'llm_calls' in tables:
        cursor.execute("SELECT COUNT(*) FROM llm_calls")
        count = cursor.fetchone()[0]
        print(f"\n📊 LLM Calls recorded: {count}")
        
        if count > 0:
            cursor.execute("SELECT * FROM llm_calls ORDER BY timestamp DESC LIMIT 5")
            rows = cursor.fetchall()
            
            print("\n📝 Sample records:")
            for row in rows:
                row_dict = dict(row)
                print(f"\n  Operation: {row_dict.get('operation', 'N/A')}")
                print(f"  Agent: {row_dict.get('agent_name', 'N/A')}")
                print(f"  Agent Role: {row_dict.get('agent_role', 'N/A')}")
                print(f"  Model: {row_dict.get('model_name', 'N/A')}")
                print(f"  Tokens: {row_dict.get('prompt_tokens', 0)} + {row_dict.get('completion_tokens', 0)}")
                print(f"  Latency: {row_dict.get('latency_ms', 0):.0f}ms")
                print(f"  Success: {row_dict.get('success', 'N/A')}")
                
            # Check field population
            print("\n📈 Field Population Check:")
            fields_to_check = [
                'agent_name', 'agent_role', 'operation', 'model_name',
                'prompt_tokens', 'completion_tokens', 'latency_ms',
                'prompt', 'response_text', 'success',
                'routing_decision', 'cache_metadata', 'quality_evaluation',
                'prompt_breakdown', 'prompt_metadata',
                'prompt_variant_id', 'test_dataset_id'
            ]
            
            for field in fields_to_check:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM llm_calls WHERE {field} IS NOT NULL AND {field} != ''")
                    populated = cursor.fetchone()[0]
                    status = "✅" if populated > 0 else "⚠️"
                    print(f"  {status} {field}: {populated}/{count} populated")
                except:
                    print(f"  ❓ {field}: column may not exist")
    
    # Check sessions table
    if 'sessions' in tables:
        cursor.execute("SELECT COUNT(*) FROM sessions")
        count = cursor.fetchone()[0]
        print(f"\n📊 Sessions recorded: {count}")
    
    conn.close()


# =============================================================================
# MAIN
# =============================================================================

async def main():
    """Run all tests."""
    
    # Basic tracking tests
    await test_track_llm_call_basic()
    await test_track_llm_call_full()
    
    # Plugin tests (may fail if plugins not in path)
    await test_resume_matching_plugin()
    await test_resume_tailoring_plugin()
    await test_query_database_plugin()
    await test_job_plugin()
    
    # Verify what was recorded
    verify_database()
    
    print("\n" + "=" * 60)
    print("✅ TEST COMPLETE")
    print("=" * 60)
    print(f"\nTest database saved at: {TEST_DB_PATH}")
    print("You can inspect it with: sqlite3 test_observatory.db")
    print("\nNext steps:")
    print("1. Fix any ❌ errors above")
    print("2. Check ⚠️ warnings for missing data")
    print("3. Run real workflows to collect baseline data")


if __name__ == "__main__":
    asyncio.run(main())