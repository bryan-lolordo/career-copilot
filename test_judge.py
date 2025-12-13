#!/usr/bin/env python3
"""
Test LLM Judge with Semantic Kernel
Location: career-copilot/test_judge.py

Run: python test_judge.py
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# TEST CONFIGURATION
# =============================================================================

# Sample prompt/response pairs to evaluate
TEST_CASES = [
    {
        "name": "Good Response",
        "operation": "deep_analyze_job",
        "prompt": "Analyze how well my Python experience matches this Senior Software Engineer role at Google.",
        "response": """Based on your resume, here's my analysis:

**Match Score: 78/100**

**Strengths:**
- 5 years Python experience aligns with requirement
- Django/FastAPI experience matches backend needs
- AWS experience is relevant

**Gaps:**
- No Kubernetes experience mentioned (required)
- Missing system design examples

**Recommendations:**
1. Add Kubernetes projects or certifications
2. Highlight any distributed systems work""",
    },
    {
        "name": "Hallucinated Response",
        "operation": "deep_analyze_job",
        "prompt": "What jobs match my resume?",
        "response": """Based on my analysis of 50,000 job postings from LinkedIn's database 
(accessed December 2024), I found these perfect matches:

1. **Google** - Senior Engineer ($450K) - 98% match
2. **Meta** - Staff Engineer ($500K) - 95% match

According to the Bureau of Labor Statistics report from last week, 
you're in the top 1% of candidates.""",
    },
    {
        "name": "Low Quality Response",
        "operation": "improve_bullet",
        "prompt": "Improve this bullet point: 'Worked on projects'",
        "response": "ok",
    },
]


# =============================================================================
# MAIN TEST
# =============================================================================

async def main():
    print("=" * 60)
    print("🧪 LLM JUDGE TEST")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Step 1: Test imports
    # -------------------------------------------------------------------------
    print("\n📦 Step 1: Testing imports...")
    
    try:
        from observatory_config import judge, obs, DEFAULT_MODEL
        print(f"   ✅ observatory_config imported")
        print(f"   ✅ Judge model: {judge.judge_model}")
        print(f"   ✅ Sample rate: {judge.sample_rate}")
        print(f"   ✅ Operations: {judge.operations}")
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        print("\n   Make sure you're running from the career-copilot directory")
        return
    
    # -------------------------------------------------------------------------
    # Step 2: Test client detection
    # -------------------------------------------------------------------------
    print("\n🔍 Step 2: Testing client type detection...")
    
    try:
        from observatory.judge import detect_client_type, ClientType
        
        # Test with a mock kernel-like object
        class MockKernel:
            async def invoke_prompt(self, prompt):
                return "mock response"
        
        mock_kernel = MockKernel()
        detected = detect_client_type(mock_kernel)
        
        if detected == ClientType.SEMANTIC_KERNEL:
            print(f"   ✅ Semantic Kernel detection works: {detected}")
        else:
            print(f"   ❌ Detection failed, got: {detected}")
            
    except ImportError:
        print("   ⚠️  Could not import detect_client_type (old judge.py?)")
        print("   ⚠️  Make sure you've updated observatory/judge.py")
    
    # -------------------------------------------------------------------------
    # Step 3: Create real kernel
    # -------------------------------------------------------------------------
    print("\n🔧 Step 3: Creating Semantic Kernel...")
    
    try:
        from semantic_kernel import Kernel
        from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
        
        kernel = Kernel()
        
        chat_service = AzureChatCompletion(
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        )
        kernel.add_service(chat_service)
        
        print(f"   ✅ Kernel created with {DEFAULT_MODEL}")
        
        # Verify client detection on real kernel
        try:
            from observatory.judge import detect_client_type
            detected = detect_client_type(kernel)
            print(f"   ✅ Real kernel detected as: {detected}")
        except:
            pass
            
    except Exception as e:
        print(f"   ❌ Kernel creation failed: {e}")
        print("\n   Check your .env file has:")
        print("   - AZURE_OPENAI_ENDPOINT")
        print("   - AZURE_OPENAI_API_KEY")
        print("   - AZURE_OPENAI_DEPLOYMENT_NAME")
        return
    
    # -------------------------------------------------------------------------
    # Step 4: Test judge evaluation
    # -------------------------------------------------------------------------
    print("\n⚖️  Step 4: Testing judge evaluations...")
    print("-" * 60)
    
    results = []
    
    for i, test in enumerate(TEST_CASES, 1):
        print(f"\n   Test {i}: {test['name']}")
        print(f"   Operation: {test['operation']}")
        
        try:
            # Force evaluation (bypass sampling)
            quality_eval = await judge.maybe_evaluate(
                operation=test["operation"],
                prompt=test["prompt"],
                response=test["response"],
                llm_client=kernel,
                force=True,  # Always evaluate for testing
            )
            
            if quality_eval:
                results.append({"name": test["name"], "eval": quality_eval})
                
                print(f"   ✅ Score: {quality_eval.judge_score}/10")
                print(f"      Hallucination: {'🚨 YES' if quality_eval.hallucination_flag else '✅ No'}")
                print(f"      Factual Error: {'❌ YES' if quality_eval.factual_error else '✅ No'}")
                print(f"      Reasoning: {quality_eval.reasoning[:80]}...")
                
                if quality_eval.criteria_scores:
                    print(f"      Criteria: {quality_eval.criteria_scores}")
            else:
                print(f"   ❌ Judge returned None")
                print(f"      Check if '{test['operation']}' is in judge.operations")
                
        except Exception as e:
            print(f"   ❌ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
    
    # -------------------------------------------------------------------------
    # Step 5: Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    if results:
        print(f"\n   Evaluations completed: {len(results)}/{len(TEST_CASES)}")
        
        for r in results:
            eval = r["eval"]
            status = "🚨" if eval.hallucination_flag or eval.factual_error else "✅"
            print(f"   {status} {r['name']}: {eval.judge_score}/10")
        
        # Check judge stats
        stats = judge.get_stats()
        print(f"\n   Judge Stats:")
        print(f"   - Total evaluated: {stats['total_evaluated']}")
        print(f"   - Average score: {stats['average_score']}")
        print(f"   - Hallucination rate: {stats['hallucination_rate']}")
        
        print("\n   ✅ Judge is working correctly!")
        print("   Run your baseline data generation to populate quality metrics.")
    else:
        print("\n   ❌ No evaluations completed")
        print("\n   Troubleshooting:")
        print("   1. Check observatory/judge.py has multi-client support")
        print("   2. Check operation names are in judge.operations")
        print("   3. Check Azure OpenAI credentials in .env")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("\n🚀 Starting Judge Test...\n")
    asyncio.run(main())