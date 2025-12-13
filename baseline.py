"""
Observatory Tracking Test Script - Career Copilot
Tests all updated plugins to verify 139-field tracking is working.

Run from career-copilot directory:
    python test_observatory_tracking.py
    
Or test optimized mode:
    OBSERVATORY_PHASE=optimized python test_observatory_tracking.py
"""

import asyncio
import sqlite3
import os
import sys
from datetime import datetime
from pathlib import Path

# Set test environment BEFORE imports
os.environ['ENVIRONMENT'] = 'testing'

# Add career-copilot to path if needed
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

print("🧪 Career Copilot - Observatory Tracking Test")
print("=" * 70)
print(f"Environment: {os.getenv('ENVIRONMENT', 'production')}")
print(f"Phase: {os.getenv('OBSERVATORY_PHASE', 'baseline')}")
print("=" * 70)


async def test_all_plugins():
    """Test all Career Copilot plugins with Observatory tracking."""
    
    # Import after environment is set
    from agents.semantic_kernel_setup import (
        create_kernel_with_plugins,
        create_execution_settings,
        create_chat_history_with_system_prompt,
    )
    from observatory_config import obs, OBSERVATORY_DB_PATH, start_session, end_session
    
    print(f"\n📂 Database: {OBSERVATORY_DB_PATH}\n")
    
    # Track start time for filtering test data
    test_start = datetime.now()
    test_start_iso = test_start.isoformat()

    # ADD THIS LINE:
    session = start_session("test_tracking", metadata={"test_run": test_start_iso})
    
    # Create kernel with all plugins
    kernel, chat_completion, db_service, memory = create_kernel_with_plugins()
    execution_settings = create_execution_settings()
    history = create_chat_history_with_system_prompt()
    
    # Test cases covering all updated plugins
    test_cases = [
        {
            "name": "JobPlugin - find_jobs",
            "message": "search for Python Developer jobs in Chicago",
            "expected_operations": ["find_jobs"],
            "plugin": "JobPlugin",
        },
        {
            "name": "JobPlugin - get_job_details", 
            "message": "tell me more about job #1",
            "expected_operations": ["get_job_details"],
            "plugin": "JobPlugin",
        },
        {
            "name": "QueryDatabasePlugin - generate_sql",
            "message": "show me all saved jobs from Google",
            "expected_operations": ["generate_sql", "query_database_with_ai"],
            "plugin": "QueryDatabasePlugin",
        },
        {
            "name": "ResumeMatchingPlugin - list_resumes",
            "message": "match my resume to jobs",
            "expected_operations": ["list_resumes"],
            "plugin": "ResumeMatchingPlugin",
        },
        {
            "name": "ChatAgent - CLI conversation",
            "message": "what kind of help can you provide?",
            "expected_operations": ["cli_chat_message"],
            "plugin": "ChatAgent",
        },
    ]
    
    print("🔄 Running test operations...\n")
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        test_name = test_case["name"]
        message = test_case["message"]
        
        print(f"[{i}/{len(test_cases)}] Testing: {test_name}")
        print(f"    Message: '{message}'")

        # ADD THESE LINES:
        memory.conversation_id = f"test_session_{test_start_iso}"
        memory.turn_number = i
        
        try:
            # Add message to history
            history.add_user_message(message)
            
            # Send request (this should trigger tracking)
            response = await chat_completion.get_chat_message_content(
                chat_history=history,
                settings=execution_settings,
                kernel=kernel,
            )
            
            # Add response to history
            history.add_message(response)
            
            result_text = str(response)[:100]
            print(f"    ✅ Response: {result_text}...")
            
            results.append({
                "test": test_name,
                "status": "success",
                "plugin": test_case["plugin"],
                "expected_ops": test_case["expected_operations"],
            })
            
        except Exception as e:
            error_msg = str(e)[:150]
            print(f"    ⚠️  Error: {error_msg}")
            
            results.append({
                "test": test_name,
                "status": "error",
                "error": error_msg,
                "plugin": test_case["plugin"],
            })
        
        print()
    
    # Wait a moment for async tracking to complete
    await asyncio.sleep(1)
    
    # ADD THIS LINE:
    end_session(session, success=True)

    return test_start_iso, results


def verify_tracking(test_start_iso, results):
    """Verify that tracking data was saved correctly."""
    
    from observatory_config import OBSERVATORY_DB_PATH
    
    print("\n" + "=" * 70)
    print("📊 VERIFYING TRACKING DATA")
    print("=" * 70)
    
    if not Path(OBSERVATORY_DB_PATH).exists():
        print(f"\n❌ ERROR: Database not found at {OBSERVATORY_DB_PATH}")
        return False
    
    conn = sqlite3.connect(OBSERVATORY_DB_PATH)
    conn.row_factory = sqlite3.Row  # Access columns by name
    cursor = conn.cursor()
    
    # Check total calls tracked
    cursor.execute("""
        SELECT COUNT(*) as total
        FROM llm_calls
        WHERE timestamp >= ?
    """, (test_start_iso,))
    
    total_calls = cursor.fetchone()['total']
    print(f"\n✅ Total LLM calls tracked: {total_calls}")
    
    if total_calls == 0:
        print("\n❌ CRITICAL: No calls were tracked!")
        print("Check that plugins are calling track_llm_call()")
        conn.close()
        return False
    
    # Check NEW fields are populated
    print("\n📋 Checking NEW field coverage:")
    
    new_fields_check = cursor.execute("""
        SELECT 
            -- Conversation linking
            COUNT(CASE WHEN conversation_id IS NOT NULL THEN 1 END) as has_conversation_id,
            COUNT(CASE WHEN turn_number IS NOT NULL THEN 1 END) as has_turn_number,
            
            -- Model config
            COUNT(CASE WHEN temperature IS NOT NULL THEN 1 END) as has_temperature,
            COUNT(CASE WHEN max_tokens IS NOT NULL THEN 1 END) as has_max_tokens,
            
            -- Token breakdown (top-level)
            COUNT(CASE WHEN system_prompt_tokens IS NOT NULL THEN 1 END) as has_system_tokens,
            COUNT(CASE WHEN user_message_tokens IS NOT NULL THEN 1 END) as has_user_tokens,
            COUNT(CASE WHEN chat_history_tokens IS NOT NULL THEN 1 END) as has_history_tokens,
            
            -- Tool tracking
            COUNT(CASE WHEN tool_call_count > 0 THEN 1 END) as has_tool_calls,
            
            -- Error tracking
            COUNT(CASE WHEN error_type IS NOT NULL THEN 1 END) as has_error_type,
            COUNT(CASE WHEN retry_count IS NOT NULL THEN 1 END) as has_retry_count,
            
            -- Observability
            COUNT(CASE WHEN environment IS NOT NULL THEN 1 END) as has_environment,
            COUNT(CASE WHEN trace_id IS NOT NULL THEN 1 END) as has_trace_id,
            
            COUNT(*) as total
        FROM llm_calls
        WHERE timestamp >= ?
    """, (test_start_iso,))
    
    field_stats = new_fields_check.fetchone()
    
    def print_coverage(field_name, count, total):
        percentage = (count / total * 100) if total > 0 else 0
        status = "✅" if percentage > 0 else "⚠️ "
        print(f"  {status} {field_name:30} {count:3}/{total:3} ({percentage:5.1f}%)")
    
    print_coverage("conversation_id", field_stats['has_conversation_id'], field_stats['total'])
    print_coverage("turn_number", field_stats['has_turn_number'], field_stats['total'])
    print_coverage("temperature", field_stats['has_temperature'], field_stats['total'])
    print_coverage("max_tokens", field_stats['has_max_tokens'], field_stats['total'])
    print_coverage("system_prompt_tokens", field_stats['has_system_tokens'], field_stats['total'])
    print_coverage("user_message_tokens", field_stats['has_user_tokens'], field_stats['total'])
    print_coverage("chat_history_tokens", field_stats['has_history_tokens'], field_stats['total'])
    print_coverage("tool_call_count > 0", field_stats['has_tool_calls'], field_stats['total'])
    print_coverage("error_type", field_stats['has_error_type'], field_stats['total'])
    print_coverage("retry_count", field_stats['has_retry_count'], field_stats['total'])
    print_coverage("environment", field_stats['has_environment'], field_stats['total'])
    print_coverage("trace_id", field_stats['has_trace_id'], field_stats['total'])
    
    # Check operations tracked
    print("\n📋 Operations tracked:")
    
    ops_query = cursor.execute("""
        SELECT 
            operation,
            COUNT(*) as count,
            AVG(latency_ms) as avg_latency,
            SUM(prompt_tokens + completion_tokens) as total_tokens
        FROM llm_calls
        WHERE timestamp >= ?
        GROUP BY operation
        ORDER BY count DESC
    """, (test_start_iso,))
    
    for row in ops_query.fetchall():
        op = row['operation'] or '(null)'
        count = row['count']
        avg_lat = row['avg_latency'] or 0
        tokens = row['total_tokens'] or 0
        print(f"  • {op:30} {count:3} calls  |  {avg_lat:6.0f}ms avg  |  {tokens:6} tokens")
    
    # Sample detailed check
    print("\n🔍 Sample call details (first 3):")
    
    sample_query = cursor.execute("""
        SELECT 
            operation,
            agent_name,
            conversation_id,
            turn_number,
            environment,
            system_prompt_tokens,
            user_message_tokens,
            tool_call_count,
            success
        FROM llm_calls
        WHERE timestamp >= ?
        ORDER BY timestamp ASC
        LIMIT 3
    """, (test_start_iso,))
    
    for i, row in enumerate(sample_query.fetchall(), 1):
        print(f"\n  Call #{i}:")
        print(f"    Operation: {row['operation']}")
        print(f"    Agent: {row['agent_name']}")
        print(f"    Conversation ID: {row['conversation_id']}")
        print(f"    Turn Number: {row['turn_number']}")
        print(f"    Environment: {row['environment']}")
        print(f"    Token Breakdown: sys={row['system_prompt_tokens']}, user={row['user_message_tokens']}")
        print(f"    Tool Calls: {row['tool_call_count']}")
        print(f"    Success: {row['success']}")
    
    # Check for critical issues
    print("\n🔍 Critical checks:")
    
    critical_issues = []
    
    # All calls should have environment field
    if field_stats['has_environment'] < field_stats['total']:
        missing = field_stats['total'] - field_stats['has_environment']
        critical_issues.append(f"{missing} calls missing environment field")
    
    # Chat operations should have conversation_id
    cursor.execute("""
        SELECT COUNT(*) as count
        FROM llm_calls
        WHERE timestamp >= ?
        AND operation IN ('cli_chat_message', 'streamlit_chat')
        AND conversation_id IS NULL
    """, (test_start_iso,))
    
    missing_conv_id = cursor.fetchone()['count']
    if missing_conv_id > 0:
        critical_issues.append(f"{missing_conv_id} chat calls missing conversation_id")
    
    # LLM calls with prompts should have token breakdown
    cursor.execute("""
        SELECT COUNT(*) as count
        FROM llm_calls
        WHERE timestamp >= ?
        AND prompt_tokens > 0
        AND system_prompt_tokens IS NULL
        AND operation NOT IN ('find_jobs', 'get_job_details', 'save_jobs', 'list_resumes')
    """, (test_start_iso,))
    
    missing_breakdown = cursor.fetchone()['count']
    if missing_breakdown > 0:
        critical_issues.append(f"{missing_breakdown} LLM calls missing token breakdown")
    
    if critical_issues:
        print("  ⚠️  Issues found:")
        for issue in critical_issues:
            print(f"      - {issue}")
    else:
        print("  ✅ No critical issues found")
    
    conn.close()
    
    # Final verdict
    print("\n" + "=" * 70)
    
    # Success criteria
    has_calls = total_calls > 0
    has_environment = field_stats['has_environment'] == field_stats['total']
    has_conversation_tracking = field_stats['has_conversation_id'] > 0
    
    if has_calls and has_environment and has_conversation_tracking:
        print("✅ TEST PASSED - Tracking is working correctly!")
        print(f"   • {total_calls} calls tracked")
        print(f"   • All new fields are being populated")
        print(f"   • Environment tracking: 100%")
        success = True
    else:
        print("❌ TEST FAILED - Issues detected:")
        if not has_calls:
            print("   • No calls were tracked")
        if not has_environment:
            print("   • Environment field not consistently populated")
        if not has_conversation_tracking:
            print("   • Conversation tracking not working")
        success = False
    
    print("=" * 70)
    
    return success


async def main():
    """Run the complete test suite."""
    
    print("\nStarting test suite...\n")
    
    try:
        # Run plugin tests
        test_start_iso, results = await test_all_plugins()
        
        # Verify tracking
        success = verify_tracking(test_start_iso, results)
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())