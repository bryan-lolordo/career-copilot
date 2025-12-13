#!/usr/bin/env python3
"""
Test Script - QueryDatabasePlugin Token Breakdown (Direct Call)

Simple test that directly instantiates and calls the plugin to verify token breakdown.

Usage:
    python test_token_breakdown_simple.py
"""

import asyncio
import sqlite3
from datetime import datetime

# Import components
from observatory_config import start_session, end_session
from services.conversation_memory import ConversationMemory
from services.database_service import DatabaseService
from agents.plugins.QueryDatabasePlugin import DatabaseQueryPlugin

# =============================================================================
# TEST CONFIGURATION
# =============================================================================

TEST_QUESTIONS = [
    "How many jobs do I have saved?",
    "Show me jobs from tech companies",
    "Which jobs are remote?",
]

# =============================================================================
# TEST RUNNER
# =============================================================================

async def test_token_breakdown():
    """Test that token breakdown fields are populated."""
    
    print("="*70)
    print("🧪 TESTING: QueryDatabasePlugin Token Breakdown (Direct)")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Create memory with context
    memory = ConversationMemory("token_breakdown_test")
    
    # Start Observatory session
    session = start_session(
        operation_type="token_breakdown_test",
        metadata={
            "test_type": "token_breakdown_validation",
            "plugin": "QueryDatabasePlugin",
            "test_timestamp": datetime.now().isoformat()
        }
    )
    
    # Initialize memory for Observatory tracking
    memory.conversation_id = session.id
    memory.turn_number = 0
    
    # Add context to ConversationMemory to test conversation_context_tokens
    memory.set_current_focus(resume_id=25, job_id=101)
    memory.context.last_action = "test_query"
    memory.context.preferred_locations = ["Chicago", "Remote"]
    
    print("📝 ConversationMemory context:")
    context_text = memory.get_context_for_prompt()
    print(f"   {context_text}")
    print(f"   Context length: {len(context_text)} chars\n")
    
    # Create database service and plugin
    db_service = DatabaseService()
    plugin = DatabaseQueryPlugin(db_service=db_service, memory=memory)
    
    print("✅ Plugin instantiated successfully\n")
    
    # Run test questions
    for i, question in enumerate(TEST_QUESTIONS, 1):
        memory.turn_number = i
        
        print(f"{'─'*70}")
        print(f"Turn {i}: {question}")
        print(f"{'─'*70}")
        
        try:
            # Directly call the plugin method
            result = await plugin.query_database_with_ai(question=question)
            
            print(f"✅ Query executed successfully")
            print(f"   Result: {str(result)[:100]}...")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        print()
        
        # Small delay between queries
        await asyncio.sleep(0.5)
    
    end_session(session, success=True)
    
    print(f"{'='*70}")
    print("📊 Checking Database for Token Breakdown...")
    print(f"{'='*70}\n")
    
    # Query the database to verify token breakdown was populated
    conn = sqlite3.connect('C:/Users/bjlol/Desktop/ai-agent-observatory/observatory.db')
    cursor = conn.cursor()
    
    # Get the calls from this session
    cursor.execute("""
        SELECT 
            agent_name,
            operation,
            system_prompt_tokens,
            user_message_tokens,
            chat_history_tokens,
            conversation_context_tokens,
            temperature,
            max_tokens,
            top_p,
            prompt_tokens,
            completion_tokens
        FROM llm_calls
        WHERE session_id = ?
        ORDER BY timestamp
    """, (session.id,))
    
    results = cursor.fetchall()
    
    if not results:
        print("⚠️  No calls found in database!")
        print(f"   Session ID: {session.id}")
        
        # Check if there are ANY recent calls
        cursor.execute("""
            SELECT COUNT(*) FROM llm_calls 
            WHERE timestamp > datetime('now', '-1 hour')
        """)
        recent_count = cursor.fetchone()[0]
        print(f"   Recent calls in last hour: {recent_count}")
        
    else:
        print(f"Found {len(results)} LLM calls:\n")
        
        for i, row in enumerate(results, 1):
            (agent, op, sys_tokens, user_tokens, hist_tokens, ctx_tokens, 
             temp, max_tok, top_p, prompt_tokens, completion_tokens) = row
            
            total_breakdown = (sys_tokens or 0) + (user_tokens or 0) + (hist_tokens or 0) + (ctx_tokens or 0)
            
            print(f"Call {i}: {agent}.{op}")
            print(f"  Token Breakdown:")
            print(f"  ├─ system_prompt_tokens: {sys_tokens}")
            print(f"  ├─ user_message_tokens: {user_tokens}")
            print(f"  ├─ chat_history_tokens: {hist_tokens}")
            print(f"  ├─ conversation_context_tokens: {ctx_tokens} {'✅' if ctx_tokens and ctx_tokens > 0 else '❌'}")
            print(f"  └─ Total breakdown: {total_breakdown} tokens")
            print(f"  Model Parameters:")
            print(f"  ├─ temperature: {temp}")
            print(f"  ├─ max_tokens: {max_tok}")
            print(f"  └─ top_p: {top_p}")
            print(f"  Totals:")
            print(f"  ├─ prompt_tokens: {prompt_tokens}")
            print(f"  └─ completion_tokens: {completion_tokens}")
            print()
        
        # Validation checks
        print(f"{'='*70}")
        print("VALIDATION RESULTS:")
        print(f"{'='*70}")
        
        sys_populated = sum(1 for r in results if r[2] is not None and r[2] > 0)
        user_populated = sum(1 for r in results if r[3] is not None and r[3] > 0)
        ctx_populated = sum(1 for r in results if r[5] is not None and r[5] > 0)
        temp_populated = sum(1 for r in results if r[6] is not None)
        
        print(f"✅ system_prompt_tokens populated: {sys_populated}/{len(results)}")
        print(f"✅ user_message_tokens populated: {user_populated}/{len(results)}")
        print(f"{'✅' if ctx_populated > 0 else '❌'} conversation_context_tokens populated: {ctx_populated}/{len(results)}")
        print(f"✅ temperature populated: {temp_populated}/{len(results)}")
        
        if ctx_populated > 0:
            print(f"\n🎉 SUCCESS: conversation_context_tokens is working!")
        else:
            print(f"\n⚠️  WARNING: conversation_context_tokens still empty")
            print(f"   Check that ConversationMemory context had content")
    
    conn.close()
    
    print(f"\n{'='*70}")
    print(f"Test completed!")
    print(f"Session ID: {session.id}")
    print(f"{'='*70}\n")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    asyncio.run(test_token_breakdown())