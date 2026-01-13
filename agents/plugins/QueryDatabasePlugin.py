# agents/plugins/QueryDatabasePlugin.py
"""
Database Query Plugin - Career Copilot
UPDATED: Complete Observatory integration with two-phase system

Special plugin: Makes LLM calls for SQL generation, so implements
full optimization pattern (caching, routing, etc.) for that operation.
"""

from semantic_kernel.functions import kernel_function
from typing import Annotated
import json
import logging
import os
import re
import sqlite3
import time
import uuid

from services.db import DB_PATH

# Observatory Integration - CORRECT imports (only what exists in observatory_config.py)
from observatory_config import (
    # Main tracking
    track_llm_call,
    
    # Optimization components (import ALL for consistency)
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
SQL_GENERATION_PROMPT_VERSION = "1.0.0"

class DatabaseQueryPlugin:
    """
    Agentic plugin that allows the AI to query the database using natural language.
    
    Special Note: This plugin makes LLM calls for SQL generation, so it implements
    the full two-phase optimization pattern for those operations.
    """
    
    def __init__(self, kernel, memory=None):
        """
        Initialize the plugin with kernel and get database schema.
        
        Args:
            kernel: Semantic Kernel instance needed for AI SQL generation
            memory: ConversationMemory instance for context tracking
        """
        self.kernel = kernel
        self.db_path = DB_PATH
        self.schema = self._get_database_schema()
        self.memory = memory

        # Create execution settings once per plugin instance
        from agents.semantic_kernel_setup import create_execution_settings
        self.exec_settings = create_execution_settings()

    def _get_database_schema(self) -> str:
        """Retrieves the database schema."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            schema_info = "DATABASE SCHEMA:\n\n"
            
            for (table_name,) in tables:
                schema_info += f"Table: {table_name}\n"
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                
                for col in columns:
                    col_id, col_name, col_type, not_null, default, pk = col
                    schema_info += f"  - {col_name} ({col_type})"
                    if pk:
                        schema_info += " [PRIMARY KEY]"
                    schema_info += "\n"
                schema_info += "\n"
            
            conn.close()
            return schema_info
            
        except Exception as e:
            return f"Error retrieving schema: {e}"
    
    def _is_safe_query(self, sql: str) -> tuple[bool, str]:
        """Validates that a SQL query is safe to execute."""
        sql_upper = sql.upper().strip()
        
        if not sql_upper.startswith("SELECT"):
            return False, "Only SELECT queries are allowed for safety. No modifications permitted."
        
        dangerous_keywords = [
            "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", 
            "CREATE", "TRUNCATE", "EXEC", "EXECUTE"
        ]
        
        for keyword in dangerous_keywords:
            if re.search(r'\b' + keyword + r'\b', sql_upper):
                return False, f"Query contains forbidden keyword: {keyword}"
        
        if ";" in sql and sql.count(";") > 1:
            return False, "Multiple SQL statements not allowed"
        
        if "SQLITE_" in sql_upper:
            return False, "Access to system tables not allowed"
        
        return True, "Query is safe"
    
    @kernel_function(
        name="query_database_with_ai",
        description=(
            "Queries the EXISTING saved jobs and resumes already in the database using SQL. "
            "Use this when the user asks about SAVED or EXISTING data: "
            "'show me saved jobs from Deloitte', 'how many resumes do I have', 'what jobs are in the database from Company X', "
            "'find jobs created today', 'show all remote positions I've saved'. "
            "DO NOT use this for searching NEW jobs from the internet - use find_jobs for that."
        )
    )
    async def query_database_with_ai(
        self,
        question: Annotated[str, "Natural language question about the database"]
    ) -> Annotated[str, "Query results formatted as text"]:
        """
        Takes a natural language question, generates SQL, executes it safely,
        and returns the results.
        
        This function makes LLM calls for SQL generation, so it implements
        the full 10-step optimization pattern.
        """
        
        # Build the SQL generation prompt
        system_prompt = """You are a SQL expert. Given a database schema and a user question, generate a safe SQL SELECT query.

RULES:
1. Generate ONLY a SELECT query (no modifications)
2. Return ONLY the SQL query, nothing else
3. Use proper SQLite syntax
4. Limit results to 50 rows maximum using LIMIT clause
5. Do not use subqueries if possible
6. Do not include markdown formatting or code blocks"""

        user_message = f"""{self.schema}

User Question: {question}

SQL Query:"""

        try:
            logger.info(f"🤖 Generating SQL for question: '{question}'")
            
            # ═══════════════════════════════════════════════════════════════
            # STEP 1: Check exact cache
            # ═══════════════════════════════════════════════════════════════
            operation = "generate_sql"
            cache_key_data = {"question": question, "schema_hash": hash(self.schema)}
            
            cached_sql, cache_meta = cache.get(
                operation=operation,
                key_data=cache_key_data
            )
            
            if cached_sql:  # None in baseline, actual SQL in optimized
                logger.info(f"✅ Cache hit! Using cached SQL.")
                generated_sql = cached_sql
                latency_ms = 1.0  # Minimal latency
                prompt_tokens = 0
                completion_tokens = 0
                
                # Track cache hit and skip to SQL execution
                track_llm_call(
                    operation=operation,
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=1.0,
                    success=True,
                    response_text=generated_sql,
                    cache_metadata=cache_meta,
                    agent_name="DatabaseQuery",
                    agent_role="analyst",
                    conversation_id=self.memory.conversation_id if self.memory else None,
                    turn_number=self.memory.turn_number if self.memory else None,
                    parent_call_id=self.memory.request_id if self.memory else None,
                    request_id=str(uuid.uuid4()),
                    trace_id=self.memory.conversation_id if self.memory else None,
                    environment=os.getenv("ENVIRONMENT", "development"),
                    metadata={
                        "phase": CURRENT_PHASE,  # ← CRITICAL
                        "cache_hit": True,
                        "question": question[:200],
                    }
                )
                
                # Skip to SQL execution (after all the LLM steps)
                # ... (will be after STEP 10)
            
            else:
                # ═══════════════════════════════════════════════════════════════
                # STEP 2: Check semantic cache (if available)
                # ═══════════════════════════════════════════════════════════════
                if semantic_cache:
                    result = await semantic_cache.get(operation=operation, prompt=question)
                    if result.hit:  # False in baseline, True in optimized if similar
                        logger.info(f"✅ Semantic cache hit ({result.similarity:.1%} similar)!")
                        generated_sql = result.response
                        latency_ms = 1.0
                        prompt_tokens = 0
                        completion_tokens = 0
                        
                        # Track semantic cache hit and skip to SQL execution
                        track_llm_call(
                            operation=operation,
                            prompt_tokens=0,
                            completion_tokens=0,
                            latency_ms=1.0,
                            success=True,
                            response_text=generated_sql,
                            cache_metadata=create_cache_metadata(
                                cache_hit=True,
                                similarity_score=result.similarity
                            ),
                            agent_name="DatabaseQuery",
                            agent_role="analyst",
                            conversation_id=self.memory.conversation_id if self.memory else None,
                            turn_number=self.memory.turn_number if self.memory else None,
                            parent_call_id=self.memory.request_id if self.memory else None,
                            request_id=str(uuid.uuid4()),
                            trace_id=self.memory.conversation_id if self.memory else None,
                            environment=os.getenv("ENVIRONMENT", "development"),
                            metadata={
                                "phase": CURRENT_PHASE,  # ← CRITICAL
                                "semantic_cache_hit": True,
                                "similarity": result.similarity,
                                "question": question[:200],
                            }
                        )
                        
                        # Skip to SQL execution
                        # ... (will be after STEP 10)
                
                # If no cache hit, proceed with LLM call
                if not cached_sql and (not semantic_cache or not result.hit if semantic_cache else True):
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
                        prompt=optimized_prompt + user_message,  # ✅ Added
                        estimated_tokens=estimate_tokens(optimized_prompt + user_message),  # ✅ Changed
                        complexity=0.3  # SQL generation is low complexity
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
                    
                    generated_sql = str(result).strip()
                    prompt_tokens = estimate_tokens(full_prompt)
                    completion_tokens = estimate_tokens(generated_sql)
                    
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
                        value=generated_sql
                    )
                    
                    if semantic_cache:
                        await semantic_cache.set(
                            operation=operation,
                            prompt=question,
                            response=generated_sql
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
                        response=generated_sql,
                        llm_client=self.kernel,
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
                        agent_name="DatabaseQuery",
                        agent_role="analyst",
                        operation=operation,
                        success=True,
                        
                        # Prompt content
                        system_prompt=optimized_prompt,
                        user_message=user_message,
                        response_text=generated_sql,
                        prompt_breakdown=prompt_breakdown,
                        
                        # Optimization tracking
                        routing_decision=routing_meta,
                        cache_metadata=None, 
                        quality_evaluation=quality_eval,
                        prompt_metadata=None,
                        
                        # Model configuration
                        temperature=0.0,  # SQL generation should be deterministic
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
                            "phase": CURRENT_PHASE,  # ← CRITICAL
                            "question": question[:200],
                            "generated_sql": generated_sql[:300],
                            "schema_length": len(self.schema),
                            "judged": quality_eval is not None,
                            "streaming_candidate": streaming_candidate,
                        }
                    )
                    
                    logger.info(f"📊 Tracked SQL generation: {latency_ms:.0f}ms, {prompt_tokens + completion_tokens} tokens")
            
            # ═══════════════════════════════════════════════════════════════
            # SQL EXECUTION (common path for all branches above)
            # ═══════════════════════════════════════════════════════════════
            
            # Clean up generated SQL
            if "```sql" in generated_sql:
                generated_sql = generated_sql.split("```sql")[1].split("```")[0].strip()
            elif "```" in generated_sql:
                generated_sql = generated_sql.split("```")[1].split("```")[0].strip()
            
            generated_sql = generated_sql.rstrip(";")
            
            logger.info(f"📝 Generated SQL: {generated_sql}")
            
            # Validate query safety
            is_safe, safety_reason = self._is_safe_query(generated_sql)
            
            if not is_safe:
                return f"❌ Cannot execute query: {safety_reason}"
            
            # Execute the query
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(generated_sql)
            rows = cursor.fetchall()
            
            if not rows:
                conn.close()
                return "✅ Query executed successfully but returned no results."
            
            column_names = [description[0] for description in cursor.description]
            conn.close()
            
            result_text = f"✅ Found {len(rows)} result(s):\n\n"
            result_text += " | ".join(column_names) + "\n"
            result_text += "-" * (len(" | ".join(column_names))) + "\n"
            
            for row in rows:
                formatted_row = []
                for value in row:
                    if value is None:
                        formatted_row.append("NULL")
                    elif isinstance(value, str) and len(value) > 50:
                        formatted_row.append(value[:47] + "...")
                    else:
                        formatted_row.append(str(value))
                
                result_text += " | ".join(formatted_row) + "\n"
            
            return result_text
            
        except sqlite3.Error as e:
            # Classify error
            error_info = classify_error(e, operation="generate_sql")
            
            # Track database error with phase metadata
            track_llm_call(
                prompt_tokens=estimate_tokens(system_prompt + user_message) if 'system_prompt' in locals() else 0,
                completion_tokens=estimate_tokens(generated_sql) if 'generated_sql' in locals() else 0,
                latency_ms=latency_ms if 'latency_ms' in locals() else 0,
                agent_name="DatabaseQuery",
                agent_role="analyst",
                operation="generate_sql",
                success=False,
                error=f"Database error: {str(e)}",
                system_prompt=system_prompt if 'system_prompt' in locals() else None,
                user_message=user_message if 'user_message' in locals() else None,
                response_text=generated_sql if 'generated_sql' in locals() else None,
                temperature=0.0,
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
                    "phase": CURRENT_PHASE,  # ← CRITICAL
                    "question": question[:200],
                    "generated_sql": generated_sql[:300] if 'generated_sql' in locals() else None,
                    "error_type": error_info['error_type'],
                }
            )
            return f"❌ Database error: {str(e)}\nGenerated SQL was: {generated_sql if 'generated_sql' in locals() else 'N/A'}"
            
        except Exception as e:
            # Classify error
            error_info = classify_error(e, operation="generate_sql")
            
            # Track general error with phase metadata
            track_llm_call(
                prompt_tokens=estimate_tokens(system_prompt + user_message) if 'system_prompt' in locals() else 0,
                completion_tokens=estimate_tokens(generated_sql) if 'generated_sql' in locals() else 0,
                latency_ms=latency_ms if 'latency_ms' in locals() else 0,
                agent_name="DatabaseQuery",
                agent_role="analyst",
                operation="generate_sql",
                success=False,
                error=str(e),
                system_prompt=system_prompt if 'system_prompt' in locals() else None,
                user_message=user_message if 'user_message' in locals() else None,
                response_text=generated_sql if 'generated_sql' in locals() else None,
                temperature=0.0,
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
                    "phase": CURRENT_PHASE,  # ← CRITICAL
                    "question": question[:200],
                    "generated_sql": generated_sql[:300] if 'generated_sql' in locals() else None,
                    "error_type": error_info['error_type'],
                }
            )
            return f"❌ Error processing query: {str(e)}"
    
    @kernel_function(
        name="get_top_matches",
        description=(
            "Retrieves the top job matches for a resume from the database by MATCH SCORE. "
            "Use when user asks: 'show my top matches', 'what are my best matches', 'show top 5 jobs for my resume'. "
            "This returns jobs sorted by their match percentage/score (highest first)."
        )
    )
    async def get_top_matches(
        self,
        resume_id: Annotated[str, "Resume ID or 'most_recent' for latest resume"] = "most_recent",
        limit: Annotated[int, "Number of top matches to return (default 5)"] = 5
    ) -> Annotated[str, "Top matched jobs sorted by score"]:
        """
        Retrieve top job matches sorted by match score.
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get resume ID
            if resume_id == "most_recent":
                cursor.execute("SELECT id, name FROM resumes ORDER BY created_at DESC LIMIT 1")
                resume_row = cursor.fetchone()
                if not resume_row:
                    conn.close()
                    return "❌ No resumes found. Please upload a resume first."
                resume_id = resume_row[0]
                resume_name = resume_row[1]
            else:
                cursor.execute("SELECT name FROM resumes WHERE id = ?", (int(resume_id),))
                resume_row = cursor.fetchone()
                if not resume_row:
                    conn.close()
                    return f"❌ Resume with ID {resume_id} not found."
                resume_name = resume_row[0]
            
            # Query matches sorted by score
            cursor.execute("""
                SELECT 
                    m.score, m.reason,
                    j.id, j.title, j.company, j.location, j.link
                FROM resume_job_matches m
                JOIN jobs j ON m.job_id = j.id
                WHERE m.resume_id = ?
                ORDER BY m.score DESC
                LIMIT ?
            """, (int(resume_id), limit))
            
            matches = cursor.fetchall()
            conn.close()
            
            if not matches:
                result = f"❌ No matches found for '{resume_name}'.\n\nRun matching first: 'match my resume'"
            else:
                # Store in memory
                if self.memory:
                    self.memory.set_current_focus(resume_id=int(resume_id))
                    for match in matches:
                        score, reason, job_id, title, company, location, link = match
                        self.memory.add_match_result({
                            'job_id': job_id, 'title': title, 'company': company,
                            'location': location, 'link': link, 'score': score, 'reason': reason
                        })
                
                # Format results
                result = f"🎯 Top {len(matches)} Matches for '{resume_name}':\n\n"
                
                for i, match in enumerate(matches, 1):
                    score, reason, job_id, title, company, location, link = match
                    result += f"{i}. **{title}** at **{company}** - {score}% match\n"
                    result += f"   📍 {location}\n"
                    result += f"   🔗 {link}\n\n"
                
                result += "\nSay 'tell me about match #1' for details or 'explain match #2' for why you matched."
            
            # Track database read operation with phase metadata
            track_llm_call(
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=(time.time() - start_time) * 1000,
                agent_name="DatabaseQuery",
                agent_role="retriever",
                operation="get_top_matches",
                success=True,
                prompt=f"Get top {limit} matches for resume {resume_id}",
                response_text=result[:500],
                conversation_id=self.memory.conversation_id if self.memory else None,
                turn_number=self.memory.turn_number if self.memory else None,
                parent_call_id=self.memory.request_id if self.memory else None,
                request_id=request_id,
                trace_id=self.memory.conversation_id if self.memory else None,
                environment=os.getenv("ENVIRONMENT", "development"),
                metadata={
                    "phase": CURRENT_PHASE,  # ← CRITICAL
                    "resume_id": resume_id,
                    "resume_name": resume_name if 'resume_name' in locals() else None,
                    "limit": limit,
                    "matches_returned": len(matches) if 'matches' in locals() else 0,
                    "is_db_read": True,
                }
            )
            
            return result
            
        except Exception as e:
            # Classify error
            error_info = classify_error(e, operation="get_top_matches")
            
            # Track error with phase metadata
            track_llm_call(
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=(time.time() - start_time) * 1000,
                agent_name="DatabaseQuery",
                agent_role="retriever",
                operation="get_top_matches",
                success=False,
                error=str(e),
                error_type=error_info['error_type'],
                error_code=error_info['error_code'],
                retry_count=0,
                conversation_id=self.memory.conversation_id if self.memory else None,
                turn_number=self.memory.turn_number if self.memory else None,
                parent_call_id=self.memory.request_id if self.memory else None,
                request_id=request_id,
                trace_id=self.memory.conversation_id if self.memory else None,
                environment=os.getenv("ENVIRONMENT", "development"),
                metadata={
                    "phase": CURRENT_PHASE,  # ← CRITICAL
                    "resume_id": resume_id,
                    "is_db_read": True,
                    "error_type": error_info['error_type'],
                }
            )
            return f"❌ Error retrieving matches: {str(e)}"
    
    @kernel_function(
        name="get_recent_saved_jobs",
        description=(
            "Retrieves recently saved jobs from the database by SAVE DATE (not match score). "
            "Use when user asks: 'show my recent jobs', 'what jobs did I save recently', 'show last 10 saved jobs'. "
            "This returns jobs sorted by when they were added to the database (newest first)."
        )
    )
    async def get_recent_saved_jobs(
        self,
        limit: Annotated[int, "Number of recent jobs to return (default 10)"] = 10
    ) -> Annotated[str, "Recently saved jobs sorted by date"]:
        """
        Retrieve recently saved jobs sorted by creation date.
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, title, company, location, link, created_at
                FROM jobs
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))
            
            jobs = cursor.fetchall()
            conn.close()
            
            if not jobs:
                result = "❌ No saved jobs found in the database."
            else:
                result = f"📅 {len(jobs)} Most Recently Saved Jobs:\n\n"
                
                for i, job in enumerate(jobs, 1):
                    job_id, title, company, location, link, created_at = job
                    result += f"{i}. **{title}** at **{company}**\n"
                    result += f"   📍 {location}\n"
                    result += f"   📅 Saved: {created_at}\n"
                    result += f"   🔗 {link}\n\n"
            
            # Track database read operation with phase metadata
            track_llm_call(
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=(time.time() - start_time) * 1000,
                agent_name="DatabaseQuery",
                agent_role="retriever",
                operation="get_recent_saved_jobs",
                success=True,
                prompt=f"Get recent {limit} saved jobs",
                response_text=result[:500],
                conversation_id=self.memory.conversation_id if self.memory else None,
                turn_number=self.memory.turn_number if self.memory else None,
                parent_call_id=self.memory.request_id if self.memory else None,
                request_id=request_id,
                trace_id=self.memory.conversation_id if self.memory else None,
                environment=os.getenv("ENVIRONMENT", "development"),
                metadata={
                    "phase": CURRENT_PHASE,  # ← CRITICAL
                    "limit": limit,
                    "jobs_returned": len(jobs) if 'jobs' in locals() else 0,
                    "is_db_read": True,
                }
            )
            
            return result
            
        except Exception as e:
            # Classify error
            error_info = classify_error(e, operation="get_recent_saved_jobs")
            
            # Track error with phase metadata
            track_llm_call(
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=(time.time() - start_time) * 1000,
                agent_name="DatabaseQuery",
                agent_role="retriever",
                operation="get_recent_saved_jobs",
                success=False,
                error=str(e),
                error_type=error_info['error_type'],
                error_code=error_info['error_code'],
                retry_count=0,
                conversation_id=self.memory.conversation_id if self.memory else None,
                turn_number=self.memory.turn_number if self.memory else None,
                parent_call_id=self.memory.request_id if self.memory else None,
                request_id=request_id,
                trace_id=self.memory.conversation_id if self.memory else None,
                environment=os.getenv("ENVIRONMENT", "development"),
                metadata={
                    "phase": CURRENT_PHASE,  # ← CRITICAL
                    "limit": limit,
                    "is_db_read": True,
                    "error_type": error_info['error_type'],
                }
            )
            return f"❌ Error retrieving recent jobs: {str(e)}"

    @kernel_function(
        name="get_database_schema",
        description="Returns the structure of the database (tables and columns) to understand what data is available"
    )
    async def get_database_schema(self) -> Annotated[str, "Database schema information"]:
        """Returns the database schema in a readable format."""
        return self.schema
    
    @kernel_function(
        name="get_database_stats",
        description="Returns statistics about the database (number of resumes, jobs, matches, etc.)"
    )
    async def get_database_stats(self) -> Annotated[str, "Database statistics"]:
        """Provides quick statistics about the database contents."""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM resumes")
            resume_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM jobs")
            job_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM resume_job_matches")
            match_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT company) FROM jobs")
            company_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT location) FROM jobs")
            location_count = cursor.fetchone()[0]
            
            conn.close()
            
            stats = f"""📊 Database Statistics:

📄 Resumes: {resume_count}
💼 Jobs: {job_count}
🎯 Matches: {match_count}
🏢 Unique Companies: {company_count}
📍 Unique Locations: {location_count}
"""
            
            # Track database read operation with phase metadata
            track_llm_call(
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=(time.time() - start_time) * 1000,
                agent_name="DatabaseQuery",
                agent_role="retriever",
                operation="get_database_stats",
                success=True,
                prompt="Get database statistics",
                response_text=stats,
                conversation_id=self.memory.conversation_id if self.memory else None,
                turn_number=self.memory.turn_number if self.memory else None,
                parent_call_id=self.memory.request_id if self.memory else None,
                request_id=request_id,
                trace_id=self.memory.conversation_id if self.memory else None,
                environment=os.getenv("ENVIRONMENT", "development"),
                metadata={
                    "phase": CURRENT_PHASE,  # ← CRITICAL
                    "resume_count": resume_count,
                    "job_count": job_count,
                    "match_count": match_count,
                    "is_db_read": True,
                }
            )
            
            return stats
            
        except Exception as e:
            # Classify error
            error_info = classify_error(e, operation="get_database_stats")
            
            # Track error with phase metadata
            track_llm_call(
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=(time.time() - start_time) * 1000,
                agent_name="DatabaseQuery",
                agent_role="retriever",
                operation="get_database_stats",
                success=False,
                error=str(e),
                error_type=error_info['error_type'],
                error_code=error_info['error_code'],
                retry_count=0,
                conversation_id=self.memory.conversation_id if self.memory else None,
                turn_number=self.memory.turn_number if self.memory else None,
                parent_call_id=self.memory.request_id if self.memory else None,
                request_id=request_id,
                trace_id=self.memory.conversation_id if self.memory else None,
                environment=os.getenv("ENVIRONMENT", "development"),
                metadata={
                    "phase": CURRENT_PHASE,  # ← CRITICAL
                    "is_db_read": True,
                    "error_type": error_info['error_type'],
                }
            )
            return f"❌ Error retrieving stats: {str(e)}"