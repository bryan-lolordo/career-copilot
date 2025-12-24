# agents/plugins/QueryDatabasePlugin.py
"""
Database Query Plugin - Career Copilot
UPDATED: Complete Observatory Tier 1, 2, 3 metrics coverage
"""

from semantic_kernel.functions import kernel_function
from typing import Annotated
import json
import logging
import os
import re
import sqlite3
import time

from services.db import DB_PATH

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
    semantic_cache,
    calculate_prefix_hash,
)

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# PROMPT VERSIONING
# =============================================================================
SQL_GENERATION_PROMPT_VERSION = "1.0.0"

# Create PromptMetadata for SQL generation operations
SQL_PROMPT_META = create_prompt_metadata(
    template_id="database_query_sql_generation",
    version=SQL_GENERATION_PROMPT_VERSION,
    compressible_sections=["RULES"],
    optimization_flags={"deterministic_sql": True},
    config_version="1.0"
) if PromptMetadata else None


class DatabaseQueryPlugin:
    """
    Agentic plugin that allows the AI to query the database using natural language.
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

        full_prompt = f"{system_prompt}\n\n{user_message}"

        try:
            print(f"\n🤖 Generating SQL for question: '{question}'")
            
            # Get execution settings for tracking
            from agents.semantic_kernel_setup import create_execution_settings
            exec_settings = create_execution_settings()

            # Track LLM call for SQL generation
            llm_start_time = time.time()
            
            # ═══════════════════════════════════════════════════════════════
            # CHECK SEMANTIC CACHE FIRST
            # ═══════════════════════════════════════════════════════════════
            cache_result = await semantic_cache.get(full_prompt, operation="generate_sql")
            
            if cache_result.hit:
                # Cache HIT - use cached response
                latency_ms = (time.time() - llm_start_time) * 1000
                generated_sql = cache_result.response
                prompt_tokens = 0  # No tokens used
                completion_tokens = 0
                cache_hit = True
                cache_key = cache_result.cache_key
                print(f"   ✅ Cache HIT ({cache_result.similarity:.1%} similar)")
            else:
                # Cache MISS - call LLM
                result = await self.kernel.invoke_prompt(full_prompt)
                latency_ms = (time.time() - llm_start_time) * 1000
                generated_sql = str(result).strip()
                prompt_tokens = len(full_prompt) // 4
                completion_tokens = len(generated_sql) // 4
                cache_hit = False
                
                # Store in cache for next time
                cache_key = await semantic_cache.set(
                    full_prompt, 
                    generated_sql, 
                    operation="generate_sql",
                    metadata={"question": question}
                )
                print(f"   💾 Cached for future use")
            # ═══════════════════════════════════════════════════════════════
            
            # Create prompt breakdown for Tier 2
            prompt_breakdown = create_prompt_breakdown(
                system_prompt=system_prompt,
                system_prompt_tokens=len(system_prompt) // 4,
                user_message=user_message,
                user_message_tokens=len(user_message) // 4,
            ) if create_prompt_breakdown else None
            
            # LLM Judge evaluation
            quality_eval = await judge.maybe_evaluate(
                operation="generate_sql",
                prompt=full_prompt,
                response=generated_sql,
                llm_client=self.kernel,
                conversation_id=self.memory.conversation_id if self.memory else None,
                turn_number=self.memory.turn_number if self.memory else None,
            )
            
            # Tier 3: Routing decision (placeholder - ready for optimization)
            routing_decision = create_routing_decision(
                chosen_model=DEFAULT_MODEL,
                alternative_models=["gpt-4o", "gpt-4o-mini"],
                reasoning="SQL generation - deterministic task",
                complexity_score=0.4
            ) if create_routing_decision else None
            
            # Tier 3: Cache metadata - WITH REAL DATA
            cache_metadata = create_cache_metadata(
                cache_hit=cache_hit,
                cache_key=cache_key,
                cache_cluster_id="sql_generation",
                similarity_score=cache_result.similarity if cache_hit else None,
            ) if create_cache_metadata else None
            
            # Track in Observatory - COMPLETE with all tiers
            track_llm_call(
                # Core metrics (Tier 1)
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                agent_name="DatabaseQuery",
                agent_role="analyst",
                operation="generate_sql",
                success=True,
                
                # Prompt analysis (Tier 2)
                system_prompt=system_prompt,
                user_message=user_message,
                response_text=generated_sql,
                prompt_metadata=SQL_PROMPT_META,
                prompt_breakdown=prompt_breakdown,  # Added: token breakdown
                
                # Quality evaluation (Tier 2) - Added
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
                request_id=self.memory.request_id if self.memory else None,
                
                # NEW: Model configuration
                temperature=0.0,  # SQL generation should be deterministic
                max_tokens=None,
                
                # NEW: Token breakdown (top-level)
                system_prompt_tokens=prompt_breakdown.system_prompt_tokens if prompt_breakdown else None,
                user_message_tokens=prompt_breakdown.user_message_tokens if prompt_breakdown else None,
                
                # NEW: Streaming
                time_to_first_token_ms=None,
                
                # NEW: Prefix hash (schema is static, question varies)
                prompt_prefix_hash=calculate_prefix_hash(system_prompt, self.schema[:1000]),
                
                # NEW: Observability
                environment=os.getenv("ENVIRONMENT", "development"),
                
                # Metadata
                metadata={
                    "question": question[:200],
                    "generated_sql": generated_sql[:300],
                    "schema_length": len(self.schema),
                    "judged": quality_eval is not None,
                    "conversation_memory": self.memory,
                    "execution_settings": self.exec_settings,
                }
            )
            
            print(f"📊 Tracked SQL generation: {latency_ms:.0f}ms, ~{prompt_tokens + completion_tokens} tokens")
            
            # Clean up generated SQL
            if "```sql" in generated_sql:
                generated_sql = generated_sql.split("```sql")[1].split("```")[0].strip()
            elif "```" in generated_sql:
                generated_sql = generated_sql.split("```")[1].split("```")[0].strip()
            
            generated_sql = generated_sql.rstrip(";")
            
            print(f"📝 Generated SQL: {generated_sql}")
            
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
            # Phase 2 Fix: Complete error tracking with token breakdown, routing, cache
            
            # Token breakdown (use cached values if available)
            _system_tokens = len(system_prompt) // 4 if 'system_prompt' in locals() else None
            _user_tokens = len(user_message) // 4 if 'user_message' in locals() else None
            
            # Routing decision (same as success path)
            _routing = create_routing_decision(
                chosen_model=DEFAULT_MODEL,
                alternative_models=["gpt-4o", "gpt-4o-mini"],
                reasoning="SQL generation - deterministic task (error path)",
                complexity_score=0.4
            ) if create_routing_decision else None
            
            # Cache metadata (use values from earlier if available)
            _cache_meta = create_cache_metadata(
                cache_hit=cache_hit if 'cache_hit' in locals() else False,
                cache_key=cache_key if 'cache_key' in locals() else generate_cache_key("generate_sql", question),
                cache_cluster_id="sql_generation",
            ) if create_cache_metadata else None
            
            track_llm_call(
                # Core metrics (Tier 1)
                prompt_tokens=len(full_prompt) // 4 if 'full_prompt' in locals() else 0,
                completion_tokens=len(generated_sql) // 4 if 'generated_sql' in locals() else 0,
                latency_ms=(time.time() - llm_start_time) * 1000 if 'llm_start_time' in locals() else 0,
                agent_name="DatabaseQuery",
                agent_role="analyst",
                operation="generate_sql",
                success=False,
                error=f"Database error: {str(e)}",
                prompt_metadata=SQL_PROMPT_META,
                
                # Prompt content (Tier 2) - Phase 2 addition
                system_prompt=system_prompt if 'system_prompt' in locals() else None,
                user_message=user_message if 'user_message' in locals() else None,
                response_text=generated_sql if 'generated_sql' in locals() else None,
                
                # Token breakdown (Tier 2) - Phase 2 addition
                system_prompt_tokens=_system_tokens,
                user_message_tokens=_user_tokens,
                
                # Model config (Tier 2)
                temperature=0.0,
                
                # Routing decision (Tier 3) - Phase 2 addition
                routing_decision=_routing,
                
                # Cache metadata (Tier 3) - Phase 2 addition
                cache_metadata=_cache_meta,
                
                # Error details
                error_type="sqlite_error",
                retry_count=0,

                # Conversation linking
                conversation_id=self.memory.conversation_id if self.memory else None,
                turn_number=self.memory.turn_number if self.memory else None,
                parent_call_id=self.memory.request_id if self.memory else None,
                request_id=self.memory.request_id if self.memory else None,
                
                # Streaming
                time_to_first_token_ms=None,
                
                # Observability
                environment=os.getenv("ENVIRONMENT", "development"),
                
                metadata={
                    "question": question[:200],
                    "generated_sql": generated_sql[:300] if 'generated_sql' in locals() else None,
                    "error_type": "sqlite_error",
                    "conversation_memory": self.memory,
                    "execution_settings": self.exec_settings
                }
            )
            return f"❌ Database error: {str(e)}\nGenerated SQL was: {generated_sql if 'generated_sql' in locals() else 'N/A'}"

        # generate_sql    
        except Exception as e:
            # Phase 2 Fix: Complete error tracking with token breakdown, routing, cache
            
            # Token breakdown (use cached values if available)
            _system_tokens = len(system_prompt) // 4 if 'system_prompt' in locals() else None
            _user_tokens = len(user_message) // 4 if 'user_message' in locals() else None
            
            # Routing decision (same as success path)
            _routing = create_routing_decision(
                chosen_model=DEFAULT_MODEL,
                alternative_models=["gpt-4o", "gpt-4o-mini"],
                reasoning="SQL generation - deterministic task (error path)",
                complexity_score=0.4
            ) if create_routing_decision else None
            
            # Cache metadata (use values from earlier if available)
            _cache_meta = create_cache_metadata(
                cache_hit=cache_hit if 'cache_hit' in locals() else False,
                cache_key=cache_key if 'cache_key' in locals() else generate_cache_key("generate_sql", question),
                cache_cluster_id="sql_generation",
            ) if create_cache_metadata else None
            
            track_llm_call(
                # Core metrics (Tier 1)
                prompt_tokens=len(full_prompt) // 4 if 'full_prompt' in locals() else 0,
                completion_tokens=len(generated_sql) // 4 if 'generated_sql' in locals() else 0,
                latency_ms=(time.time() - llm_start_time) * 1000 if 'llm_start_time' in locals() else 0,
                agent_name="DatabaseQuery",
                agent_role="analyst",
                operation="generate_sql",
                success=False,
                error=str(e),
                prompt_metadata=SQL_PROMPT_META,
                
                # Prompt content (Tier 2) - Phase 2 addition
                system_prompt=system_prompt if 'system_prompt' in locals() else None,
                user_message=user_message if 'user_message' in locals() else None,
                response_text=generated_sql if 'generated_sql' in locals() else None,
                
                # Token breakdown (Tier 2) - Phase 2 addition
                system_prompt_tokens=_system_tokens,
                user_message_tokens=_user_tokens,
                
                # Model config (Tier 2)
                temperature=0.0,
                
                # Routing decision (Tier 3) - Phase 2 addition
                routing_decision=_routing,
                
                # Cache metadata (Tier 3) - Phase 2 addition
                cache_metadata=_cache_meta,
                
                # ERROR CLASSIFICATION
                **classify_error(e, operation="generate_sql"),
                retry_count=0,

                # Conversation linking
                conversation_id=self.memory.conversation_id if self.memory else None,
                turn_number=self.memory.turn_number if self.memory else None,
                parent_call_id=self.memory.request_id if self.memory else None,
                request_id=self.memory.request_id if self.memory else None,
                
                # Streaming
                time_to_first_token_ms=None,
                
                # Observability
                environment=os.getenv("ENVIRONMENT", "development"),
                
                metadata={
                    "question": question[:200],
                    "generated_sql": generated_sql[:300] if 'generated_sql' in locals() else None,
                    "error_type": "general_error",
                    "conversation_memory": self.memory,
                    "execution_settings": self.exec_settings
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
            
            # Track database read operation
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
                routing_decision=None,
                cache_metadata=None,
                quality_evaluation=None,
                prompt_variant_id=None,
                test_dataset_id=None,

                # NEW: Conversation linking
                conversation_id=self.memory.conversation_id if self.memory else None,
                turn_number=self.memory.turn_number if self.memory else None,
                parent_call_id=self.memory.request_id if self.memory else None,
                request_id=self.memory.request_id if self.memory else None,
                
                # NEW: Streaming
                time_to_first_token_ms=None,
                
                # NEW: Observability
                environment=os.getenv("ENVIRONMENT", "development"),
                
                metadata={
                    "resume_id": resume_id,
                    "resume_name": resume_name if 'resume_name' in locals() else None,
                    "limit": limit,
                    "matches_returned": len(matches) if 'matches' in locals() else 0,
                    "is_db_read": True,
                    "conversation_memory": self.memory,
                    "execution_settings": self.exec_settings
                }
            )
            
            return result

        # get_top_matches    
        except Exception as e:
            track_llm_call(
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=(time.time() - start_time) * 1000,
                agent_name="DatabaseQuery",
                agent_role="retriever",
                operation="get_top_matches",
                success=False,
                error=str(e),
                
                # NEW: Error details
                **classify_error(e, operation="get_top_matches"),
                retry_count=0,

                # NEW: Conversation linking
                conversation_id=self.memory.conversation_id if self.memory else None,
                turn_number=self.memory.turn_number if self.memory else None,
                parent_call_id=self.memory.request_id if self.memory else None,
                request_id=self.memory.request_id if self.memory else None,
                
                # NEW: Streaming
                time_to_first_token_ms=None,
                
                # NEW: Observability
                environment=os.getenv("ENVIRONMENT", "development"),
                
                metadata={"resume_id": resume_id, "is_db_read": True, "conversation_memory": self.memory, "execution_settings": self.exec_settings}
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
            
            # Track database read operation
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
                routing_decision=None,
                cache_metadata=None,
                quality_evaluation=None,
                prompt_variant_id=None,
                test_dataset_id=None,

                # NEW: Conversation linking
                conversation_id=self.memory.conversation_id if self.memory else None,
                turn_number=self.memory.turn_number if self.memory else None,
                parent_call_id=self.memory.request_id if self.memory else None,
                request_id=self.memory.request_id if self.memory else None,
                
                # NEW: Streaming
                time_to_first_token_ms=None,
                
                # NEW: Observability
                environment=os.getenv("ENVIRONMENT", "development"),
                
                metadata={
                    "limit": limit,
                    "jobs_returned": len(jobs) if 'jobs' in locals() else 0,
                    "is_db_read": True,
                    "conversation_memory": self.memory,
                    "execution_settings": self.exec_settings
                }
            )
            
            return result

        # get_recent_saved_jobs    
        except Exception as e:
            track_llm_call(
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=(time.time() - start_time) * 1000,
                agent_name="DatabaseQuery",
                agent_role="retriever",
                operation="get_recent_saved_jobs",
                success=False,
                error=str(e),
                
                # NEW: Error details
                **classify_error(e, operation="get_recent_saved_jobs"),
                retry_count=0,

                # NEW: Conversation linking
                conversation_id=self.memory.conversation_id if self.memory else None,
                turn_number=self.memory.turn_number if self.memory else None,
                parent_call_id=self.memory.request_id if self.memory else None,
                request_id=self.memory.request_id if self.memory else None,
                
                # NEW: Streaming
                time_to_first_token_ms=None,
                
                # NEW: Observability
                environment=os.getenv("ENVIRONMENT", "development"),
                
                metadata={"limit": limit, "is_db_read": True, "conversation_memory": self.memory, "execution_settings": self.exec_settings}
                
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
            
            # Track database read operation
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
                routing_decision=None,
                cache_metadata=None,
                quality_evaluation=None,
                prompt_variant_id=None,
                test_dataset_id=None,

                # NEW: Conversation linking
                conversation_id=self.memory.conversation_id if self.memory else None,
                turn_number=self.memory.turn_number if self.memory else None,
                parent_call_id=self.memory.request_id if self.memory else None,
                request_id=self.memory.request_id if self.memory else None,
                
                # NEW: Streaming
                time_to_first_token_ms=None,
                
                # NEW: Observability
                environment=os.getenv("ENVIRONMENT", "development"),
                
                metadata={
                    "resume_count": resume_count,
                    "job_count": job_count,
                    "match_count": match_count,
                    "is_db_read": True,
                    "conversation_memory": self.memory,
                    "execution_settings": self.exec_settings
                }
            )
            
            return stats

        # get_database_stats    
        except Exception as e:
            track_llm_call(
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=(time.time() - start_time) * 1000,
                agent_name="DatabaseQuery",
                agent_role="retriever",
                operation="get_database_stats",
                success=False,
                error=str(e),
                
                # NEW: Error details
                **classify_error(e, operation="get_database_stats"),
                retry_count=0,

                # NEW: Conversation linking
                conversation_id=self.memory.conversation_id if self.memory else None,
                turn_number=self.memory.turn_number if self.memory else None,
                parent_call_id=self.memory.request_id if self.memory else None,
                request_id=self.memory.request_id if self.memory else None,
                
                # NEW: Streaming
                time_to_first_token_ms=None,
                
                # NEW: Observability
                environment=os.getenv("ENVIRONMENT", "development"),
                
                metadata={"is_db_read": True, "conversation_memory": self.memory, "execution_settings": self.exec_settings}
            )
            return f"❌ Error retrieving stats: {str(e)}"