# agents/plugins/QueryDatabasePlugin.py
"""
Database Query Plugin - Career Copilot
UPDATED: Complete Observatory Tier 2 metrics coverage
"""

import sqlite3
import json
import re
import time
import os
from semantic_kernel.functions import kernel_function
from typing import Annotated
from services.db import DB_PATH

# Observatory Integration - Updated imports
from observatory_config import (
    track_llm_call,
    create_prompt_metadata,
    PromptMetadata
)

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
            
            # Track LLM call for SQL generation
            llm_start_time = time.time()
            
            result = await self.kernel.invoke_prompt(full_prompt)
            
            latency_ms = (time.time() - llm_start_time) * 1000
            generated_sql = str(result).strip()
            
            # Estimate tokens
            prompt_tokens = len(full_prompt) // 4
            completion_tokens = len(generated_sql) // 4
            
            # Track in Observatory - SINGLE CALL with all data
            track_llm_call(
                # Core metrics (model auto-detected from env)
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                agent_name="DatabaseQuery",
                operation="generate_sql",
                
                # Prompt analysis
                system_prompt=system_prompt,
                user_message=user_message,
                response_text=generated_sql,
                prompt_metadata=SQL_PROMPT_META,
                
                # Metadata
                metadata={
                    "question": question[:200],
                    "generated_sql": generated_sql[:300],
                    "schema_length": len(self.schema)
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
                return f"❌ Query blocked for safety: {safety_reason}"
            
            print(f"✅ Query validated as safe")
            
            # Execute query
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(generated_sql)
            rows = cursor.fetchall()
            
            if not rows:
                conn.close()
                return f"✅ Query executed successfully.\n\nNo results found for: {question}"
            
            # Format results
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
            return f"❌ Database error: {str(e)}\nGenerated SQL was: {generated_sql if 'generated_sql' in locals() else 'N/A'}"
            
        except Exception as e:
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
                return f"❌ No matches found for '{resume_name}'.\n\nRun matching first: 'match my resume'"
            
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
            
            return result
            
        except Exception as e:
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
                return "❌ No saved jobs found in the database."
            
            result = f"📅 {len(jobs)} Most Recently Saved Jobs:\n\n"
            
            for i, job in enumerate(jobs, 1):
                job_id, title, company, location, link, created_at = job
                result += f"{i}. **{title}** at **{company}**\n"
                result += f"   📍 {location}\n"
                result += f"   📅 Saved: {created_at}\n"
                result += f"   🔗 {link}\n\n"
            
            return result
            
        except Exception as e:
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
            return stats
            
        except Exception as e:
            return f"❌ Error retrieving stats: {str(e)}"