# generate_baseline_data.py
"""
Generate Observatory Baseline Data with Real LLM Calls
Location: career-copilot/generate_baseline_data.py

This script makes REAL calls to Azure OpenAI to populate Observatory
with authentic baseline metrics for analysis.

Usage:
    python generate_baseline_data.py              # Full run (~50 calls)
    python generate_baseline_data.py --quick      # Quick run (~20 calls)
    python generate_baseline_data.py --jobs-only  # Only job operations
"""

import asyncio
import argparse
import random
import time
import sys
from datetime import datetime

# =============================================================================
# SAMPLE DATA FOR VARIATION
# =============================================================================

SEARCH_QUERIES = [
    ("AI Engineer", "Chicago, IL"),
]

DB_QUERIES = [
    "Show me all saved jobs",
    "How many resumes do I have?",
    "Find jobs from tech companies",
    "Show remote positions",
    "What jobs did I save recently?",
    "Show jobs in Chicago",
    "Find senior level positions",
    "List all companies I have jobs from",
    "Show jobs with salary info",
    "Find Python-related jobs",
]

TAILORING_REQUESTS = [
    "Make this bullet point more impactful",
    "Add quantifiable metrics to my experience",
    "Emphasize my leadership experience",
    "Focus on my technical skills",
    "Highlight my achievements better",
    "Make this sound more senior-level",
    "Add more action verbs",
    "Quantify my impact with numbers",
]

# =============================================================================
# VISUAL HELPERS
# =============================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'


def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}  {text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.END}\n")


def print_section(text, emoji="📍"):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{emoji} {text}{Colors.END}")
    print(f"{Colors.DIM}{'─'*50}{Colors.END}")


def print_progress(current, total, description, status="running"):
    bar_length = 20
    filled = int(bar_length * current / total)
    bar = "█" * filled + "░" * (bar_length - filled)
    
    if status == "success":
        color = Colors.GREEN
        icon = "✅"
    elif status == "error":
        color = Colors.RED
        icon = "❌"
    else:
        color = Colors.YELLOW
        icon = "⏳"
    
    print(f"  {icon} [{bar}] {current}/{total} {color}{description[:40]:<40}{Colors.END}")


def print_stats(stats):
    print(f"\n{Colors.BOLD}{'─'*50}{Colors.END}")
    print(f"{Colors.BOLD}📊 Operation Statistics:{Colors.END}")
    print(f"{'─'*50}")
    
    total_calls = sum(s['success'] + s['error'] for s in stats.values())
    total_success = sum(s['success'] for s in stats.values())
    total_errors = sum(s['error'] for s in stats.values())
    total_time = sum(s['time'] for s in stats.values())
    
    for op, data in stats.items():
        success_rate = (data['success'] / (data['success'] + data['error']) * 100) if (data['success'] + data['error']) > 0 else 0
        avg_time = (data['time'] / (data['success'] + data['error'])) if (data['success'] + data['error']) > 0 else 0
        
        color = Colors.GREEN if success_rate == 100 else Colors.YELLOW if success_rate >= 80 else Colors.RED
        print(f"  {op:<25} {color}{data['success']:>3}✓ {data['error']:>2}✗  {success_rate:>5.1f}%  avg {avg_time:>6.0f}ms{Colors.END}")
    
    print(f"{'─'*50}")
    print(f"{Colors.BOLD}  {'TOTAL':<25} {total_success:>3}✓ {total_errors:>2}✗  {total_success/total_calls*100 if total_calls else 0:>5.1f}%  {total_time/1000:>6.1f}s{Colors.END}")


# =============================================================================
# MAIN SCRIPT
# =============================================================================

async def main(quick_mode=False, jobs_only=False):
    print_header("🚀 Observatory Baseline Data Generator")
    
    print(f"{Colors.DIM}Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.END}")
    print(f"{Colors.DIM}Mode: {'Quick' if quick_mode else 'Full'} {'(Jobs Only)' if jobs_only else ''}{Colors.END}")
    
    # =========================================================================
    # INITIALIZE
    # =========================================================================
    print_section("Initializing Components", "🔧")
    
    try:
        from observatory_config import obs, start_session, end_session
        print_progress(1, 4, "Observatory config loaded", "success")
        
        from services.conversation_memory import get_memory_manager
        print_progress(2, 4, "Memory manager loaded", "success")
        
        from agents.semantic_kernel_setup import create_kernel_with_plugins
        print_progress(3, 4, "Kernel setup loaded", "success")
        
        # Create kernel with plugins
        memory_manager = get_memory_manager()
        memory = memory_manager.get_session("baseline_generation")
        kernel, chat_completion, db_service, memory = create_kernel_with_plugins(memory)
        print_progress(4, 4, "Kernel initialized", "success")
        
    except Exception as e:
        print(f"\n{Colors.RED}❌ Initialization failed: {e}{Colors.END}")
        return
    
    # Get plugin instances
    from agents.plugins.JobPlugin import JobPlugin
    from agents.plugins.QueryDatabasePlugin import DatabaseQueryPlugin
    from agents.plugins.ResumeMatchingPlugin import ResumeMatchingPlugin
    from agents.plugins.ResumeTailoringPlugin import ResumeTailoringPlugin
    
    job_plugin = JobPlugin(context=memory.context)
    db_plugin = DatabaseQueryPlugin(kernel, memory)
    matching_plugin = ResumeMatchingPlugin(kernel, db_service, memory)
    tailoring_plugin = ResumeTailoringPlugin(kernel, memory)
    
    # Stats tracking
    stats = {
        "Job Search": {"success": 0, "error": 0, "time": 0},
        "Database Query": {"success": 0, "error": 0, "time": 0},
        "Quick Match": {"success": 0, "error": 0, "time": 0},
        "Deep Analysis": {"success": 0, "error": 0, "time": 0},
        "Resume Tailoring": {"success": 0, "error": 0, "time": 0},
    }
    
    # Determine counts based on mode
    if quick_mode:
        job_count, db_count, match_count, tailor_count = 3, 4, 2, 2
    else:
        job_count, db_count, match_count, tailor_count = 5, 8, 5, 5
    
    # Start Observatory session
    session = start_session("baseline_data_generation", metadata={
        "mode": "quick" if quick_mode else "full",
        "jobs_only": jobs_only
    })
    
    overall_start = time.time()
    
    try:
        # =====================================================================
        # 1. JOB SEARCHES
        # =====================================================================
        print_section(f"Job Searches (0/{job_count})", "🔍")
        
        queries = random.sample(SEARCH_QUERIES, min(job_count, len(SEARCH_QUERIES)))
        
        for i, (query, location) in enumerate(queries, 1):
            start = time.time()
            try:
                print_progress(i, job_count, f"{query} in {location}", "running")
                result = await job_plugin.find_jobs(query, location, num_results=3)
                elapsed = (time.time() - start) * 1000
                stats["Job Search"]["success"] += 1
                stats["Job Search"]["time"] += elapsed
                # Reprint with success
                print(f"\033[F", end="")  # Move cursor up
                print_progress(i, job_count, f"{query} in {location}", "success")
            except Exception as e:
                elapsed = (time.time() - start) * 1000
                stats["Job Search"]["error"] += 1
                stats["Job Search"]["time"] += elapsed
                print(f"\033[F", end="")
                print_progress(i, job_count, f"Error: {str(e)[:30]}", "error")
            
            await asyncio.sleep(1.5)  # Rate limiting
        
        if jobs_only:
            end_session(session, success=True)
            print_stats(stats)
            return
        
        # =====================================================================
        # 2. DATABASE QUERIES
        # =====================================================================
        print_section(f"Database Queries (0/{db_count})", "💾")
        
        queries = random.sample(DB_QUERIES, min(db_count, len(DB_QUERIES)))
        
        for i, query in enumerate(queries, 1):
            start = time.time()
            try:
                print_progress(i, db_count, query, "running")
                result = await db_plugin.query_database_with_ai(query)
                elapsed = (time.time() - start) * 1000
                stats["Database Query"]["success"] += 1
                stats["Database Query"]["time"] += elapsed
                print(f"\033[F", end="")
                print_progress(i, db_count, query, "success")
            except Exception as e:
                elapsed = (time.time() - start) * 1000
                stats["Database Query"]["error"] += 1
                stats["Database Query"]["time"] += elapsed
                print(f"\033[F", end="")
                print_progress(i, db_count, f"Error: {str(e)[:30]}", "error")
            
            await asyncio.sleep(1)
        
        # =====================================================================
        # 3. RESUME MATCHING (requires saved resumes and jobs)
        # =====================================================================
        print_section(f"Resume Matching (0/{match_count})", "🎯")
        
        # Check for available resumes
        resumes = db_service.list_all_resumes() if hasattr(db_service, 'list_all_resumes') else []
        jobs = db_service.get_all_jobs()[:10] if hasattr(db_service, 'get_all_jobs') else []
        
        if not resumes or not jobs:
            print(f"  {Colors.YELLOW}⚠️  Skipping: Need saved resumes and jobs in database{Colors.END}")
        else:
            resume = resumes[0]
            resume_text = resume.get('text', resume.get('content', ''))
            
            for i, job in enumerate(jobs[:match_count], 1):
                # Quick score
                start = time.time()
                try:
                    print_progress(i, match_count, f"Scoring: {job.get('title', 'Unknown')[:30]}", "running")
                    result = await matching_plugin._quick_score_job_match(resume_text, job)
                    elapsed = (time.time() - start) * 1000
                    stats["Quick Match"]["success"] += 1
                    stats["Quick Match"]["time"] += elapsed
                    print(f"\033[F", end="")
                    print_progress(i, match_count, f"Score: {result.get('score', '?')}/100 - {job.get('title', '')[:25]}", "success")
                except Exception as e:
                    elapsed = (time.time() - start) * 1000
                    stats["Quick Match"]["error"] += 1
                    stats["Quick Match"]["time"] += elapsed
                    print(f"\033[F", end="")
                    print_progress(i, match_count, f"Error: {str(e)[:30]}", "error")
                
                await asyncio.sleep(1)
                
                # Deep analysis (every other job to save costs)
                if i % 2 == 1 and i <= match_count:
                    start = time.time()
                    try:
                        print_progress(i, match_count, f"Deep analyzing...", "running")
                        result = await matching_plugin._deep_analyze_job_match(
                            resume_text, job, result.get('score', 50)
                        )
                        elapsed = (time.time() - start) * 1000
                        stats["Deep Analysis"]["success"] += 1
                        stats["Deep Analysis"]["time"] += elapsed
                        print(f"\033[F", end="")
                        print_progress(i, match_count, f"Deep: {result.get('score', '?')}/100", "success")
                    except Exception as e:
                        elapsed = (time.time() - start) * 1000
                        stats["Deep Analysis"]["error"] += 1
                        stats["Deep Analysis"]["time"] += elapsed
                        print(f"\033[F", end="")
                        print_progress(i, match_count, f"Deep error: {str(e)[:25]}", "error")
                    
                    await asyncio.sleep(1.5)
        
        # =====================================================================
        # 4. RESUME TAILORING
        # =====================================================================
        print_section(f"Resume Tailoring (0/{tailor_count})", "✏️")
        
        if not resumes or not jobs:
            print(f"  {Colors.YELLOW}⚠️  Skipping: Need saved resumes and jobs in database{Colors.END}")
        else:
            resume = resumes[0]
            resume_text = resume.get('text', resume.get('content', ''))
            
            requests = random.sample(TAILORING_REQUESTS, min(tailor_count, len(TAILORING_REQUESTS)))
            
            for i, (request, job) in enumerate(zip(requests, jobs[:tailor_count]), 1):
                start = time.time()
                try:
                    print_progress(i, tailor_count, request[:35], "running")
                    result = await tailoring_plugin.improve_resume_bullet(
                        resume_text=resume_text[:2000],
                        job_description=job.get('description', '')[:1500],
                        job_title=job.get('title', 'Software Engineer'),
                        company=job.get('company', 'Unknown'),
                        user_request=request
                    )
                    elapsed = (time.time() - start) * 1000
                    stats["Resume Tailoring"]["success"] += 1
                    stats["Resume Tailoring"]["time"] += elapsed
                    print(f"\033[F", end="")
                    print_progress(i, tailor_count, request[:35], "success")
                except Exception as e:
                    elapsed = (time.time() - start) * 1000
                    stats["Resume Tailoring"]["error"] += 1
                    stats["Resume Tailoring"]["time"] += elapsed
                    print(f"\033[F", end="")
                    print_progress(i, tailor_count, f"Error: {str(e)[:30]}", "error")
                
                await asyncio.sleep(1.5)
        
        # =====================================================================
        # COMPLETE
        # =====================================================================
        end_session(session, success=True)
        
        total_time = time.time() - overall_start
        
        print_header("✅ Baseline Generation Complete")
        print_stats(stats)
        
        print(f"\n{Colors.BOLD}⏱️  Total Time: {total_time:.1f}s{Colors.END}")
        print(f"{Colors.DIM}Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.END}")
        print(f"\n{Colors.CYAN}📊 View results in your Observatory dashboard!{Colors.END}\n")
        
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}⚠️  Interrupted by user{Colors.END}")
        end_session(session, success=False, error="User interrupted")
        print_stats(stats)
        
    except Exception as e:
        print(f"\n{Colors.RED}❌ Fatal error: {e}{Colors.END}")
        end_session(session, success=False, error=str(e))
        print_stats(stats)
        raise


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Observatory baseline data")
    parser.add_argument("--quick", action="store_true", help="Quick run (~20 calls)")
    parser.add_argument("--jobs-only", action="store_true", help="Only job search operations")
    args = parser.parse_args()
    
    asyncio.run(main(quick_mode=args.quick, jobs_only=args.jobs_only))