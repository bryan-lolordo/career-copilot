#!/usr/bin/env python3
"""
Career Copilot - Multi-Turn Scenario Test Runner
Location: career-copilot/run_scenarios.py

Runs multi-turn test scenarios through the full chatbot pipeline,
logging all calls to Observatory with test metadata.

Usage:
    python run_scenarios.py                     # Run all scenarios
    python run_scenarios.py --scenario job_search_explore_save  # Run one
    python run_scenarios.py --list              # List available scenarios
    python run_scenarios.py --dry-run           # Preview without running
"""

import json
import asyncio
import argparse
import time
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional

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


def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}  {text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.END}\n")


def print_scenario_header(scenario_id: str, description: str, turn_count: int):
    print(f"\n{Colors.BOLD}{Colors.CYAN}┌{'─'*68}┐{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}│ 📋 {scenario_id:<62} │{Colors.END}")
    print(f"{Colors.CYAN}│ {description[:64]:<64} │{Colors.END}")
    print(f"{Colors.CYAN}│ Turns: {turn_count:<59} │{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}└{'─'*68}┘{Colors.END}")


def print_turn(turn_num: int, user_input: str, expected_tools: List[str]):
    print(f"\n{Colors.BOLD}{Colors.YELLOW}── Turn {turn_num} {'─'*55}{Colors.END}")
    print(f"{Colors.BOLD}🧑 USER:{Colors.END} {user_input}")
    if expected_tools:
        print(f"{Colors.DIM}   Expected: {', '.join(expected_tools)}{Colors.END}")


def print_response(response: str, latency_ms: float):
    # Truncate long responses for display
    display = response[:500] + "..." if len(response) > 500 else response
    print(f"{Colors.GREEN}🤖 ASSISTANT:{Colors.END} {display}")
    print(f"{Colors.DIM}   ({latency_ms:.0f}ms){Colors.END}")


def print_error(error: str):
    print(f"{Colors.RED}❌ ERROR: {error}{Colors.END}")


def print_success(message: str):
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")


def print_summary(results: List[Dict]):
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}📊 SCENARIO TEST SUMMARY{Colors.END}")
    print(f"{'='*70}")
    
    total = len(results)
    passed = sum(1 for r in results if r["success"])
    failed = total - passed
    
    for r in results:
        status = f"{Colors.GREEN}✅ PASS{Colors.END}" if r["success"] else f"{Colors.RED}❌ FAIL{Colors.END}"
        print(f"  {status} {r['scenario_id']:<40} ({r['turns_completed']}/{r['total_turns']} turns, {r['duration']:.1f}s)")
        if r.get("error"):
            print(f"       {Colors.RED}Error: {r['error'][:60]}{Colors.END}")
    
    print(f"{'─'*70}")
    print(f"{Colors.BOLD}  Total: {total} | Passed: {passed} | Failed: {failed}{Colors.END}")
    print(f"{'='*70}\n")


# =============================================================================
# SCENARIO RUNNER
# =============================================================================

class ScenarioRunner:
    """Runs multi-turn test scenarios through Career Copilot."""
    
    def __init__(self, scenario_file: str = "career_copilot_scenarios.json"):
        self.scenario_file = scenario_file
        self.scenarios = []
        self.results = []
        
    def load_scenarios(self) -> List[Dict]:
        """Load scenarios from JSON file."""
        with open(self.scenario_file, 'r') as f:
            self.scenarios = json.load(f)
        return self.scenarios
    
    def list_scenarios(self):
        """Print available scenarios."""
        scenarios = self.load_scenarios()
        print(f"\n{Colors.BOLD}Available Scenarios:{Colors.END}\n")
        for s in scenarios:
            print(f"  • {Colors.CYAN}{s['scenario_id']}{Colors.END}")
            print(f"    {s['description']}")
            print(f"    {len(s['conversation'])} turns\n")
    
    async def run_scenario(self, scenario: Dict, dry_run: bool = False) -> Dict:
        """
        Run a single scenario through the chatbot.
        
        Args:
            scenario: Scenario dict with conversation turns
            dry_run: If True, just preview without executing
            
        Returns:
            Result dict with success, turns_completed, errors
        """
        scenario_id = scenario["scenario_id"]
        description = scenario["description"]
        conversation = scenario["conversation"]
        
        print_scenario_header(scenario_id, description, len(conversation))
        
        if dry_run:
            for turn in conversation:
                print_turn(turn["turn"], turn["user"], turn.get("expected_tools", []))
                print(f"{Colors.DIM}   [DRY RUN - would execute here]{Colors.END}")
            return {
                "scenario_id": scenario_id,
                "success": True,
                "turns_completed": len(conversation),
                "total_turns": len(conversation),
                "duration": 0,
                "dry_run": True
            }
        
        # Import here to avoid loading if just listing/dry-run
        from observatory_config import start_session, end_session
        from services.conversation_memory import get_memory_manager
        from agents.semantic_kernel_setup import (
            create_kernel_with_plugins,
            create_chat_history_with_system_prompt,
            create_execution_settings,
        )
        
        # Create fresh kernel and memory for this scenario
        memory_manager = get_memory_manager()
        memory = memory_manager.get_session(f"scenario_{scenario_id}")
        kernel, chat_completion, db_service, memory = create_kernel_with_plugins(memory)
        
        # Create chat history (separate from memory, matches chatbot.py pattern)
        history = create_chat_history_with_system_prompt()
        execution_settings = create_execution_settings()
        
        # Start Observatory session for this scenario
        session = start_session(
            operation_type="scenario_test",
            metadata={
                "scenario_id": scenario_id,
                "description": description,
                "total_turns": len(conversation),
                "is_test": True,
                "test_dataset_id": f"scenario_{scenario_id}",
            }
        )
        
        result = {
            "scenario_id": scenario_id,
            "success": True,
            "turns_completed": 0,
            "total_turns": len(conversation),
            "duration": 0,
            "error": None,
        }
        
        start_time = time.time()
        
        try:
            for turn in conversation:
                turn_num = turn["turn"]
                user_input = turn["user"]
                expected_tools = turn.get("expected_tools", [])
                
                print_turn(turn_num, user_input, expected_tools)
                
                # Tag test metadata in memory context
                memory.context.test_metadata = {
                    "scenario_id": scenario_id,
                    "turn": turn_num,
                    "expected_tools": expected_tools,
                    "is_test": True,
                }
                
                # Execute turn through the chatbot
                turn_start = time.time()
                
                try:
                    # Use the same pattern as your chatbot
                    response = await self._execute_turn(
                        kernel, 
                        chat_completion, 
                        history,
                        execution_settings,
                        user_input,
                    )
                    
                    latency_ms = (time.time() - turn_start) * 1000
                    print_response(response, latency_ms)
                    result["turns_completed"] += 1
                    
                except Exception as e:
                    print_error(f"Turn {turn_num} failed: {e}")
                    result["error"] = str(e)
                    result["success"] = False
                    break
                
                # Small delay between turns to avoid rate limits
                await asyncio.sleep(0.5)
            
            result["duration"] = time.time() - start_time
            
            end_session(session, success=result["success"])
            print_success(f"Scenario completed: {result['turns_completed']}/{result['total_turns']} turns")
            
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            result["duration"] = time.time() - start_time
            end_session(session, success=False, error=str(e))
            print_error(f"Scenario failed: {e}")
        
        return result
    
    async def _execute_turn(
        self, 
        kernel, 
        chat_completion,
        history,
        execution_settings,
        user_input: str,
    ) -> str:
        """
        Execute a single turn through the chatbot.
        
        This mimics your chatbot's message handling.
        """
        # Add user message to history
        history.add_user_message(user_input)
        
        # Get response from chat completion
        response = await chat_completion.get_chat_message_content(
            chat_history=history,
            settings=execution_settings,
            kernel=kernel,
        )
        
        # Add assistant response to history for next turn
        assistant_message = str(response)
        history.add_assistant_message(assistant_message)
        
        return assistant_message
    
    async def run_all(self, dry_run: bool = False) -> List[Dict]:
        """Run all scenarios."""
        scenarios = self.load_scenarios()
        self.results = []
        
        for scenario in scenarios:
            result = await self.run_scenario(scenario, dry_run)
            self.results.append(result)
        
        return self.results
    
    async def run_by_id(self, scenario_id: str, dry_run: bool = False) -> Optional[Dict]:
        """Run a specific scenario by ID."""
        scenarios = self.load_scenarios()
        
        for scenario in scenarios:
            if scenario["scenario_id"] == scenario_id:
                result = await self.run_scenario(scenario, dry_run)
                self.results = [result]
                return result
        
        print(f"{Colors.RED}❌ Scenario not found: {scenario_id}{Colors.END}")
        return None


# =============================================================================
# MAIN
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Run Career Copilot multi-turn test scenarios"
    )
    parser.add_argument(
        "--scenario", "-s",
        help="Run a specific scenario by ID"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available scenarios"
    )
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Preview scenarios without executing"
    )
    parser.add_argument(
        "--file", "-f",
        default="career_copilot_scenarios.json",
        help="Path to scenarios JSON file"
    )
    
    args = parser.parse_args()
    
    runner = ScenarioRunner(scenario_file=args.file)
    
    if args.list:
        runner.list_scenarios()
        return
    
    print_header("🎭 Career Copilot Scenario Test Runner")
    print(f"{Colors.DIM}Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.END}")
    print(f"{Colors.DIM}Scenarios: {args.file}{Colors.END}")
    
    if args.scenario:
        result = await runner.run_by_id(args.scenario, dry_run=args.dry_run)
        if result:
            runner.results = [result]
    else:
        await runner.run_all(dry_run=args.dry_run)
    
    if runner.results and not args.dry_run:
        print_summary(runner.results)
        print(f"{Colors.CYAN}📊 View results in Observatory dashboard!{Colors.END}\n")


if __name__ == "__main__":
    asyncio.run(main())