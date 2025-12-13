"""
Conversation Memory System for Career Copilot Chatbot

This module provides stateful conversation tracking to enable natural,
context-aware interactions across multiple turns.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ConversationIntent(Enum):
    """Track the user's current intent/goal in the conversation"""
    GENERAL_CHAT = "general_chat"
    JOB_SEARCH = "job_search"
    RESUME_MATCHING = "resume_matching"
    MATCH_ANALYSIS = "match_analysis"
    RESUME_TAILORING = "resume_tailoring"
    RESUME_MANAGEMENT = "resume_management"
    JOB_MANAGEMENT = "job_management"


@dataclass
class ConversationContext:
    """Stores the current context of what's being discussed"""
    # Current focus
    current_resume_id: Optional[int] = None
    current_job_id: Optional[int] = None
    current_match_id: Optional[int] = None
    
    # Recent items for "show me more" / "the previous one"
    recent_job_ids: List[int] = field(default_factory=list)
    recent_resume_ids: List[int] = field(default_factory=list)
    recent_match_results: List[Dict] = field(default_factory=list)
    
    # Plugin-specific context
    active_resume_id: Optional[int] = None  # For backwards compatibility
    active_job_id: Optional[int] = None      # For backwards compatibility
    last_match_analysis: Optional[Dict] = None  # Detailed analysis for "why?" questions
    last_tailoring_suggestions: Optional[Dict] = None  # Suggestions for "apply #2"
    last_query_results: Optional[List] = None  # Last database query results
    
    # NEW: Job search exploration context (for browsing jobs before saving)
    last_searched_jobs: Optional[List[Dict]] = None  # Jobs from last search
    last_search_query: Optional[str] = None  # Query used
    last_search_location: Optional[str] = None  # Location used
    active_job_index: Optional[int] = None  # Currently viewing job index
    
    # NEW: Multi-step matching workflow context
    available_resumes: Optional[List[Dict]] = None  # Resumes for selection
    selected_resume_for_matching: Optional[Dict] = None  # Resume selected for matching
    awaiting_resume_selection: bool = False  # Waiting for user to pick resume
    awaiting_job_filter_selection: bool = False  # Waiting for user to pick job filter
    selected_job_ids_for_matching: Optional[List[int]] = None  # Filtered job IDs
    
    # Confirmation handling (keep for backward compatibility but less used now)
    awaiting_confirmation: bool = False
    pending_action: Optional[Any] = None  # Can be dict or string
    
    # Last action performed
    last_action: Optional[str] = None
    last_action_result: Optional[Dict] = None
    
    # User preferences (learned over time)
    preferred_locations: List[str] = field(default_factory=list)
    preferred_job_types: List[str] = field(default_factory=list)
    
    # Current intent
    intent: ConversationIntent = ConversationIntent.GENERAL_CHAT
    
    # Metadata
    session_start: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def request_confirmation(self, action: Any):
        """Ask for confirmation before performing an action."""
        self.awaiting_confirmation = True
        self.pending_action = action

    def confirm(self):
        """Mark confirmation as granted."""
        self.awaiting_confirmation = False
        self.pending_action = None

    def cancel(self):
        """Cancel a pending action."""
        self.awaiting_confirmation = False
        self.pending_action = None


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation"""
    timestamp: datetime
    user_message: str
    assistant_message: str
    intent: ConversationIntent
    plugins_used: List[str]
    context_snapshot: Dict[str, Any]


class ConversationMemory:
    """
    Manages conversation state and history for natural, context-aware interactions.
    
    Features:
    - Track what's being discussed (jobs, resumes, matches)
    - Remember recent items for "show me the next one" queries
    - Maintain conversation history for context
    - Learn user preferences over time
    - Enable follow-up questions without re-explaining
    """
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.context = ConversationContext()
        self.history: List[ConversationTurn] = []
        self.pending_actions: List[Dict] = []  # Actions to be executed by UI

        # ADD THESE TWO LINES:
        self.conversation_id = None  # Set by chatbot/kernel before each turn
        self.turn_number = 0         # Incremented by chatbot/kernel for each user message
    
    def update_context(self, **kwargs):
        """Update any aspect of the conversation context"""
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)
        self.context.last_updated = datetime.now()
    
    def add_turn(self, user_message: str, assistant_message: str, 
                 plugins_used: List[str], intent: Optional[ConversationIntent] = None):
        """Record a conversation turn"""
        turn = ConversationTurn(
            timestamp=datetime.now(),
            user_message=user_message,
            assistant_message=assistant_message,
            intent=intent or self.context.intent,
            plugins_used=plugins_used,
            context_snapshot=self._get_context_snapshot()
        )
        self.history.append(turn)
    
    def set_current_focus(self, resume_id: Optional[int] = None, 
                         job_id: Optional[int] = None,
                         match_id: Optional[int] = None):
        """Set what the user is currently focused on"""
        if resume_id is not None:
            self.context.current_resume_id = resume_id
            self.context.active_resume_id = resume_id  # Backwards compatibility
            if resume_id not in self.context.recent_resume_ids:
                self.context.recent_resume_ids.insert(0, resume_id)
                self.context.recent_resume_ids = self.context.recent_resume_ids[:10]
        
        if job_id is not None:
            self.context.current_job_id = job_id
            self.context.active_job_id = job_id  # Backwards compatibility
            if job_id not in self.context.recent_job_ids:
                self.context.recent_job_ids.insert(0, job_id)
                self.context.recent_job_ids = self.context.recent_job_ids[:10]
        
        if match_id is not None:
            self.context.current_match_id = match_id
    
    def add_match_result(self, match_result: Dict):
        """Store a match result for later reference"""
        self.context.recent_match_results.insert(0, match_result)
        self.context.recent_match_results = self.context.recent_match_results[:20]
    
    def set_match_analysis(self, analysis: Dict):
        """Store detailed match analysis for 'why?' questions"""
        self.context.last_match_analysis = analysis
    
    def set_tailoring_suggestions(self, suggestions: Dict):
        """Store tailoring suggestions for 'apply #2' commands"""
        self.context.last_tailoring_suggestions = suggestions
    
    def set_query_results(self, results: List):
        """Store query results for 'show me more' requests"""
        self.context.last_query_results = results
    
    def get_current_job(self) -> Optional[int]:
        """Get the job currently being discussed"""
        return self.context.current_job_id or self.context.active_job_id
    
    def get_current_resume(self) -> Optional[int]:
        """Get the resume currently being discussed"""
        return self.context.current_resume_id or self.context.active_resume_id
    
    def get_current_match(self) -> Optional[int]:
        """Get the match currently being analyzed"""
        return self.context.current_match_id
    
    def get_recent_jobs(self, limit: int = 5) -> List[int]:
        """Get recently discussed jobs"""
        return self.context.recent_job_ids[:limit]
    
    def get_recent_matches(self, limit: int = 5) -> List[Dict]:
        """Get recent match results"""
        return self.context.recent_match_results[:limit]
    
    def get_match_analysis(self) -> Optional[Dict]:
        """Get the last detailed match analysis"""
        return self.context.last_match_analysis
    
    def get_tailoring_suggestions(self) -> Optional[Dict]:
        """Get the last tailoring suggestions"""
        return self.context.last_tailoring_suggestions
    
    def add_pending_action(self, action_type: str, params: Dict):
        """Queue an action for the UI to execute"""
        self.pending_actions.append({
            'type': action_type,
            'params': params,
            'timestamp': datetime.now()
        })
    
    def get_pending_actions(self) -> List[Dict]:
        """Get and clear pending actions"""
        actions = self.pending_actions.copy()
        self.pending_actions.clear()
        return actions
    
    def learn_preference(self, preference_type: str, value: str):
        """Learn a user preference from the conversation"""
        if preference_type == "location":
            if value not in self.context.preferred_locations:
                self.context.preferred_locations.append(value)
        elif preference_type == "job_type":
            if value not in self.context.preferred_job_types:
                self.context.preferred_job_types.append(value)
    
    def get_context_for_prompt(self) -> str:
        """Generate a context summary to include in the LLM prompt"""
        context_parts = []
        
        if self.context.current_resume_id:
            context_parts.append(f"Currently discussing resume ID: {self.context.current_resume_id}")
        
        if self.context.current_job_id:
            context_parts.append(f"Currently discussing job ID: {self.context.current_job_id}")
        
        if self.context.current_match_id:
            context_parts.append(f"Currently analyzing match ID: {self.context.current_match_id}")
        
        if self.context.recent_job_ids:
            context_parts.append(f"Recently viewed jobs: {self.context.recent_job_ids[:3]}")
        
        if self.context.last_action:
            context_parts.append(f"Last action: {self.context.last_action}")
        
        if self.context.preferred_locations:
            context_parts.append(f"User's preferred locations: {', '.join(self.context.preferred_locations)}")
        
        if self.context.preferred_job_types:
            context_parts.append(f"User's preferred job types: {', '.join(self.context.preferred_job_types)}")
        
        return "\n".join(context_parts) if context_parts else "No prior context."
    
    def detect_intent(self, user_message: str) -> ConversationIntent:
        """Detect user intent from their message"""
        message_lower = user_message.lower()
        
        # Intent detection keywords
        if any(word in message_lower for word in ["search for jobs", "find jobs", "look for jobs", "job openings"]):
            return ConversationIntent.JOB_SEARCH
        
        if any(word in message_lower for word in ["match my resume", "how do i match", "compatibility", "matching score"]):
            return ConversationIntent.RESUME_MATCHING
        
        if any(word in message_lower for word in ["why did i get", "explain the match", "why is my score", "match analysis"]):
            return ConversationIntent.MATCH_ANALYSIS
        
        if any(word in message_lower for word in ["tailor my resume", "customize resume", "improve my resume for"]):
            return ConversationIntent.RESUME_TAILORING
        
        if any(word in message_lower for word in ["upload resume", "my resumes", "resume manager"]):
            return ConversationIntent.RESUME_MANAGEMENT
        
        if any(word in message_lower for word in ["saved jobs", "my jobs", "bookmarked"]):
            return ConversationIntent.JOB_MANAGEMENT
        
        return ConversationIntent.GENERAL_CHAT
    
    def _get_context_snapshot(self) -> Dict[str, Any]:
        """Get current context as a dictionary"""
        return {
            'current_resume_id': self.context.current_resume_id,
            'current_job_id': self.context.current_job_id,
            'current_match_id': self.context.current_match_id,
            'recent_job_ids': self.context.recent_job_ids[:5],
            'intent': self.context.intent.value,
        }
    
    def get_conversation_summary(self) -> str:
        """Generate a summary of the conversation for display"""
        summary_parts = [
            f"Session started: {self.context.session_start.strftime('%I:%M %p')}",
            f"Turns: {len(self.history)}",
            f"Current intent: {self.context.intent.value}"
        ]
        
        if self.context.recent_job_ids:
            summary_parts.append(f"Jobs viewed: {len(self.context.recent_job_ids)}")
        
        if self.context.recent_match_results:
            summary_parts.append(f"Matches run: {len(self.context.recent_match_results)}")
        
        return " | ".join(summary_parts)


class ConversationMemoryManager:
    """
    Manages multiple conversation sessions.
    In production, this would be backed by a database.
    """
    
    def __init__(self):
        self._sessions: Dict[str, ConversationMemory] = {}
    
    def get_session(self, session_id: str) -> ConversationMemory:
        """Get or create a conversation session"""
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationMemory(session_id)
        return self._sessions[session_id]
    
    def clear_session(self, session_id: str):
        """Clear a conversation session"""
        if session_id in self._sessions:
            del self._sessions[session_id]


# Singleton instance for Streamlit
_global_memory_manager = ConversationMemoryManager()


def get_memory_manager() -> ConversationMemoryManager:
    """Get the global conversation memory manager"""
    return _global_memory_manager