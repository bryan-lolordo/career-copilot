# 🚀 Career Copilot - Agentic AI Job Search Assistant

**[📺 View Live Demo on LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7389604326559817728/)**

An intelligent agentic AI system powered by **Microsoft Semantic Kernel** and **Azure OpenAI** that autonomously helps users find jobs, match resumes to opportunities, and manage job searches through natural language conversations and automated tool orchestration.

## 🎯 What is Career Copilot?

Career Copilot is an **agentic AI system** that demonstrates production-ready patterns for autonomous AI applications. The system uses natural language to understand intent, automatically selects appropriate tools, and maintains conversation context across multi-turn dialogues.

**Key Technologies:**
- 🤖 Microsoft Semantic Kernel for AI orchestration
- 🧠 Azure OpenAI (GPT-4) for language understanding
- 🎨 Streamlit for web interface
- 🔍 SerpAPI for job search data

## 💡 Why This Project?

This project demonstrates **production-ready agentic AI patterns**:

✅ **Autonomous Decision Making** - AI selects tools without hardcoded commands  
✅ **Multi-Turn Context** - Maintains conversation state across interactions  
✅ **Self-Improvement** - AI critiques and refines its own outputs  
✅ **Safe Code Generation** - Natural language to SQL with injection prevention  
✅ **Plugin Architecture** - Modular, extensible design  

---

## 🏗️ Agentic AI Architecture

### Core Agent System

The system uses **Microsoft Semantic Kernel** as the orchestration framework, enabling the AI to:

1. **Understand intent** from natural language
2. **Automatically select tools** (plugins) based on context
3. **Chain multiple operations** to accomplish complex tasks
4. **Maintain conversation state** across multi-turn dialogues
5. **Self-improve** by critiquing and refining its own outputs

### Plugin Ecosystem
```
┌─────────────────────────────────────────────────┐
│          Azure OpenAI (GPT-4)                   │
│          Semantic Kernel Agent                  │
└───────────────┬─────────────────────────────────┘
                │
                ├─> JobPlugin (SerpAPI integration)
                ├─> ResumeMatchingPlugin (AI scoring)
                ├─> QueryDatabasePlugin (NL to SQL)
                ├─> ResumeTailoringPlugin (Content optimization)
                ├─> SelfImprovingMatchPlugin (Iterative refinement)
                ├─> ResumePreprocessorPlugin (Text extraction)
                └─> JobPreprocessorPlugin (Data cleaning)
```

Each plugin exposes kernel functions that the AI agent can autonomously invoke based on user intent.

---

## ✨ Key Features

### 🤖 Agentic Capabilities

**Multi-Turn Conversational Memory**
- Maintains context across conversation history
- Tracks active jobs, resumes, and match results
- Handles follow-up questions without re-querying ("tell me more about #2")
- Reference resolution ("this job", "my resume", "the previous one")

**Autonomous Tool Orchestration**
- AI automatically selects appropriate plugins based on intent
- Chains multiple tool calls to complete complex workflows
- No hardcoded command syntax - pure natural language

**Self-Improving Workflows**
- `SelfImprovingMatchPlugin` implements AI self-critique loops
- Iteratively refines resume-job match analysis
- AI critic evaluates quality and provides refinement guidance
- Demonstrates meta-learning patterns

### 🎯 Resume Matching with AI Scoring

**Multi-Dimensional Scoring Algorithm**
```
Match Score = weighted_average(
    skill_alignment,      # Technical & domain skills
    experience_match,     # Years and role level
    role_fit,            # Job responsibilities alignment
    cultural_fit         # Company values & work style
)
```

**Detailed AI Reasoning**
- Explains *why* each score was assigned
- Identifies strengths and gaps
- Provides actionable improvement suggestions

**Match Workflow**
1. User initiates: "match my resume"
2. Agent lists available resumes
3. User selects resume by natural reference ("the first one")
4. Agent offers job filtering options
5. Agent autonomously executes matching against all jobs
6. Results stored in database with full provenance

### 🧠 Natural Language to SQL

**Safe Query Generation**
```
User: "Show me jobs from Deloitte with salary over $120k"
       ↓
AI analyzes schema + user intent
       ↓
Generates: SELECT * FROM jobs 
           WHERE company LIKE '%Deloitte%' 
           AND salary_max > 120000
       ↓
Validates (SELECT only, no injection)
       ↓
Executes and formats results
```

**Safety Features**
- Read-only queries (no DELETE/UPDATE/DROP)
- SQL injection prevention through validation
- Schema-aware query generation
- Graceful error handling

### 🔄 Self-Improving Match Plugin

**Iterative Refinement Process**
```python
for iteration in range(max_iterations):
    # 1. Deep analysis
    analysis = analyze_match(resume, job, prior_guidance)
    
    # 2. AI critic evaluates
    critique = critic.evaluate(analysis)
    
    # 3. If quality insufficient, refine
    if critique.quality_score < threshold:
        guidance.append(critique.suggestions)
        continue
    else:
        break

return final_improved_match
```

This demonstrates meta-learning: the AI learns from its own mistakes within a single session.

### 📊 Job Search & Management

**Job Discovery**
- Search via SerpAPI (Google Jobs integration)
- Save interesting positions to local database
- Bulk operations (save all, delete by company)

**Job Preprocessing**
- Automatic extraction of structured data
- Text cleaning and normalization
- Duplicate detection

---

## 🛠️ Installation & Setup

### Prerequisites
```bash
Python 3.8+
Azure OpenAI API access
SerpAPI key (for job search)
```

### Quick Start

1. **Clone the repository**
```bash
git clone <repository-url>
cd career_copilot
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys:
# - AZURE_OPENAI_KEY
# - AZURE_OPENAI_BASE_URL
# - AZURE_OPENAI_DEPLOYMENT_NAME
# - SERPAPI_KEY
```

5. **Run the application**
```bash
streamlit run streamlit_app.py
```

Navigate to `http://localhost:8501`

---

## 💬 Example Conversations

### Job Search & Exploration
```
User: search for Python Developer jobs in Chicago
AI: [searches SerpAPI] Found 15 Python Developer positions in Chicago...

User: tell me about the second one
AI: [calls get_job_details with job_number=2] This is a Senior Python Developer role at Acme Corp...

User: save jobs 1, 2, and 5
AI: [saves selected jobs to database] ✅ Saved 3 jobs to your database
```

### Resume Matching with Context
```
User: match my resume
AI: 📄 Available resumes:
    1. AI Engineer Resume
    Which resume would you like to match?

User: the first one
AI: ✅ Selected "AI Engineer Resume"
    How would you like to filter jobs?
    - All jobs in database (23 jobs)
    - Only unmatched jobs (15 jobs)
    - Filter by keyword

User: only unmatched jobs
AI: [executes matching] ✅ Matched your resume to 15 jobs
    Top matches:
    1. Senior AI Engineer at Microsoft (94/100)
    2. ML Engineer at OpenAI (89/100)
    ...

User: why did I get 94% for the first one?
AI: [retrieves stored reasoning] You scored 94% because:
    - Skills: 95/100 (Strong match on Python, PyTorch, Azure)
    - Experience: 98/100 (8 years matches their 7-10 requirement)
    - Role Fit: 90/100 (Your agentic AI work aligns with their projects)
```

### Natural Language Database Queries
```
User: show me all jobs from Deloitte
AI: [generates SQL] Found 8 jobs from Deloitte...

User: which ones scored above 85?
AI: [queries matches table] Here are 3 Deloitte jobs with scores > 85...

User: when did I save these?
AI: [adds date filter to query] These jobs were saved between Oct 15-20...
```

### Self-Improvement
```
User: improve the match for job #1
AI: 🤖 Running self-improvement (2 iterations)...
    
    Iteration 1:
    - Initial score: 87/100
    - AI Critic: "Skills section lacks depth on ML frameworks"
    
    Iteration 2:
    - Refined analysis with better framework evaluation
    - New score: 91/100
    - Quality improved by 4 points ✅
```

---

## 🎯 Agentic AI Patterns Demonstrated

### 1. **Autonomous Function Calling**
The agent decides which plugin functions to call based on semantic understanding of user intent, without explicit command syntax.

### 2. **Multi-Step Workflows**
Complex tasks (like resume matching) are broken into guided multi-step flows where the agent manages state between steps.

### 3. **Context Awareness**
The `ConversationMemory` service maintains rich context:
- Active resume/job references
- Recent search results
- Pending multi-step workflows
- User preferences learned over time

### 4. **Self-Critique Loops**
The `SelfImprovingMatchPlugin` implements a critic-refine loop:
```
Initial Analysis → AI Critic → Refinement Guidance → Re-Analysis → Final Output
```

### 5. **Natural Language API**
All database interactions use natural language → SQL generation, demonstrating LLM-powered API layers.

### 6. **Graceful Degradation**
When the agent can't fulfill a request, it explains why and offers alternatives rather than failing silently.

---

## 🛠️ Technical Stack

### AI/ML Layer
```
├─ Microsoft Semantic Kernel (agent orchestration)
├─ Azure OpenAI Service (GPT-4)
│  ├─ Chat completion with function calling
│  ├─ Few-shot prompting for structured outputs
│  └─ Token management & optimization
├─ Prompt Engineering
│  ├─ System prompts with role definition
│  ├─ Few-shot examples for consistency
│  └─ Chain-of-thought reasoning
└─ Vector similarity (planned feature)
```

### Backend
```
├─ Python 3.8+
├─ SQLite (job & resume database)
├─ SerpAPI (job search provider)
├─ pdfplumber & docx2txt (document parsing)
└─ asyncio (async plugin execution)
```

### Frontend
```
├─ Streamlit (multi-page web app)
├─ Custom HTML/CSS components
└─ Real-time UI updates
```

---

## 📁 Project Structure
```
career_copilot/
├── agents/
│   ├── plugins/                         # Semantic Kernel plugins
│   │   ├── JobPlugin.py                 # Job search & management
│   │   ├── ResumeMatchingPlugin.py      # AI-powered matching
│   │   ├── QueryDatabasePlugin.py       # NL to SQL
│   │   ├── ResumeTailoringPlugin.py     # Resume optimization
│   │   ├── SelfImprovingMatchPlugin.py  # Self-critique loops
│   │   ├── ResumePreprocessorPlugin.py  # Resume text processing
│   │   ├── JobPreprocessorPlugin.py     # Job data cleaning
│   │   └── __init__.py
│   ├── semantic_kernel_setup.py         # Kernel configuration & CLI
│   └── __init__.py
│
├── services/
│   ├── chatbot.py                       # Main agent orchestration
│   ├── enhanced_chatbot.py              # Advanced conversational agent
│   ├── conversation_memory.py           # Conversation state management
│   ├── database_service.py              # Data access layer
│   ├── db.py                            # SQLite operations
│   ├── job_api.py                       # SerpAPI integration
│   ├── resume_parser.py                 # Document parsing
│   └── __init__.py
│
├── pages/                               # Streamlit multi-page UI
│   ├── 1_💬_Chatbot.py
│   ├── 2_📄_Resume_Manager.py
│   ├── 3_🔍_Job_Search.py
│   ├── 4_💾_Saved_Jobs.py
│   ├── 5_🎯_Resume_Matching.py
│   ├── 6_🔬_Match_Analysis.py
│   └── 7_✏️_Resume_Tailoring.py
│
├── ui/                                  # Reusable UI components
│   ├── components.py
│   ├── utils.py
│   └── __init__.py
│
├── data/                                # Local data and database
│   ├── Bryan LoLordo - AI Engineer.docx # Example resume
│   └── career_copilot.db                # SQLite database
│
├── documents/                           # Project documents & templates
│   └── career_copilot-search_service/
│       ├── parameters.json
│       └── template.json
│
├── pics/                                # Images and demo media
│   ├── career_copilot.mp4
│   ├── match_analysis.png
│   ├── match_analysis_2.png
│   ├── resume_matching.png
│   └── resume_tailoring.png
│
├── tests/                               # Test suite
│   ├── backup.py
│   ├── prompt.py
│   ├── smoke_test.py
│   ├── test_baseline.py
│   ├── test_career_copilot.py
│   ├── test_career_copilot_custom.py
│   ├── test_config.json
│   └── __init__.py
│
├── .env                                 # Environment variables (not tracked)
├── ARCHITECTURE.md                      # Detailed architecture docs
├── LICENSE
├── llm_judge.py                         # LLM evaluation script
├── observatory_config.py                # Observatory integration config
├── README.md
├── requirements.txt
├── streamlit_app.py                     # Web app entry point
└── __pycache__/                         # Python bytecode cache (not tracked)
```

---

## 📊 Database Schema
```sql
-- Resumes
CREATE TABLE resumes (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    content TEXT NOT NULL,
    file_type TEXT,
    uploaded_at TIMESTAMP,
    preprocessed_content TEXT
);

-- Jobs
CREATE TABLE jobs (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    company TEXT NOT NULL,
    location TEXT,
    description TEXT,
    link TEXT,
    created_at TIMESTAMP,
    preprocessed_description TEXT
);

-- Resume-Job Matches
CREATE TABLE resume_matches (
    id INTEGER PRIMARY KEY,
    resume_id INTEGER,
    job_id INTEGER,
    overall_score INTEGER,
    skill_alignment INTEGER,
    experience_match INTEGER,
    role_fit INTEGER,
    reasoning TEXT,
    matched_at TIMESTAMP,
    FOREIGN KEY (resume_id) REFERENCES resumes(id),
    FOREIGN KEY (job_id) REFERENCES jobs(id)
);
```

---

## 🔐 Security Considerations

1. **SQL Injection Prevention**: All database queries are validated before execution
2. **Read-Only Queries**: Plugin system only allows SELECT statements
3. **API Key Management**: Environment variables for sensitive credentials
4. **Input Sanitization**: User inputs are validated before processing
5. **Token Limits**: Text truncation prevents excessive API costs

---

## 🚧 Roadmap

**Planned Features:**
- Vector embeddings for semantic job search
- RAG (Retrieval-Augmented Generation) for resume writing assistance
- Multi-agent collaboration (separate agents for search, analysis, writing)
- Streaming responses for real-time feedback
- Cover letter generation tailored to job descriptions
- Interview preparation based on job requirements
- Salary negotiation guidance using market data

---

## 📚 Architecture Documentation

For detailed technical architecture, see [ARCHITECTURE.md](ARCHITECTURE.md)

Topics covered:
- System design patterns
- Data flow diagrams
- Component hierarchy
- Plugin development guide
- UI component structure
- User journey flows

---

## 🙏 Acknowledgments

**AI Frameworks**
- [Microsoft Semantic Kernel](https://github.com/microsoft/semantic-kernel) - Agent orchestration
- [Azure OpenAI Service](https://azure.microsoft.com/en-us/products/ai-services/openai-service) - LLM provider

**Infrastructure**
- [Streamlit](https://streamlit.io/) - Web UI framework
- [SerpAPI](https://serpapi.com/) - Job search data
- [pdfplumber](https://github.com/jsvine/pdfplumber) - PDF parsing
- [docx2txt](https://github.com/ankushshah89/python-docx2txt) - DOCX parsing

---

## 📄 License

MIT License

---

## 👤 Author

**Bryan LoLordo**
- Specialization: Agentic AI Systems, GenAI Engineering
- Focus: Production-ready AI agents with Microsoft Semantic Kernel

---

**Built with ❤️ using Agentic AI patterns**

*Demonstrating enterprise-grade AI agent development* 🎯