# MedGuide v2.0

**AI-Powered Medical Learning Companion**

---

## 📋 Table of Contents

- [Problem](#-problem)
- [Solution](#-solution)
- [Architecture](#-architecture)
- [Features](#-features)
- [Setup Instructions](#-setup-instructions)
- [Usage Examples](#-usage-examples)
- [ADK Concepts](#-adk-concepts-demonstrated)
- [Project Structure](#-project-structure)
- [Safety Design](#-safety-design)
- [API Reference](#-api-reference)

---

## 🎯 Problem

Medical students face a **retention crisis**:

- **70% of content forgotten** within days without reinforcement
- **4,000+ hours** invested in learning with suboptimal methods
- **Fragmented resources** across textbooks, lectures, and papers
- **No personalization** to individual knowledge gaps
- **Evidence disconnect** from current research

Traditional studying requires constant context-switching between explanation-seeking, self-assessment, literature review, and planning—each demanding different approaches and tools.

---

## 💡 Solution

MedGuide provides **coordinated AI tutoring** through 6 specialized agents:

| Agent | Responsibility |
|-------|----------------|
| **Educator** | Adaptive concept explanations |
| **Quiz Master** | Assessment generation with feedback |
| **Researcher** | PubMed literature search |
| **Planner** | Personalized study schedules |
| **Chat Handler** | Natural conversation |
| **Router** | Intent classification |

Orchestrated through an **explicit state machine** with safety boundaries and observability.

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MEDGUIDE ORCHESTRATOR                           │
│                        (State Machine Pattern)                          │
│                                                                         │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐  │
│  │  INPUT  │──▶│ SAFETY  │──▶│ INTENT  │──▶│  MODE   │──▶│ MEMORY  │  │
│  │VALIDATE │   │  CHECK  │   │CLASSIFY │   │ EXECUTE │   │ UPDATE  │  │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘  │
│                     │              │              │                     │
│                Boundaries    Deterministic   6 Agents                   │
│                + Refusals    + LLM Fallback                             │
├─────────────────────────────────────────────────────────────────────────┤
│                        AGENT LAYER                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │ Educator │  │   Quiz   │  │ Research │  │ Planner  │  │   Chat   │ │
│  │  Agent   │  │  Agent   │  │  Agent   │  │  Agent   │  │  Agent   │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘ │
│       │             │             │             │             │        │
│   Concepts      Quizzes      PubMed API    Study Plans    Greetings   │
├─────────────────────────────────────────────────────────────────────────┤
│                     INFRASTRUCTURE LAYER                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐        │
│  │  Observability  │  │ Adaptive Memory │  │  Safety Layer   │        │
│  │  (JSONL Logs)   │  │(Spaced Repet.)  │  │  (Boundaries)   │        │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘        │
└─────────────────────────────────────────────────────────────────────────┘
```

### State Machine Modes

```python
class AgentMode(Enum):
    IDLE = auto()      # Greetings, simple interactions
    EXPLAIN = auto()   # Medical concept explanations
    QUIZ = auto()      # Assessment generation
    SEARCH = auto()    # PubMed literature retrieval
    PLAN = auto()      # Study schedule creation
    HISTORY = auto()   # Progress review
    REFUSE = auto()    # Safety boundary enforcement
```

### Data Flow

```
User Input
    │
    ▼
┌──────────────────┐
│ Input Validation │ ─── Reject if too long/empty
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Safety Check    │ ─── REFUSE if diagnosis/treatment/emergency
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│Intent Classifier │ ─── Deterministic rules first, LLM fallback
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Mode Execution  │ ─── Route to appropriate agent
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Memory Update   │ ─── Record topic, update spaced repetition
└────────┬─────────┘
         │
         ▼
    Response
```

---

## ✨ Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Adaptive Explanations** | Content calibrated to beginner/intermediate/advanced |
| **Quiz Generation** | MCQs with clinical vignettes + flashcards |
| **Literature Search** | Real PubMed integration with citations |
| **Study Planning** | Personalized 5-7 day schedules |
| **Progress Tracking** | Learning history with recommendations |
| **Spaced Repetition** | Optimized review scheduling |

### Production Features

| Feature | Description |
|---------|-------------|
| **Safety Boundaries** | Hard refusals for diagnosis/treatment |
| **Structured Logging** | JSONL with correlation IDs |
| **Hybrid Classification** | Deterministic + LLM fallback |
| **Formal Contracts** | Typed input/output dataclasses |
| **Mode State Machine** | Explicit transitions with logging |

---

## 🚀 Setup Instructions

### Prerequisites

- Python 3.11+
- Google API Key (Gemini)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/medguide.git
cd medguide

# Install dependencies
pip install google-adk google-generativeai

# Set API key
export GOOGLE_API_KEY="your-api-key-here"
```

### Running the Agent

```bash
# Using ADK CLI
adk run medguide

# Or run directly
python -c "from medguide import root_agent; print(root_agent)"
```

### Verify Installation

```bash
# Check syntax
python -m py_compile medguide/agent.py

# Run tests (if available)
python -m pytest tests/
```

---

## 💬 Usage Examples

### Basic Interactions

```
[user]: hi
[MedGuide]: Hey! What would you like to learn about today?

[user]: explain heart failure
[MedGuide]: Heart failure occurs when the heart cannot pump blood 
effectively to meet the body's demands...
[Detailed explanation with clinical relevance]

[user]: quiz me on that
[MedGuide]: 
Question 1: A 65-year-old patient presents with bilateral leg 
edema, orthopnea, and an S3 gallop. Which type of heart failure 
is most likely?
A) Systolic heart failure
B) Diastolic heart failure
...
```

### Literature Search

```
[user]: find papers on SGLT2 inhibitors in heart failure
[MedGuide]: Searching PubMed...

Found 5 relevant articles:

1. "SGLT2 Inhibitors in Heart Failure" (Smith et al., 2023)
   - Key finding: 25% reduction in hospitalization
   - PMID: 12345678

2. "Dapagliflozin Outcomes Trial" (Johnson et al., 2022)
   - Key finding: Mortality benefit in HFrEF
   - PMID: 23456789
...
```

### Safety Boundaries

```
[user]: do I have diabetes?
[MedGuide]: I can't provide diagnoses. I'm designed for medical 
education, not clinical assessment. For health concerns, please 
consult a healthcare provider who can properly evaluate you.

[user]: what medication should I take for my headache?
[MedGuide]: I can't recommend treatments or medications. Treatment 
decisions require a healthcare provider who knows your complete 
medical history. I can explain how headache medications work 
educationally if that helps.
```

---

## 🔧 ADK Concepts Demonstrated

| # | Concept | Implementation | Location |
|---|---------|----------------|----------|
| 1 | **Multi-Agent System** | 6 specialized agents | `agent.py:L950-1020` |
| 2 | **Custom Tools** | `search_pubmed` with NCBI API | `agent.py:L680-780` |
| 3 | **Session State** | Context propagation via state dict | `agent.py:L1150-1180` |
| 4 | **Long-Term Memory** | JSON persistence + spaced repetition | `agent.py:L450-600` |
| 5 | **Orchestration** | State machine with mode transitions | `agent.py:L1100-1280` |
| 6 | **Observability** | Structured JSONL logging | `agent.py:L200-300` |

### Code Examples

**Multi-Agent Definition:**
```python
explain_agent = LlmAgent(
    model=Config.MODEL,
    name="explainer",
    description="Medical concept explanation",
    instruction=f"{AgentPrompts.SYSTEM}\n\n{AgentPrompts.EXPLAIN}",
    output_key="explanation"
)
```

**Custom Tool with Contract:**
```python
def search_pubmed(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    PRECONDITIONS:
    - Query must be non-empty medical term
    - Max results between 1-10
    
    POSTCONDITIONS:
    - Returns dict with: query, count, articles[]
    - Each article has: pmid, title, abstract, authors, journal, year, url
    """
```

**State Machine Transition:**
```python
def _transition_mode(self, new_mode: AgentMode, reason: str):
    old_mode = self._current_mode
    self._current_mode = new_mode
    obs.log_mode_transition(old_mode, new_mode, reason)
```

---

## 📁 Project Structure

```
medguide/
├── medguide/
│   ├── __init__.py          # Module exports
│   └── agent.py              # Main agent (1,285 lines)
├── agent.py                  # Root-level agent copy
├── requirements.txt          # Dependencies
├── README.md                 # This file
├── KAGGLE_WRITEUP.md         # Competition writeup
├── learner_memory.json       # Persistent learner state (auto-generated)
└── agent_logs.jsonl          # Structured logs (auto-generated)
```

### Key Components

| File | Description | Lines |
|------|-------------|-------|
| `agent.py` | Main orchestrator + all agents | 1,285 |
| `learner_memory.json` | Spaced repetition state | Auto |
| `agent_logs.jsonl` | Observability logs | Auto |

---

## 🛡 Safety Design

### Hard Boundaries (Always Refuse)

| Pattern | Response |
|---------|----------|
| Diagnosis requests | Redirect to healthcare provider |
| Treatment recommendations | Explain educational-only purpose |
| Emergency situations | Direct to 911/ER immediately |
| Drug dosing | Refuse with explanation |

### Soft Boundaries (Add Disclaimer)

| Pattern | Action |
|---------|--------|
| Symptom discussions | Append disclaimer |
| Drug information | Append disclaimer |
| Procedure descriptions | Append disclaimer |

### Implementation

```python
class SafetyBoundary:
    REFUSE_PATTERNS = [
        r"\bdiagnos(e|is)\b",
        r"\bshould i take\b",
        r"\bemergency\b",
    ]
    
    @classmethod
    def evaluate(cls, text: str) -> SafetyFlag:
        # Pattern matching against boundaries
        ...
```

---

## 📚 API Reference

### AgentMode (Enum)

```python
IDLE      # Greetings, simple interactions
EXPLAIN   # Medical concept explanations  
QUIZ      # Assessment generation
SEARCH    # PubMed literature retrieval
PLAN      # Study schedule creation
HISTORY   # Progress review
REFUSE    # Safety boundary enforcement
```

### SafetyFlag (Enum)

```python
SAFE              # No concerns
NEEDS_DISCLAIMER  # Add educational disclaimer
REFUSE_DIAGNOSIS  # Hard refuse - diagnosis request
REFUSE_TREATMENT  # Hard refuse - treatment request
REFUSE_EMERGENCY  # Hard refuse - emergency situation
```

### ToolRegistry

```python
search_pubmed(query: str, max_results: int = 5) -> Dict[str, Any]
save_study_plan(content: str, topic: str = "Medical") -> Dict[str, Any]
```

### MemoryManager

```python
record_study(topic: str, activity: str, performance: float = None)
get_due_topics() -> List[str]
get_progress_summary() -> str
get_recommendations() -> str
```

---

## 📊 Observability

### Log Format

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "correlation_id": "a1b2c3d4",
  "event_type": "MODE_TRANSITION",
  "component": "Orchestrator",
  "message": "IDLE → EXPLAIN",
  "duration_ms": 45.2
}
```

### Event Types

| Type | Description |
|------|-------------|
| `REQUEST_START` | New request received |
| `SAFETY_CHECK` | Safety evaluation result |
| `INTENT_CLASSIFIED` | Intent determined |
| `MODE_TRANSITION` | State machine transition |
| `TOOL_CALL` | External tool invocation |
| `MEMORY_UPDATE` | Learning record saved |
| `REQUEST_COMPLETE` | Request finished |

---


## 📄 License

MIT License

