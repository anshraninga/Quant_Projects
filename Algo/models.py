"""
models.py
---------
All internal dataclasses and Pydantic response models for Alpha-Seeker.

Internal types (dataclasses) flow through the LangGraph graph.
Pydantic models are used exclusively at the FastAPI boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from operator import add
from typing import Annotated, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel


# ─────────────────────────────────────────────────────────
# INTERNAL DATACLASSES  (used inside the graph)
# ─────────────────────────────────────────────────────────

@dataclass
class AgentThesis:
    agent:          str                      # "technical" | "sentiment" | "fundamental"
    direction:      str                      # "BULLISH" | "BEARISH" | "NEUTRAL"
    confidence:     float                    # 0.0 – 1.0
    reasoning:      str
    key_signals:    list[str]  = field(default_factory=list)
    raw_data:       dict       = field(default_factory=dict)
    rag_status:     str        = "PENDING"   # "SUPPORTED" | "CONTRADICTED" | "INSUFFICIENT_EVIDENCE"
    revision_flag:  bool       = False       # True if Corrective RAG revised the thesis


@dataclass
class VerificationResult:
    status:       str            # "SUPPORTED" | "CONTRADICTED" | "INSUFFICIENT_EVIDENCE"
    explanation:  str
    source_urls:  list[str]  = field(default_factory=list)
    n_docs_checked: int      = 0


@dataclass
class DebateRound:
    challenger:       str    # agent name
    challenged:       str    # agent name
    challenge:        str
    response:         str
    confidence_delta: float  # change applied to challenged agent's confidence (negative = reduced)


@dataclass
class DebateTranscript:
    rounds:             list[DebateRound]  = field(default_factory=list)
    confidence_changes: dict               = field(default_factory=dict)
    consensus_reached:  bool               = False


@dataclass
class FinalRecommendation:
    direction:            str        # "LONG" | "SHORT" | "STAY OUT"
    conviction:           float      # 0.0 – 1.0
    reasoning:            str
    key_risks:            list[str]  = field(default_factory=list)
    conditions_to_revisit: str       = ""
    agent_agreement:      str        = "SPLIT"  # "CONSENSUS" | "SPLIT" | "OPPOSED"


# ─────────────────────────────────────────────────────────
# LANGGRAPH STATE
# ─────────────────────────────────────────────────────────

class AlphaState(TypedDict):
    symbol:               str
    tech_thesis:          Optional[AgentThesis]
    sent_thesis:          Optional[AgentThesis]
    fund_thesis:          Optional[AgentThesis]
    tech_verification:    Optional[list[VerificationResult]]
    sent_verification:    Optional[list[VerificationResult]]
    fund_verification:    Optional[list[VerificationResult]]
    debate_transcript:    Optional[DebateTranscript]
    memory_context:       Optional[str]
    final_recommendation: Optional[FinalRecommendation]
    memory_record_id:     Optional[int]
    started_at:           str
    errors:               Annotated[list[str], add]  # reducer: parallel nodes append, not overwrite


# ─────────────────────────────────────────────────────────
# PYDANTIC RESPONSE MODELS  (FastAPI boundary only)
# ─────────────────────────────────────────────────────────

class AgentOut(BaseModel):
    direction:        str
    confidence:       float
    reasoning:        str
    key_signals:      list[str]
    rag_status:       str
    revision_applied: bool


class DebateRoundOut(BaseModel):
    challenger:       str
    challenged:       str
    challenge:        str
    response:         str
    confidence_delta: float


class DebateOut(BaseModel):
    rounds:            list[DebateRoundOut]
    consensus_reached: bool


class FinalOut(BaseModel):
    direction:             str
    conviction:            float
    reasoning:             str
    key_risks:             list[str]
    conditions_to_revisit: str
    agent_agreement:       str


class MemoryOut(BaseModel):
    context_summary: str
    record_id:       Optional[int]


class MetadataOut(BaseModel):
    duration_seconds:   float
    llm_calls:          int
    estimated_cost_usd: float
    rag_docs_checked:   int
    errors:             list[str]


class AnalysisResponse(BaseModel):
    symbol:    str
    timestamp: str
    final:     FinalOut
    agents:    dict[str, AgentOut]
    debate:    DebateOut
    memory:    MemoryOut
    metadata:  MetadataOut


class AgentAccuracyOut(BaseModel):
    correct: int
    total:   int
    pct:     float


class MemoryRecordOut(BaseModel):
    id:               int
    timestamp:        str
    final_direction:  Optional[str]
    final_conviction: Optional[float]
    outcome_24h:      Optional[float]
    outcome_correct:  Optional[int]


class MemoryStatsResponse(BaseModel):
    symbol:          str
    total_analyses:  int
    agent_accuracy:  dict[str, AgentAccuracyOut]
    recent:          list[MemoryRecordOut]


class HealthResponse(BaseModel):
    status:         str
    rag_docs:       int
    memory_records: int
