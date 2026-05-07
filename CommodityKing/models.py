"""
All shared data models for CommodityKing.

Pydantic models are used for API request/response validation.
Dataclasses are used for internal agent-to-agent communication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST MODELS
# ══════════════════════════════════════════════════════════════════════════════

class AnalyseRequest(BaseModel):
    symbols: list[str] = Field(
        ...,
        min_length=1,
        description="List of commodity symbols to analyse, e.g. ['wheat', 'gold']",
    )


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL AGENT TYPES (dataclasses — fast, no validation overhead)
# ══════════════════════════════════════════════════════════════════════════════

Direction = Literal["BULLISH", "BEARISH", "NEUTRAL", "BULLISH_SUPPLY", "BEARISH_SUPPLY"]
TimeHorizon = Literal["SHORT", "MEDIUM"]
RAGStatus = Literal["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE", "NOT_CHECKED"]


@dataclass
class AgentThesis:
    """Output produced by each of the three agents."""
    agent: Literal["geopolitical", "weather", "fundamentals"]
    direction: Direction
    confidence: float                   # 0.0 – 1.0
    headline: str

    # Geopolitical fields
    analysis_24h: str         = ""
    analysis_7d: str          = ""
    historical_analog: str    = ""
    key_risks: list[str]      = field(default_factory=list)
    surprise_score: float          = 0.0
    surprise_interp: str           = ""
    surprise_terms: list[str]      = field(default_factory=list)
    surprise_kl: float             = 0.0
    surprise_history_source: str   = "insufficient"  # "chroma_cache" | "insufficient"
    kalman_result: dict       = field(default_factory=dict)
    n_articles: int           = 0

    # Weather fields
    critical_regions: list[str]  = field(default_factory=list)
    forecast_outlook: str        = ""
    supply_impact: str           = ""
    weather_zscores: dict        = field(default_factory=dict)
    max_zscore: float            = 0.0
    weather_relevant: bool       = True

    # Fundamentals fields
    supply_analysis: str         = ""
    demand_analysis: str         = ""
    inventory_analysis: str      = ""
    ssi_result: dict             = field(default_factory=dict)
    hmm_result: dict             = field(default_factory=dict)
    cointegration_results: list  = field(default_factory=list)

    # Set by RAG node after agent runs
    rag_status: RAGStatus        = "NOT_CHECKED"

    # Error tracking
    error: Optional[str]         = None


@dataclass
class VerificationResult:
    """One claim verified against the RAG knowledge base."""
    claim: str
    verdict: RAGStatus
    supporting_text: str    = ""
    source: str             = ""
    similarity_score: float = 0.0


@dataclass
class DebateRound:
    challenger: Literal["geopolitical", "weather", "fundamentals"]
    challenged: Literal["geopolitical", "weather", "fundamentals"]
    challenge_text: str
    response_text: str
    confidence_before: float
    confidence_after: float
    revised: bool


@dataclass
class DebateTranscript:
    occurred: bool
    rounds: int
    transcript: list[DebateRound] = field(default_factory=list)
    final_directions: dict        = field(default_factory=dict)  # agent -> direction
    skip_reason: str              = ""


# ══════════════════════════════════════════════════════════════════════════════
# QUANTITATIVE MODEL OUTPUTS (nested inside CommodityReport)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class KalmanOutput:
    latest_signal: float
    is_signal: bool
    trend: Literal["up", "down"]
    signal_sigma: float


@dataclass
class HMMOutput:
    current_regime: int
    regime_label: str
    confidence: float
    is_high_volatility: bool


@dataclass
class SSIOutput:
    ssi: float
    level: Literal["normal", "elevated", "high", "extreme"]
    direction: Literal["bullish", "bearish"]
    weather_component: float
    geo_component: float
    inventory_component: float


@dataclass
class BayesianSurpriseOutput:
    surprise_score: float
    top_surprise_terms: list[str]
    interpretation: str
    kl_divergence: float
    history_source: str = "insufficient"  # "chroma_cache" | "insufficient"


@dataclass
class CointegrationOutput:
    pair: str
    cointegrated: bool
    pvalue: float           = 1.0
    spread_zscore: float    = 0.0
    signal: str             = ""
    mean_reverting: bool    = False


@dataclass
class HistoricalAnalog:
    found: bool
    event_id: str           = ""
    event: str              = ""
    date: str               = ""
    similarity: float       = 0.0
    price_impact_then: str  = ""
    context: str            = ""
    resolution: str         = ""


# ══════════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FinalRecommendation:
    direction: Direction
    conviction: float
    time_horizon: TimeHorizon
    reasoning: str
    upside_risks: list[str]     = field(default_factory=list)
    downside_risks: list[str]   = field(default_factory=list)
    watch_list: list[str]       = field(default_factory=list)


@dataclass
class CommodityReport:
    symbol: str
    commodity_name: str
    generated_at: str
    duration_seconds: float

    # Report sections (written by synthesis node)
    executive_summary: str      = ""
    full_report_text: str       = ""

    # Price context (current_price, change_24h_pct, change_7d_pct, etc.)
    price_context: dict         = field(default_factory=dict)

    # Agent outputs
    geo_thesis: Optional[AgentThesis]      = None
    weather_thesis: Optional[AgentThesis]  = None
    fund_thesis: Optional[AgentThesis]     = None

    # Quant outputs
    kalman: Optional[KalmanOutput]                 = None
    hmm: Optional[HMMOutput]                       = None
    ssi: Optional[SSIOutput]                       = None
    bayesian_surprise: Optional[BayesianSurpriseOutput] = None
    cointegration: list[CointegrationOutput]       = field(default_factory=list)

    # Debate
    debate: Optional[DebateTranscript]     = None

    # Historical analog
    historical_analog: Optional[HistoricalAnalog]  = None

    # Final recommendation
    final_recommendation: Optional[FinalRecommendation] = None

    # Metadata
    data_sources_used: list[str]   = field(default_factory=list)
    llm_calls: int                 = 0
    estimated_cost_usd: float      = 0.0
    errors: list[str]              = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# API RESPONSE MODELS (Pydantic — serialised to JSON for the client)
# ══════════════════════════════════════════════════════════════════════════════

class PriceContextResponse(BaseModel):
    current_price: float
    unit: str
    change_24h_pct: float
    change_7d_pct: float
    kalman_trend: str
    signal_sigma: float
    regime: str


class AgentResponse(BaseModel):
    direction: str
    confidence: float
    headline: str
    rag_status: str
    error: Optional[str] = None

    # Geopolitical
    analysis_24h: str           = ""
    analysis_7d: str            = ""
    historical_analog: str      = ""
    key_risks: list[str]        = Field(default_factory=list)
    surprise_score: float       = 0.0
    n_articles: int             = 0

    # Weather
    critical_regions: list[str] = Field(default_factory=list)
    forecast_outlook: str       = ""
    supply_impact: str          = ""
    max_zscore: float           = 0.0
    weather_relevant: bool      = True

    # Fundamentals
    supply_analysis: str        = ""
    demand_analysis: str        = ""
    inventory_analysis: str     = ""
    ssi: Optional[float]        = None
    ssi_level: Optional[str]    = None
    regime: Optional[str]       = None


class DebateResponse(BaseModel):
    occurred: bool
    rounds: int
    transcript: list[dict]      = Field(default_factory=list)
    skip_reason: str            = ""


class QuantitativeResponse(BaseModel):
    kalman: dict                = Field(default_factory=dict)
    hmm: dict                   = Field(default_factory=dict)
    ssi: dict                   = Field(default_factory=dict)
    bayesian_surprise: dict     = Field(default_factory=dict)
    cointegration: list[dict]   = Field(default_factory=list)


class HistoricalAnalogResponse(BaseModel):
    found: bool
    event: str              = ""
    date: str               = ""
    similarity: float       = 0.0
    price_impact_then: str  = ""
    context: str            = ""
    resolution: str         = ""


class FinalRecommendationResponse(BaseModel):
    direction: str
    conviction: float
    time_horizon: str
    reasoning: str
    upside_risks: list[str]     = Field(default_factory=list)
    downside_risks: list[str]   = Field(default_factory=list)
    watch_list: list[str]       = Field(default_factory=list)


class MetadataResponse(BaseModel):
    data_sources_used: list[str]    = Field(default_factory=list)
    llm_calls: int                  = 0
    estimated_cost_usd: float       = 0.0
    errors: list[str]               = Field(default_factory=list)


class CommodityAnalysisResponse(BaseModel):
    symbol: str
    commodity_name: str
    generated_at: str
    duration_seconds: float
    executive_summary: str
    price_context: PriceContextResponse
    agents: dict[str, AgentResponse]
    debate: DebateResponse
    quantitative: QuantitativeResponse
    historical_analog: HistoricalAnalogResponse
    final_recommendation: FinalRecommendationResponse
    full_report_text: str
    metadata: MetadataResponse


class AnalyseResponse(BaseModel):
    analyses: list[CommodityAnalysisResponse]
    total_duration_seconds: float
    commodities_analysed: int


class CommodityListItem(BaseModel):
    symbol: str
    name: str
    ticker: str
    unit: str
    category: str
    weather_relevant: bool
    usda_relevant: bool
    eia_relevant: bool
    n_weather_regions: int
    cointegrated_with: list[str]


class CommodityListResponse(BaseModel):
    commodities: list[CommodityListItem]
    total: int


class HealthResponse(BaseModel):
    status: str
    chroma_docs: int
    commodities_supported: int
    config_warnings: list[str]
