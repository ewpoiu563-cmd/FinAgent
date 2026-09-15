"""Source-aware orchestration primitives for FinAgent."""

from .document_catalog import DocumentCatalog, DocumentRecord, load_default_catalog
from .context_builder import (
    CONTEXT_MAX_CHARS,
    CONTEXT_RECENT_MESSAGE_LIMIT,
    ContextBuildResult,
    ContextBuilder,
)
from .context_compressor import (
    COMPRESSION_MAX_CHARS,
    CONVERSATION_SUMMARY_MAX_CHARS,
    SUMMARY_KEEP_RECENT_MESSAGES,
    SUMMARY_TRIGGER_MESSAGES,
    ContextCompressor,
)
from .entity_resolver import EntityResolver
from .entrypoint import OrchestrationEntrypoint, build_default_entrypoint
from .evidence import (
    EvidenceBundle,
    EvidenceItem,
    model_prior_evidence,
    merge_evidence_bundles,
    normalize_rag_tool_result,
    normalize_sql_tool_result,
    normalize_web_tool_result,
    validate_rag_evidence_scope,
)
from .financial_amounts import (
    AmountSemanticError,
    ExchangeRate,
    MoneyAmount,
    amount_ratio,
    convert_currency,
    extract_money_amounts,
    extract_sql_money_amounts,
    normalize_currency,
    normalize_scale,
    subtract_money,
)
from .financial_calculator import (
    CalculationPlanError,
    CalculationResult,
    DeterministicCalculation,
    execute_calculation,
    parse_calculation,
    requires_deterministic_calculation,
)
from .executors import DirectRAGExecutionResult, DirectRAGExecutor, is_direct_rag_plan
from .rag_policy import (
    RAGPolicyExecutionResult,
    RAGRetryFallbackPolicy,
    LLMRAGQueryRewriter,
    build_rag_query,
    constrain_rag_query,
    rewrite_rag_query,
)
from .llm_router import LLMSourceRouter
from .models import EntityResolution, PlanMode, RouterDecision, SourcePlan, SourceType
from .source_planner import SourcePlanner
from .session_state import RedisSessionStore, SessionState
from .source_scope import (
    CatalogIndexConsistencyReport,
    SourceScopeResolver,
    validate_catalog_index_consistency,
)
from .tool_registry import ToolRegistry, ToolSpec, build_default_tool_registry
from .sufficiency import EvidenceSufficiencyEvaluator, SufficiencyResult
from .sufficiency import SQLResultSufficiencyEvaluator
from .sql_executor import (
    ControlledSQLRepairDecider,
    DirectSQLExecutionResult,
    DirectSQLExecutor,
    SQLRepairDecision,
    is_direct_sql_plan,
)
from .sql_fallback import SQLWebFallbackExecutionResult, SQLWebFallbackPolicy
from .web_fallback import (
    ControlledWebFallbackExecutor,
    DirectWebExecutor,
    WebFallbackExecutionResult,
    is_direct_web_plan,
    is_simple_web_factual_query,
    select_web_evidence,
)
from .web_planner import (
    WebResearchPlan,
    WebResearchPlanner,
    WebResearchPlanningError,
    WebSubquestion,
    WebTaskComplexity,
    classify_web_task,
    materialize_subquestion,
)
from .web_research import (
    StructuredWebResearchExecutor,
    StructuredWebResearchResult,
    WebFinding,
    is_structured_web_plan,
)
from .finding_extractor import FindingExtractor, is_atomic_expected_output
from .direct_execution import DirectExecutionResult, DirectNoToolExecutor, is_no_tool_plan
from .multi_source import (
    MultiSourceExecutionResult,
    MultiSourceOrchestrator,
    is_supported_multi_source_plan,
)
from .multi_source_synthesis import (
    ClaimEvidenceValidationError,
    MultiSourceClaim,
    MultiSourceSynthesisResult,
    MultiSourceSynthesizer,
)
from .state import AgentState, FallbackRecord, SourceStatus
from .task_execution import TaskPlanExecutionResult, TaskPlanExecutor
from .requirement_synthesis import RequirementSynthesisResult, RequirementSynthesizer
from .task_planning import TaskPlan, TaskPlanner, TaskRequirement, TaskStatus
from .task_decomposition import (
    DecomposedRequirement,
    LightweightTaskDecomposer,
    TaskDecompositionError,
)
from .temporal import TemporalReplacement, TemporalResolution, TemporalResolver
from .tool_necessity import (
    ToolDecisionMode,
    ToolNecessityDecision,
    ToolNecessityError,
    ToolNecessityPolicy,
)
from .trace import (
    TRACE_SCHEMA_VERSION,
    JsonlTraceSink,
    TraceContext,
    TraceEvent,
    TraceRecorder,
    get_current_trace_context,
    get_current_trace_source,
    get_current_trace_task_id,
    record_trace_event,
    trace_execution_scope,
    trace_task_scope,
)
from .trace_summary import SUMMARY_SCHEMA_VERSION, RunSummary, TraceSummaryBuilder
from .trace_metrics import (
    AGGREGATE_SCHEMA_VERSION,
    AggregateMetrics,
    DistributionStats,
    RunMetricsAggregator,
)

__all__ = [
    "DocumentCatalog",
    "CONTEXT_RECENT_MESSAGE_LIMIT",
    "CONTEXT_MAX_CHARS",
    "COMPRESSION_MAX_CHARS",
    "CONVERSATION_SUMMARY_MAX_CHARS",
    "ContextBuildResult",
    "ContextBuilder",
    "ContextCompressor",
    "SUMMARY_KEEP_RECENT_MESSAGES",
    "SUMMARY_TRIGGER_MESSAGES",
    "DocumentRecord",
    "DirectRAGExecutionResult",
    "DirectRAGExecutor",
    "DirectSQLExecutionResult",
    "DirectSQLExecutor",
    "ControlledSQLRepairDecider",
    "SQLRepairDecision",
    "SQLWebFallbackExecutionResult",
    "SQLWebFallbackPolicy",
    "ControlledWebFallbackExecutor",
    "DirectWebExecutor",
    "StructuredWebResearchExecutor",
    "StructuredWebResearchResult",
    "WebFinding",
    "EvidenceBundle",
    "EvidenceItem",
    "AmountSemanticError",
    "ExchangeRate",
    "MoneyAmount",
    "amount_ratio",
    "convert_currency",
    "extract_money_amounts",
    "extract_sql_money_amounts",
    "normalize_currency",
    "normalize_scale",
    "subtract_money",
    "CalculationPlanError",
    "CalculationResult",
    "DeterministicCalculation",
    "execute_calculation",
    "parse_calculation",
    "requires_deterministic_calculation",
    "model_prior_evidence",
    "EvidenceSufficiencyEvaluator",
    "EntityResolution",
    "EntityResolver",
    "LLMSourceRouter",
    "OrchestrationEntrypoint",
    "PlanMode",
    "RAGPolicyExecutionResult",
    "RAGRetryFallbackPolicy",
    "LLMRAGQueryRewriter",
    "build_rag_query",
    "constrain_rag_query",
    "RouterDecision",
    "SourcePlan",
    "SourcePlanner",
    "RedisSessionStore",
    "SessionState",
    "SourceScopeResolver",
    "SourceType",
    "SQLResultSufficiencyEvaluator",
    "SufficiencyResult",
    "ToolRegistry",
    "ToolSpec",
    "build_default_tool_registry",
    "build_default_entrypoint",
    "is_direct_rag_plan",
    "is_direct_sql_plan",
    "is_direct_web_plan",
    "is_simple_web_factual_query",
    "select_web_evidence",
    "load_default_catalog",
    "merge_evidence_bundles",
    "normalize_rag_tool_result",
    "normalize_sql_tool_result",
    "normalize_web_tool_result",
    "rewrite_rag_query",
    "validate_catalog_index_consistency",
    "validate_rag_evidence_scope",
    "CatalogIndexConsistencyReport",
    "WebFallbackExecutionResult",
    "WebResearchPlan",
    "WebResearchPlanner",
    "WebResearchPlanningError",
    "WebSubquestion",
    "WebTaskComplexity",
    "classify_web_task",
    "is_structured_web_plan",
    "materialize_subquestion",
    "FindingExtractor",
    "is_atomic_expected_output",
    "AgentState",
    "FallbackRecord",
    "SourceStatus",
    "MultiSourceExecutionResult",
    "MultiSourceOrchestrator",
    "is_supported_multi_source_plan",
    "ClaimEvidenceValidationError",
    "MultiSourceClaim",
    "MultiSourceSynthesisResult",
    "MultiSourceSynthesizer",
    "DirectExecutionResult",
    "DirectNoToolExecutor",
    "is_no_tool_plan",
    "TaskPlanExecutionResult",
    "TaskPlanExecutor",
    "RequirementSynthesisResult",
    "RequirementSynthesizer",
    "TaskPlan",
    "TaskPlanner",
    "TaskRequirement",
    "TaskStatus",
    "DecomposedRequirement",
    "LightweightTaskDecomposer",
    "TaskDecompositionError",
    "ToolDecisionMode",
    "ToolNecessityDecision",
    "ToolNecessityError",
    "ToolNecessityPolicy",
    "TemporalReplacement",
    "TemporalResolution",
    "TemporalResolver",
    "TRACE_SCHEMA_VERSION",
    "JsonlTraceSink",
    "TraceContext",
    "TraceEvent",
    "TraceRecorder",
    "get_current_trace_context",
    "get_current_trace_source",
    "get_current_trace_task_id",
    "record_trace_event",
    "trace_execution_scope",
    "trace_task_scope",
    "SUMMARY_SCHEMA_VERSION",
    "RunSummary",
    "TraceSummaryBuilder",
    "AGGREGATE_SCHEMA_VERSION",
    "AggregateMetrics",
    "DistributionStats",
    "RunMetricsAggregator",
]
