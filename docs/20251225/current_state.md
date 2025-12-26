# CrucibleEnsemble - Current State Documentation

**Date**: 2025-12-25
**Version**: 0.3.0

## Overview

CrucibleEnsemble is a multi-model ensemble prediction library with configurable voting strategies for AI reliability research. Built on the BEAM VM, it leverages Elixir's lightweight processes to achieve massive parallelism with minimal overhead.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CrucibleEnsemble                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                     Main Entry Point                      │ │
│  │  lib/ensemble.ex - CrucibleEnsemble module                │ │
│  │  - predict/2, predict/3                                   │ │
│  │  - predict_async/2, predict_async/3                       │ │
│  │  - predict_stream/2                                       │ │
│  └──────────────────────────────────────────────────────────┘ │
│                              │                                 │
│         ┌────────────────────┼────────────────────┐           │
│         ▼                    ▼                    ▼            │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│  │   Executor   │     │   Strategy   │     │    Vote      │   │
│  │              │     │              │     │              │   │
│  │ - parallel   │     │ - parallel   │     │ - majority   │   │
│  │ - sequential │     │ - sequential │     │ - weighted   │   │
│  │              │     │ - hedged     │     │ - unanimous  │   │
│  │              │     │ - cascade    │     │ - semantic   │   │
│  │              │     │              │     │ - ranked     │   │
│  └──────────────┘     └──────────────┘     └──────────────┘   │
│         │                    │                    │            │
│         ▼                    ▼                    ▼            │
│  ┌────────────────────────────────────────────────────────┐   │
│  │                   Supporting Modules                    │   │
│  │  - Normalize (response normalization)                  │   │
│  │  - Similarity (text similarity algorithms)             │   │
│  │  - Pricing (cost tracking)                             │   │
│  │  - Metrics (telemetry integration)                     │   │
│  └────────────────────────────────────────────────────────┘   │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐   │
│  │                Pipeline Integration                     │   │
│  │  - Stage (CrucibleEnsemble.Stage)                      │   │
│  │    Uses CrucibleIR.Reliability.Ensemble config         │   │
│  └────────────────────────────────────────────────────────┘   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Module Inventory

### Core Module

**File**: `/home/home/p/g/North-Shore-AI/crucible_ensemble/lib/ensemble.ex`
**Module**: `CrucibleEnsemble`
**Lines**: 1-529

Key Functions:
- `predict/2` (Line 149) - Synchronous ensemble prediction with options
- `predict/3` (Line 249) - Prediction with CrucibleIR.Reliability.Ensemble config
- `predict_async/2` (Line 285) - Asynchronous prediction returning Task
- `predict_async/3` (Line 298) - Async with CrucibleIR config
- `predict_stream/2` (Line 330) - Streaming results with early stopping

### Execution Module

**File**: `/home/home/p/g/North-Shore-AI/crucible_ensemble/lib/crucible_ensemble/executor.ex`
**Module**: `CrucibleEnsemble.Executor`
**Lines**: 1-366

Key Functions:
- `execute_parallel/3` (Line 51) - Execute all models concurrently
- `call_model/3` (Line 110) - Call single model with telemetry
- `execute_sequential/3` (Line 206) - Execute until consensus reached

### Strategy Module

**File**: `/home/home/p/g/North-Shore-AI/crucible_ensemble/lib/crucible_ensemble/strategy.ex`
**Module**: `CrucibleEnsemble.Strategy`
**Lines**: 1-318

Key Functions:
- `parallel/3` (Line 41) - Execute all models simultaneously
- `sequential/3` (Line 75) - Execute one at a time until consensus
- `hedged/4` (Line 110) - Primary with backup hedges
- `cascade/3` (Line 178) - Priority order with early stopping

### Voting Modules

#### Main Vote Module
**File**: `/home/home/p/g/North-Shore-AI/crucible_ensemble/lib/crucible_ensemble/vote.ex`
**Modules**:
- `CrucibleEnsemble.Vote` (Lines 1-180)
- `CrucibleEnsemble.Vote.Majority` (Lines 182-236)
- `CrucibleEnsemble.Vote.Weighted` (Lines 238-313)
- `CrucibleEnsemble.Vote.BestConfidence` (Lines 315-371)
- `CrucibleEnsemble.Vote.Unanimous` (Lines 373-426)
- `CrucibleEnsemble.Vote.Custom` (Lines 428-449)

Key Functions:
- `apply_strategy/3` (Line 58) - Apply voting strategy to responses
- `consensus_strength/1` (Line 116) - Calculate consensus for distribution
- `sufficient_consensus?/2` (Line 153) - Check if consensus meets threshold

#### Semantic Similarity Voting
**File**: `/home/home/p/g/North-Shore-AI/crucible_ensemble/lib/crucible_ensemble/vote/semantic_similarity.ex`
**Module**: `CrucibleEnsemble.Vote.SemanticSimilarity`
**Lines**: 1-172

Key Functions:
- `aggregate/2` (Line 63) - Cluster responses by similarity and vote

#### Ranked Choice Voting
**File**: `/home/home/p/g/North-Shore-AI/crucible_ensemble/lib/crucible_ensemble/vote/ranked_choice.ex`
**Module**: `CrucibleEnsemble.Vote.RankedChoice`
**Lines**: 1-301

Key Functions:
- `aggregate/2` (Line 78) - Aggregate using instant-runoff or Borda count

### Supporting Modules

#### Normalize Module
**File**: `/home/home/p/g/North-Shore-AI/crucible_ensemble/lib/crucible_ensemble/normalize.ex`
**Module**: `CrucibleEnsemble.Normalize`
**Lines**: 1-322

Key Functions:
- `normalize/2` (Line 39) - Normalize response using strategy
- `extract_numeric/1` (Line 85) - Extract numeric value from text
- `parse_json/1` (Line 123) - Parse JSON response
- `extract_boolean/1` (Line 160) - Extract boolean yes/no
- `text_similarity/2` (Line 215) - Levenshtein-based similarity
- `normalize_result/2` (Line 250) - Normalize from result map
- `extract_response_text/1` (Line 271) - Extract text from formats

#### Similarity Module
**File**: `/home/home/p/g/North-Shore-AI/crucible_ensemble/lib/crucible_ensemble/similarity.ex`
**Module**: `CrucibleEnsemble.Similarity`
**Lines**: 1-453

Key Functions:
- `levenshtein_similarity/3` (Line 54) - Edit distance-based similarity
- `jaccard_similarity/2` (Line 98) - Set-based word overlap
- `cosine_similarity/2` (Line 135) - Term frequency vector similarity
- `compute/3` (Line 179) - Compute using specified metric
- `similarity_matrix/2` (Line 205) - Create pairwise similarity matrix
- `cluster_by_threshold/3` (Line 249) - Cluster texts by similarity
- `find_representative/3` (Line 278) - Find centroid of cluster

#### Pricing Module
**File**: `/home/home/p/g/North-Shore-AI/crucible_ensemble/lib/crucible_ensemble/pricing.ex`
**Module**: `CrucibleEnsemble.Pricing`
**Lines**: 1-300

Key Functions:
- `calculate_cost/2` (Line 106) - Calculate cost for model response
- `calculate_cost_breakdown/2` (Line 138) - Detailed cost breakdown
- `get_prices/1` (Line 166) - Get pricing for model
- `aggregate_costs/1` (Line 194) - Aggregate costs from results
- `estimate_cost/3` (Line 252) - Estimate cost before execution

#### Metrics Module
**File**: `/home/home/p/g/North-Shore-AI/crucible_ensemble/lib/crucible_ensemble/metrics.ex`
**Module**: `CrucibleEnsemble.Metrics`
**Lines**: 1-393

Key Functions:
- `attach_handlers/0` (Line 28) - Attach telemetry handlers
- `handle_event/4` (Line 60) - Handle telemetry events
- `record_prediction/1` (Line 179) - Record prediction metadata
- `record_model_response/4` (Line 203) - Record model response
- `aggregate_stats/1` (Line 241) - Calculate aggregate statistics
- `export_to_csv/2` (Line 291) - Export metrics to CSV
- `summary_report/1` (Line 345) - Create summary report

### Pipeline Stage Module

**File**: `/home/home/p/g/North-Shore-AI/crucible_ensemble/lib/crucible_ensemble/stage.ex`
**Module**: `CrucibleEnsemble.Stage`
**Lines**: 1-271

Key Functions:
- `run/2` (Line 89) - Run ensemble voting on context
- `describe/1` (Line 132) - Stage metadata for introspection

## Voting Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `:majority` | Most common response wins | Factual questions, classification |
| `:weighted` | Responses weighted by confidence | Open-ended questions |
| `:best_confidence` | Highest confidence response | Latency-critical applications |
| `:unanimous` | All models must agree | High-stakes decisions |
| `:semantic_similarity` | Groups by textual similarity | Varied phrasing answers |
| `:ranked_choice` | Instant-runoff or Borda count | Multiple valid approaches |

## Execution Strategies

| Strategy | Description | Tradeoff |
|----------|-------------|----------|
| `:parallel` | All models simultaneously | Fast completion, all called |
| `:sequential` | One at a time until consensus | Variable latency, lower cost |
| `:hedged` | Primary with backup hedges | Optimized P99, controlled overhead |
| `:cascade` | Priority order, early stop | Fast and cheap, may miss consensus |

## Normalization Strategies

| Strategy | Description |
|----------|-------------|
| `:lowercase_trim` | Case-insensitive, trim whitespace |
| `:numeric` | Extract numeric values |
| `:boolean` | Extract yes/no answers |
| `:json` | Parse JSON responses |
| `{:custom, fn}` | Custom normalization function |

## Dependencies

From `mix.exs`:
```elixir
{:crucible_ir, "~> 0.1.1"}
{:jason, "~> 1.4"}
{:telemetry, "~> 1.2"}
{:ex_doc, "~> 0.31", only: :dev}
{:dialyxir, "~> 1.4", only: :dev}
{:mox, "~> 1.1", only: :test}
```

## Test Coverage

### Test Files
- `/home/home/p/g/North-Shore-AI/crucible_ensemble/test/ensemble_test.exs` - Integration tests (skipped, require API keys)
- `/home/home/p/g/North-Shore-AI/crucible_ensemble/test/crucible_ensemble/vote_test.exs` - Voting strategy tests (212 lines)
- `/home/home/p/g/North-Shore-AI/crucible_ensemble/test/crucible_ensemble/stage_test.exs` - Stage tests (554 lines)
- `/home/home/p/g/North-Shore-AI/crucible_ensemble/test/crucible_ensemble/similarity_test.exs` - Similarity tests (228 lines)
- `/home/home/p/g/North-Shore-AI/crucible_ensemble/test/crucible_ensemble/vote/ranked_choice_test.exs` - Ranked choice tests (245 lines)
- `/home/home/p/g/North-Shore-AI/crucible_ensemble/test/crucible_ensemble/vote/semantic_similarity_test.exs` - Semantic similarity tests (281 lines)
- `/home/home/p/g/North-Shore-AI/crucible_ensemble/test/crucible_ensemble/normalize_test.exs` - Normalize tests
- `/home/home/p/g/North-Shore-AI/crucible_ensemble/test/crucible_ensemble/pricing_test.exs` - Pricing tests
- `/home/home/p/g/North-Shore-AI/crucible_ensemble/test/crucible_ensemble/metrics_test.exs` - Metrics tests

## Current CrucibleIR Integration

The `CrucibleEnsemble.Stage` module uses `CrucibleIR.Reliability.Ensemble` configuration:

```elixir
%CrucibleIR.Reliability.Ensemble{
  strategy: :majority,
  execution_mode: :parallel,
  models: [:gemini_flash, :openai_gpt4o_mini],
  min_agreement: 0.7,
  timeout_ms: 5000,
  weights: %{...},
  options: %{...}
}
```

The Stage expects context with:
- `experiment.reliability.ensemble` - Ensemble configuration
- Either `outputs` (existing model responses) or `query` (to execute models)

The Stage adds to context:
- `ensemble_result` - Full voting result
- `consensus` - Consensus score (0.0-1.0)
- `answer` - Final answer selected by voting
- `ensemble_metadata` - Additional metadata

## Note on Crucible.Stage Behaviour

The current `CrucibleEnsemble.Stage` does **NOT** implement the `Crucible.Stage` behaviour from crucible_framework. It has its own similar interface:

- `run/2` - Takes `(context :: map(), opts :: map())`, returns `{:ok, map()} | {:error, term()}`
- `describe/1` - Takes `(opts :: map())`, returns `map()`

The crucible_framework's `Crucible.Stage` behaviour expects:
- `run/2` - Takes `(context :: Crucible.Context.t(), opts :: map())`, returns `{:ok, Crucible.Context.t()} | {:error, term()}`
- `describe/1` (optional) - Takes `(opts :: map())`, returns `map()`

A proper integration would require implementing the behaviour and using `Crucible.Context.t()` instead of plain maps.
