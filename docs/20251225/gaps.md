# CrucibleEnsemble - Gap Analysis

**Date**: 2025-12-25
**Version**: 0.3.0

## Overview

This document identifies gaps, missing features, and areas for improvement in the CrucibleEnsemble library.

---

## Critical Gap: Crucible.Stage Behaviour Not Implemented

### Issue

The current `CrucibleEnsemble.Stage` module does **NOT** implement the `Crucible.Stage` behaviour from crucible_framework. While it has similar functions (`run/2` and `describe/1`), it:

1. Does not declare `@behaviour Crucible.Stage`
2. Uses plain maps instead of `Crucible.Context.t()` struct
3. Cannot be used directly in crucible_framework pipelines

### Current Implementation

```elixir
# lib/crucible_ensemble/stage.ex
defmodule CrucibleEnsemble.Stage do
  # MISSING: @behaviour Crucible.Stage

  @spec run(map(), map()) :: {:ok, map()} | {:error, term()}
  def run(context, opts \\ %{}) when is_map(context) and is_map(opts) do
    # Uses plain map context, not Crucible.Context.t()
  end
end
```

### Expected Implementation

```elixir
# lib/crucible_ensemble/stage.ex
defmodule CrucibleEnsemble.Stage do
  @behaviour Crucible.Stage  # <-- REQUIRED

  alias Crucible.Context

  @impl true
  @spec run(Context.t(), map()) :: {:ok, Context.t()} | {:error, term()}
  def run(%Context{} = ctx, opts \\ %{}) do
    # Use Context struct and helpers
    # Store results in ctx.artifacts, ctx.metrics, ctx.assigns
  end

  @impl true
  def describe(opts) do
    # Return stage metadata
  end
end
```

### Impact

- Cannot integrate with crucible_framework's `Crucible.Pipeline.Runner`
- Cannot be composed with other stages like `Crucible.Stage.Bench`, `Crucible.Stage.DataLoad`
- Cannot leverage `Crucible.Context` helper functions

---

## Missing Features

### 1. No crucible_framework Dependency

**File**: `mix.exs`

The library does not depend on crucible_framework:

```elixir
defp deps do
  [
    {:crucible_ir, "~> 0.1.1"},  # Has CrucibleIR, but not framework
    # MISSING: {:crucible_framework, "~> X.X"}
  ]
end
```

This prevents implementing the `Crucible.Stage` behaviour.

### 2. Stage Does Not Use Crucible.Context Helper Functions

The `Crucible.Context` module provides ergonomic helpers:
- `put_artifact/3` - Store results in artifacts
- `put_metric/3` - Store metrics
- `mark_stage_complete/2` - Track stage completion
- `assign/2` - Phoenix-style assigns

Current implementation manually manipulates map fields instead.

### 3. No Integration with Crucible.Context.artifacts

Crucible.Stage implementations typically store results in:
- `ctx.artifacts` - For final results (ensemble answer, vote result)
- `ctx.metrics` - For telemetry data (consensus, latency, cost)

Current implementation just puts everything in the returned context map.

### 4. Missing Stage Completion Tracking

Other stages call `Crucible.Context.mark_stage_complete(ctx, :stage_name)`. The ensemble stage does not.

---

## Incomplete Features

### 1. Mock LLM Implementation Only

**File**: `lib/crucible_ensemble/executor.ex` (Lines 311-329)

The `make_llm_request/2` function is a mock:

```elixir
defp make_llm_request(query, _opts) do
  # This is a placeholder for the actual req_llm integration
  # Mock response for compilation
  %{
    text: "Mock response: #{query}",
    usage: %{input_tokens: String.length(query), output_tokens: 50},
    finish_reason: "stop"
  }
end
```

**Impact**: Cannot make real LLM calls without external configuration.

### 2. Application Module Not Started

**File**: `mix.exs` (Lines 23-27)

```elixir
def application do
  [
    extra_applications: [:logger]
    # MISSING: mod: {CrucibleEnsemble.Application, []}
  ]
end
```

The Application module exists but isn't started, so:
- Telemetry handlers are not auto-attached
- Supervision tree is not started

### 3. No Connection Pooling

**File**: `lib/crucible_ensemble/application.ex` (Lines 13-16)

```elixir
children = [
  # Future: Add workers for connection pooling, circuit breakers, etc.
  # {CrucibleEnsemble.Worker, arg}
]
```

Connection pooling is planned but not implemented.

---

## Documentation Gaps

### 1. README Does Not Document crucible_framework Integration

The README documents the Stage module but does not explain:
- It doesn't implement `Crucible.Stage` behaviour
- How to integrate with crucible_framework pipelines
- The difference between plain map context and `Crucible.Context.t()`

### 2. No Migration Guide for Behaviour Implementation

No documentation on how to:
- Add crucible_framework dependency
- Refactor Stage to use `Crucible.Context.t()`
- Handle backwards compatibility

---

## Test Gaps

### 1. Integration Tests Are Skipped

**File**: `test/ensemble_test.exs`

All integration tests are tagged `@tag :skip` because they require API keys:

```elixir
@tag :skip
test "basic prediction with default settings" do
  # This would require actual API keys to run
end
```

### 2. No Tests for Crucible.Stage Behaviour Compliance

When implementing the behaviour, tests should verify:
- Returns `{:ok, %Crucible.Context{}}` on success
- Stores results in `ctx.artifacts` and `ctx.metrics`
- Marks stage complete
- Handles Context struct properly

### 3. Missing Property-Based Tests

The README mentions property tests but none exist:

```elixir
property "majority voting is deterministic" do
  # Does not exist
end
```

---

## Code Quality Issues

### 1. Duplicated Levenshtein Implementation

Both `CrucibleEnsemble.Normalize` and `CrucibleEnsemble.Similarity` have their own Levenshtein distance implementations:

- `lib/crucible_ensemble/normalize.ex` Lines 287-321
- `lib/crucible_ensemble/similarity.ex` Lines 314-348

### 2. Inconsistent Option Handling

Some functions expect keyword lists, others expect maps:
- `run/2` in Stage expects `opts :: map()`
- `aggregate/2` in Vote expects `opts :: keyword()`

### 3. Missing Typespec for Custom Strategy

The type definition allows `{module(), keyword()}` but there's no documented interface for custom strategy modules beyond `CrucibleEnsemble.Vote.Custom` behaviour.

---

## Summary of Required Changes

### Must Have

1. **Add crucible_framework dependency** to enable `Crucible.Stage` behaviour
2. **Implement `Crucible.Stage` behaviour** in `CrucibleEnsemble.Stage`
3. **Refactor to use `Crucible.Context.t()`** instead of plain maps
4. **Store results in artifacts** using `Crucible.Context.put_artifact/3`
5. **Add comprehensive tests** for behaviour compliance

### Should Have

1. Start Application in supervision tree
2. Deduplicate Levenshtein implementation
3. Consistent option handling (all keyword or all map)
4. Property-based tests for voting determinism

### Nice to Have

1. Real LLM integration (not just mock)
2. Connection pooling implementation
3. Migration guide documentation
4. Integration test suite with mock API
