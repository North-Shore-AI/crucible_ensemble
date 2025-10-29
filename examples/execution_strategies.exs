#!/usr/bin/env elixir

# Execution Strategies Example
#
# This example demonstrates different execution strategies:
# - Parallel: All models run simultaneously
# - Sequential: Models run one at a time until consensus
# - Hedged: Primary model with backup hedges
# - Cascade: Priority order with early stopping
#
# Note: This uses mock responses since it doesn't require actual API keys.
#
# Run:
#   mix run examples/execution_strategies.exs

Application.ensure_all_started(:crucible_ensemble)

IO.puts("\n=== Execution Strategies Comparison ===\n")

question = "What is the capital of France?"
models = [:gemini_flash, :openai_gpt4o_mini, :anthropic_haiku]

# Strategy 1: Parallel Execution (default)
IO.puts("1. PARALLEL EXECUTION")
IO.puts("   All models run simultaneously")
IO.puts("   " <> String.duplicate("-", 50))

start_time = System.monotonic_time(:millisecond)

{:ok, result} =
  CrucibleEnsemble.predict(
    question,
    models: models,
    execution: :parallel
  )

elapsed = System.monotonic_time(:millisecond) - start_time

IO.puts("   Answer: #{result.answer}")
IO.puts("   Consensus: #{Float.round(result.metadata.consensus * 100, 2)}%")
IO.puts("   Total Latency: #{elapsed}ms")
IO.puts("   Cost: $#{Float.round(result.metadata.cost_usd, 6)}")
IO.puts("   All #{length(models)} models called")
IO.puts("")

# Strategy 2: Sequential Execution
IO.puts("2. SEQUENTIAL EXECUTION")
IO.puts("   Models run one at a time until consensus")
IO.puts("   " <> String.duplicate("-", 50))

start_time = System.monotonic_time(:millisecond)

{:ok, result} =
  CrucibleEnsemble.predict(
    question,
    models: models,
    execution: :sequential,
    min_consensus: 0.7
  )

elapsed = System.monotonic_time(:millisecond) - start_time

IO.puts("   Answer: #{result.answer}")
IO.puts("   Consensus: #{Float.round(result.metadata.consensus * 100, 2)}%")
IO.puts("   Total Latency: #{elapsed}ms")
IO.puts("   Cost: $#{Float.round(result.metadata.cost_usd, 6)}")
IO.puts("   Models called: #{result.metadata.successes}")
IO.puts("   Note: May stop early if consensus reached")
IO.puts("")

# Strategy 3: Hedged Execution
IO.puts("3. HEDGED EXECUTION")
IO.puts("   Primary model with backup hedges for tail latency")
IO.puts("   " <> String.duplicate("-", 50))

start_time = System.monotonic_time(:millisecond)

{:ok, result} =
  CrucibleEnsemble.predict(
    question,
    models: models,
    execution: :hedged,
    hedge_delay_ms: 500
  )

elapsed = System.monotonic_time(:millisecond) - start_time

IO.puts("   Answer: #{result.answer}")
IO.puts("   Total Latency: #{elapsed}ms")
IO.puts("   Cost: $#{Float.round(result.metadata.cost_usd, 6)}")
IO.puts("   Note: Uses first successful response")
IO.puts("")

# Strategy 4: Cascade Execution
IO.puts("4. CASCADE EXECUTION")
IO.puts("   Priority order with early stopping on high confidence")
IO.puts("   " <> String.duplicate("-", 50))

# Order models by priority (fast -> slow)
priority_models = [:gemini_flash, :openai_gpt4o_mini, :anthropic_haiku]

start_time = System.monotonic_time(:millisecond)

{:ok, result} =
  CrucibleEnsemble.predict(
    question,
    models: priority_models,
    execution: :cascade,
    confidence_threshold: 0.85,
    min_models: 2
  )

elapsed = System.monotonic_time(:millisecond) - start_time

IO.puts("   Answer: #{result.answer}")
IO.puts("   Total Latency: #{elapsed}ms")
IO.puts("   Cost: $#{Float.round(result.metadata.cost_usd, 6)}")
IO.puts("   Models called: #{result.metadata.successes}")
IO.puts("   Note: Stops when high confidence achieved")
IO.puts("")

# Performance Comparison Table
IO.puts("\n=== Performance Summary ===\n")

strategies = [
  {:parallel, "Maximum quality, higher cost"},
  {:sequential, "Adaptive cost, higher latency"},
  {:hedged, "Optimized P99 latency"},
  {:cascade, "Fast and cost-efficient"}
]

IO.puts("Strategy      | Tradeoff")
IO.puts(String.duplicate("-", 60))

Enum.each(strategies, fn {strategy, description} ->
  IO.puts("#{String.pad_trailing(to_string(strategy), 12)} | #{description}")
end)

IO.puts("\n=== Example Complete ===\n")
