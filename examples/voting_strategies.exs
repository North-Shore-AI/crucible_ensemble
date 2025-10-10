#!/usr/bin/env elixir

# Voting Strategies Comparison Example
#
# This example demonstrates different voting strategies and shows
# how they produce different results for the same query.
#
# Run:
#   elixir examples/voting_strategies.exs

Mix.install([
  {:crucible_ensemble, path: "."}
])

{:ok, _} = Application.ensure_all_started(:crucible_ensemble)

IO.puts("\n=== Voting Strategies Comparison ===\n")

# Test question
question = "Is artificial intelligence dangerous? Answer yes or no."
models = [:gemini_flash, :openai_gpt4o_mini, :anthropic_haiku]

IO.puts("Question: #{question}")
IO.puts("Models: #{inspect(models)}")
IO.puts("\n")

# Strategy 1: Majority Voting
IO.puts("1. MAJORITY VOTING")
IO.puts("   Most common response wins")
IO.puts("   " <> String.duplicate("-", 40))

{:ok, result} = CrucibleEnsemble.predict(question, strategy: :majority, models: models)

IO.puts("   Answer: #{result.answer}")
IO.puts("   Consensus: #{Float.round(result.metadata.consensus * 100, 2)}%")
IO.puts("   Votes: #{inspect(result.metadata.votes)}")
IO.puts("   Cost: $#{Float.round(result.metadata.cost_usd, 6)}")
IO.puts("")

# Strategy 2: Weighted Voting
IO.puts("2. WEIGHTED VOTING")
IO.puts("   Responses weighted by confidence")
IO.puts("   " <> String.duplicate("-", 40))

{:ok, result} = CrucibleEnsemble.predict(question, strategy: :weighted, models: models)

IO.puts("   Answer: #{result.answer}")
IO.puts("   Consensus: #{Float.round(result.metadata.consensus * 100, 2)}%")
IO.puts("   Cost: $#{Float.round(result.metadata.cost_usd, 6)}")
IO.puts("")

# Strategy 3: Best Confidence
IO.puts("3. BEST CONFIDENCE")
IO.puts("   Highest confidence response selected")
IO.puts("   " <> String.duplicate("-", 40))

{:ok, result} = CrucibleEnsemble.predict(question, strategy: :best_confidence, models: models)

IO.puts("   Answer: #{result.answer}")
IO.puts("   Consensus: #{Float.round(result.metadata.consensus * 100, 2)}%")
IO.puts("   Cost: $#{Float.round(result.metadata.cost_usd, 6)}")
IO.puts("")

# Strategy 4: Unanimous
IO.puts("4. UNANIMOUS VOTING")
IO.puts("   All models must agree")
IO.puts("   " <> String.duplicate("-", 40))

case CrucibleEnsemble.predict(question, strategy: :unanimous, models: models) do
  {:ok, result} ->
    IO.puts("   Answer: #{result.answer}")
    IO.puts("   All models agreed!")
    IO.puts("   Cost: $#{Float.round(result.metadata.cost_usd, 6)}")

  {:error, _reason} ->
    IO.puts("   No unanimous consensus reached")
    IO.puts("   Models had differing opinions")
end

IO.puts("\n")

# Demonstrate cost vs quality tradeoff
IO.puts("=== Cost vs Quality Analysis ===\n")

math_question = "What is 123 * 456?"

# High quality: Use expensive models
{:ok, high_quality} =
  CrucibleEnsemble.predict(
    math_question,
    models: [:openai_gpt4o, :anthropic_sonnet],
    strategy: :majority
  )

# Low cost: Use cheaper models
{:ok, low_cost} =
  CrucibleEnsemble.predict(
    math_question,
    models: [:gemini_flash, :openai_gpt4o_mini],
    strategy: :majority
  )

IO.puts("High Quality Ensemble:")
IO.puts("  Models: [:openai_gpt4o, :anthropic_sonnet]")
IO.puts("  Answer: #{high_quality.answer}")
IO.puts("  Consensus: #{Float.round(high_quality.metadata.consensus * 100, 2)}%")
IO.puts("  Cost: $#{Float.round(high_quality.metadata.cost_usd, 6)}")
IO.puts("")

IO.puts("Low Cost Ensemble:")
IO.puts("  Models: [:gemini_flash, :openai_gpt4o_mini]")
IO.puts("  Answer: #{low_cost.answer}")
IO.puts("  Consensus: #{Float.round(low_cost.metadata.consensus * 100, 2)}%")
IO.puts("  Cost: $#{Float.round(low_cost.metadata.cost_usd, 6)}")
IO.puts("")

cost_ratio = high_quality.metadata.cost_usd / low_cost.metadata.cost_usd
IO.puts("Cost Ratio: #{Float.round(cost_ratio, 2)}x")

IO.puts("\n=== Example Complete ===\n")
