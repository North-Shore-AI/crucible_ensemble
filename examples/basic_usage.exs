#!/usr/bin/env elixir

# Basic Usage Example for CrucibleEnsemble Library
#
# This example demonstrates the simplest way to use CrucibleEnsemble for
# multi-model predictions with voting.
#
# Note: This uses mock responses since it doesn't require actual API keys.
# For real usage with API keys, set:
#   export GEMINI_API_KEY="your-key"
#   export OPENAI_API_KEY="your-key"
#   export ANTHROPIC_API_KEY="your-key"
#
# Run:
#   mix run examples/basic_usage.exs

# Ensure the application is started
Application.ensure_all_started(:crucible_ensemble)

IO.puts("\n=== Ensemble Basic Usage Example ===\n")

# Example 1: Simple factual question with default settings
IO.puts("Example 1: Simple Question")
IO.puts("---------------------------")

{:ok, result} = CrucibleEnsemble.predict("What is 2+2?")

IO.puts("Question: What is 2+2?")
IO.puts("Answer: #{result.answer}")
IO.puts("Consensus: #{Float.round(result.metadata.consensus * 100, 2)}%")
IO.puts("Cost: $#{Float.round(result.metadata.cost_usd, 6)}")
IO.puts("Latency: #{result.metadata.latency_ms}ms")
IO.puts("Successes: #{result.metadata.successes}/#{length(result.metadata.models_used)}")

IO.puts("\n")

# Example 2: Using specific models
IO.puts("Example 2: Custom Model Selection")
IO.puts("----------------------------------")

{:ok, result} =
  CrucibleEnsemble.predict(
    "What is the capital of France?",
    models: [:gemini_flash, :openai_gpt4o_mini]
  )

IO.puts("Question: What is the capital of France?")
IO.puts("Answer: #{result.answer}")
IO.puts("Models used: #{inspect(result.metadata.models_used)}")
IO.puts("Consensus: #{Float.round(result.metadata.consensus * 100, 2)}%")

IO.puts("\n")

# Example 3: Weighted voting for open-ended questions
IO.puts("Example 3: Weighted Voting Strategy")
IO.puts("------------------------------------")

{:ok, result} =
  CrucibleEnsemble.predict(
    "Explain the concept of recursion in one sentence.",
    strategy: :weighted,
    models: [:openai_gpt4o_mini, :anthropic_haiku, :gemini_flash]
  )

IO.puts("Question: Explain the concept of recursion in one sentence.")
IO.puts("Answer: #{result.answer}")
IO.puts("Strategy: #{result.metadata.strategy}")
IO.puts("Consensus: #{Float.round(result.metadata.consensus * 100, 2)}%")

IO.puts("\n")

# Example 4: Yes/No question with boolean normalization
IO.puts("Example 4: Boolean Question")
IO.puts("---------------------------")

{:ok, result} =
  CrucibleEnsemble.predict(
    "Is Elixir a functional programming language? Answer yes or no.",
    normalization: :boolean,
    models: [:gemini_flash, :openai_gpt4o_mini]
  )

IO.puts("Question: Is Elixir a functional programming language?")
IO.puts("Answer: #{result.answer}")
IO.puts("Votes: #{inspect(result.metadata.votes)}")

IO.puts("\n")

# Example 5: Asynchronous prediction
IO.puts("Example 5: Asynchronous Predictions")
IO.puts("------------------------------------")

questions = [
  "What is 10 * 10?",
  "What is 5 + 5?",
  "What is 100 - 50?"
]

# Start all predictions concurrently
tasks =
  Enum.map(questions, fn question ->
    {question, CrucibleEnsemble.predict_async(question)}
  end)

# Wait for all results
results =
  Enum.map(tasks, fn {question, task} ->
    {:ok, result} = Task.await(task, 10_000)
    {question, result}
  end)

# Display results
Enum.each(results, fn {question, result} ->
  IO.puts("Q: #{question}")
  IO.puts("A: #{result.answer} (#{Float.round(result.metadata.consensus * 100)}% consensus)")
end)

IO.puts("\n=== Example Complete ===\n")
