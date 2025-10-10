#!/usr/bin/env elixir

# Research Experiment Example
#
# This example demonstrates how to conduct a research experiment
# comparing ensemble reliability vs single model performance.
#
# Run:
#   elixir examples/research_experiment.exs

Mix.install([
  {:crucible_ensemble, path: "."}
])

{:ok, _} = Application.ensure_all_started(:crucible_ensemble)

IO.puts("\n=== Ensemble Reliability Research Experiment ===\n")

# Test dataset: Math problems with known correct answers
test_cases = [
  {"What is 15 * 23?", "345"},
  {"What is 144 / 12?", "12"},
  {"What is 7^3?", "343"},
  {"What is sqrt(256)?", "16"},
  {"What is 100 - 37?", "63"}
]

IO.puts("Test Dataset: #{length(test_cases)} math problems")
IO.puts("Models: gemini_flash, openai_gpt4o_mini, anthropic_haiku")
IO.puts("\n")

# Run experiment
results =
  Enum.map(test_cases, fn {question, expected} ->
    IO.puts("Testing: #{question}")

    # Single model baseline (fastest model)
    {:ok, single_result} =
      CrucibleEnsemble.predict(
        question,
        models: [:gemini_flash],
        normalization: :numeric
      )

    # 3-model ensemble
    {:ok, ensemble_result} =
      CrucibleEnsemble.predict(
        question,
        models: [:gemini_flash, :openai_gpt4o_mini, :anthropic_haiku],
        strategy: :majority,
        normalization: :numeric
      )

    # Parse expected answer
    {expected_num, _} = Float.parse(expected)

    # Check correctness
    single_correct = abs(single_result.answer - expected_num) < 0.01
    ensemble_correct = abs(ensemble_result.answer - expected_num) < 0.01

    IO.puts("  Single Model: #{single_result.answer} - #{if single_correct, do: "✓", else: "✗"}")

    IO.puts(
      "  Ensemble: #{ensemble_result.answer} (#{Float.round(ensemble_result.metadata.consensus * 100)}% consensus) - #{if ensemble_correct, do: "✓", else: "✗"}"
    )

    IO.puts("")

    %{
      question: question,
      expected: expected_num,
      single_correct: single_correct,
      ensemble_correct: ensemble_correct,
      single_cost: single_result.metadata.cost_usd,
      ensemble_cost: ensemble_result.metadata.cost_usd,
      consensus: ensemble_result.metadata.consensus
    }
  end)

# Calculate aggregate statistics
single_accuracy = Enum.count(results, & &1.single_correct) / length(results)
ensemble_accuracy = Enum.count(results, & &1.ensemble_correct) / length(results)

total_single_cost = Enum.sum(Enum.map(results, & &1.single_cost))
total_ensemble_cost = Enum.sum(Enum.map(results, & &1.ensemble_cost))

avg_consensus = Enum.sum(Enum.map(results, & &1.consensus)) / length(results)

# Display results
IO.puts("\n=== Experimental Results ===\n")

IO.puts("Accuracy:")
IO.puts("  Single Model: #{Float.round(single_accuracy * 100, 2)}%")
IO.puts("  Ensemble (3 models): #{Float.round(ensemble_accuracy * 100, 2)}%")

IO.puts(
  "  Improvement: +#{Float.round((ensemble_accuracy - single_accuracy) * 100, 2)} percentage points"
)

IO.puts("")

IO.puts("Cost:")
IO.puts("  Single Model Total: $#{Float.round(total_single_cost, 6)}")
IO.puts("  Ensemble Total: $#{Float.round(total_ensemble_cost, 6)}")
IO.puts("  Cost Multiplier: #{Float.round(total_ensemble_cost / total_single_cost, 2)}x")
IO.puts("")

IO.puts("Consensus:")
IO.puts("  Average Consensus: #{Float.round(avg_consensus * 100, 2)}%")
IO.puts("  (Higher consensus = more agreement between models)")
IO.puts("")

# Theoretical reliability calculation
IO.puts("=== Theoretical Reliability Analysis ===\n")

# Assume 85% accuracy per model
individual_accuracy = 0.85
ensemble_size = 3

# Probability of majority being correct
# P(2 or more correct) = P(2 correct) + P(3 correct)
p_2_correct = 3 * individual_accuracy ** 2 * (1 - individual_accuracy)
p_3_correct = individual_accuracy ** 3
theoretical_ensemble_accuracy = p_2_correct + p_3_correct

IO.puts("Assumptions:")
IO.puts("  Individual model accuracy: #{Float.round(individual_accuracy * 100, 2)}%")
IO.puts("  Ensemble size: #{ensemble_size} models")
IO.puts("  Voting strategy: Majority (2-of-3)")
IO.puts("")

IO.puts("Theoretical Ensemble Accuracy:")
IO.puts("  P(2 of 3 correct): #{Float.round(p_2_correct * 100, 2)}%")
IO.puts("  P(3 of 3 correct): #{Float.round(p_3_correct * 100, 2)}%")
IO.puts("  Total: #{Float.round(theoretical_ensemble_accuracy * 100, 2)}%")
IO.puts("")

IO.puts("Key Insight:")

IO.puts(
  "  Ensemble voting can increase reliability from #{Float.round(individual_accuracy * 100)}% to #{Float.round(theoretical_ensemble_accuracy * 100, 2)}%"
)

IO.puts(
  "  This represents a #{Float.round((1 - (1 - theoretical_ensemble_accuracy) / (1 - individual_accuracy)) * 100)}% reduction in error rate!"
)

IO.puts("\n=== Experiment Complete ===\n")

# Export results for further analysis
metrics = %{
  test_cases: length(test_cases),
  single_accuracy: single_accuracy,
  ensemble_accuracy: ensemble_accuracy,
  total_single_cost: total_single_cost,
  total_ensemble_cost: total_ensemble_cost,
  avg_consensus: avg_consensus,
  theoretical_accuracy: theoretical_ensemble_accuracy
}

IO.puts("\nMetrics Summary:")
IO.inspect(metrics, pretty: true)
