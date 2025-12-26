# Implementation Prompt: CrucibleEnsemble Stage Integration

**Date**: 2025-12-25
**Target**: Implement `Crucible.Stage` behaviour wrapper for CrucibleEnsemble

---

## Objective

Create a proper `Crucible.Stage` behaviour implementation that wraps `CrucibleEnsemble` voting functionality for use in crucible_framework pipelines.

---

## Required Reading

Before starting, read these files in order:

### 1. Crucible.Stage Behaviour (Reference Implementation)
```
/home/home/p/g/North-Shore-AI/crucible_framework/lib/crucible/stage.ex
```
- Lines 1-18: The behaviour definition
- Defines `@callback run/2` and `@callback describe/1`

### 2. Crucible.Context Struct
```
/home/home/p/g/North-Shore-AI/crucible_framework/lib/crucible/context.ex
```
- Lines 1-101: Struct definition and type
- Lines 107-119: `put_metric/3`
- Lines 219-231: `put_artifact/3`
- Lines 357-362: `mark_stage_complete/2`

### 3. Example Stage Implementation
```
/home/home/p/g/North-Shore-AI/crucible_framework/lib/crucible/stage/bench.ex
```
- Lines 1-96: Full stage implementation example
- Line 43: `@behaviour Crucible.Stage`
- Lines 52-86: `run/2` implementation pattern
- Lines 88-96: `describe/1` implementation

### 4. CrucibleEnsemble Main Module
```
/home/home/p/g/North-Shore-AI/crucible_ensemble/lib/ensemble.ex
```
- Lines 1-55: Module documentation
- Lines 59-95: Type definitions
- Lines 149-242: `predict/2` implementation

### 5. Current Stage Implementation (to be wrapped/refactored)
```
/home/home/p/g/North-Shore-AI/crucible_ensemble/lib/crucible_ensemble/stage.ex
```
- Lines 1-271: Current implementation using plain maps
- Lines 89-108: Current `run/2` function
- Lines 132-164: Current `describe/1` function

### 6. Vote Module
```
/home/home/p/g/North-Shore-AI/crucible_ensemble/lib/crucible_ensemble/vote.ex
```
- Lines 56-99: `apply_strategy/3` - main voting function
- Lines 116-136: `consensus_strength/1`

### 7. CrucibleIR Ensemble Config
```
/home/home/p/g/North-Shore-AI/crucible_ensemble/deps/crucible_ir/lib/crucible_ir/reliability/ensemble.ex
```
(Or check the crucible_ir library for the Ensemble struct definition)

### 8. Current Tests for Stage
```
/home/home/p/g/North-Shore-AI/crucible_ensemble/test/crucible_ensemble/stage_test.exs
```
- Lines 1-554: Comprehensive test cases to maintain compatibility

### 9. Mix Configuration
```
/home/home/p/g/North-Shore-AI/crucible_ensemble/mix.exs
```
- Lines 29-42: Current dependencies (need to add crucible_framework)

---

## Implementation Steps

### Step 1: Add Dependency

Edit `/home/home/p/g/North-Shore-AI/crucible_ensemble/mix.exs`:

Add crucible_framework to deps (Line ~34):
```elixir
defp deps do
  [
    # Core dependencies
    {:crucible_framework, "~> 0.5.0"},  # ADD THIS
    {:crucible_ir, "~> 0.1.1"},
    {:jason, "~> 1.4"},
    {:telemetry, "~> 1.2"},
    # ...
  ]
end
```

### Step 2: Write Tests First (TDD)

Create new test file:
```
/home/home/p/g/North-Shore-AI/crucible_ensemble/test/crucible_ensemble/stage_behaviour_test.exs
```

Test cases to write:

```elixir
defmodule CrucibleEnsemble.StageBehaviourTest do
  use ExUnit.Case, async: true

  alias Crucible.Context
  alias CrucibleIR.Reliability.Ensemble, as: EnsembleConfig
  alias CrucibleIR.Experiment
  alias CrucibleIR.BackendRef

  # Helper to create minimal valid context
  defp build_context(overrides \\ %{}) do
    experiment = Map.get(overrides, :experiment, %Experiment{
      id: "test_exp",
      backend: %BackendRef{id: :mock},
      reliability: %{
        ensemble: %EnsembleConfig{
          strategy: :majority,
          execution_mode: :parallel
        }
      }
    })

    %Context{
      experiment_id: "test_exp",
      run_id: "test_run_#{System.unique_integer()}",
      experiment: experiment,
      outputs: Map.get(overrides, :outputs, [])
    }
  end

  describe "Crucible.Stage behaviour compliance" do
    test "implements run/2 callback" do
      assert function_exported?(CrucibleEnsemble.Stage, :run, 2)
    end

    test "implements describe/1 callback" do
      assert function_exported?(CrucibleEnsemble.Stage, :describe, 1)
    end
  end

  describe "run/2 with Crucible.Context" do
    test "returns {:ok, %Context{}} on success" do
      ctx = build_context(%{
        outputs: [
          %{response: "4", model: :model1},
          %{response: "4", model: :model2},
          %{response: "5", model: :model3}
        ]
      })

      assert {:ok, %Context{} = result_ctx} = CrucibleEnsemble.Stage.run(ctx, %{})
    end

    test "stores ensemble_result in artifacts" do
      ctx = build_context(%{
        outputs: [
          %{response: "4", model: :model1},
          %{response: "4", model: :model2}
        ]
      })

      {:ok, result_ctx} = CrucibleEnsemble.Stage.run(ctx, %{})

      assert Context.has_artifact?(result_ctx, :ensemble_result)
      ensemble_result = Context.get_artifact(result_ctx, :ensemble_result)
      assert ensemble_result.strategy == :majority
    end

    test "stores consensus in metrics" do
      ctx = build_context(%{
        outputs: [
          %{response: "yes", model: :model1},
          %{response: "yes", model: :model2}
        ]
      })

      {:ok, result_ctx} = CrucibleEnsemble.Stage.run(ctx, %{})

      assert Context.has_metric?(result_ctx, :consensus)
      assert Context.get_metric(result_ctx, :consensus) == 1.0
    end

    test "stores answer in assigns" do
      ctx = build_context(%{
        outputs: [
          %{response: "paris", model: :model1},
          %{response: "paris", model: :model2}
        ]
      })

      {:ok, result_ctx} = CrucibleEnsemble.Stage.run(ctx, %{})

      assert result_ctx.assigns[:answer] == "paris"
    end

    test "marks stage as complete" do
      ctx = build_context(%{
        outputs: [%{response: "4", model: :model1}]
      })

      {:ok, result_ctx} = CrucibleEnsemble.Stage.run(ctx, %{})

      assert Context.stage_completed?(result_ctx, :ensemble_voting)
    end

    test "returns error for missing ensemble config" do
      experiment = %Experiment{
        id: "test",
        backend: %BackendRef{id: :mock},
        reliability: %{}  # Missing ensemble config
      }

      ctx = %Context{
        experiment_id: "test",
        run_id: "run1",
        experiment: experiment,
        outputs: [%{response: "4", model: :m1}]
      }

      assert {:error, :missing_ensemble_config} = CrucibleEnsemble.Stage.run(ctx, %{})
    end

    test "returns error for empty outputs" do
      ctx = build_context(%{outputs: []})

      assert {:error, :no_responses} = CrucibleEnsemble.Stage.run(ctx, %{})
    end
  end

  describe "run/2 with different strategies" do
    test "majority voting" do
      ctx = build_context(%{
        outputs: [
          %{response: "A", model: :m1},
          %{response: "A", model: :m2},
          %{response: "B", model: :m3}
        ]
      })

      {:ok, result_ctx} = CrucibleEnsemble.Stage.run(ctx, %{})
      assert result_ctx.assigns[:answer] == "a"
      assert_in_delta Context.get_metric(result_ctx, :consensus), 0.666, 0.01
    end

    test "weighted voting" do
      experiment = %Experiment{
        id: "test",
        backend: %BackendRef{id: :mock},
        reliability: %{
          ensemble: %EnsembleConfig{
            strategy: :weighted,
            execution_mode: :parallel
          }
        }
      }

      ctx = %Context{
        experiment_id: "test",
        run_id: "run1",
        experiment: experiment,
        outputs: [
          %{response: "A", model: :m1, confidence: 0.9},
          %{response: "B", model: :m2, confidence: 0.6},
          %{response: "B", model: :m3, confidence: 0.5}
        ]
      }

      {:ok, result_ctx} = CrucibleEnsemble.Stage.run(ctx, %{})
      assert result_ctx.assigns[:answer] == "b"  # B has higher total weight
    end

    test "semantic similarity voting" do
      experiment = %Experiment{
        id: "test",
        backend: %BackendRef{id: :mock},
        reliability: %{
          ensemble: %EnsembleConfig{
            strategy: :semantic_similarity,
            execution_mode: :parallel
          }
        }
      }

      ctx = %Context{
        experiment_id: "test",
        run_id: "run1",
        experiment: experiment,
        outputs: [
          %{response: "hello", model: :m1},
          %{response: "hallo", model: :m2},
          %{response: "world", model: :m3}
        ]
      }

      {:ok, result_ctx} = CrucibleEnsemble.Stage.run(ctx, %{similarity_threshold: 0.7})

      # hello and hallo should cluster
      ensemble_result = Context.get_artifact(result_ctx, :ensemble_result)
      assert ensemble_result.strategy == :semantic_similarity
    end
  end

  describe "describe/1" do
    test "returns stage metadata" do
      desc = CrucibleEnsemble.Stage.describe(%{})

      assert desc.name == "ensemble_voting"
      assert desc.description =~ "ensemble"
      assert :majority in desc.strategies
      assert :parallel in desc.execution_modes
    end
  end

  describe "backwards compatibility" do
    # Ensure existing plain-map usage still works via wrapper
    test "accepts outputs key in context" do
      ctx = build_context(%{
        outputs: [%{response: "test", model: :m1}]
      })

      assert {:ok, _} = CrucibleEnsemble.Stage.run(ctx, %{})
    end

    test "accepts responses key in context" do
      experiment = %Experiment{
        id: "test",
        backend: %BackendRef{id: :mock},
        reliability: %{
          ensemble: %EnsembleConfig{strategy: :majority}
        }
      }

      # Responses might be stored differently
      ctx = %Context{
        experiment_id: "test",
        run_id: "run1",
        experiment: experiment,
        outputs: [%{response: "test", model: :m1}]
      }

      assert {:ok, _} = CrucibleEnsemble.Stage.run(ctx, %{})
    end
  end
end
```

### Step 3: Implement the Stage

Refactor `/home/home/p/g/North-Shore-AI/crucible_ensemble/lib/crucible_ensemble/stage.ex`:

```elixir
defmodule CrucibleEnsemble.Stage do
  @moduledoc """
  Crucible.Stage behaviour implementation for ensemble voting.

  This stage integrates CrucibleEnsemble voting functionality into
  crucible_framework pipelines, allowing ensemble voting to be composed
  with other stages like DataLoad, BackendCall, Bench, etc.

  ## Context Requirements

  The stage expects the following in the Crucible.Context:

  - `experiment.reliability.ensemble` - CrucibleIR.Reliability.Ensemble configuration
  - `outputs` - Model responses to vote on (list of maps with `:response` key)

  ## Output

  Stores in the context:

  - `artifacts[:ensemble_result]` - The full voting result map
  - `metrics[:consensus]` - Consensus score (0.0 to 1.0)
  - `metrics[:ensemble_latency_us]` - Voting latency in microseconds
  - `assigns[:answer]` - The final answer selected by voting

  ## Examples

      # In a pipeline
      stages = [
        {Crucible.Stage.DataLoad, %{}},
        {Crucible.Stage.BackendCall, %{}},
        {CrucibleEnsemble.Stage, %{normalization: :lowercase_trim}},
        {Crucible.Stage.Bench, %{}}
      ]

      {:ok, ctx} = Crucible.Pipeline.Runner.run(experiment, stages: stages)
      ctx.assigns[:answer]
      # => "4"

  """

  @behaviour Crucible.Stage

  require Logger

  alias Crucible.Context
  alias CrucibleIR.Reliability.Ensemble, as: EnsembleConfig

  @impl true
  @spec run(Context.t(), map()) :: {:ok, Context.t()} | {:error, term()}
  def run(%Context{experiment: experiment} = ctx, opts) do
    start_time = System.monotonic_time(:microsecond)

    with {:ok, ensemble_config} <- extract_ensemble_config(experiment),
         {:ok, responses} <- get_responses(ctx),
         {:ok, vote_result} <- apply_voting(responses, ensemble_config, opts) do
      end_time = System.monotonic_time(:microsecond)
      latency_us = end_time - start_time

      # Store results using Context helpers
      updated_ctx =
        ctx
        |> Context.put_artifact(:ensemble_result, vote_result)
        |> Context.put_metric(:consensus, vote_result.consensus)
        |> Context.put_metric(:ensemble_latency_us, latency_us)
        |> Context.put_metric(:ensemble_strategy, vote_result.strategy)
        |> Context.assign(:answer, vote_result.answer)
        |> Context.assign(:ensemble_metadata, Map.get(vote_result, :metadata, %{}))
        |> Context.mark_stage_complete(:ensemble_voting)

      Logger.debug(
        "Ensemble voting complete: strategy=#{vote_result.strategy}, " <>
          "consensus=#{Float.round(vote_result.consensus, 4)}, " <>
          "latency_us=#{latency_us}"
      )

      {:ok, updated_ctx}
    else
      {:error, reason} = error ->
        Logger.warning("Ensemble voting failed: #{inspect(reason)}")
        error
    end
  end

  @impl true
  @spec describe(map()) :: map()
  def describe(_opts \\ %{}) do
    %{
      name: "ensemble_voting",
      description: "Multi-model ensemble voting stage using CrucibleEnsemble",
      version: "0.4.0",
      behaviour: Crucible.Stage,
      inputs: [
        {:context, :outputs, "List of model response maps"},
        {:experiment, :reliability, :ensemble, "EnsembleConfig struct"}
      ],
      outputs: [
        {:artifact, :ensemble_result, "Full voting result map"},
        {:metric, :consensus, "Consensus score 0.0-1.0"},
        {:metric, :ensemble_latency_us, "Voting latency"},
        {:metric, :ensemble_strategy, "Strategy used"},
        {:assign, :answer, "Final answer"}
      ],
      config_type: CrucibleIR.Reliability.Ensemble,
      strategies: [
        :majority,
        :weighted,
        :best_confidence,
        :unanimous,
        :semantic_similarity,
        :ranked_choice
      ],
      execution_modes: [
        :parallel,
        :sequential,
        :hedged,
        :cascade
      ]
    }
  end

  # Private helper functions

  defp extract_ensemble_config(%{reliability: %{ensemble: %EnsembleConfig{} = config}}) do
    {:ok, config}
  end

  defp extract_ensemble_config(%{reliability: %{ensemble: config}}) when is_map(config) do
    # Try to convert map to struct
    try do
      {:ok, struct(EnsembleConfig, config)}
    rescue
      _ -> {:error, {:invalid_ensemble_config, config}}
    end
  end

  defp extract_ensemble_config(_) do
    {:error, :missing_ensemble_config}
  end

  defp get_responses(%Context{outputs: outputs}) when is_list(outputs) and outputs != [] do
    {:ok, outputs}
  end

  defp get_responses(%Context{outputs: []}) do
    {:error, :no_responses}
  end

  defp get_responses(%Context{outputs: nil}) do
    {:error, :missing_outputs}
  end

  defp apply_voting(responses, %EnsembleConfig{} = config, opts) do
    # Build voting options
    voting_opts =
      opts
      |> Map.to_list()
      |> Keyword.put_new(:normalization, :lowercase_trim)
      |> Keyword.put_new(:return_original_answer, false)
      |> maybe_add_weights(config)

    # Apply the voting strategy
    CrucibleEnsemble.Vote.apply_strategy(responses, config.strategy, voting_opts)
  end

  defp maybe_add_weights(opts, %EnsembleConfig{strategy: :weighted, weights: weights})
       when is_map(weights) and map_size(weights) > 0 do
    Keyword.put(opts, :weights, weights)
  end

  defp maybe_add_weights(opts, _config), do: opts
end
```

### Step 4: Update mix.exs Application Config

Edit `/home/home/p/g/North-Shore-AI/crucible_ensemble/mix.exs` (Lines 23-27):

```elixir
def application do
  [
    extra_applications: [:logger],
    mod: {CrucibleEnsemble.Application, []}  # ADD THIS
  ]
end
```

### Step 5: Update README.md

Add a new section to `/home/home/p/g/North-Shore-AI/crucible_ensemble/README.md`:

After the "Pipeline Stage Usage" section (around Line 97), add:

```markdown
### Crucible.Stage Behaviour Integration (v0.4.0+)

CrucibleEnsemble.Stage now implements the `Crucible.Stage` behaviour from crucible_framework,
enabling seamless integration with pipelines:

```elixir
# Define pipeline stages
stages = [
  {Crucible.Stage.DataLoad, %{source: :gsm8k}},
  {Crucible.Stage.BackendCall, %{timeout: 5000}},
  {CrucibleEnsemble.Stage, %{normalization: :numeric}},
  {Crucible.Stage.Bench, %{tests: [:ttest, :bootstrap]}}
]

# Run experiment
{:ok, ctx} = CrucibleFramework.run(experiment, stages: stages)

# Access results
ctx.assigns[:answer]                              # => "42"
Crucible.Context.get_metric(ctx, :consensus)      # => 1.0
Crucible.Context.get_artifact(ctx, :ensemble_result)  # => %{strategy: :majority, ...}
```

The stage stores results using Crucible.Context helpers:
- `artifacts[:ensemble_result]` - Full voting result
- `metrics[:consensus]` - Consensus score (0.0-1.0)
- `metrics[:ensemble_latency_us]` - Voting latency
- `assigns[:answer]` - Final answer

Stage completion is tracked via `Crucible.Context.stage_completed?(ctx, :ensemble_voting)`.
```

### Step 6: Run Quality Checks

After implementation, run:

```bash
# Compile with warnings as errors
cd /home/home/p/g/North-Shore-AI/crucible_ensemble
mix deps.get
mix compile --warnings-as-errors

# Run tests
mix test

# Check formatting
mix format --check-formatted

# Run Credo (strict mode)
mix credo --strict

# Run Dialyzer
mix dialyzer
```

All must pass with no warnings or errors.

---

## Quality Requirements Checklist

- [ ] All tests pass: `mix test` returns 0 failures
- [ ] No compiler warnings: `mix compile --warnings-as-errors` succeeds
- [ ] Code formatted: `mix format --check-formatted` succeeds
- [ ] Credo passes: `mix credo --strict` returns no issues
- [ ] Dialyzer passes: `mix dialyzer` returns no warnings
- [ ] README updated with Stage behaviour documentation
- [ ] `@behaviour Crucible.Stage` declared
- [ ] `@impl true` on both `run/2` and `describe/1`
- [ ] Uses `Crucible.Context.t()` struct, not plain maps
- [ ] Results stored in `artifacts` using `put_artifact/3`
- [ ] Metrics stored using `put_metric/3`
- [ ] Stage completion marked with `mark_stage_complete/2`
- [ ] Backwards compatible with existing tests in `stage_test.exs`

---

## File Summary

| File | Action |
|------|--------|
| `mix.exs` | Add `crucible_framework` dependency, update application config |
| `lib/crucible_ensemble/stage.ex` | Refactor to implement `Crucible.Stage` behaviour |
| `test/crucible_ensemble/stage_behaviour_test.exs` | New test file for behaviour compliance |
| `README.md` | Add documentation for Stage behaviour integration |

---

## Notes for Implementation Agent

1. **Read ALL required files before starting** - Understanding the Crucible.Context structure is critical
2. **Write tests first** - Follow TDD approach strictly
3. **Keep backwards compatibility** - Existing tests in `stage_test.exs` must continue to pass
4. **Use Context helpers** - Never manipulate context struct fields directly
5. **Follow existing patterns** - Look at `Crucible.Stage.Bench` for conventions
6. **Run quality checks after each change** - Catch issues early
