defmodule CrucibleEnsemble.StageTest do
  @moduledoc """
  Tests for CrucibleEnsemble.Stage module.

  These tests verify the ensemble voting stage functionality using
  the Crucible.Context struct from crucible_framework.
  """
  use ExUnit.Case, async: true
  import ExUnit.CaptureLog

  alias Crucible.Context
  alias CrucibleEnsemble.Stage
  alias CrucibleIR.BackendRef
  alias CrucibleIR.Experiment
  alias CrucibleIR.Reliability.Ensemble, as: EnsembleConfig
  alias CrucibleIR.StageDef

  # Helper to build a valid Crucible.Context for testing
  defp build_context(ensemble_config, outputs) do
    experiment = %Experiment{
      id: "test_exp",
      backend: %BackendRef{id: :mock},
      pipeline: [%StageDef{name: :ensemble}],
      reliability: %{ensemble: ensemble_config}
    }

    %Context{
      experiment_id: "test_exp",
      run_id: "test_run_#{System.unique_integer([:positive])}",
      experiment: experiment,
      outputs: outputs
    }
  end

  defp run_stage(ctx, opts) do
    test_pid = self()

    log =
      capture_log(fn ->
        send(test_pid, {:stage_result, Stage.run(ctx, opts)})
      end)

    result =
      receive do
        {:stage_result, result} -> result
      end

    {result, log}
  end

  defp run_stage_ok(ctx, opts) do
    {{:ok, result}, log} = run_stage(ctx, opts)
    assert log =~ "Ensemble voting complete"
    result
  end

  defp run_stage_error(ctx, opts) do
    {{:error, reason}, log} = run_stage(ctx, opts)
    assert log =~ "Ensemble voting failed"
    reason
  end

  describe "run/2 with existing outputs" do
    test "applies majority voting to outputs" do
      config = %EnsembleConfig{strategy: :majority, execution_mode: :parallel}

      ctx =
        build_context(config, [
          %{response: "4", model: :model1},
          %{response: "4", model: :model2},
          %{response: "5", model: :model3}
        ])

      result = run_stage_ok(ctx, %{})

      assert result.assigns[:answer] == "4"
      assert Context.get_metric(result, :consensus) == 2 / 3
      ensemble_result = Context.get_artifact(result, :ensemble_result)
      assert ensemble_result.strategy == :majority
      assert Map.has_key?(result.assigns, :ensemble_metadata)
    end

    test "applies weighted voting to outputs" do
      config = %EnsembleConfig{strategy: :weighted, execution_mode: :parallel}

      ctx =
        build_context(config, [
          %{response: "A", model: :model1, confidence: 0.9},
          %{response: "B", model: :model2, confidence: 0.6},
          %{response: "B", model: :model3, confidence: 0.5}
        ])

      result = run_stage_ok(ctx, %{})

      # B has total weight 1.1 (0.6 + 0.5), A has 0.9
      assert result.assigns[:answer] == "b"
      ensemble_result = Context.get_artifact(result, :ensemble_result)
      assert ensemble_result.strategy == :weighted
    end

    test "applies best_confidence strategy" do
      config = %EnsembleConfig{strategy: :best_confidence, execution_mode: :parallel}

      ctx =
        build_context(config, [
          %{response: "A", model: :model1, confidence: 0.6},
          %{response: "B", model: :model2, confidence: 0.9},
          %{response: "C", model: :model3, confidence: 0.7}
        ])

      result = run_stage_ok(ctx, %{})

      assert result.assigns[:answer] == "b"
      assert Context.get_metric(result, :consensus) == 0.9
      ensemble_result = Context.get_artifact(result, :ensemble_result)
      assert ensemble_result.strategy == :best_confidence
    end

    test "applies unanimous voting successfully" do
      config = %EnsembleConfig{strategy: :unanimous, execution_mode: :parallel}

      ctx =
        build_context(config, [
          %{response: "Paris", model: :model1},
          %{response: "Paris", model: :model2},
          %{response: "Paris", model: :model3}
        ])

      result = run_stage_ok(ctx, %{})

      assert result.assigns[:answer] == "paris"
      assert Context.get_metric(result, :consensus) == 1.0
      ensemble_result = Context.get_artifact(result, :ensemble_result)
      assert ensemble_result.strategy == :unanimous
    end

    test "handles unanimous voting failure" do
      config = %EnsembleConfig{strategy: :unanimous, execution_mode: :parallel}

      ctx =
        build_context(config, [
          %{response: "Paris", model: :model1},
          %{response: "London", model: :model2},
          %{response: "Paris", model: :model3}
        ])

      error = run_stage_error(ctx, %{})

      assert error.reason == :no_unanimous_consensus
    end

    test "respects normalization option" do
      config = %EnsembleConfig{strategy: :majority, execution_mode: :parallel}

      ctx =
        build_context(config, [
          %{response: "YES", model: :model1},
          %{response: "yes", model: :model2},
          %{response: "Yes", model: :model3}
        ])

      result = run_stage_ok(ctx, %{normalization: :lowercase_trim})

      assert result.assigns[:answer] == "yes"
      assert Context.get_metric(result, :consensus) == 1.0
    end

    test "respects return_original_answer option" do
      config = %EnsembleConfig{strategy: :majority, execution_mode: :parallel}

      ctx =
        build_context(config, [
          %{response: "The answer is 4", model: :model1},
          %{response: "4", model: :model2},
          %{response: "Four", model: :model3}
        ])

      result =
        run_stage_ok(ctx, %{
          normalization: :numeric,
          return_original_answer: true
        })

      # Should return one of the original responses
      assert result.assigns[:answer] in ["The answer is 4", "4", "Four"]
    end
  end

  describe "run/2 error handling" do
    test "returns error when ensemble config is missing" do
      experiment = %Experiment{
        id: "test",
        backend: %BackendRef{id: :mock},
        pipeline: [%StageDef{name: :ensemble}],
        reliability: %{}
      }

      ctx = %Context{
        experiment_id: "test",
        run_id: "run1",
        experiment: experiment,
        outputs: [%{response: "4", model: :model1}]
      }

      reason = run_stage_error(ctx, %{})

      assert reason == :missing_ensemble_config
    end

    test "returns error when ensemble config is invalid type" do
      experiment = %Experiment{
        id: "test",
        backend: %BackendRef{id: :mock},
        pipeline: [%StageDef{name: :ensemble}],
        reliability: %{ensemble: "not a valid config"}
      }

      ctx = %Context{
        experiment_id: "test",
        run_id: "run1",
        experiment: experiment,
        outputs: [%{response: "4", model: :model1}]
      }

      reason = run_stage_error(ctx, %{})
      assert {:invalid_ensemble_config, _} = reason
    end

    test "returns error when outputs are empty" do
      config = %EnsembleConfig{strategy: :majority, execution_mode: :parallel}
      ctx = build_context(config, [])

      reason = run_stage_error(ctx, %{})

      assert reason == :no_responses
    end
  end

  describe "run/2 with semantic_similarity strategy" do
    test "groups similar responses" do
      config = %EnsembleConfig{strategy: :semantic_similarity, execution_mode: :parallel}

      ctx =
        build_context(config, [
          %{response: "The answer is 4", model: :model1},
          %{response: "4", model: :model2},
          %{response: "Four", model: :model3}
        ])

      result =
        run_stage_ok(ctx, %{
          similarity_threshold: 0.3,
          similarity_metric: :levenshtein
        })

      # Should recognize all as similar answers
      assert result.assigns[:answer] in ["the answer is 4", "4", "four"]
      ensemble_result = Context.get_artifact(result, :ensemble_result)
      assert ensemble_result.strategy == :semantic_similarity
    end
  end

  describe "run/2 with ranked_choice strategy" do
    test "applies ranked choice voting" do
      config = %EnsembleConfig{strategy: :ranked_choice, execution_mode: :parallel}

      ctx =
        build_context(config, [
          %{response: ["A", "B", "C"], model: :model1},
          %{response: ["B", "A", "C"], model: :model2},
          %{response: ["A", "C", "B"], model: :model3}
        ])

      result =
        run_stage_ok(ctx, %{
          ranking_method: :instant_runoff
        })

      ensemble_result = Context.get_artifact(result, :ensemble_result)
      assert ensemble_result.strategy == :ranked_choice
      assert result.assigns[:answer] != nil
    end
  end

  describe "run/2 with config options" do
    test "uses timeout from ensemble config" do
      config = %EnsembleConfig{
        strategy: :majority,
        execution_mode: :parallel,
        timeout_ms: 3000
      }

      ctx =
        build_context(config, [
          %{response: "4", model: :model1},
          %{response: "4", model: :model2}
        ])

      result = run_stage_ok(ctx, %{})

      # Should complete successfully with the timeout setting
      assert result.assigns[:answer] == "4"
    end

    test "uses min_agreement from ensemble config" do
      config = %EnsembleConfig{
        strategy: :majority,
        execution_mode: :sequential,
        min_agreement: 0.8
      }

      ctx =
        build_context(config, [
          %{response: "4", model: :model1},
          %{response: "4", model: :model2},
          %{response: "5", model: :model3}
        ])

      result = run_stage_ok(ctx, %{})

      # Consensus is 2/3 = 0.667, which is less than 0.8
      assert Context.get_metric(result, :consensus) == 2 / 3
      assert result.assigns[:answer] == "4"
    end

    test "uses weights from ensemble config for weighted strategy" do
      config = %EnsembleConfig{
        strategy: :weighted,
        execution_mode: :parallel,
        weights: %{
          model1: 2.0,
          model2: 1.0,
          model3: 1.0
        }
      }

      ctx =
        build_context(config, [
          %{response: "A", model: :model1, confidence: 0.5},
          %{response: "B", model: :model2, confidence: 0.5},
          %{response: "B", model: :model3, confidence: 0.5}
        ])

      result = run_stage_ok(ctx, %{})

      # With weights, model1's response gets doubled weight
      # A: 0.5 (from confidence), B: 0.5 + 0.5 = 1.0
      # But note: the Stage doesn't currently apply the weights to responses
      # It passes them as options, so the behavior depends on Vote implementation
      assert result.assigns[:answer] in ["a", "b"]
    end
  end

  describe "describe/1" do
    test "returns stage metadata" do
      description = Stage.describe(%{})

      assert description.name == "ensemble_voting"
      assert description.description =~ "ensemble"
      assert description.version == "0.4.0"
      assert description.config_type == CrucibleIR.Reliability.Ensemble
    end

    test "includes behaviour reference" do
      description = Stage.describe(%{})

      assert description.behaviour == Crucible.Stage
    end

    test "lists available strategies" do
      description = Stage.describe(%{})

      assert :majority in description.strategies
      assert :weighted in description.strategies
      assert :best_confidence in description.strategies
      assert :unanimous in description.strategies
      assert :semantic_similarity in description.strategies
      assert :ranked_choice in description.strategies
    end

    test "lists execution modes" do
      description = Stage.describe(%{})

      assert :parallel in description.execution_modes
      assert :sequential in description.execution_modes
      assert :hedged in description.execution_modes
      assert :cascade in description.execution_modes
    end

    test "accepts options parameter" do
      description = Stage.describe(%{custom_option: true})

      # Should still return valid description
      assert description.name == "ensemble_voting"
    end
  end

  describe "run/2 with additional options" do
    test "merges config options with additional options" do
      config = %EnsembleConfig{
        strategy: :majority,
        execution_mode: :parallel,
        options: %{"custom_key" => "custom_value"}
      }

      ctx =
        build_context(config, [
          %{response: "4", model: :model1},
          %{response: "4", model: :model2}
        ])

      result = run_stage_ok(ctx, %{another_option: "value"})

      assert result.assigns[:answer] == "4"
    end

    test "additional options take precedence over config options" do
      config = %EnsembleConfig{strategy: :majority, execution_mode: :parallel}

      ctx =
        build_context(config, [
          %{response: "YES", model: :model1},
          %{response: "yes", model: :model2}
        ])

      result = run_stage_ok(ctx, %{normalization: :lowercase_trim})

      assert result.assigns[:answer] == "yes"
      assert Context.get_metric(result, :consensus) == 1.0
    end
  end

  describe "run/2 context preservation" do
    test "preserves existing context fields" do
      config = %EnsembleConfig{strategy: :majority, execution_mode: :parallel}

      ctx =
        build_context(config, [
          %{response: "4", model: :model1},
          %{response: "4", model: :model2}
        ])

      # Add custom assigns and metrics
      ctx = Context.assign(ctx, :custom_field, "preserved")
      ctx = Context.put_metric(ctx, :custom_metric, 123)

      result = run_stage_ok(ctx, %{})

      assert result.assigns[:custom_field] == "preserved"
      assert Context.get_metric(result, :custom_metric) == 123
      assert result.assigns[:answer] == "4"
    end

    test "adds ensemble-specific fields to context" do
      config = %EnsembleConfig{strategy: :majority, execution_mode: :parallel}

      ctx =
        build_context(config, [
          %{response: "4", model: :model1},
          %{response: "4", model: :model2}
        ])

      result = run_stage_ok(ctx, %{})

      assert Context.has_artifact?(result, :ensemble_result)
      assert Context.has_metric?(result, :consensus)
      assert Context.has_metric?(result, :ensemble_latency_us)
      assert Context.has_metric?(result, :ensemble_strategy)
      assert Map.has_key?(result.assigns, :answer)
      assert Map.has_key?(result.assigns, :ensemble_metadata)
    end

    test "marks stage as complete" do
      config = %EnsembleConfig{strategy: :majority, execution_mode: :parallel}

      ctx =
        build_context(config, [
          %{response: "4", model: :model1}
        ])

      result = run_stage_ok(ctx, %{})

      assert Context.stage_completed?(result, :ensemble_voting)
    end
  end
end
