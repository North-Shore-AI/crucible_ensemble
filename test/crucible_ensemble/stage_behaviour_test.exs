defmodule CrucibleEnsemble.StageBehaviourTest do
  @moduledoc """
  Tests for Crucible.Stage behaviour compliance in CrucibleEnsemble.Stage.

  These tests verify that the Stage module properly implements the
  Crucible.Stage behaviour from crucible_framework and correctly uses
  the Crucible.Context struct.
  """
  use ExUnit.Case, async: true
  import ExUnit.CaptureLog

  alias Crucible.Context
  alias CrucibleIR.BackendRef
  alias CrucibleIR.Experiment
  alias CrucibleIR.Reliability.Ensemble, as: EnsembleConfig
  alias CrucibleIR.StageDef

  # Helper to create minimal valid context
  defp build_context(overrides) do
    experiment =
      Map.get(overrides, :experiment, %Experiment{
        id: "test_exp",
        backend: %BackendRef{id: :mock},
        pipeline: [%StageDef{name: :ensemble}],
        reliability: %{
          ensemble: %EnsembleConfig{
            strategy: :majority,
            execution_mode: :parallel
          }
        }
      })

    %Context{
      experiment_id: "test_exp",
      run_id: "test_run_#{System.unique_integer([:positive])}",
      experiment: experiment,
      outputs: Map.get(overrides, :outputs, [])
    }
  end

  setup_all do
    Code.ensure_compiled!(CrucibleEnsemble.Stage)
    :ok
  end

  defp run_stage(ctx, opts) do
    test_pid = self()

    log =
      capture_log(fn ->
        send(test_pid, {:stage_result, CrucibleEnsemble.Stage.run(ctx, opts)})
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

  describe "Crucible.Stage behaviour compliance" do
    test "implements run/2 callback" do
      assert function_exported?(CrucibleEnsemble.Stage, :run, 2)
    end

    test "implements describe/1 callback" do
      assert function_exported?(CrucibleEnsemble.Stage, :describe, 1)
    end

    test "declares @behaviour Crucible.Stage" do
      behaviours = CrucibleEnsemble.Stage.__info__(:attributes)[:behaviour] || []
      assert Crucible.Stage in behaviours
    end
  end

  describe "run/2 with Crucible.Context" do
    test "returns {:ok, %Context{}} on success" do
      ctx =
        build_context(%{
          outputs: [
            %{response: "4", model: :model1},
            %{response: "4", model: :model2},
            %{response: "5", model: :model3}
          ]
        })

      result_ctx = run_stage_ok(ctx, %{})
      assert %Context{} = result_ctx
    end

    test "stores ensemble_result in artifacts" do
      ctx =
        build_context(%{
          outputs: [
            %{response: "4", model: :model1},
            %{response: "4", model: :model2}
          ]
        })

      result_ctx = run_stage_ok(ctx, %{})

      assert Context.has_artifact?(result_ctx, :ensemble_result)
      ensemble_result = Context.get_artifact(result_ctx, :ensemble_result)
      assert ensemble_result.strategy == :majority
    end

    test "stores consensus in metrics" do
      ctx =
        build_context(%{
          outputs: [
            %{response: "yes", model: :model1},
            %{response: "yes", model: :model2}
          ]
        })

      result_ctx = run_stage_ok(ctx, %{})

      assert Context.has_metric?(result_ctx, :consensus)
      assert Context.get_metric(result_ctx, :consensus) == 1.0
    end

    test "stores ensemble_latency_us in metrics" do
      ctx =
        build_context(%{
          outputs: [
            %{response: "yes", model: :model1},
            %{response: "yes", model: :model2}
          ]
        })

      result_ctx = run_stage_ok(ctx, %{})

      assert Context.has_metric?(result_ctx, :ensemble_latency_us)
      latency = Context.get_metric(result_ctx, :ensemble_latency_us)
      assert is_integer(latency) and latency >= 0
    end

    test "stores ensemble_strategy in metrics" do
      ctx =
        build_context(%{
          outputs: [
            %{response: "yes", model: :model1},
            %{response: "yes", model: :model2}
          ]
        })

      result_ctx = run_stage_ok(ctx, %{})

      assert Context.has_metric?(result_ctx, :ensemble_strategy)
      assert Context.get_metric(result_ctx, :ensemble_strategy) == :majority
    end

    test "stores answer in assigns" do
      ctx =
        build_context(%{
          outputs: [
            %{response: "paris", model: :model1},
            %{response: "paris", model: :model2}
          ]
        })

      result_ctx = run_stage_ok(ctx, %{})

      assert result_ctx.assigns[:answer] == "paris"
    end

    test "stores ensemble_metadata in assigns" do
      ctx =
        build_context(%{
          outputs: [
            %{response: "4", model: :model1},
            %{response: "4", model: :model2}
          ]
        })

      result_ctx = run_stage_ok(ctx, %{})

      assert Map.has_key?(result_ctx.assigns, :ensemble_metadata)
    end

    test "marks stage as complete" do
      ctx =
        build_context(%{
          outputs: [%{response: "4", model: :model1}]
        })

      result_ctx = run_stage_ok(ctx, %{})

      assert Context.stage_completed?(result_ctx, :ensemble_voting)
    end

    test "returns error for missing ensemble config" do
      experiment = %Experiment{
        id: "test",
        backend: %BackendRef{id: :mock},
        pipeline: [%StageDef{name: :ensemble}],
        # Missing ensemble config
        reliability: %{}
      }

      ctx = %Context{
        experiment_id: "test",
        run_id: "run1",
        experiment: experiment,
        outputs: [%{response: "4", model: :m1}]
      }

      reason = run_stage_error(ctx, %{})
      assert reason == :missing_ensemble_config
    end

    test "returns error for empty outputs" do
      ctx = build_context(%{outputs: []})

      reason = run_stage_error(ctx, %{})
      assert reason == :no_responses
    end

    test "returns error for nil outputs" do
      experiment = %Experiment{
        id: "test",
        backend: %BackendRef{id: :mock},
        pipeline: [%StageDef{name: :ensemble}],
        reliability: %{
          ensemble: %EnsembleConfig{strategy: :majority}
        }
      }

      # Context without outputs set will have outputs: []
      ctx = %Context{
        experiment_id: "test",
        run_id: "run1",
        experiment: experiment,
        outputs: []
      }

      reason = run_stage_error(ctx, %{})
      assert reason == :no_responses
    end
  end

  describe "run/2 with different strategies" do
    test "majority voting normalizes and finds winner" do
      ctx =
        build_context(%{
          outputs: [
            %{response: "A", model: :m1},
            %{response: "A", model: :m2},
            %{response: "B", model: :m3}
          ]
        })

      result_ctx = run_stage_ok(ctx, %{})
      assert result_ctx.assigns[:answer] == "a"
      assert_in_delta Context.get_metric(result_ctx, :consensus), 0.666, 0.01
    end

    test "weighted voting respects confidence scores" do
      experiment = %Experiment{
        id: "test",
        backend: %BackendRef{id: :mock},
        pipeline: [%StageDef{name: :ensemble}],
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

      result_ctx = run_stage_ok(ctx, %{})
      # B has total weight 1.1 (0.6 + 0.5), A has 0.9
      assert result_ctx.assigns[:answer] == "b"
    end

    test "best_confidence selects highest confidence response" do
      experiment = %Experiment{
        id: "test",
        backend: %BackendRef{id: :mock},
        pipeline: [%StageDef{name: :ensemble}],
        reliability: %{
          ensemble: %EnsembleConfig{
            strategy: :best_confidence,
            execution_mode: :parallel
          }
        }
      }

      ctx = %Context{
        experiment_id: "test",
        run_id: "run1",
        experiment: experiment,
        outputs: [
          %{response: "A", model: :m1, confidence: 0.6},
          %{response: "B", model: :m2, confidence: 0.9},
          %{response: "C", model: :m3, confidence: 0.7}
        ]
      }

      result_ctx = run_stage_ok(ctx, %{})
      assert result_ctx.assigns[:answer] == "b"
      assert Context.get_metric(result_ctx, :consensus) == 0.9
    end

    test "unanimous voting succeeds when all agree" do
      experiment = %Experiment{
        id: "test",
        backend: %BackendRef{id: :mock},
        pipeline: [%StageDef{name: :ensemble}],
        reliability: %{
          ensemble: %EnsembleConfig{
            strategy: :unanimous,
            execution_mode: :parallel
          }
        }
      }

      ctx = %Context{
        experiment_id: "test",
        run_id: "run1",
        experiment: experiment,
        outputs: [
          %{response: "Paris", model: :m1},
          %{response: "Paris", model: :m2},
          %{response: "Paris", model: :m3}
        ]
      }

      result_ctx = run_stage_ok(ctx, %{})
      assert result_ctx.assigns[:answer] == "paris"
      assert Context.get_metric(result_ctx, :consensus) == 1.0
    end

    test "unanimous voting fails when models disagree" do
      experiment = %Experiment{
        id: "test",
        backend: %BackendRef{id: :mock},
        pipeline: [%StageDef{name: :ensemble}],
        reliability: %{
          ensemble: %EnsembleConfig{
            strategy: :unanimous,
            execution_mode: :parallel
          }
        }
      }

      ctx = %Context{
        experiment_id: "test",
        run_id: "run1",
        experiment: experiment,
        outputs: [
          %{response: "Paris", model: :m1},
          %{response: "London", model: :m2}
        ]
      }

      error = run_stage_error(ctx, %{})
      assert error.reason == :no_unanimous_consensus
    end

    test "semantic similarity voting groups similar responses" do
      experiment = %Experiment{
        id: "test",
        backend: %BackendRef{id: :mock},
        pipeline: [%StageDef{name: :ensemble}],
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

      result_ctx = run_stage_ok(ctx, %{similarity_threshold: 0.7})

      ensemble_result = Context.get_artifact(result_ctx, :ensemble_result)
      assert ensemble_result.strategy == :semantic_similarity
    end

    test "ranked_choice voting applies instant-runoff" do
      experiment = %Experiment{
        id: "test",
        backend: %BackendRef{id: :mock},
        pipeline: [%StageDef{name: :ensemble}],
        reliability: %{
          ensemble: %EnsembleConfig{
            strategy: :ranked_choice,
            execution_mode: :parallel
          }
        }
      }

      ctx = %Context{
        experiment_id: "test",
        run_id: "run1",
        experiment: experiment,
        outputs: [
          %{response: ["A", "B", "C"], model: :m1},
          %{response: ["B", "A", "C"], model: :m2},
          %{response: ["A", "C", "B"], model: :m3}
        ]
      }

      result_ctx = run_stage_ok(ctx, %{ranking_method: :instant_runoff})

      ensemble_result = Context.get_artifact(result_ctx, :ensemble_result)
      assert ensemble_result.strategy == :ranked_choice
      assert result_ctx.assigns[:answer] != nil
    end
  end

  describe "describe/1" do
    test "returns stage metadata" do
      desc = CrucibleEnsemble.Stage.describe(%{})

      assert desc.name == "ensemble_voting"
      assert is_binary(desc.description)
      assert desc.description =~ "ensemble"
    end

    test "includes behaviour reference" do
      desc = CrucibleEnsemble.Stage.describe(%{})

      assert desc.behaviour == Crucible.Stage
    end

    test "lists available strategies" do
      desc = CrucibleEnsemble.Stage.describe(%{})

      assert :majority in desc.strategies
      assert :weighted in desc.strategies
      assert :best_confidence in desc.strategies
      assert :unanimous in desc.strategies
      assert :semantic_similarity in desc.strategies
      assert :ranked_choice in desc.strategies
    end

    test "lists execution modes" do
      desc = CrucibleEnsemble.Stage.describe(%{})

      assert :parallel in desc.execution_modes
      assert :sequential in desc.execution_modes
      assert :hedged in desc.execution_modes
      assert :cascade in desc.execution_modes
    end

    test "specifies config_type" do
      desc = CrucibleEnsemble.Stage.describe(%{})

      assert desc.config_type == CrucibleIR.Reliability.Ensemble
    end

    test "includes version" do
      desc = CrucibleEnsemble.Stage.describe(%{})

      assert is_binary(desc.version)
    end
  end

  describe "Context preservation" do
    test "preserves existing metrics" do
      ctx =
        build_context(%{
          outputs: [%{response: "4", model: :model1}]
        })

      ctx = Context.put_metric(ctx, :pre_existing_metric, 42)

      result_ctx = run_stage_ok(ctx, %{})

      assert Context.get_metric(result_ctx, :pre_existing_metric) == 42
    end

    test "preserves existing artifacts" do
      ctx =
        build_context(%{
          outputs: [%{response: "4", model: :model1}]
        })

      ctx = Context.put_artifact(ctx, :pre_existing_artifact, "some data")

      result_ctx = run_stage_ok(ctx, %{})

      assert Context.get_artifact(result_ctx, :pre_existing_artifact) == "some data"
    end

    test "preserves existing assigns" do
      ctx =
        build_context(%{
          outputs: [%{response: "4", model: :model1}]
        })

      ctx = Context.assign(ctx, :custom_key, "custom_value")

      result_ctx = run_stage_ok(ctx, %{})

      assert result_ctx.assigns[:custom_key] == "custom_value"
    end

    test "preserves outputs list" do
      outputs = [
        %{response: "4", model: :model1},
        %{response: "4", model: :model2}
      ]

      ctx = build_context(%{outputs: outputs})

      result_ctx = run_stage_ok(ctx, %{})

      assert result_ctx.outputs == outputs
    end
  end

  describe "options handling" do
    test "respects normalization option" do
      ctx =
        build_context(%{
          outputs: [
            %{response: "YES", model: :model1},
            %{response: "yes", model: :model2},
            %{response: "Yes", model: :model3}
          ]
        })

      result_ctx = run_stage_ok(ctx, %{normalization: :lowercase_trim})

      assert result_ctx.assigns[:answer] == "yes"
      assert Context.get_metric(result_ctx, :consensus) == 1.0
    end

    test "respects return_original_answer option" do
      ctx =
        build_context(%{
          outputs: [
            %{response: "The answer is 4", model: :model1},
            %{response: "4", model: :model2}
          ]
        })

      result_ctx =
        run_stage_ok(ctx, %{
          normalization: :numeric,
          return_original_answer: true
        })

      # Should return one of the original responses
      assert result_ctx.assigns[:answer] in ["The answer is 4", "4"]
    end
  end
end
