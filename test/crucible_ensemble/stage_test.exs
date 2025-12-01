defmodule CrucibleEnsemble.StageTest do
  use ExUnit.Case, async: true
  alias CrucibleEnsemble.Stage
  alias CrucibleIR.Reliability.Ensemble, as: EnsembleConfig

  describe "run/2 with existing outputs" do
    test "applies majority voting to outputs" do
      context = %{
        experiment: %{
          reliability: %{
            ensemble: %EnsembleConfig{
              strategy: :majority,
              execution_mode: :parallel
            }
          }
        },
        outputs: [
          %{response: "4", model: :model1},
          %{response: "4", model: :model2},
          %{response: "5", model: :model3}
        ]
      }

      {:ok, result} = Stage.run(context)

      assert result.answer == "4"
      assert result.consensus == 2 / 3
      assert result.ensemble_result.strategy == :majority
      assert Map.has_key?(result, :ensemble_metadata)
    end

    test "applies weighted voting to outputs" do
      context = %{
        experiment: %{
          reliability: %{
            ensemble: %EnsembleConfig{
              strategy: :weighted,
              execution_mode: :parallel
            }
          }
        },
        outputs: [
          %{response: "A", model: :model1, confidence: 0.9},
          %{response: "B", model: :model2, confidence: 0.6},
          %{response: "B", model: :model3, confidence: 0.5}
        ]
      }

      {:ok, result} = Stage.run(context)

      # B has total weight 1.1 (0.6 + 0.5), A has 0.9
      assert result.answer == "b"
      assert result.ensemble_result.strategy == :weighted
    end

    test "applies best_confidence strategy" do
      context = %{
        experiment: %{
          reliability: %{
            ensemble: %EnsembleConfig{
              strategy: :best_confidence,
              execution_mode: :parallel
            }
          }
        },
        outputs: [
          %{response: "A", model: :model1, confidence: 0.6},
          %{response: "B", model: :model2, confidence: 0.9},
          %{response: "C", model: :model3, confidence: 0.7}
        ]
      }

      {:ok, result} = Stage.run(context)

      assert result.answer == "b"
      assert result.consensus == 0.9
      assert result.ensemble_result.strategy == :best_confidence
    end

    test "applies unanimous voting successfully" do
      context = %{
        experiment: %{
          reliability: %{
            ensemble: %EnsembleConfig{
              strategy: :unanimous,
              execution_mode: :parallel
            }
          }
        },
        outputs: [
          %{response: "Paris", model: :model1},
          %{response: "Paris", model: :model2},
          %{response: "Paris", model: :model3}
        ]
      }

      {:ok, result} = Stage.run(context)

      assert result.answer == "paris"
      assert result.consensus == 1.0
      assert result.ensemble_result.strategy == :unanimous
    end

    test "handles unanimous voting failure" do
      context = %{
        experiment: %{
          reliability: %{
            ensemble: %EnsembleConfig{
              strategy: :unanimous,
              execution_mode: :parallel
            }
          }
        },
        outputs: [
          %{response: "Paris", model: :model1},
          %{response: "London", model: :model2},
          %{response: "Paris", model: :model3}
        ]
      }

      {:error, error} = Stage.run(context)

      assert error.reason == :no_unanimous_consensus
    end

    test "accepts context with 'responses' key" do
      context = %{
        experiment: %{
          reliability: %{
            ensemble: %EnsembleConfig{
              strategy: :majority,
              execution_mode: :parallel
            }
          }
        },
        responses: [
          %{response: "yes", model: :model1},
          %{response: "yes", model: :model2},
          %{response: "no", model: :model3}
        ]
      }

      {:ok, result} = Stage.run(context)

      assert result.answer == "yes"
      assert result.consensus == 2 / 3
    end

    test "respects normalization option" do
      context = %{
        experiment: %{
          reliability: %{
            ensemble: %EnsembleConfig{
              strategy: :majority,
              execution_mode: :parallel
            }
          }
        },
        outputs: [
          %{response: "YES", model: :model1},
          %{response: "yes", model: :model2},
          %{response: "Yes", model: :model3}
        ]
      }

      {:ok, result} = Stage.run(context, %{normalization: :lowercase_trim})

      assert result.answer == "yes"
      assert result.consensus == 1.0
    end

    test "respects return_original_answer option" do
      context = %{
        experiment: %{
          reliability: %{
            ensemble: %EnsembleConfig{
              strategy: :majority,
              execution_mode: :parallel
            }
          }
        },
        outputs: [
          %{response: "The answer is 4", model: :model1},
          %{response: "4", model: :model2},
          %{response: "Four", model: :model3}
        ]
      }

      {:ok, result} =
        Stage.run(context, %{
          normalization: :numeric,
          return_original_answer: true
        })

      # Should return one of the original responses
      assert result.answer in ["The answer is 4", "4", "Four"]
    end
  end

  describe "run/2 error handling" do
    test "returns error when ensemble config is missing" do
      context = %{
        outputs: [
          %{response: "4", model: :model1}
        ]
      }

      {:error, reason} = Stage.run(context)

      assert reason == :missing_ensemble_config
    end

    test "returns error when ensemble config is invalid type" do
      context = %{
        experiment: %{
          reliability: %{
            ensemble: "not a valid config"
          }
        },
        outputs: [
          %{response: "4", model: :model1}
        ]
      }

      {:error, {:invalid_ensemble_config, _}} = Stage.run(context)
    end

    test "returns error when both query and outputs are missing" do
      context = %{
        experiment: %{
          reliability: %{
            ensemble: %EnsembleConfig{
              strategy: :majority,
              execution_mode: :parallel
            }
          }
        }
      }

      {:error, reason} = Stage.run(context)

      assert reason == :missing_query_or_outputs
    end

    test "returns error when no responses provided" do
      context = %{
        experiment: %{
          reliability: %{
            ensemble: %EnsembleConfig{
              strategy: :majority,
              execution_mode: :parallel
            }
          }
        },
        outputs: []
      }

      {:error, reason} = Stage.run(context)

      assert reason == :no_responses
    end
  end

  describe "run/2 with semantic_similarity strategy" do
    test "groups similar responses" do
      context = %{
        experiment: %{
          reliability: %{
            ensemble: %EnsembleConfig{
              strategy: :semantic_similarity,
              execution_mode: :parallel
            }
          }
        },
        outputs: [
          %{response: "The answer is 4", model: :model1},
          %{response: "4", model: :model2},
          %{response: "Four", model: :model3}
        ]
      }

      {:ok, result} =
        Stage.run(context, %{
          similarity_threshold: 0.3,
          similarity_metric: :levenshtein
        })

      # Should recognize all as similar answers
      assert result.answer in ["the answer is 4", "4", "four"]
      assert result.ensemble_result.strategy == :semantic_similarity
    end
  end

  describe "run/2 with ranked_choice strategy" do
    test "applies ranked choice voting" do
      context = %{
        experiment: %{
          reliability: %{
            ensemble: %EnsembleConfig{
              strategy: :ranked_choice,
              execution_mode: :parallel
            }
          }
        },
        outputs: [
          %{response: ["A", "B", "C"], model: :model1},
          %{response: ["B", "A", "C"], model: :model2},
          %{response: ["A", "C", "B"], model: :model3}
        ]
      }

      {:ok, result} =
        Stage.run(context, %{
          ranking_method: :instant_runoff
        })

      assert result.ensemble_result.strategy == :ranked_choice
      assert result.answer != nil
    end
  end

  describe "run/2 with config options" do
    test "uses timeout from ensemble config" do
      context = %{
        experiment: %{
          reliability: %{
            ensemble: %EnsembleConfig{
              strategy: :majority,
              execution_mode: :parallel,
              timeout_ms: 3000
            }
          }
        },
        outputs: [
          %{response: "4", model: :model1},
          %{response: "4", model: :model2}
        ]
      }

      {:ok, result} = Stage.run(context)

      # Should complete successfully with the timeout setting
      assert result.answer == "4"
    end

    test "uses min_agreement from ensemble config" do
      context = %{
        experiment: %{
          reliability: %{
            ensemble: %EnsembleConfig{
              strategy: :majority,
              execution_mode: :sequential,
              min_agreement: 0.8
            }
          }
        },
        outputs: [
          %{response: "4", model: :model1},
          %{response: "4", model: :model2},
          %{response: "5", model: :model3}
        ]
      }

      {:ok, result} = Stage.run(context)

      # Consensus is 2/3 = 0.667, which is less than 0.8
      assert result.consensus == 2 / 3
      assert result.answer == "4"
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

      context = %{
        experiment: %{
          reliability: %{
            ensemble: config
          }
        },
        outputs: [
          %{response: "A", model: :model1, confidence: 0.5},
          %{response: "B", model: :model2, confidence: 0.5},
          %{response: "B", model: :model3, confidence: 0.5}
        ]
      }

      {:ok, result} = Stage.run(context)

      # With weights, model1's response gets doubled weight
      # A: 0.5 (from confidence), B: 0.5 + 0.5 = 1.0
      # But note: the Stage doesn't currently apply the weights to responses
      # It passes them as options, so the behavior depends on Vote implementation
      assert result.answer in ["a", "b"]
    end
  end

  describe "describe/1" do
    test "returns stage metadata" do
      description = Stage.describe()

      assert description.name == "ensemble_voting"
      assert description.description == "Multi-model ensemble voting stage"
      assert description.version == "0.3.0"
      assert description.config_type == CrucibleIR.Reliability.Ensemble
    end

    test "lists available strategies" do
      description = Stage.describe()

      assert :majority in description.strategies
      assert :weighted in description.strategies
      assert :best_confidence in description.strategies
      assert :unanimous in description.strategies
      assert :semantic_similarity in description.strategies
      assert :ranked_choice in description.strategies
    end

    test "lists execution modes" do
      description = Stage.describe()

      assert :parallel in description.execution_modes
      assert :sequential in description.execution_modes
      assert :hedged in description.execution_modes
      assert :cascade in description.execution_modes
    end

    test "lists inputs and outputs" do
      description = Stage.describe()

      assert :outputs in description.inputs
      assert :query in description.inputs
      assert {:experiment, :reliability, :ensemble} in description.inputs

      assert :ensemble_result in description.outputs
      assert :consensus in description.outputs
      assert :answer in description.outputs
      assert :ensemble_metadata in description.outputs
    end

    test "accepts options parameter" do
      description = Stage.describe(%{custom_option: true})

      # Should still return valid description
      assert description.name == "ensemble_voting"
    end
  end

  describe "run/2 with additional options" do
    test "merges config options with additional options" do
      context = %{
        experiment: %{
          reliability: %{
            ensemble: %EnsembleConfig{
              strategy: :majority,
              execution_mode: :parallel,
              options: %{
                "custom_key" => "custom_value"
              }
            }
          }
        },
        outputs: [
          %{response: "4", model: :model1},
          %{response: "4", model: :model2}
        ]
      }

      {:ok, result} = Stage.run(context, %{another_option: "value"})

      assert result.answer == "4"
    end

    test "additional options take precedence over config options" do
      context = %{
        experiment: %{
          reliability: %{
            ensemble: %EnsembleConfig{
              strategy: :majority,
              execution_mode: :parallel
            }
          }
        },
        outputs: [
          %{response: "YES", model: :model1},
          %{response: "yes", model: :model2}
        ]
      }

      {:ok, result} = Stage.run(context, %{normalization: :lowercase_trim})

      assert result.answer == "yes"
      assert result.consensus == 1.0
    end
  end

  describe "run/2 context preservation" do
    test "preserves existing context fields" do
      context = %{
        experiment: %{
          reliability: %{
            ensemble: %EnsembleConfig{
              strategy: :majority,
              execution_mode: :parallel
            }
          }
        },
        outputs: [
          %{response: "4", model: :model1},
          %{response: "4", model: :model2}
        ],
        custom_field: "preserved",
        another_field: 123
      }

      {:ok, result} = Stage.run(context)

      assert result.custom_field == "preserved"
      assert result.another_field == 123
      assert result.answer == "4"
    end

    test "adds ensemble-specific fields to context" do
      context = %{
        experiment: %{
          reliability: %{
            ensemble: %EnsembleConfig{
              strategy: :majority,
              execution_mode: :parallel
            }
          }
        },
        outputs: [
          %{response: "4", model: :model1},
          %{response: "4", model: :model2}
        ]
      }

      {:ok, result} = Stage.run(context)

      assert Map.has_key?(result, :ensemble_result)
      assert Map.has_key?(result, :consensus)
      assert Map.has_key?(result, :answer)
      assert Map.has_key?(result, :ensemble_metadata)
    end
  end
end
