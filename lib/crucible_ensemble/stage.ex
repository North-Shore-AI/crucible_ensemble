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
  - `metrics[:ensemble_strategy]` - Strategy used for voting
  - `assigns[:answer]` - The final answer selected by voting
  - `assigns[:ensemble_metadata]` - Additional metadata from voting

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
      {:error, _reason} = error ->
        Logger.warning("Ensemble voting failed: #{inspect(error)}")
        error
    end
  end

  @impl true
  @spec describe(map()) :: map()
  def describe(_opts \\ %{}) do
    %{
      __schema_version__: "1.0.0",
      name: :ensemble_voting,
      description: "Multi-model ensemble voting stage using CrucibleEnsemble",
      required: [],
      optional: [:normalization, :timeout_ms, :min_responses],
      types: %{
        normalization: {:enum, [:none, :lowercase, :trim, :lowercase_trim]},
        timeout_ms: :integer,
        min_responses: :integer
      },
      defaults: %{
        normalization: :none,
        timeout_ms: 30_000,
        min_responses: 1
      },
      version: "0.5.0",
      __extensions__: %{
        ensemble: %{
          behaviour: Crucible.Stage,
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
          ],
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
          ]
        }
      }
    }
  end

  # ============================================================================
  # Private Helpers
  # ============================================================================

  defp extract_ensemble_config(%{reliability: %{ensemble: %EnsembleConfig{} = config}}) do
    {:ok, config}
  end

  defp extract_ensemble_config(%{reliability: %{ensemble: config}}) when is_map(config) do
    # Try to convert map to struct
    {:ok, struct(EnsembleConfig, config)}
  rescue
    _ -> {:error, {:invalid_ensemble_config, config}}
  end

  defp extract_ensemble_config(%{reliability: %{ensemble: config}}) do
    # Non-map config (e.g., string or other invalid type)
    {:error, {:invalid_ensemble_config, config}}
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
    {:error, :no_responses}
  end

  defp apply_voting(responses, %EnsembleConfig{} = config, opts) do
    # Build voting options from map opts
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
