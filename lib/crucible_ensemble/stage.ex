defmodule CrucibleEnsemble.Stage do
  @moduledoc """
  Pipeline stage for ensemble voting.

  Implements stage behaviour for use in crucible_framework pipelines.
  Uses CrucibleIR.Reliability.Ensemble configuration.

  ## Context Requirements

  The stage expects the following in the context:

  - `experiment.reliability.ensemble` - CrucibleIR.Reliability.Ensemble configuration
  - `outputs` - Model responses to vote on (list of maps with `:response` key)

  Or alternatively:

  - `query` - Query string to execute on models
  - `experiment.reliability.ensemble` - Configuration including model list

  ## Returns

  Updates context with:

  - `ensemble_result` - The voting result
  - `consensus` - Consensus score (0.0 to 1.0)
  - `answer` - The final answer selected by voting

  ## Examples

      # With existing model outputs
      context = %{
        experiment: %{
          reliability: %{
            ensemble: %CrucibleIR.Reliability.Ensemble{
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

      {:ok, updated_context} = CrucibleEnsemble.Stage.run(context)
      updated_context.answer
      # => "4"

      # With query execution
      context = %{
        query: "What is 2+2?",
        experiment: %{
          reliability: %{
            ensemble: %CrucibleIR.Reliability.Ensemble{
              strategy: :majority,
              execution_mode: :parallel,
              models: [:gemini_flash, :openai_gpt4o_mini]
            }
          }
        }
      }

      {:ok, updated_context} = CrucibleEnsemble.Stage.run(context)

  """

  alias CrucibleIR.Reliability.Ensemble, as: EnsembleConfig

  @doc """
  Runs ensemble voting on model responses.

  Expects context to have:
  - `experiment.reliability.ensemble` - Ensemble configuration
  - Either `outputs` (model responses) or `query` (to execute models)

  Returns updated context with ensemble results.

  ## Options

  - `:normalization` - Response normalization strategy (default: `:lowercase_trim`)
  - `:return_original_answer` - Return original text instead of normalized (default: `false`)
  - `:api_keys` - Map of model => API key for query execution
  - `:telemetry_metadata` - Additional telemetry metadata

  """
  @spec run(map(), map()) :: {:ok, map()} | {:error, term()}
  def run(context, opts \\ %{}) when is_map(context) and is_map(opts) do
    # Convert opts to keyword list if it's a map
    opts_keyword = if is_map(opts), do: Map.to_list(opts), else: opts

    with {:ok, ensemble_config} <- extract_ensemble_config(context),
         {:ok, responses} <- get_or_execute_responses(context, ensemble_config, opts_keyword),
         {:ok, vote_result} <- apply_voting(responses, ensemble_config, opts_keyword) do
      # Update context with results
      updated_context =
        context
        |> Map.put(:ensemble_result, vote_result)
        |> Map.put(:consensus, vote_result.consensus)
        |> Map.put(:answer, vote_result.answer)
        |> Map.put(:ensemble_metadata, Map.get(vote_result, :metadata, %{}))

      {:ok, updated_context}
    else
      {:error, _reason} = error -> error
    end
  end

  @doc """
  Describes this stage for introspection.

  Returns metadata about the stage including name, purpose, and configuration options.

  ## Options

  Currently unused but reserved for future extensions.

  ## Examples

      iex> CrucibleEnsemble.Stage.describe()
      %{
        name: "ensemble_voting",
        description: "Multi-model ensemble voting stage",
        inputs: [:outputs, :query],
        outputs: [:ensemble_result, :consensus, :answer],
        config_type: CrucibleIR.Reliability.Ensemble
      }

  """
  @spec describe(map()) :: map()
  def describe(_opts \\ %{}) do
    %{
      name: "ensemble_voting",
      description: "Multi-model ensemble voting stage",
      version: "0.3.0",
      inputs: [
        :outputs,
        :query,
        {:experiment, :reliability, :ensemble}
      ],
      outputs: [
        :ensemble_result,
        :consensus,
        :answer,
        :ensemble_metadata
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

  defp extract_ensemble_config(context) do
    case get_in(context, [:experiment, :reliability, :ensemble]) do
      %EnsembleConfig{} = config ->
        {:ok, config}

      nil ->
        {:error, :missing_ensemble_config}

      other ->
        {:error, {:invalid_ensemble_config, other}}
    end
  end

  defp get_or_execute_responses(context, ensemble_config, opts) do
    cond do
      # Case 1: Already have outputs/responses
      Map.has_key?(context, :outputs) ->
        {:ok, context.outputs}

      Map.has_key?(context, :responses) ->
        {:ok, context.responses}

      # Case 2: Have query, need to execute
      Map.has_key?(context, :query) ->
        execute_ensemble(context.query, ensemble_config, opts)

      true ->
        {:error, :missing_query_or_outputs}
    end
  end

  defp execute_ensemble(query, %EnsembleConfig{} = config, opts) do
    # Build options from ensemble config
    ensemble_opts =
      opts
      |> Keyword.put(:models, config.models || [])
      |> Keyword.put(:strategy, config.strategy)
      |> Keyword.put(:execution, config.execution_mode)
      |> maybe_put(:timeout, config.timeout_ms)
      |> maybe_put(:min_consensus, config.min_agreement)
      |> merge_config_options(config.options)

    # Execute ensemble prediction
    case CrucibleEnsemble.predict(query, ensemble_opts) do
      {:ok, result} ->
        # Extract responses from result if available
        # Otherwise create mock responses from the result
        responses = [
          %{
            response: result.answer,
            model: :ensemble,
            metadata: result.metadata
          }
        ]

        {:ok, responses}

      {:error, _reason} = error ->
        error
    end
  end

  defp apply_voting(responses, %EnsembleConfig{} = config, opts) do
    # Build voting options
    voting_opts =
      opts
      |> Keyword.put_new(:normalization, :lowercase_trim)
      |> Keyword.put_new(:return_original_answer, false)

    # Add weights if using weighted strategy
    voting_opts =
      if config.strategy == :weighted and config.weights do
        Keyword.put(voting_opts, :weights, config.weights)
      else
        voting_opts
      end

    # Apply the voting strategy
    case CrucibleEnsemble.Vote.apply_strategy(responses, config.strategy, voting_opts) do
      {:ok, vote_result} ->
        {:ok, vote_result}

      {:error, _reason} = error ->
        error
    end
  end

  defp maybe_put(keyword, _key, nil), do: keyword

  defp maybe_put(keyword, key, value) do
    Keyword.put(keyword, key, value)
  end

  defp merge_config_options(keyword, nil), do: keyword

  defp merge_config_options(keyword, options) when is_map(options) do
    Enum.reduce(options, keyword, fn {key, value}, acc ->
      atom_key = if is_binary(key), do: String.to_atom(key), else: key
      Keyword.put_new(acc, atom_key, value)
    end)
  end

  defp merge_config_options(keyword, _options), do: keyword
end
