defmodule CrucibleEnsemble.Executor do
  @moduledoc """
  Concurrent execution of model predictions.

  Handles parallel model invocation using BEAM processes, with proper
  timeout handling, error isolation, and telemetry integration.
  """

  require Logger
  alias CrucibleEnsemble.Pricing

  @type model :: atom()
  @type query :: String.t()
  @type model_result :: %{
          model: model(),
          response: String.t(),
          latency_us: non_neg_integer(),
          cost: float(),
          metadata: map()
        }
  @type task_result :: {:ok, model_result()} | {:error, map()}

  @doc """
  Execute query on multiple models in parallel.

  Spawns concurrent tasks for each model and collects results within
  the specified timeout period.

  ## Options

    * `:timeout` - Timeout per model in milliseconds (default: 5000)
    * `:api_keys` - Map of model => API key (default: from config)
    * `:temperature` - Sampling temperature (default: 0.7)
    * `:max_tokens` - Maximum tokens in response (default: 1000)
    * `:telemetry_metadata` - Additional metadata for telemetry events

  ## Examples

      iex> CrucibleEnsemble.Executor.execute_parallel(
      ...>   "What is 2+2?",
      ...>   [:gemini_flash, :openai_gpt4o_mini],
      ...>   timeout: 5000
      ...> )
      [
        {:ok, %{model: :gemini_flash, response: "4", latency_us: 234000, cost: 0.00001}},
        {:ok, %{model: :openai_gpt4o_mini, response: "4", latency_us: 456000, cost: 0.00002}}
      ]

  """
  @spec execute_parallel(query(), [model()], keyword()) :: [task_result()]
  def execute_parallel(query, models, opts \\ []) do
    timeout = Keyword.get(opts, :timeout, 5_000)
    telemetry_meta = Keyword.get(opts, :telemetry_metadata, %{})

    # Emit start event
    :telemetry.execute(
      [:crucible_ensemble, :executor, :start],
      %{system_time: System.system_time()},
      Map.merge(telemetry_meta, %{models: models, query_length: String.length(query)})
    )

    start_time = System.monotonic_time(:microsecond)

    # Execute all models concurrently
    results =
      models
      |> Task.async_stream(
        fn model ->
          call_model(model, query, opts)
        end,
        timeout: timeout,
        max_concurrency: length(models),
        on_timeout: :kill_task
      )
      |> Enum.map(fn
        {:ok, result} -> result
        {:exit, reason} -> {:error, %{error: :task_exit, reason: reason}}
      end)

    end_time = System.monotonic_time(:microsecond)
    total_latency = end_time - start_time

    # Emit stop event
    :telemetry.execute(
      [:crucible_ensemble, :executor, :stop],
      %{duration: total_latency},
      Map.merge(telemetry_meta, %{
        models: models,
        results_count: length(results),
        successes: count_successes(results),
        failures: count_failures(results)
      })
    )

    results
  end

  @doc """
  Call a single model with the given query.

  This is the core function that interfaces with req_llm.
  Includes telemetry, error handling, and cost calculation.

  ## Options

  Same as `execute_parallel/3`, plus model-specific API key handling.

  """
  @spec call_model(model(), query(), keyword()) :: task_result()
  def call_model(model, query, opts \\ []) do
    start_time = System.monotonic_time(:microsecond)
    telemetry_meta = Keyword.get(opts, :telemetry_metadata, %{})

    # Emit model start event
    :telemetry.execute(
      [:crucible_ensemble, :model, :start],
      %{system_time: System.system_time()},
      Map.merge(telemetry_meta, %{model: model, query_length: String.length(query)})
    )

    try do
      # Get API key for model
      api_key = get_api_key(model, opts)

      # Prepare request options
      request_opts = [
        model: map_model_name(model),
        api_key: api_key,
        temperature: Keyword.get(opts, :temperature, 0.7),
        max_tokens: Keyword.get(opts, :max_tokens, 1000)
      ]

      # Make the LLM call
      # Note: This is a simplified version. In production, you'd use the actual req_llm client
      response = make_llm_request(query, request_opts)

      end_time = System.monotonic_time(:microsecond)
      latency_us = end_time - start_time

      # Calculate cost
      cost = Pricing.calculate_cost(model, response)

      result = %{
        model: model,
        response: extract_text(response),
        latency_us: latency_us,
        cost: cost,
        metadata: %{
          usage: extract_usage(response),
          finish_reason: extract_finish_reason(response)
        }
      }

      # Emit success event
      :telemetry.execute(
        [:crucible_ensemble, :model, :stop],
        %{duration: latency_us},
        Map.merge(telemetry_meta, %{
          model: model,
          cost: cost,
          success: true
        })
      )

      {:ok, result}
    rescue
      e ->
        end_time = System.monotonic_time(:microsecond)
        latency_us = end_time - start_time

        error_result = %{
          model: model,
          error: Exception.message(e),
          error_type: e.__struct__,
          latency_us: latency_us
        }

        # Emit exception event
        :telemetry.execute(
          [:crucible_ensemble, :model, :exception],
          %{duration: latency_us},
          Map.merge(telemetry_meta, %{
            model: model,
            error: Exception.message(e),
            error_type: e.__struct__
          })
        )

        Logger.warning("Model #{model} failed: #{Exception.message(e)}")

        {:error, error_result}
    end
  end

  @doc """
  Execute models sequentially until consensus is reached.

  More cost-efficient than parallel execution when early consensus is likely.

  ## Options

    * `:min_consensus` - Minimum consensus threshold to stop (default: 0.7)
    * `:max_models` - Maximum number of models to call (default: length(models))

  """
  @spec execute_sequential(query(), [model()], keyword()) :: [task_result()]
  def execute_sequential(query, models, opts \\ []) do
    min_consensus = Keyword.get(opts, :min_consensus, 0.7)
    max_models = Keyword.get(opts, :max_models, length(models))

    execute_sequential_helper(query, models, [], 0, max_models, min_consensus, opts)
  end

  defp execute_sequential_helper(_query, [], results, _count, _max, _threshold, _opts) do
    Enum.reverse(results)
  end

  defp execute_sequential_helper(_query, _models, results, count, max, _threshold, _opts)
       when count >= max do
    Enum.reverse(results)
  end

  defp execute_sequential_helper(query, [model | rest], results, count, max, threshold, opts) do
    # Call next model
    result = call_model(model, query, opts)
    new_results = [result | results]

    # Check if we have consensus
    if should_stop?(new_results, threshold) do
      Enum.reverse(new_results)
    else
      execute_sequential_helper(query, rest, new_results, count + 1, max, threshold, opts)
    end
  end

  defp should_stop?(results, threshold) do
    # Simple heuristic: check if we have strong agreement
    # In practice, you'd use the voting module here
    successes =
      Enum.filter(results, fn
        {:ok, _} -> true
        _ -> false
      end)

    if length(successes) < 2 do
      false
    else
      # Extract responses and check for agreement
      responses =
        successes
        |> Enum.map(fn {:ok, result} -> result.response end)
        |> Enum.frequencies()

      if map_size(responses) > 0 do
        max_count = responses |> Map.values() |> Enum.max()
        total = length(successes)
        consensus = max_count / total
        consensus >= threshold
      else
        false
      end
    end
  end

  # Private helper functions

  defp get_api_key(model, opts) do
    # Try to get API key from options first
    api_keys = Keyword.get(opts, :api_keys, %{})

    case Map.get(api_keys, model) do
      nil ->
        # Fall back to environment variables
        env_var = model_to_env_var(model)
        System.get_env(env_var) || raise "No API key found for model #{model}"

      key ->
        key
    end
  end

  defp model_to_env_var(model) do
    case model do
      :gemini_flash -> "GEMINI_API_KEY"
      :gemini_pro -> "GEMINI_API_KEY"
      :openai_gpt4o_mini -> "OPENAI_API_KEY"
      :openai_gpt4o -> "OPENAI_API_KEY"
      :openai_gpt4 -> "OPENAI_API_KEY"
      :anthropic_haiku -> "ANTHROPIC_API_KEY"
      :anthropic_sonnet -> "ANTHROPIC_API_KEY"
      :anthropic_opus -> "ANTHROPIC_API_KEY"
      _ -> "#{String.upcase(to_string(model))}_API_KEY"
    end
  end

  defp map_model_name(model) do
    # Map internal model names to API model names
    case model do
      :gemini_flash -> "gemini-2.0-flash-exp"
      :gemini_pro -> "gemini-pro"
      :openai_gpt4o_mini -> "gpt-4o-mini"
      :openai_gpt4o -> "gpt-4o"
      :openai_gpt4 -> "gpt-4"
      :anthropic_haiku -> "claude-3-haiku-20240307"
      :anthropic_sonnet -> "claude-3-5-sonnet-20241022"
      :anthropic_opus -> "claude-3-opus-20240229"
      _ -> to_string(model)
    end
  end

  defp make_llm_request(query, _opts) do
    # This is a placeholder for the actual req_llm integration
    # In production, this would use ReqLLM.chat/3 or similar
    #
    # For now, we'll create a mock response structure
    # Real implementation would be:
    # client = ReqLLM.new(opts[:model], api_key: opts[:api_key])
    # ReqLLM.chat(client, query, Keyword.take(opts, [:temperature, :max_tokens]))

    # Mock response for compilation
    %{
      text: "Mock response: #{query}",
      usage: %{
        input_tokens: String.length(query),
        output_tokens: 50
      },
      finish_reason: "stop"
    }
  end

  defp extract_text(response) do
    cond do
      Map.has_key?(response, :text) -> response.text
      Map.has_key?(response, :response) -> response.response
      Map.has_key?(response, :content) -> response.content
      Map.has_key?(response, "text") -> response["text"]
      Map.has_key?(response, "response") -> response["response"]
      Map.has_key?(response, "content") -> response["content"]
      true -> ""
    end
  end

  defp extract_usage(response) do
    cond do
      Map.has_key?(response, :usage) -> response.usage
      Map.has_key?(response, "usage") -> response["usage"]
      true -> %{input_tokens: 0, output_tokens: 0}
    end
  end

  defp extract_finish_reason(response) do
    cond do
      Map.has_key?(response, :finish_reason) -> response.finish_reason
      Map.has_key?(response, "finish_reason") -> response["finish_reason"]
      true -> "unknown"
    end
  end

  defp count_successes(results) do
    Enum.count(results, fn
      {:ok, _} -> true
      _ -> false
    end)
  end

  defp count_failures(results) do
    Enum.count(results, fn
      {:error, _} -> true
      _ -> false
    end)
  end
end
