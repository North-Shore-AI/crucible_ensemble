defmodule CrucibleEnsemble.Strategy do
  @moduledoc """
  Execution strategies for coordinating multiple model calls.

  Provides different approaches to orchestrating ensemble predictions:
  - **Parallel**: Execute all models simultaneously (maximize quality)
  - **Sequential**: Execute one at a time until consensus (minimize cost)
  - **Hedged**: Primary model with backup hedges (optimize P99 latency)
  - **Cascade**: Execute in priority order with early stopping (adaptive)
  """

  alias CrucibleEnsemble.Executor
  alias CrucibleEnsemble.Vote

  @type strategy :: :parallel | :sequential | :hedged | :cascade
  @type query :: String.t()
  @type model :: atom()
  @type result :: {:ok, map()} | {:error, term()}

  @doc """
  Execute all models in parallel and wait for all to complete.

  **Pros**: Maximum consensus quality, fastest time to completion
  **Cons**: Highest cost (all models always called)

  ## Options

    * `:timeout` - Timeout per model in milliseconds (default: 5000)
    * `:voting_strategy` - How to aggregate results (default: :majority)

  ## Examples

      iex> CrucibleEnsemble.Strategy.parallel(
      ...>   "What is 2+2?",
      ...>   [:gemini_flash, :openai_gpt4o_mini, :anthropic_haiku]
      ...> )
      {:ok, %{answer: "4", consensus: 1.0, results: [...]}}

  """
  @spec parallel(query(), [model()], keyword()) :: result()
  def parallel(query, models, opts \\ []) do
    # Execute all models concurrently
    results = Executor.execute_parallel(query, models, opts)

    # Aggregate using voting strategy
    aggregate_results(results, opts)
  end

  @doc """
  Execute models sequentially until consensus is reached.

  Calls models one at a time and stops when sufficient consensus is achieved.
  More cost-efficient for queries where models are likely to agree.

  **Pros**: Lower cost, adaptive to task difficulty
  **Cons**: Higher latency, may not achieve full consensus

  ## Options

    * `:min_consensus` - Minimum consensus to stop early (default: 0.7)
    * `:max_models` - Maximum number of models to call (default: all)
    * `:voting_strategy` - How to aggregate results (default: :majority)

  ## Examples

      iex> CrucibleEnsemble.Strategy.sequential(
      ...>   "What is the capital of France?",
      ...>   [:gemini_flash, :openai_gpt4o_mini, :anthropic_haiku],
      ...>   min_consensus: 0.8
      ...> )
      {:ok, %{answer: "Paris", consensus: 1.0, models_called: 2}}

  """
  @spec sequential(query(), [model()], keyword()) :: result()
  def sequential(query, models, opts \\ []) do
    # Execute models sequentially with early stopping
    results = Executor.execute_sequential(query, models, opts)

    # Aggregate results
    aggregate_results(results, opts)
  end

  @doc """
  Execute primary model with hedged backup calls.

  Starts with a primary model, then launches backup requests after a delay
  to protect against tail latency. This is a classic "request hedging" pattern.

  **Pros**: Optimizes P99 latency, controlled cost overhead
  **Cons**: More complex, requires tuning hedge delay

  ## Options

    * `:hedge_delay_ms` - Delay before starting hedge requests (default: 1000)
    * `:max_hedges` - Maximum number of hedge requests (default: 2)
    * `:timeout` - Total timeout for all requests (default: 5000)

  ## Examples

      iex> CrucibleEnsemble.Strategy.hedged(
      ...>   "Translate 'hello' to French",
      ...>   :openai_gpt4o_mini,
      ...>   [:gemini_flash, :anthropic_haiku],
      ...>   hedge_delay_ms: 500
      ...> )
      {:ok, %{answer: "bonjour", primary_succeeded: true, hedges_used: 0}}

  """
  @spec hedged(query(), model(), [model()], keyword()) :: result()
  def hedged(query, primary, backups, opts \\ []) do
    hedge_delay = Keyword.get(opts, :hedge_delay_ms, 1_000)
    timeout = Keyword.get(opts, :timeout, 5_000)

    # Start primary request
    primary_task =
      Task.async(fn ->
        Executor.call_model(primary, query, opts)
      end)

    # Wait for hedge delay
    case Task.yield(primary_task, hedge_delay) do
      {:ok, result} ->
        # Primary completed within hedge delay
        aggregate_results([result], opts)

      nil ->
        # Primary still running, start hedges
        backup_tasks =
          Enum.map(backups, fn model ->
            Task.async(fn ->
              Executor.call_model(model, query, opts)
            end)
          end)

        all_tasks = [primary_task | backup_tasks]
        remaining_timeout = timeout - hedge_delay

        # Wait for first successful result
        case wait_for_first_success(all_tasks, remaining_timeout) do
          {:ok, result} ->
            # Cancel remaining tasks
            Enum.each(all_tasks, &Task.shutdown(&1, :brutal_kill))
            aggregate_results([result], opts)

          {:error, _} = error ->
            error
        end
    end
  end

  @doc """
  Execute models in priority order with early stopping on high confidence.

  Calls models from highest to lowest priority, stopping when a model
  returns a high-confidence result. Good for heterogeneous ensembles
  where some models are clearly better for certain tasks.

  **Pros**: Fast and cheap, leverages model strengths
  **Cons**: May miss consensus, requires model ranking

  ## Options

    * `:confidence_threshold` - Confidence to stop early (default: 0.9)
    * `:min_models` - Minimum models to call (default: 1)
    * `:voting_strategy` - How to aggregate if multiple called (default: :weighted)

  ## Examples

      iex> CrucibleEnsemble.Strategy.cascade(
      ...>   "Write a complex algorithm",
      ...>   [:openai_gpt4, :anthropic_sonnet, :gemini_flash],
      ...>   confidence_threshold: 0.85
      ...> )
      {:ok, %{answer: "...", confidence: 0.92, model_used: :openai_gpt4}}

  """
  @spec cascade(query(), [model()], keyword()) :: result()
  def cascade(query, models, opts \\ []) do
    confidence_threshold = Keyword.get(opts, :confidence_threshold, 0.9)
    min_models = Keyword.get(opts, :min_models, 1)

    cascade_helper(query, models, [], min_models, confidence_threshold, opts)
  end

  # Private helper functions

  defp cascade_helper(_query, [], results, _min_models, _threshold, opts) do
    # No more models, aggregate what we have
    aggregate_results(results, opts)
  end

  defp cascade_helper(query, [model | rest], results, min_models, threshold, opts) do
    # Call next model
    result = Executor.call_model(model, query, opts)
    new_results = [result | results]

    # Check if we should stop
    should_continue =
      length(new_results) < min_models ||
        not high_confidence_result?(result, threshold)

    if should_continue do
      cascade_helper(query, rest, new_results, min_models, threshold, opts)
    else
      aggregate_results(new_results, opts)
    end
  end

  defp high_confidence_result?({:ok, result}, threshold) do
    confidence = Map.get(result, :confidence, 0.0)
    confidence >= threshold
  end

  defp high_confidence_result?({:error, _}, _threshold), do: false

  defp wait_for_first_success(tasks, timeout) do
    start_time = System.monotonic_time(:millisecond)

    case wait_for_success_helper(tasks, timeout, start_time) do
      {:ok, _result} = success ->
        success

      :timeout ->
        {:error, :all_requests_timed_out}
    end
  end

  defp wait_for_success_helper([], _timeout, _start_time), do: :timeout

  defp wait_for_success_helper(tasks, timeout, start_time) do
    elapsed = System.monotonic_time(:millisecond) - start_time
    remaining = max(0, timeout - elapsed)

    if remaining == 0 do
      :timeout
    else
      # Check all tasks
      case Task.yield_many(tasks, remaining) do
        [] ->
          :timeout

        results ->
          # Look for first success
          success =
            Enum.find_value(results, fn {_task, result} ->
              case result do
                {:ok, {:ok, _} = success_result} -> success_result
                _ -> nil
              end
            end)

          case success do
            {:ok, _} = result ->
              result

            nil ->
              # No success yet, filter out completed tasks and continue
              still_running =
                results
                |> Enum.filter(fn {_task, result} -> result == nil end)
                |> Enum.map(fn {task, _} -> task end)

              wait_for_success_helper(still_running, timeout, start_time)
          end
      end
    end
  end

  defp aggregate_results(results, opts) do
    # Separate successes and failures
    {successes, failures} = partition_results(results)

    if Enum.empty?(successes) do
      {:error,
       %{
         reason: :all_models_failed,
         failures: failures
       }}
    else
      # Extract successful responses
      responses =
        Enum.map(successes, fn {:ok, result} -> result end)

      # Apply voting strategy
      voting_strategy = Keyword.get(opts, :voting_strategy, :majority)
      normalization = Keyword.get(opts, :normalization, :lowercase_trim)

      voting_opts = [
        normalization: normalization
      ]

      case Vote.apply_strategy(responses, voting_strategy, voting_opts) do
        {:ok, vote_result} ->
          # Add metadata about execution
          enhanced_result =
            Map.merge(vote_result, %{
              total_models: length(results),
              successful_models: length(successes),
              failed_models: length(failures),
              all_responses: responses
            })

          {:ok, enhanced_result}

        {:error, _} = error ->
          error
      end
    end
  end

  defp partition_results(results) do
    Enum.split_with(results, fn
      {:ok, _} -> true
      {:error, _} -> false
    end)
  end
end
