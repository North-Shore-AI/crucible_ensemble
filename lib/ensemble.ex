defmodule CrucibleEnsemble do
  @moduledoc """
  Multi-model ensemble prediction with configurable voting strategies.

  Research infrastructure for AI reliability experiments using BEAM concurrency.

  ## Overview

  Ensemble enables reliable AI predictions by querying multiple language models
  concurrently and aggregating their responses using sophisticated voting strategies.
  This approach dramatically improves reliability compared to single-model systems.

  ## Quick Start

      # Basic usage with default settings (majority voting)
      {:ok, result} = CrucibleEnsemble.predict("What is 2+2?")

      # Use specific models
      {:ok, result} = CrucibleEnsemble.predict(
        "What is 2+2?",
        models: [:gemini_flash, :openai_gpt4o_mini, :anthropic_haiku]
      )

      # Use weighted voting strategy
      {:ok, result} = CrucibleEnsemble.predict(
        "Explain quantum computing",
        strategy: :weighted,
        models: [:openai_gpt4o, :anthropic_sonnet]
      )

  ## Voting Strategies

  - `:majority` - Most common response wins (default)
  - `:weighted` - Responses weighted by confidence scores
  - `:best_confidence` - Highest confidence response
  - `:unanimous` - All models must agree

  ## Execution Strategies

  - `:parallel` - All models execute simultaneously (default)
  - `:sequential` - One at a time until consensus
  - `:hedged` - Primary with backup hedges
  - `:cascade` - Priority order with early stopping

  ## Features

  - **High Reliability**: Ensemble voting reduces error rates exponentially
  - **Cost Tracking**: Automatic cost calculation per model and ensemble
  - **Telemetry**: Comprehensive instrumentation for research analysis
  - **Fault Tolerance**: Graceful degradation when models fail
  - **BEAM Concurrency**: Leverages Elixir's lightweight processes
  """

  alias CrucibleEnsemble.{Strategy, Pricing}

  @type query :: String.t()
  @type model :: atom()
  @type voting_strategy ::
          :majority | :weighted | :best_confidence | :unanimous | {module(), keyword()}
  @type execution_strategy :: :parallel | :sequential | :hedged | :cascade

  @type options :: [
          models: [model()],
          strategy: voting_strategy(),
          execution: execution_strategy(),
          timeout: pos_integer(),
          min_responses: pos_integer(),
          normalization: atom(),
          api_keys: map(),
          telemetry_metadata: map()
        ]

  @type result :: %{
          answer: String.t(),
          metadata: %{
            consensus: float(),
            votes: map(),
            latency_ms: pos_integer(),
            cost_usd: float(),
            models_used: [model()],
            successes: pos_integer(),
            failures: pos_integer()
          }
        }

  @default_models [:gemini_flash, :openai_gpt4o_mini, :anthropic_haiku]
  @default_voting_strategy :majority
  @default_execution_strategy :parallel
  @default_timeout 5_000

  @doc """
  Execute ensemble prediction with synchronous blocking.

  Queries multiple language models in parallel and aggregates their responses
  using the specified voting strategy.

  ## Options

    * `:models` - List of model identifiers (default: #{inspect(@default_models)})
    * `:strategy` - Voting strategy (default: #{@default_voting_strategy})
    * `:execution` - Execution strategy (default: #{@default_execution_strategy})
    * `:timeout` - Per-model timeout in ms (default: #{@default_timeout})
    * `:min_responses` - Minimum successful responses required (default: ceil(length(models) / 2))
    * `:normalization` - Response normalization strategy (default: :lowercase_trim)
    * `:api_keys` - Map of model => API key (default: from environment)
    * `:telemetry_metadata` - Additional metadata for telemetry events

  ## Examples

      # Simple question with default settings (requires API keys)
      # {:ok, result} = CrucibleEnsemble.predict("What is 2+2?")
      # result.answer
      # => "4"
      # result.metadata.consensus
      # => 1.0

      # Custom configuration (example only, would require API keys to run)
      # CrucibleEnsemble.predict(
      #   "Is Elixir a functional language?",
      #   models: [:gemini_flash, :openai_gpt4o],
      #   strategy: :weighted,
      #   execution: :parallel,
      #   timeout: 3000
      # )
      # => {:ok, %{answer: "yes", metadata: %{consensus: 1.0, ...}}}

  ## Error Handling

  Returns `{:error, reason}` if:
  - All models fail
  - Insufficient responses for voting
  - Invalid configuration

  """
  @spec predict(query(), options()) :: {:ok, result()} | {:error, term()}
  def predict(query, opts \\ []) do
    start_time = System.monotonic_time(:microsecond)
    telemetry_meta = prepare_telemetry_metadata(query, opts)

    # Emit start event
    :telemetry.execute(
      [:crucible_ensemble, :predict, :start],
      %{system_time: System.system_time()},
      telemetry_meta
    )

    # Extract options
    models = Keyword.get(opts, :models, @default_models)
    voting_strategy = Keyword.get(opts, :strategy, @default_voting_strategy)
    execution_strategy = Keyword.get(opts, :execution, @default_execution_strategy)

    # Execute based on strategy
    execution_opts =
      opts
      |> Keyword.put(:voting_strategy, voting_strategy)
      |> Keyword.put_new(:timeout, @default_timeout)

    result =
      case execution_strategy do
        :parallel ->
          Strategy.parallel(query, models, execution_opts)

        :sequential ->
          Strategy.sequential(query, models, execution_opts)

        :hedged ->
          [primary | backups] = models
          Strategy.hedged(query, primary, backups, execution_opts)

        :cascade ->
          Strategy.cascade(query, models, execution_opts)

        _ ->
          {:error, {:invalid_execution_strategy, execution_strategy}}
      end

    end_time = System.monotonic_time(:microsecond)
    duration_us = end_time - start_time

    # Process result and emit telemetry
    case result do
      {:ok, vote_result} ->
        # Calculate total cost
        responses = Map.get(vote_result, :all_responses, [])
        cost_info = calculate_ensemble_cost(responses)

        # Build final result
        final_result = %{
          answer: vote_result.answer,
          metadata: %{
            consensus: vote_result.consensus,
            votes: Map.get(vote_result, :votes, %{}),
            latency_ms: div(duration_us, 1_000),
            cost_usd: cost_info.total_usd,
            models_used: models,
            successes: Map.get(vote_result, :successful_models, 0),
            failures: Map.get(vote_result, :failed_models, 0),
            strategy: voting_strategy,
            execution: execution_strategy
          }
        }

        # Emit success event
        :telemetry.execute(
          [:crucible_ensemble, :predict, :stop],
          %{duration: duration_us},
          Map.merge(telemetry_meta, %{
            consensus: vote_result.consensus,
            total_cost: cost_info.total_usd,
            successes: Map.get(vote_result, :successful_models, 0),
            failures: Map.get(vote_result, :failed_models, 0)
          })
        )

        {:ok, final_result}

      {:error, reason} ->
        # Emit exception event
        :telemetry.execute(
          [:crucible_ensemble, :predict, :exception],
          %{duration: duration_us},
          Map.merge(telemetry_meta, %{error: inspect(reason)})
        )

        {:error, reason}
    end
  end

  @doc """
  Execute ensemble prediction asynchronously.

  Returns a Task that can be awaited later. Useful for concurrent
  operations or when you need to do other work while waiting.

  ## Examples

      # Start prediction
      task = CrucibleEnsemble.predict_async("What is the capital of France?")

      # Do other work...
      other_work()

      # Get result
      {:ok, result} = Task.await(task, 10_000)

      # Multiple concurrent predictions
      tasks = Enum.map(questions, &CrucibleEnsemble.predict_async/1)
      results = Task.await_many(tasks, 10_000)

  """
  @spec predict_async(query(), options()) :: Task.t()
  def predict_async(query, opts \\ []) do
    Task.async(fn ->
      predict(query, opts)
    end)
  end

  @doc """
  Execute ensemble prediction with streaming results.

  Returns a Stream that emits results as models complete. Enables
  early stopping and progressive result processing.

  ## Options

  Same as `predict/2`, plus:
    * `:early_stop_threshold` - Consensus threshold to stop early (default: 1.0)

  ## Examples

      # Stream results as they arrive
      stream = CrucibleEnsemble.predict_stream("Complex question?", models: [:model1, :model2, :model3])

      Enum.each(stream, fn
        {:response, model, response} ->
          IO.puts "Got response from \#{model}"
        {:consensus, consensus} ->
          IO.puts "Current consensus: \#{consensus}"
        {:complete, final_result} ->
          IO.puts "Final answer: \#{final_result.answer}"
      end)

  """
  @spec predict_stream(query(), options()) :: Enumerable.t()
  def predict_stream(query, opts \\ []) do
    models = Keyword.get(opts, :models, @default_models)
    timeout = Keyword.get(opts, :timeout, @default_timeout)
    early_stop = Keyword.get(opts, :early_stop_threshold, 1.0)

    Stream.resource(
      # Start function: spawn all model tasks
      fn ->
        tasks =
          Enum.map(models, fn model ->
            task =
              Task.async(fn ->
                CrucibleEnsemble.Executor.call_model(model, query, opts)
              end)

            {task, model}
          end)

        {tasks, [], timeout}
      end,
      # Next function: yield results as they complete
      fn
        {[], results, _timeout} ->
          # All tasks complete, emit final result
          voting_strategy = Keyword.get(opts, :strategy, @default_voting_strategy)
          normalization = Keyword.get(opts, :normalization, :lowercase_trim)

          case aggregate_stream_results(results, voting_strategy, normalization) do
            {:ok, final_result} ->
              {[{:complete, final_result}], :halt}

            {:error, _} = error ->
              {[error], :halt}
          end

        {tasks, results, timeout} ->
          # Wait for next result
          case Task.yield_many(tasks, timeout) do
            [] ->
              # Timeout - complete with what we have
              {:halt, {[], results, 0}}

            completions ->
              # Process completions
              {new_results, still_running} = process_completions(completions, results)

              # Check for early stop
              if should_early_stop?(new_results, early_stop) do
                # Cancel remaining tasks
                Enum.each(still_running, fn {task, _} -> Task.shutdown(task, :brutal_kill) end)
                {:halt, {[], new_results, 0}}
              else
                # Emit new results
                events =
                  Enum.map(new_results -- results, fn {:ok, result} ->
                    {:response, result.model, result}
                  end)

                {events, {still_running, new_results, timeout}}
              end
          end

        :halt ->
          {:halt, :halt}
      end,
      # Cleanup function
      fn
        {tasks, _results, _timeout} ->
          Enum.each(tasks, fn {task, _} -> Task.shutdown(task, :brutal_kill) end)

        :halt ->
          :ok
      end
    )
  end

  # Private helper functions

  defp prepare_telemetry_metadata(query, opts) do
    telemetry_meta = Keyword.get(opts, :telemetry_metadata, %{})
    models = Keyword.get(opts, :models, @default_models)

    Map.merge(telemetry_meta, %{
      query_length: String.length(query),
      models: models
    })
  end

  defp calculate_ensemble_cost(responses) do
    results =
      Enum.map(responses, fn response ->
        {:ok, response}
      end)

    Pricing.aggregate_costs(results)
  end

  defp process_completions(completions, existing_results) do
    {new_results, still_running} =
      Enum.reduce(completions, {existing_results, []}, fn
        {_task, {:ok, result}}, {results, running} ->
          {[result | results], running}

        {task, nil}, {results, running} ->
          # Task still running
          {results, [{task, :unknown_model} | running]}

        {_task, _error}, {results, running} ->
          # Task failed, ignore
          {results, running}
      end)

    {new_results, still_running}
  end

  defp should_early_stop?(results, threshold) do
    if length(results) < 2 do
      false
    else
      successes =
        Enum.filter(results, fn
          {:ok, _} -> true
          _ -> false
        end)

      if length(successes) < 2 do
        false
      else
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
  end

  defp aggregate_stream_results(results, voting_strategy, normalization) do
    # Filter successes
    responses =
      results
      |> Enum.filter(fn
        {:ok, _} -> true
        _ -> false
      end)
      |> Enum.map(fn {:ok, result} -> result end)

    if Enum.empty?(responses) do
      {:error, :no_successful_responses}
    else
      CrucibleEnsemble.Vote.apply_strategy(responses, voting_strategy,
        normalization: normalization
      )
    end
  end
end
