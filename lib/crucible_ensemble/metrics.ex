defmodule CrucibleEnsemble.Metrics do
  @moduledoc """
  Telemetry integration and metrics collection for ensemble predictions.

  Provides comprehensive instrumentation for research-grade data collection,
  including latency tracking, cost accumulation, consensus measurement,
  and failure analysis.
  """

  require Logger

  @doc """
  Attach default telemetry handlers for ensemble events.

  Call this function in your application startup to enable automatic
  metrics collection and logging.

  ## Examples

      # In your application.ex
      def start(_type, _args) do
        CrucibleEnsemble.Metrics.attach_handlers()
        # ... rest of supervision tree
      end

  """
  @spec attach_handlers() :: :ok
  def attach_handlers do
    events = [
      [:crucible_ensemble, :predict, :start],
      [:crucible_ensemble, :predict, :stop],
      [:crucible_ensemble, :predict, :exception],
      [:crucible_ensemble, :executor, :start],
      [:crucible_ensemble, :executor, :stop],
      [:crucible_ensemble, :model, :start],
      [:crucible_ensemble, :model, :stop],
      [:crucible_ensemble, :model, :exception],
      [:crucible_ensemble, :vote, :complete],
      [:crucible_ensemble, :consensus, :reached],
      [:crucible_ensemble, :consensus, :failed]
    ]

    :telemetry.attach_many(
      "ensemble-metrics-handler",
      events,
      &__MODULE__.handle_event/4,
      nil
    )

    :ok
  end

  @doc """
  Handle telemetry events.

  This is called automatically when handlers are attached.
  You can also implement custom handlers by attaching to specific events.
  """
  @spec handle_event([atom()], map(), map(), any()) :: :ok
  def handle_event([:crucible_ensemble, :predict, :start], _measurements, metadata, _config) do
    Logger.debug("Ensemble prediction started",
      models: metadata.models,
      query_length: metadata.query_length
    )

    :ok
  end

  def handle_event([:crucible_ensemble, :predict, :stop], measurements, metadata, _config) do
    duration_ms = measurements.duration / 1_000

    Logger.info("Ensemble prediction completed",
      duration_ms: Float.round(duration_ms, 2),
      consensus: metadata.consensus,
      total_cost: metadata.total_cost,
      successes: metadata.successes,
      failures: metadata.failures
    )

    :ok
  end

  def handle_event([:crucible_ensemble, :predict, :exception], measurements, metadata, _config) do
    Logger.error("Ensemble prediction failed",
      error: metadata.error,
      duration_ms: Float.round(measurements.duration / 1_000, 2)
    )

    :ok
  end

  def handle_event([:crucible_ensemble, :model, :start], _measurements, metadata, _config) do
    Logger.debug("Model call started",
      model: metadata.model
    )

    :ok
  end

  def handle_event([:crucible_ensemble, :model, :stop], measurements, metadata, _config) do
    duration_ms = measurements.duration / 1_000

    Logger.debug("Model call completed",
      model: metadata.model,
      duration_ms: Float.round(duration_ms, 2),
      cost: metadata.cost
    )

    :ok
  end

  def handle_event([:crucible_ensemble, :model, :exception], measurements, metadata, _config) do
    duration_ms = measurements.duration / 1_000

    Logger.warning("Model call failed",
      model: metadata.model,
      error: Map.get(metadata, :error, "unknown"),
      duration_ms: Float.round(duration_ms, 2)
    )

    :ok
  end

  def handle_event([:crucible_ensemble, :vote, :complete], _measurements, metadata, _config) do
    Logger.debug("Voting completed",
      strategy: metadata.strategy,
      consensus: metadata.consensus
    )

    :ok
  end

  def handle_event([:crucible_ensemble, :consensus, :reached], _measurements, metadata, _config) do
    Logger.info("Consensus reached",
      consensus: metadata.consensus,
      threshold: metadata.threshold
    )

    :ok
  end

  def handle_event([:crucible_ensemble, :consensus, :failed], _measurements, metadata, _config) do
    Logger.warning("Consensus failed",
      consensus: metadata.consensus,
      threshold: metadata.threshold
    )

    :ok
  end

  def handle_event(_event, _measurements, _metadata, _config) do
    :ok
  end

  @doc """
  Record a prediction event with full metadata.

  Use this to manually record prediction events if you're not using
  the standard CrucibleEnsemble.predict/2 API.

  ## Parameters

    * `metadata` - Map containing prediction details

  ## Expected Metadata Keys

    * `:query` - The input query
    * `:models` - List of models used
    * `:strategy` - Voting strategy
    * `:answer` - Final answer
    * `:consensus` - Consensus score (0.0-1.0)
    * `:duration_us` - Total duration in microseconds
    * `:total_cost` - Total cost in USD
    * `:successes` - Number of successful model calls
    * `:failures` - Number of failed model calls

  """
  @spec record_prediction(map()) :: :ok
  def record_prediction(metadata) do
    duration = Map.get(metadata, :duration_us, 0)

    :telemetry.execute(
      [:crucible_ensemble, :predict, :stop],
      %{duration: duration},
      metadata
    )

    :ok
  end

  @doc """
  Record an individual model response.

  ## Parameters

    * `model` - Model identifier
    * `latency_us` - Response latency in microseconds
    * `cost` - Cost in USD
    * `success` - Whether the call succeeded

  """
  @spec record_model_response(atom(), non_neg_integer(), float(), boolean()) :: :ok
  def record_model_response(model, latency_us, cost, success) do
    event_name =
      if success do
        [:crucible_ensemble, :model, :stop]
      else
        [:crucible_ensemble, :model, :exception]
      end

    :telemetry.execute(
      event_name,
      %{duration: latency_us},
      %{model: model, cost: cost, success: success}
    )

    :ok
  end

  @doc """
  Calculate aggregate statistics from collected metrics.

  This is a helper for research analysis. In production, you'd typically
  use a metrics aggregation service like StatsD, Prometheus, or DataDog.

  ## Returns

  A map containing:
    * `:total_predictions` - Total number of predictions
    * `:avg_latency_ms` - Average latency in milliseconds
    * `:p50_latency_ms` - Median latency
    * `:p95_latency_ms` - 95th percentile latency
    * `:p99_latency_ms` - 99th percentile latency
    * `:total_cost` - Total cost in USD
    * `:avg_cost` - Average cost per prediction
    * `:avg_consensus` - Average consensus score
    * `:success_rate` - Percentage of successful predictions

  """
  @spec aggregate_stats([map()]) :: map()
  def aggregate_stats(predictions) when is_list(predictions) do
    if Enum.empty?(predictions) do
      %{
        total_predictions: 0,
        avg_latency_ms: 0.0,
        p50_latency_ms: 0.0,
        p95_latency_ms: 0.0,
        p99_latency_ms: 0.0,
        total_cost: 0.0,
        avg_cost: 0.0,
        avg_consensus: 0.0,
        success_rate: 0.0
      }
    else
      latencies =
        predictions
        |> Enum.map(&Map.get(&1, :duration_us, 0))
        |> Enum.map(&(&1 / 1_000))
        |> Enum.sort()

      costs = Enum.map(predictions, &Map.get(&1, :total_cost, 0.0))
      consensuses = Enum.map(predictions, &Map.get(&1, :consensus, 0.0))
      successes = Enum.count(predictions, &Map.get(&1, :success, true))

      %{
        total_predictions: length(predictions),
        avg_latency_ms: average(latencies),
        p50_latency_ms: percentile(latencies, 50),
        p95_latency_ms: percentile(latencies, 95),
        p99_latency_ms: percentile(latencies, 99),
        total_cost: Enum.sum(costs),
        avg_cost: average(costs),
        avg_consensus: average(consensuses),
        success_rate: successes / length(predictions) * 100
      }
    end
  end

  @doc """
  Export metrics to CSV format.

  Useful for offline analysis in tools like Excel, R, or Python.

  ## Parameters

    * `predictions` - List of prediction metadata maps
    * `path` - Output file path

  """
  @spec export_to_csv([map()], String.t()) :: :ok | {:error, term()}
  def export_to_csv(predictions, path) do
    headers = [
      "timestamp",
      "query",
      "models",
      "strategy",
      "answer",
      "consensus",
      "duration_ms",
      "total_cost",
      "successes",
      "failures"
    ]

    rows =
      Enum.map(predictions, fn pred ->
        [
          Map.get(pred, :timestamp, ""),
          Map.get(pred, :query, ""),
          Enum.join(Map.get(pred, :models, []), ";"),
          to_string(Map.get(pred, :strategy, "")),
          Map.get(pred, :answer, ""),
          to_string(Map.get(pred, :consensus, 0.0)),
          to_string(Map.get(pred, :duration_us, 0) / 1_000),
          to_string(Map.get(pred, :total_cost, 0.0)),
          to_string(Map.get(pred, :successes, 0)),
          to_string(Map.get(pred, :failures, 0))
        ]
      end)

    csv_content =
      [headers | rows]
      |> Enum.map(&Enum.join(&1, ","))
      |> Enum.join("\n")

    File.write(path, csv_content)
  end

  @doc """
  Create a summary report from metrics.

  Returns a human-readable string summarizing ensemble performance.

  ## Examples

      iex> predictions = [
      ...>   %{duration_us: 1000000, total_cost: 0.001, consensus: 0.8},
      ...>   %{duration_us: 1500000, total_cost: 0.0015, consensus: 1.0}
      ...> ]
      iex> CrucibleEnsemble.Metrics.summary_report(predictions)
      # Returns formatted report string

  """
  @spec summary_report([map()]) :: String.t()
  def summary_report(predictions) do
    stats = aggregate_stats(predictions)

    """
    Ensemble Performance Summary
    ============================

    Total Predictions: #{stats.total_predictions}

    Latency Metrics:
      - Average: #{Float.round(stats.avg_latency_ms, 2)} ms
      - P50 (Median): #{Float.round(stats.p50_latency_ms, 2)} ms
      - P95: #{Float.round(stats.p95_latency_ms, 2)} ms
      - P99: #{Float.round(stats.p99_latency_ms, 2)} ms

    Cost Metrics:
      - Total Cost: $#{Float.round(stats.total_cost, 4)}
      - Average Cost: $#{Float.round(stats.avg_cost, 6)}

    Consensus Metrics:
      - Average Consensus: #{Float.round(stats.avg_consensus * 100, 2)}%
      - Success Rate: #{Float.round(stats.success_rate, 2)}%
    """
  end

  # Private helper functions

  defp average([]), do: 0.0

  defp average(list) do
    Enum.sum(list) / length(list)
  end

  defp percentile([], _p), do: 0.0

  defp percentile(sorted_list, p) when p >= 0 and p <= 100 do
    k = (length(sorted_list) - 1) * (p / 100)
    f = Float.floor(k)
    c = Float.ceil(k)

    if f == c do
      Enum.at(sorted_list, trunc(k))
    else
      lower = Enum.at(sorted_list, trunc(f))
      upper = Enum.at(sorted_list, trunc(c))
      lower + (upper - lower) * (k - f)
    end
  end
end
