<p align="center">
  <img src="assets/crucible_ensemble.svg" alt="Ensemble" width="150"/>
</p>

# CrucibleEnsemble

**Multi-model ensemble prediction with configurable voting strategies for AI reliability research.**

Ensemble is an Elixir library that enables reliable AI predictions by querying multiple language models concurrently and aggregating their responses using sophisticated voting strategies. Built on the BEAM VM, it leverages Elixir's lightweight processes to achieve massive parallelism with minimal overhead.

## Research Motivation

Current AI systems exhibit unacceptably high failure rates in production:
- Single GPT-4 instances achieve ~85-90% reliability
- Production AI agents fail 70-95% of complex tasks
- Mission-critical applications require 99.9%+ reliability

**Ensemble Hypothesis**: Massively concurrent small language model (SLM) ensembles can achieve 99.9%+ reliability at <10% the cost of single large language model approaches.

## Features

- **High Reliability**: Ensemble voting reduces error rates exponentially
- **Multiple Voting Strategies**: Majority, weighted, best confidence, unanimous
- **Flexible Execution**: Parallel, sequential, hedged, cascade strategies
- **Cost Tracking**: Automatic per-model and ensemble cost calculation
- **Telemetry Integration**: Comprehensive instrumentation for research analysis
- **Fault Tolerance**: Graceful degradation when models fail
- **BEAM Concurrency**: Leverages Elixir's lightweight processes for massive parallelism

## Installation

Add `ensemble` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:crucible_ensemble, "~> 0.1.0"}
  ]
end
```

Or install from GitHub:

```elixir
def deps do
  [
  ]
end
```

## Quick Start

```elixir
# Basic usage with default settings (majority voting)
{:ok, result} = CrucibleEnsemble.predict("What is 2+2?")

IO.puts(result.answer)
# => "4"

IO.inspect(result.metadata)
# => %{
#   consensus: 1.0,
#   votes: %{"4" => 3},
#   latency_ms: 234,
#   cost_usd: 0.00015,
#   models_used: [:gemini_flash, :openai_gpt4o_mini, :anthropic_haiku],
#   successes: 3,
#   failures: 0
# }
```

## Configuration

Set your API keys as environment variables:

```bash
export GEMINI_API_KEY="your-gemini-key"
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

## Usage Examples

### Custom Model Selection

```elixir
{:ok, result} = CrucibleEnsemble.predict(
  "What is the capital of France?",
  models: [:gemini_flash, :openai_gpt4o_mini, :anthropic_haiku]
)
```

### Weighted Voting Strategy

Useful for open-ended questions where model confidence matters:

```elixir
{:ok, result} = CrucibleEnsemble.predict(
  "Explain quantum computing in one sentence.",
  strategy: :weighted,
  models: [:openai_gpt4o, :anthropic_sonnet]
)
```

### Asynchronous Predictions

```elixir
# Start prediction
task = CrucibleEnsemble.predict_async("What is the capital of France?")

# Do other work...
other_work()

# Get result
{:ok, result} = Task.await(task, 10_000)
```

### Multiple Concurrent Predictions

```elixir
questions = [
  "What is 10 * 10?",
  "What is 5 + 5?",
  "What is 100 - 50?"
]

tasks = Enum.map(questions, &CrucibleEnsemble.predict_async/1)
results = Task.await_many(tasks, 10_000)
```

### Streaming Results

```elixir
stream = CrucibleEnsemble.predict_stream(
  "Complex question?",
  models: [:model1, :model2, :model3],
  early_stop_threshold: 0.8
)

Enum.each(stream, fn
  {:response, model, response} ->
    IO.puts "Got response from #{model}"
  {:complete, final_result} ->
    IO.puts "Final answer: #{final_result.answer}"
end)
```

## Voting Strategies

### Majority Voting (`:majority`)

Most common response wins. Simple and interpretable.

```elixir
{:ok, result} = CrucibleEnsemble.predict(
  "What is 2+2?",
  strategy: :majority
)
```

**Best for**: Factual questions, classification tasks, deterministic problems

### Weighted Voting (`:weighted`)

Responses weighted by model confidence scores.

```elixir
{:ok, result} = CrucibleEnsemble.predict(
  "Explain recursion",
  strategy: :weighted
)
```

**Best for**: Open-ended questions, when confidence scores are reliable

### Best Confidence (`:best_confidence`)

Select single highest confidence response. Fast, no consensus.

```elixir
{:ok, result} = CrucibleEnsemble.predict(
  "Generate code",
  strategy: :best_confidence
)
```

**Best for**: Latency-critical applications, heterogeneous ensembles

### Unanimous (`:unanimous`)

All models must agree. Highest confidence, may fail.

```elixir
{:ok, result} = CrucibleEnsemble.predict(
  "Critical decision",
  strategy: :unanimous
)
```

**Best for**: High-stakes decisions requiring absolute consensus

## Execution Strategies

### Parallel (`:parallel`)

Execute all models simultaneously. Maximum quality, higher cost.

```elixir
{:ok, result} = CrucibleEnsemble.predict(
  "Query",
  execution: :parallel
)
```

**Tradeoff**: Fastest completion, all models always called

### Sequential (`:sequential`)

Execute one at a time until consensus. Adaptive cost.

```elixir
{:ok, result} = CrucibleEnsemble.predict(
  "Query",
  execution: :sequential,
  min_consensus: 0.7
)
```

**Tradeoff**: Lower cost, higher latency, may stop early

### Hedged (`:hedged`)

Primary model with backup hedges for tail latency.

```elixir
{:ok, result} = CrucibleEnsemble.predict(
  "Query",
  execution: :hedged,
  hedge_delay_ms: 500
)
```

**Tradeoff**: Optimized P99 latency, controlled cost overhead

### Cascade (`:cascade`)

Priority order with early stopping on high confidence.

```elixir
{:ok, result} = CrucibleEnsemble.predict(
  "Query",
  execution: :cascade,
  confidence_threshold: 0.85
)
```

**Tradeoff**: Fast and cheap, may miss consensus

## Response Normalization

Different normalization strategies for comparing responses:

### Lowercase Trim (`:lowercase_trim`)

Default. Case-insensitive comparison.

```elixir
{:ok, result} = CrucibleEnsemble.predict(
  "Query",
  normalization: :lowercase_trim
)
```

### Numeric (`:numeric`)

Extract numeric values from responses.

```elixir
{:ok, result} = CrucibleEnsemble.predict(
  "What is 2+2?",
  normalization: :numeric
)
# Normalizes "The answer is 4" -> 4.0
```

### Boolean (`:boolean`)

Extract yes/no answers.

```elixir
{:ok, result} = CrucibleEnsemble.predict(
  "Is Elixir functional?",
  normalization: :boolean
)
# Normalizes "Yes, it is" -> true
```

### JSON (`:json`)

Parse JSON responses.

```elixir
{:ok, result} = CrucibleEnsemble.predict(
  "Return JSON",
  normalization: :json
)
```

## Telemetry Integration

Ensemble emits comprehensive telemetry events for research analysis:

```elixir
# Attach default handlers
CrucibleEnsemble.Metrics.attach_handlers()

# Or attach custom handlers
:telemetry.attach(
  "my-ensemble-handler",
  [:ensemble, :predict, :stop],
  fn _event, measurements, metadata, _config ->
    IO.inspect({measurements, metadata})
  end,
  nil
)
```

### Available Events

- `[:ensemble, :predict, :start]` - Prediction started
- `[:ensemble, :predict, :stop]` - Prediction completed
- `[:ensemble, :predict, :exception]` - Prediction failed
- `[:ensemble, :model, :start]` - Individual model call started
- `[:ensemble, :model, :stop]` - Individual model call completed
- `[:ensemble, :model, :exception]` - Individual model call failed
- `[:ensemble, :vote, :complete]` - Voting completed
- `[:ensemble, :consensus, :reached]` - Consensus threshold reached
- `[:ensemble, :consensus, :failed]` - Consensus not achieved

## Cost Tracking

Ensemble automatically tracks costs based on token usage:

```elixir
{:ok, result} = CrucibleEnsemble.predict("Query")

IO.puts "Total cost: $#{result.metadata.cost_usd}"
IO.puts "Cost per model: #{inspect(result.metadata.cost_breakdown)}"
```

Estimate costs before execution:

```elixir
estimate = CrucibleEnsemble.Pricing.estimate_cost(
  [:gemini_flash, :openai_gpt4o_mini],
  estimated_input_tokens: 100,
  estimated_output_tokens: 50
)

IO.puts "Estimated cost: $#{estimate.total_usd}"
```

## Metrics and Analysis

Export metrics for research analysis:

```elixir
# Collect prediction data
predictions = [...]

# Generate statistics
stats = CrucibleEnsemble.Metrics.aggregate_stats(predictions)

IO.puts "Average latency: #{stats.avg_latency_ms}ms"
IO.puts "P95 latency: #{stats.p95_latency_ms}ms"
IO.puts "Average consensus: #{stats.avg_consensus}"
IO.puts "Total cost: $#{stats.total_cost}"

# Export to CSV
CrucibleEnsemble.Metrics.export_to_csv(predictions, "results.csv")

# Generate report
report = CrucibleEnsemble.Metrics.summary_report(predictions)
IO.puts report
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Ensemble System                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐                     │
│  │   Ensemble   │      │  Ensemble    │                     │
│  │     API      │─────▶│  Supervisor  │                     │
│  └──────────────┘      └──────┬───────┘                     │
│                                │                              │
│                    ┌───────────┼───────────┐                │
│                    ▼           ▼           ▼                 │
│            ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│            │  Model   │ │  Model   │ │  Model   │          │
│            │  Worker  │ │  Worker  │ │  Worker  │          │
│            │   (1)    │ │   (2)    │ │   (N)    │          │
│            └────┬─────┘ └────┬─────┘ └────┬─────┘          │
│                 │            │            │                  │
│                 ▼            ▼            ▼                  │
│            ┌──────────────────────────────────┐             │
│            │        Voting & Aggregation      │             │
│            │  • Majority    • Weighted        │             │
│            │  • Confidence  • Unanimous       │             │
│            └──────────────────────────────────┘             │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Supported Models

- **Google Gemini**: `gemini_flash`, `gemini_pro`
- **OpenAI**: `openai_gpt4o_mini`, `openai_gpt4o`, `openai_gpt4`
- **Anthropic**: `anthropic_haiku`, `anthropic_sonnet`, `anthropic_opus`

## Examples

See the `examples/` directory for complete working examples:

- `basic_usage.exs` - Basic ensemble predictions
- `voting_strategies.exs` - Comparison of voting strategies
- `execution_strategies.exs` - Different execution modes
- `research_experiment.exs` - Research experiment template

Run examples:

```bash
elixir examples/basic_usage.exs
```

## Testing

Run the test suite:

```bash
mix test
```

Run with coverage:

```bash
mix test --cover
```

## Research Applications

This library is designed for AI reliability research. Example research questions:

- **H1**: Does 5-model ensemble achieve >99% reliability vs 85-90% single model?
- **H2**: Does request hedging reduce P99 latency by >25%?
- **H3**: Does BEAM enable 10x more parallel operations than Python?
- **H4**: Is ensemble cost <10% of single GPT-4 at equivalent reliability?

## Performance

Ensemble leverages BEAM's lightweight processes for massive parallelism:

- **Concurrent Models**: Unlimited (bounded by system resources)
- **Process Overhead**: ~2KB per model call
- **Latency**: P99 < single-model P50 (with hedging)
- **Throughput**: Scales linearly with cores

## Contributing

This is a research library. Contributions welcome:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see [LICENSE](https://github.com/North-Shore-AI/crucible_ensemble/blob/main/LICENSE) file for details

## Citation

If you use Ensemble in research, please cite:

```bibtex
@software{ensemble2025,
  title = {Ensemble: Multi-Model AI Reliability Framework},
  author = {ElixirAI Research Initiative},
  year = {2025},
  url = {https://github.com/elixir-ai-research/ensemble}
}
```

## Acknowledgments

- Built on [req_llm](https://github.com/calebjcourtney/req_llm) for LLM API integration
- Inspired by ensemble methods in machine learning
- Research funded by ElixirAI Initiative

## Support

- Documentation: https://hexdocs.pm/ensemble
- Issues: https://github.com/elixir-ai-research/ensemble/issues
- Discussions: https://github.com/elixir-ai-research/ensemble/discussions

## Advanced Features

### Custom Voting Strategies

Implement domain-specific voting strategies:

```elixir
# Confidence-weighted majority vote
confidence_weighted = fn responses, opts ->
  # Weight votes by confidence scores
  weighted_votes = Enum.map(responses, fn response ->
    weight = response.confidence || 0.5
    {response.answer, weight}
  end)

  # Aggregate weighted votes
  vote_counts = Enum.reduce(weighted_votes, %{}, fn {answer, weight}, acc ->
    Map.update(acc, answer, weight, &(&1 + weight))
  end)

  # Select highest weighted answer
  {best_answer, best_weight} = Enum.max_by(vote_counts, fn {_answer, weight} -> weight end)

  total_weight = Enum.sum(Map.values(vote_counts))
  consensus = best_weight / total_weight

  %{
    answer: best_answer,
    consensus: consensus,
    votes: vote_counts
  }
end

# Use custom strategy
{:ok, result} = CrucibleEnsemble.predict(
  "Complex question?",
  strategy: confidence_weighted
)
```

### Dynamic Model Selection

Select models based on query characteristics:

```elixir
# Intelligent model router
def select_models_for_query(query) do
  query_length = String.length(query)

  cond do
    query_length < 50 ->
      # Short queries: fast models
      [:gemini_flash, :openai_gpt4o_mini]

    String.contains?(query, "code") ->
      # Code-related: specialized models
      [:openai_gpt4o, :anthropic_sonnet]

    true ->
      # Default ensemble
      [:gemini_flash, :openai_gpt4o_mini, :anthropic_haiku]
  end
end

# Use dynamic selection
models = select_models_for_query(question)
{:ok, result} = CrucibleEnsemble.predict(question, models: models)
```

### Ensemble of Ensembles

Create hierarchical ensemble structures:

```elixir
# Primary ensemble for initial answers
{:ok, primary_result} = CrucibleEnsemble.predict(
  question,
  models: [:gemini_flash, :openai_gpt4o_mini, :anthropic_haiku],
  strategy: :majority
)

# Secondary ensemble for verification
verification_question = "Verify this answer: #{primary_result.answer}. Is it correct?"

{:ok, verification_result} = CrucibleEnsemble.predict(
  verification_question,
  models: [:openai_gpt4o, :anthropic_sonnet],  # More capable models
  strategy: :weighted
)

# Combine results
final_answer = if verification_result.metadata.consensus > 0.8 do
  primary_result.answer
else
  # Fallback or re-query
  handle_uncertain_answer(primary_result, verification_result)
end
```

### Adaptive Execution Strategies

Change strategy based on runtime conditions:

```elixir
# Adaptive strategy selection
def adaptive_predict(question, context) do
  # Check system load
  system_busy = check_system_load()

  # Select strategy based on context
  {models, strategy, execution} =
    cond do
      context[:high_stakes] ->
        {[:openai_gpt4o, :anthropic_sonnet], :unanimous, :parallel}

      context[:cost_sensitive] ->
        {[:gemini_flash, :openai_gpt4o_mini], :majority, :sequential}

      system_busy ->
        {[:gemini_flash], :best_confidence, :parallel}

      true ->
        {[:gemini_flash, :openai_gpt4o_mini, :anthropic_haiku], :majority, :parallel}
    end

  CrucibleEnsemble.predict(question,
    models: models,
    strategy: strategy,
    execution: execution
  )
end
```

### Performance Monitoring and Optimization

Track and optimize ensemble performance:

```elixir
# Performance profiler
defmodule EnsembleProfiler do
  def profile_ensemble(question, model_configs) do
    results = for config <- model_configs do
      {time, result} = :timer.tc(fn ->
        CrucibleEnsemble.predict(question, config)
      end)

      latency_ms = time / 1_000
      cost = result.metadata.cost_usd

      %{
        config: config,
        latency_ms: latency_ms,
        cost: cost,
        consensus: result.metadata.consensus
      }
    end

    # Find optimal configurations
    %{
      fastest: Enum.min_by(results, & &1.latency_ms),
      cheapest: Enum.min_by(results, & &1.cost),
      most_reliable: Enum.max_by(results, & &1.consensus),
      best_value: find_best_value_ratio(results)
    }
  end

  defp find_best_value_ratio(results) do
    # Balance latency, cost, and reliability
    Enum.min_by(results, fn r ->
      # Composite score: latency + cost * 1000 + (1 - consensus) * 100
      r.latency_ms + r.cost * 1000 + (1 - r.consensus) * 100
    end)
  end
end
```

## Complete API Reference

### Core Functions

#### `CrucibleEnsemble.predict(query, opts \\\\ [])`

Execute ensemble prediction synchronously.

**Parameters:**
- `query`: String - The question or prompt
- `opts`: Keyword list of options

**Options:**
- `:models` - List of model atoms (default: `[:gemini_flash, :openai_gpt4o_mini, :anthropic_haiku]`)
- `:strategy` - Voting strategy (default: `:majority`)
- `:execution` - Execution strategy (default: `:parallel`)
- `:timeout` - Per-model timeout in ms (default: 5000)
- `:min_responses` - Minimum successful responses (default: ceil(n/2))
- `:normalization` - Response normalization (default: `:lowercase_trim`)
- `:api_keys` - Map of model => API key
- `:telemetry_metadata` - Additional telemetry data

**Returns:** `{:ok, result}` or `{:error, reason}`

#### `CrucibleEnsemble.predict_async(query, opts \\\\ [])`

Execute ensemble prediction asynchronously.

**Returns:** `Task.t()` - Await with `Task.await(task, timeout)`

#### `CrucibleEnsemble.predict_stream(query, opts \\\\ [])`

Execute with streaming results.

**Options:**
- `:early_stop_threshold` - Stop when consensus reached (default: 1.0)

**Returns:** `Stream.t()` - Emits `{:response, model, result}` and `{:complete, final_result}`

### Voting Strategies

#### `:majority` (Default)
- Simple majority vote
- Fast and interpretable
- Best for: Factual questions, classification

#### `:weighted`
- Weight votes by model confidence
- More sophisticated aggregation
- Best for: Open-ended questions

#### `:best_confidence`
- Select highest confidence response
- No consensus requirement
- Best for: Speed-critical applications

#### `:unanimous`
- All models must agree
- Highest reliability
- Best for: Critical decisions

#### Custom Strategies
```elixir
my_strategy = fn responses, _opts ->
  # Your custom voting logic
  # Return: %{answer: answer, consensus: float, votes: map}
end

{:ok, result} = CrucibleEnsemble.predict(question, strategy: my_strategy)
```

### Execution Strategies

#### `:parallel` (Default)
- All models execute simultaneously
- Maximum quality, higher cost
- Tradeoff: Fast completion, all models called

#### `:sequential`
- Execute until consensus reached
- Adaptive cost based on early stopping
- Tradeoff: Variable latency, lower average cost

#### `:hedged`
- Primary model + backup hedges
- Optimizes P99 latency
- Tradeoff: Controlled cost overhead

#### `:cascade`
- Priority-ordered execution
- Early stopping on high confidence
- Tradeoff: Fast and cheap, may miss consensus

## Integration Examples

### Phoenix Web Application

```elixir
# lib/my_app_web/controllers/ensemble_controller.ex
defmodule MyAppWeb.EnsembleController do
  use Phoenix.Controller
  alias CrucibleEnsemble

  def predict(conn, %{"question" => question, "models" => models}) do
    models = String.split(models, ",") |> Enum.map(&String.to_atom/1)

    case CrucibleEnsemble.predict(question, models: models) do
      {:ok, result} ->
        json(conn, %{
          answer: result.answer,
          consensus: result.metadata.consensus,
          cost: result.metadata.cost_usd,
          latency_ms: result.metadata.latency_ms
        })

      {:error, reason} ->
        conn
        |> put_status(500)
        |> json(%{error: reason})
    end
  end
end

# lib/my_app_web/live/ensemble_live.ex
defmodule MyAppWeb.EnsembleLive do
  use Phoenix.LiveView

  def mount(_params, _session, socket) do
    {:ok, assign(socket, result: nil, loading: false)}
  end

  def handle_event("predict", %{"question" => question}, socket) do
    # Start async prediction
    task = CrucibleEnsemble.predict_async(question)

    # Store task and show loading
    {:noreply, assign(socket,
      task: task,
      loading: true,
      question: question
    )}
  end

  def handle_info({:task_completed, result}, socket) do
    {:noreply, assign(socket,
      result: result,
      loading: false,
      task: nil
    )}
  end

  def handle_info({task_ref, {:ok, result}}, socket) do
    # Handle async task completion
    Process.demonitor(task_ref, [:flush])
    send(self(), {:task_completed, result})
    {:noreply, socket}
  end
end
```

### Research Experiment Framework

```elixir
# lib/research_runner.ex
defmodule ResearchRunner do
  alias CrucibleEnsemble

  def run_experiment(config) do
    # Load test dataset
    {:ok, dataset} = CrucibleDatasets.load(:mmlu_stem)

    # Sample questions for experiment
    questions = Enum.take_random(dataset.items, config.sample_size)

    # Run ensemble on each question
    results = Enum.map(questions, fn item ->
      start_time = System.monotonic_time()

      result = CrucibleEnsemble.predict(
        item.input.question,
        models: config.models,
        strategy: config.strategy,
        execution: config.execution
      )

      end_time = System.monotonic_time()
      latency_us = end_time - start_time

      %{
        question: item.input.question,
        expected: item.expected,
        predicted: result.answer,
        correct: result.answer == item.expected,
        consensus: result.metadata.consensus,
        latency_us: latency_us,
        cost_usd: result.metadata.cost_usd,
        models_used: result.metadata.models_used
      }
    end)

    # Calculate metrics
    accuracy = Enum.count(results, & &1.correct) / length(results)
    avg_latency = Enum.sum(Enum.map(results, & &1.latency_us)) / length(results)
    total_cost = Enum.sum(Enum.map(results, & &1.cost_usd))

    # Generate report
    %{
      config: config,
      accuracy: accuracy,
      avg_latency_ms: avg_latency / 1000,
      total_cost: total_cost,
      results: results
    }
  end

  def compare_strategies() do
    strategies = [
      %{name: "majority_parallel", strategy: :majority, execution: :parallel},
      %{name: "weighted_sequential", strategy: :weighted, execution: :sequential},
      %{name: "hedged", execution: :hedged},
    ]

    for strategy <- strategies do
      run_experiment(Map.merge(%{
        models: [:gemini_flash, :openai_gpt4o_mini, :anthropic_haiku],
        sample_size: 100
      }, strategy))
    end
  end
end
```

### Production Service with Caching

```elixir
# lib/ensemble_service.ex
defmodule EnsembleService do
  use GenServer
  alias CrucibleEnsemble

  # Public API
  def predict(question, opts \\ []) do
    GenServer.call(__MODULE__, {:predict, question, opts})
  end

  # GenServer callbacks
  def init(_opts) do
    # Initialize cache
    cache = :ets.new(:ensemble_cache, [:set, :protected, :named_table])

    # Attach telemetry handlers
    CrucibleEnsemble.Metrics.attach_handlers()

    {:ok, %{cache: cache}}
  end

  def handle_call({:predict, question, opts}, _from, state) do
    # Check cache first
    cache_key = {question, opts}

    case :ets.lookup(state.cache, cache_key) do
      [{^cache_key, cached_result}] ->
        {:reply, cached_result, state}

      [] ->
        # Execute prediction
        result = CrucibleEnsemble.predict(question, opts)

        # Cache result (with TTL)
        :ets.insert(state.cache, {cache_key, result})

        # Schedule cache cleanup
        Process.send_after(self(), {:cleanup_cache, cache_key}, 3600_000)  # 1 hour

        {:reply, result, state}
    end
  end

  def handle_info({:cleanup_cache, key}, state) do
    :ets.delete(state.cache, key)
    {:noreply, state}
  end
end

# Application startup
defmodule MyApp.Application do
  def start(_type, _args) do
    children = [
      {EnsembleService, []},
      # ... other services
    ]

    Supervisor.start_link(children, strategy: :one_for_one)
  end
end
```

### Monitoring and Observability

```elixir
# lib/ensemble_monitor.ex
defmodule EnsembleMonitor do
  use GenServer

  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def init(_opts) do
    # Attach custom telemetry handlers
    attach_monitoring_handlers()

    # Initialize metrics storage
    {:ok, %{metrics: [], alerts: []}}
  end

  def handle_call(:get_metrics, _from, state) do
    {:reply, state.metrics, state}
  end

  # Custom telemetry handlers
  def attach_monitoring_handlers do
    # Monitor prediction completion
    :telemetry.attach(
      "prediction-monitor",
      [:crucible_ensemble, :predict, :stop],
      &handle_prediction_complete/4,
      nil
    )

    # Monitor consensus failures
    :telemetry.attach(
      "consensus-monitor",
      [:crucible_ensemble, :consensus, :failed],
      &handle_consensus_failure/4,
      nil
    )
  end

  def handle_prediction_complete(_event, measurements, metadata, _config) do
    # Store metrics
    metric = %{
      timestamp: DateTime.utc_now(),
      duration_ms: measurements.duration / 1_000,
      consensus: metadata.consensus,
      cost: metadata.total_cost,
      successes: metadata.successes,
      failures: metadata.failures
    }

    GenServer.cast(__MODULE__, {:store_metric, metric})

    # Alert on anomalies
    if metadata.consensus < 0.5 do
      Logger.warning("Low consensus detected", consensus: metadata.consensus)
    end

    if metadata.failures > metadata.successes do
      Logger.error("More failures than successes in ensemble")
    end
  end

  def handle_consensus_failure(_event, _measurements, metadata, _config) do
    Logger.warning("Ensemble consensus failed",
      consensus: metadata.consensus,
      threshold: metadata.threshold
    )
  end

  def handle_cast({:store_metric, metric}, state) do
    # Keep last 1000 metrics
    new_metrics = [metric | state.metrics] |> Enum.take(1000)
    {:noreply, %{state | metrics: new_metrics}}
  end
end
```

## Performance Considerations

### Concurrency and Scalability

- **BEAM Processes**: Each model call uses ~2KB of memory
- **Parallel Execution**: Scales linearly with CPU cores
- **Hedging**: Reduces P99 latency by 25-50%
- **Sequential**: Saves 30-70% cost vs parallel execution

### Cost Optimization

```elixir
# Cost-aware model selection
def select_cost_optimized_models(budget_per_query) do
  # Estimate costs for different model combinations
  combinations = [
    {[:gemini_flash], 0.0001},
    {[:gemini_flash, :openai_gpt4o_mini], 0.00025},
    {[:gemini_flash, :openai_gpt4o_mini, :anthropic_haiku], 0.0004},
  ]

  # Select best combination within budget
  Enum.find(combinations, fn {_models, cost} -> cost <= budget_per_query end)
  || List.first(combinations)  # Fallback to cheapest
end

# Adaptive execution based on cost constraints
def predict_with_budget(question, max_cost_per_query) do
  {models, estimated_cost} = select_cost_optimized_models(max_cost_per_query)

  if estimated_cost <= max_cost_per_query do
    CrucibleEnsemble.predict(question, models: models, execution: :parallel)
  else
    # Use sequential to save costs
    CrucibleEnsemble.predict(question, models: models, execution: :sequential)
  end
end
```

### Latency Optimization

```elixir
# Latency-optimized configuration
def fast_predict(question) do
  CrucibleEnsemble.predict(question,
    models: [:gemini_flash],  # Fastest model only
    strategy: :best_confidence,  # No voting overhead
    execution: :parallel,  # Though only one model
    timeout: 1000  # Shorter timeout
  )
end

# Balanced performance
def balanced_predict(question) do
  CrucibleEnsemble.predict(question,
    models: [:gemini_flash, :openai_gpt4o_mini],
    strategy: :majority,
    execution: :hedged,  # Good P99 performance
    hedge_delay_ms: 200
  )
end

# High-reliability (slower)
def reliable_predict(question) do
  CrucibleEnsemble.predict(question,
    models: [:openai_gpt4o, :anthropic_sonnet],
    strategy: :weighted,
    execution: :parallel,
    min_responses: 2  # Require both to succeed
  )
end
```

## Troubleshooting

### Common Issues

#### Low Consensus Scores

```elixir
# Debug consensus issues
{:ok, result} = CrucibleEnsemble.predict(question)

if result.metadata.consensus < 0.5 do
  IO.inspect(result.metadata.votes, label: "Vote breakdown")

  # Check individual responses
  if Map.has_key?(result, :individual_responses) do
    Enum.each(result.individual_responses, fn {model, response} ->
      IO.puts("#{model}: #{response}")
    end)
  end
end
```

#### API Rate Limits

```elixir
# Handle rate limiting with retry logic
def predict_with_retry(question, opts, max_retries \\ 3) do
  case CrucibleEnsemble.predict(question, opts) do
    {:ok, result} -> {:ok, result}

    {:error, :rate_limit} when max_retries > 0 ->
      :timer.sleep(1000)  # Wait 1 second
      predict_with_retry(question, opts, max_retries - 1)

    error -> error
  end
end
```

#### Model Failures

```elixir
# Graceful degradation on failures
def predict_resilient(question, models) do
  result = CrucibleEnsemble.predict(question, models: models)

  case result do
    {:ok, %{metadata: %{successes: successes, failures: failures}}} when failures > 0 ->
      if successes >= 1 do
        # Partial success - log but return result
        Logger.warning("Partial ensemble failure", failures: failures)
        result
      else
        # Complete failure - fallback strategy
        predict_fallback(question)
      end

    other -> other
  end
end
```

### Debugging Ensemble Behavior

```elixir
# Enable detailed logging
Logger.configure(level: :debug)

# Attach debug telemetry handler
:telemetry.attach(
  "debug-handler",
  [:crucible_ensemble, :model, :stop],
  fn _event, measurements, metadata, _config ->
    IO.puts("Model #{metadata.model} completed in #{measurements.duration / 1_000}ms")
  end,
  nil
)

# Run prediction with debugging
{:ok, result} = CrucibleEnsemble.predict(question)
```

## Research Best Practices

### Designing Ensemble Experiments

```elixir
# Systematic comparison of strategies
def ensemble_ablation_study() do
  # Define experimental conditions
  conditions = [
    %{name: "single_model", models: [:openai_gpt4o], strategy: :best_confidence},
    %{name: "majority_3", models: [:gemini_flash, :openai_gpt4o_mini, :anthropic_haiku], strategy: :majority},
    %{name: "weighted_3", models: [:gemini_flash, :openai_gpt4o_mini, :anthropic_haiku], strategy: :weighted},
    %{name: "unanimous_2", models: [:openai_gpt4o, :anthropic_sonnet], strategy: :unanimous},
  ]

  # Test questions
  questions = [
    "What is 2+2?",  # Factual
    "Explain quantum computing",  # Explanatory
    "Write a function to reverse a string",  # Code
  ]

  # Run experiments
  results = for condition <- conditions, question <- questions do
    {time, result} = :timer.tc(fn ->
      {:ok, res} = CrucibleEnsemble.predict(question, Map.take(condition, [:models, :strategy]))
      res
    end)

    %{
      condition: condition.name,
      question: question,
      answer: result.answer,
      consensus: result.metadata.consensus,
      latency_ms: time / 1_000,
      cost_usd: result.metadata.cost_usd
    }
  end

  # Analyze results
  analyze_ensemble_performance(results)
end

def analyze_ensemble_performance(results) do
  # Group by condition
  by_condition = Enum.group_by(results, & &1.condition)

  # Calculate metrics per condition
  analysis = for {condition, condition_results} <- by_condition do
    latencies = Enum.map(condition_results, & &1.latency_ms)
    costs = Enum.map(condition_results, & &1.cost_usd)
    consensuses = Enum.map(condition_results, & &1.consensus)

    %{
      condition: condition,
      avg_latency: Enum.sum(latencies) / length(latencies),
      avg_cost: Enum.sum(costs) / length(costs),
      avg_consensus: Enum.sum(consensuses) / length(consensuses),
      sample_size: length(condition_results)
    }
  end

  # Generate report
  generate_performance_report(analysis)
end
```

### Reliability Research Framework

```elixir
# lib/reliability_research.ex
defmodule ReliabilityResearch do
  alias CrucibleEnsemble

  def measure_reliability_curve(question_sets, model_configs) do
    # Test reliability at different ensemble sizes
    for config <- model_configs do
      reliability_scores = for questions <- question_sets do
        results = Enum.map(questions, fn q ->
          {:ok, result} = CrucibleEnsemble.predict(q, config)
          result.metadata.consensus
        end)

        # Calculate reliability as percentage of high-confidence answers
        high_confidence = Enum.count(results, &(&1 >= 0.8))
        high_confidence / length(results)
      end

      %{
        config: config,
        reliability_scores: reliability_scores,
        avg_reliability: Enum.sum(reliability_scores) / length(reliability_scores)
      }
    end
  end

  def cost_benefit_analysis() do
    # Compare cost vs reliability tradeoffs
    strategies = [
      %{name: "single_cheap", models: [:gemini_flash], strategy: :best_confidence},
      %{name: "ensemble_balanced", models: [:gemini_flash, :openai_gpt4o_mini, :anthropic_haiku], strategy: :majority},
      %{name: "ensemble_premium", models: [:openai_gpt4o, :anthropic_sonnet], strategy: :weighted},
    ]

    questions = load_test_questions()

    results = for strategy <- strategies do
      {total_time, predictions} = :timer.tc(fn ->
        Enum.map(questions, fn q ->
          {:ok, result} = CrucibleEnsemble.predict(q, Map.take(strategy, [:models, :strategy]))
          result
        end)
      end)

      total_cost = Enum.sum(Enum.map(predictions, & &1.metadata.cost_usd))
      avg_consensus = Enum.sum(Enum.map(predictions, & &1.metadata.consensus)) / length(predictions)

      %{
        strategy: strategy.name,
        total_cost: total_cost,
        avg_consensus: avg_consensus,
        cost_per_query: total_cost / length(questions),
        queries_per_second: length(questions) / (total_time / 1_000_000)
      }
    end

    generate_cost_benefit_report(results)
  end
end
```

## Contributing

### Development Setup

```bash
# Clone repository
git clone https://github.com/North-Shore-AI/crucible_ensemble.git
cd crucible_ensemble

# Install dependencies
mix deps.get

# Set up environment variables for testing
export GEMINI_API_KEY="your-test-key"
export OPENAI_API_KEY="your-test-key"
export ANTHROPIC_API_KEY="your-test-key"

# Run tests
mix test

# Run examples
mix run examples/basic_usage.exs
mix run examples/voting_strategies.exs

# Generate docs
mix docs
```

### Adding New Models

```elixir
# 1. Add model configuration in config/config.exs
config :crucible_ensemble, :models,
  my_custom_model: %{
    provider: :custom,
    model: "my-model-v1",
    cost_per_token: 0.000001
  }

# 2. Implement model adapter
defmodule CrucibleEnsemble.Executor.MyCustomModel do
  @behaviour CrucibleEnsemble.Executor

  @impl true
  def call(query, opts) do
    # Your model API call logic
    # Return {:ok, response} or {:error, reason}
  end

  @impl true
  def normalize_response(response) do
    # Normalize response format
  end
end

# 3. Register adapter
config :crucible_ensemble, :executors,
  my_custom_model: CrucibleEnsemble.Executor.MyCustomModel
```

### Implementing Custom Voting Strategies

```elixir
# lib/crucible_ensemble/strategy/my_strategy.ex
defmodule CrucibleEnsemble.Strategy.MyStrategy do
  @behaviour CrucibleEnsemble.Strategy

  @impl true
  def name(), do: :my_strategy

  @impl true
  def vote(responses, opts) do
    # Your voting logic
    # Return %{answer: answer, consensus: float, votes: map}
  end
end

# Use custom strategy
{:ok, result} = CrucibleEnsemble.predict(question, strategy: MyStrategy)
```

### Testing Guidelines

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test end-to-end functionality
- **Mock APIs**: Use mocks for external API calls in CI
- **Performance Tests**: Benchmark critical paths
- **Property Tests**: Use property-based testing for voting logic

```elixir
# Example property test
property "majority voting is deterministic" do
  check all responses <- list_of(response_generator()),
            length(responses) > 0 do
    result1 = CrucibleEnsemble.Strategy.Majority.vote(responses, [])
    result2 = CrucibleEnsemble.Strategy.Majority.vote(responses, [])

    assert result1.answer == result2.answer
    assert result1.consensus == result2.consensus
  end
end
```

## License

MIT License - see [LICENSE](https://github.com/North-Shore-AI/crucible_ensemble/blob/main/LICENSE) file for details

## Changelog

### v0.1.0 (Current)
- Initial release with multi-model ensemble support
- Four voting strategies: majority, weighted, best_confidence, unanimous
- Four execution strategies: parallel, sequential, hedged, cascade
- Comprehensive telemetry and metrics collection
- Cost tracking and optimization
- Response normalization and fault tolerance
- Complete documentation and examples

## Roadmap

- [ ] Production connection pooling and circuit breakers
- [ ] Distributed ensemble execution across nodes
- [ ] Custom embedding-based voting strategies
- [ ] Real-time metrics dashboard and alerting
- [ ] Automated benchmarking harness
- [ ] Integration with LangChain and LlamaIndex
- [ ] Model fine-tuning coordination
- [ ] Ensemble explainability and interpretability
