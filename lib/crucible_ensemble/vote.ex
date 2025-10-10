defmodule CrucibleEnsemble.Vote do
  @moduledoc """
  Voting and aggregation strategies for ensemble predictions.

  Implements multiple voting algorithms to combine responses from different models:
  - Majority voting (most common response wins)
  - Weighted voting (confidence-weighted selection)
  - Best confidence (highest confidence response)
  - Unanimous (all models must agree)
  - Custom aggregation functions
  """

  alias CrucibleEnsemble.Normalize

  @type strategy ::
          :majority | :weighted | :best_confidence | :unanimous | {module(), keyword()}
  @type response :: map()
  @type vote_result :: %{
          answer: any(),
          strategy: atom(),
          consensus: float(),
          metadata: map()
        }

  @doc """
  Apply voting strategy to aggregate responses.

  ## Parameters

    * `responses` - List of model response maps
    * `strategy` - Voting strategy to use (default: `:majority`)
    * `opts` - Additional options for normalization and voting

  ## Options

    * `:normalization` - Normalization strategy (default: `:lowercase_trim`)
    * `:min_consensus` - Minimum consensus threshold (default: 0.5)

  ## Examples

      iex> responses = [
      ...>   %{response: "4", model: :gemini_flash},
      ...>   %{response: "4", model: :openai_gpt4o_mini},
      ...>   %{response: "4", model: :anthropic_haiku}
      ...> ]
      iex> CrucibleEnsemble.Vote.apply_strategy(responses, :majority)
      {:ok, %{answer: "4", consensus: 1.0, strategy: :majority, votes: %{"4" => 3}}}

  """
  @spec apply_strategy([response()], strategy(), keyword()) ::
          {:ok, vote_result()} | {:error, term()}
  def apply_strategy(responses, strategy \\ :majority, opts \\ [])

  def apply_strategy([], _strategy, _opts) do
    {:error, :no_responses}
  end

  def apply_strategy(responses, :majority, opts) do
    CrucibleEnsemble.Vote.Majority.aggregate(responses, opts)
  end

  def apply_strategy(responses, :weighted, opts) do
    CrucibleEnsemble.Vote.Weighted.aggregate(responses, opts)
  end

  def apply_strategy(responses, :best_confidence, opts) do
    CrucibleEnsemble.Vote.BestConfidence.aggregate(responses, opts)
  end

  def apply_strategy(responses, :unanimous, opts) do
    CrucibleEnsemble.Vote.Unanimous.aggregate(responses, opts)
  end

  def apply_strategy(responses, {module, custom_opts}, opts) do
    combined_opts = Keyword.merge(opts, custom_opts)
    module.aggregate(responses, combined_opts)
  end

  @doc """
  Calculate consensus strength for a vote distribution.

  Returns a value between 0.0 and 1.0 indicating how much agreement there is.

  ## Examples

      iex> CrucibleEnsemble.Vote.consensus_strength(%{"4" => 3})
      1.0

      iex> CrucibleEnsemble.Vote.consensus_strength(%{"yes" => 2, "no" => 1})
      0.6666666666666666

  """
  @spec consensus_strength(map()) :: float()
  def consensus_strength(vote_distribution) when map_size(vote_distribution) == 0 do
    0.0
  end

  def consensus_strength(vote_distribution) do
    total_votes =
      vote_distribution
      |> Map.values()
      |> Enum.sum()

    max_votes =
      vote_distribution
      |> Map.values()
      |> Enum.max()

    if total_votes > 0 do
      max_votes / total_votes
    else
      0.0
    end
  end

  @doc """
  Check if consensus meets minimum threshold.

  ## Examples

      iex> result = %{consensus: 0.75}
      iex> CrucibleEnsemble.Vote.sufficient_consensus?(result, 0.5)
      true

      iex> result = %{consensus: 0.4}
      iex> CrucibleEnsemble.Vote.sufficient_consensus?(result, 0.5)
      false

  """
  @spec sufficient_consensus?(vote_result(), float()) :: boolean()
  def sufficient_consensus?(%{consensus: consensus}, threshold) do
    consensus >= threshold
  end
end

defmodule CrucibleEnsemble.Vote.Majority do
  @moduledoc """
  Majority voting: most common response wins.

  This is the simplest and most interpretable voting strategy.
  Each model gets one vote, and the most frequent response is selected.
  """

  alias CrucibleEnsemble.Normalize

  @doc """
  Aggregate responses using majority voting.
  """
  @spec aggregate([map()], keyword()) :: {:ok, map()} | {:error, term()}
  def aggregate([], _opts), do: {:error, :no_responses}

  def aggregate(responses, opts) do
    normalization = Keyword.get(opts, :normalization, :lowercase_trim)

    # Normalize all responses
    normalized_pairs =
      Enum.map(responses, fn resp ->
        normalized = Normalize.normalize_result(resp, normalization)
        {normalized, resp}
      end)

    # Count frequencies
    frequencies =
      normalized_pairs
      |> Enum.map(fn {normalized, _} -> normalized end)
      |> Enum.frequencies()

    # Find winner
    {winner, count} = Enum.max_by(frequencies, fn {_resp, cnt} -> cnt end)

    # Calculate consensus
    total = length(responses)
    consensus = count / total

    # Get sample original response for metadata
    {_, sample_response} =
      Enum.find(normalized_pairs, fn {norm, _} -> norm == winner end)

    {:ok,
     %{
       answer: winner,
       strategy: :majority,
       consensus: consensus,
       votes: frequencies,
       total_responses: total,
       winning_count: count,
       sample_response: sample_response
     }}
  end
end

defmodule CrucibleEnsemble.Vote.Weighted do
  @moduledoc """
  Weighted voting: responses weighted by confidence scores.

  Models with higher confidence have more influence on the final decision.
  Useful when model confidence scores are well-calibrated.
  """

  alias CrucibleEnsemble.Normalize

  @doc """
  Aggregate responses using confidence-weighted voting.
  """
  @spec aggregate([map()], keyword()) :: {:ok, map()} | {:error, term()}
  def aggregate([], _opts), do: {:error, :no_responses}

  def aggregate(responses, opts) do
    normalization = Keyword.get(opts, :normalization, :lowercase_trim)

    # Extract response + confidence pairs
    pairs =
      Enum.map(responses, fn resp ->
        normalized = Normalize.normalize_result(resp, normalization)
        confidence = extract_confidence(resp)
        {normalized, confidence, resp}
      end)

    # Calculate weighted scores
    scores =
      Enum.reduce(pairs, %{}, fn {response, conf, _original}, acc ->
        Map.update(acc, response, conf, fn existing -> existing + conf end)
      end)

    # Find winner
    {winner, score} = Enum.max_by(scores, fn {_resp, sc} -> sc end)

    # Calculate total confidence
    total_confidence =
      pairs
      |> Enum.map(fn {_, conf, _} -> conf end)
      |> Enum.sum()

    effective_confidence = if total_confidence > 0, do: score / total_confidence, else: 0.0

    # Get sample original response
    {_, _, sample_response} =
      Enum.find(pairs, fn {norm, _, _} -> norm == winner end)

    {:ok,
     %{
       answer: winner,
       strategy: :weighted,
       consensus: effective_confidence,
       scores: scores,
       total_confidence: total_confidence,
       winning_score: score,
       sample_response: sample_response
     }}
  end

  defp extract_confidence(response) do
    cond do
      Map.has_key?(response, :confidence) ->
        response.confidence

      Map.has_key?(response, "confidence") ->
        response["confidence"]

      Map.has_key?(response, :metadata) && Map.has_key?(response.metadata, :confidence) ->
        response.metadata.confidence

      true ->
        1.0
    end
  end
end

defmodule CrucibleEnsemble.Vote.BestConfidence do
  @moduledoc """
  Best confidence selection: select highest confidence response.

  This is not true voting - it simply picks the single response
  with the highest confidence score. Useful for early stopping
  and latency optimization.
  """

  alias CrucibleEnsemble.Normalize

  @doc """
  Select the response with the highest confidence score.
  """
  @spec aggregate([map()], keyword()) :: {:ok, map()} | {:error, term()}
  def aggregate([], _opts), do: {:error, :no_responses}

  def aggregate(responses, opts) do
    normalization = Keyword.get(opts, :normalization, :lowercase_trim)

    # Find response with highest confidence
    best =
      Enum.max_by(responses, fn resp ->
        extract_confidence(resp)
      end)

    confidence = extract_confidence(best)
    answer = Normalize.normalize_result(best, normalization)

    {:ok,
     %{
       answer: answer,
       strategy: :best_confidence,
       consensus: confidence,
       confidence: confidence,
       selected_model: Map.get(best, :model),
       sample_response: best,
       total_responses: length(responses)
     }}
  end

  defp extract_confidence(response) do
    cond do
      Map.has_key?(response, :confidence) ->
        response.confidence

      Map.has_key?(response, "confidence") ->
        response["confidence"]

      Map.has_key?(response, :metadata) && Map.has_key?(response.metadata, :confidence) ->
        response.metadata.confidence

      true ->
        1.0
    end
  end
end

defmodule CrucibleEnsemble.Vote.Unanimous do
  @moduledoc """
  Unanimous voting: all models must agree.

  This strategy requires all models to produce the same (normalized) response.
  Provides highest confidence but may fail frequently.
  """

  alias CrucibleEnsemble.Normalize

  @doc """
  Check if all responses are identical (after normalization).
  """
  @spec aggregate([map()], keyword()) :: {:ok, map()} | {:error, term()}
  def aggregate([], _opts), do: {:error, :no_responses}

  def aggregate(responses, opts) do
    normalization = Keyword.get(opts, :normalization, :lowercase_trim)

    # Normalize all responses
    normalized =
      Enum.map(responses, fn resp ->
        Normalize.normalize_result(resp, normalization)
      end)

    # Check if all are the same
    unique = Enum.uniq(normalized)

    case unique do
      [single_answer] ->
        # All models agree
        {:ok,
         %{
           answer: single_answer,
           strategy: :unanimous,
           consensus: 1.0,
           total_responses: length(responses),
           sample_response: hd(responses)
         }}

      multiple ->
        # Models disagree - no consensus
        frequencies = Enum.frequencies(normalized)

        {:error,
         %{
           reason: :no_unanimous_consensus,
           frequencies: frequencies,
           total_responses: length(responses),
           unique_answers: length(multiple)
         }}
    end
  end
end

defmodule CrucibleEnsemble.Vote.Custom do
  @moduledoc """
  Behaviour for custom voting strategies.

  Implement this behaviour to create custom aggregation logic.

  ## Example

      defmodule MyApp.SemanticVoting do
        @behaviour CrucibleEnsemble.Vote.Custom

        def aggregate(responses, opts) do
          # Custom aggregation logic
          # Return {:ok, result} or {:error, reason}
        end
      end

  """

  @callback aggregate(responses :: [map()], opts :: keyword()) ::
              {:ok, map()} | {:error, term()}
end
