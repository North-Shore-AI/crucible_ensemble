defmodule CrucibleEnsemble.Vote.SemanticSimilarity do
  @moduledoc """
  Semantic similarity voting: group responses by textual similarity.

  Unlike exact-match voting strategies, semantic similarity uses text similarity
  algorithms to group responses that are semantically equivalent but may differ
  in phrasing, formatting, or minor variations.

  This strategy is particularly effective for:
  - Mathematical answers with varied phrasing ("42" vs "The answer is 42")
  - Code with different formatting or comments
  - Classifications with equivalent labels ("positive" vs "affirmative")
  - Translations with synonyms

  ## Algorithm

  1. Normalize all responses using the specified normalization strategy
  2. Compute pairwise similarity matrix between all responses
  3. Cluster responses by similarity threshold
  4. Select largest cluster as the winner
  5. Calculate consensus as winner_cluster_size / total_responses
  6. Return representative answer from winning cluster

  ## Options

    * `:similarity_threshold` - Minimum similarity for grouping (default: 0.85)
    * `:similarity_metric` - Algorithm to use: :levenshtein, :jaccard, :cosine (default: :levenshtein)
    * `:normalization` - Pre-processing strategy (default: :lowercase_trim)

  ## Examples

      # Varied phrasing of same answer
      responses = [
        %{response: "The answer is 42", model: :gemini},
        %{response: "42", model: :openai},
        %{response: "Forty-two", model: :anthropic}
      ]

      {:ok, result} = SemanticSimilarity.aggregate(responses,
        similarity_threshold: 0.6,
        similarity_metric: :levenshtein
      )

      result.consensus
      # => 1.0 (all three recognized as equivalent)

      result.answer
      # => "42" (representative from cluster)

  """

  alias CrucibleEnsemble.{Normalize, Similarity}

  @default_threshold 0.85
  @default_metric :levenshtein

  @doc """
  Aggregate responses using semantic similarity clustering.

  Groups responses by similarity and selects the largest cluster as the winner.
  """
  @spec aggregate([map()], keyword()) :: {:ok, map()} | {:error, term()}
  def aggregate([], _opts), do: {:error, :no_responses}

  def aggregate(responses, opts) do
    normalization = Keyword.get(opts, :normalization, :lowercase_trim)
    threshold = Keyword.get(opts, :similarity_threshold, @default_threshold)
    metric = Keyword.get(opts, :similarity_metric, @default_metric)

    # Normalize all responses
    normalized_pairs =
      Enum.map(responses, fn resp ->
        normalized = Normalize.normalize_result(resp, normalization)
        {normalized, resp}
      end)

    # Extract just the normalized texts for clustering
    normalized_texts = Enum.map(normalized_pairs, fn {normalized, _} -> to_string(normalized) end)

    # Cluster by similarity
    clusters = Similarity.cluster_by_threshold(normalized_texts, threshold, metric)

    # Find the largest cluster (winner)
    winner_cluster =
      if Enum.empty?(clusters) do
        # Fallback: treat all as separate
        Enum.map(0..(length(responses) - 1), fn i -> [i] end)
        |> Enum.max_by(&length/1)
      else
        Enum.max_by(clusters, &length/1)
      end

    # Calculate consensus
    winner_size = length(winner_cluster)
    total = length(responses)
    consensus = winner_size / total

    # Get representative answer from winning cluster
    representative_idx = find_best_representative(winner_cluster, normalized_texts, metric)

    {representative_normalized, representative_response} =
      Enum.at(normalized_pairs, representative_idx)

    # Emit telemetry
    :telemetry.execute(
      [:crucible_ensemble, :vote, :complete],
      %{cluster_count: length(clusters), winner_size: winner_size},
      %{
        strategy: :semantic_similarity,
        consensus: consensus,
        threshold: threshold,
        metric: metric
      }
    )

    result_map = %{
      answer: representative_normalized,
      strategy: :semantic_similarity,
      consensus: consensus,
      clusters: clusters,
      winner_cluster: winner_cluster,
      total_responses: total,
      similarity_threshold: threshold,
      similarity_metric: metric,
      sample_response: representative_response
    }

    final_map =
      if Keyword.get(opts, :return_original_answer, false) do
        Map.put(result_map, :answer, Normalize.extract_response_text(representative_response))
      else
        result_map
      end

    {:ok, final_map}
  end

  # Private helper functions

  defp find_best_representative(cluster, texts, metric) do
    if length(cluster) == 1 do
      # Single item, return it
      hd(cluster)
    else
      # Find the text with highest average similarity to others in cluster
      {best_idx, _score} =
        cluster
        |> Enum.map(&calculate_text_avg_similarity(&1, cluster, texts, metric))
        |> Enum.max_by(fn {_i, score} -> score end)

      best_idx
    end
  end

  defp calculate_text_avg_similarity(i, cluster, texts, metric) do
    text_i = Enum.at(texts, i)
    other_indices = Enum.reject(cluster, &(&1 == i))

    avg_similarity =
      if Enum.empty?(other_indices) do
        1.0
      else
        other_indices
        |> Enum.map(fn j -> Similarity.compute(text_i, Enum.at(texts, j), metric) end)
        |> then(&(Enum.sum(&1) / length(&1)))
      end

    {i, avg_similarity}
  end
end
