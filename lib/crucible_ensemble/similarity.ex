defmodule CrucibleEnsemble.Similarity do
  @moduledoc """
  Text similarity algorithms for semantic comparison of responses.

  Provides multiple similarity metrics for comparing model responses,
  enabling semantic clustering and consensus detection beyond exact matching.

  ## Similarity Metrics

    * **Levenshtein** - Edit distance-based similarity, good for typos and minor variations
    * **Jaccard** - Set-based similarity using word overlap
    * **Cosine** - Vector space similarity based on term frequencies

  ## Usage

      iex> Similarity.levenshtein_similarity("hello", "hallo")
      0.8

      iex> Float.round(Similarity.jaccard_similarity("hello world", "hello there"), 3)
      0.333

      iex> Similarity.cluster_by_threshold(["hello", "hallo", "world"], 0.7, :levenshtein)
      [[0, 1], [2]]

  """

  @type similarity_metric :: :levenshtein | :jaccard | :cosine
  @type similarity_score :: float()
  @type cluster :: [non_neg_integer()]

  @doc """
  Calculate Levenshtein similarity between two strings.

  Returns a value between 0.0 (completely different) and 1.0 (identical).
  Based on normalized edit distance.

  ## Options

    * `:normalize` - Convert to lowercase before comparing (default: false)

  ## Examples

      iex> CrucibleEnsemble.Similarity.levenshtein_similarity("hello", "hello")
      1.0

      iex> CrucibleEnsemble.Similarity.levenshtein_similarity("hello", "hallo")
      0.8

      iex> CrucibleEnsemble.Similarity.levenshtein_similarity("Hello", "hello", normalize: true)
      1.0

  """
  @spec levenshtein_similarity(String.t(), String.t(), keyword()) :: similarity_score()
  def levenshtein_similarity(text1, text2, opts \\ []) do
    normalize = Keyword.get(opts, :normalize, false)

    {text1, text2} =
      if normalize do
        {String.downcase(text1), String.downcase(text2)}
      else
        {text1, text2}
      end

    # Handle edge cases
    cond do
      text1 == text2 ->
        1.0

      String.length(text1) == 0 and String.length(text2) == 0 ->
        1.0

      String.length(text1) == 0 or String.length(text2) == 0 ->
        0.0

      true ->
        distance = levenshtein_distance(text1, text2)
        max_length = max(String.length(text1), String.length(text2))
        1.0 - distance / max_length
    end
  end

  @doc """
  Calculate Jaccard similarity between two strings.

  Treats strings as sets of words (tokens) and computes the Jaccard index:
  |intersection| / |union|

  ## Examples

      iex> Float.round(CrucibleEnsemble.Similarity.jaccard_similarity("hello world", "hello there"), 3)
      0.333

      iex> CrucibleEnsemble.Similarity.jaccard_similarity("abc", "xyz")
      0.0

  """
  @spec jaccard_similarity(String.t(), String.t()) :: similarity_score()
  def jaccard_similarity(text1, text2) do
    # Tokenize and normalize
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)

    # Handle edge cases
    cond do
      MapSet.size(tokens1) == 0 and MapSet.size(tokens2) == 0 ->
        1.0

      MapSet.size(tokens1) == 0 or MapSet.size(tokens2) == 0 ->
        0.0

      true ->
        intersection = MapSet.intersection(tokens1, tokens2)
        union = MapSet.union(tokens1, tokens2)

        MapSet.size(intersection) / MapSet.size(union)
    end
  end

  @doc """
  Calculate cosine similarity between two strings.

  Treats strings as term frequency vectors and computes the cosine
  of the angle between them.

  ## Examples

      iex> CrucibleEnsemble.Similarity.cosine_similarity("hello world", "hello world")
      1.0

      iex> CrucibleEnsemble.Similarity.cosine_similarity("abc", "xyz")
      0.0

  """
  @spec cosine_similarity(String.t(), String.t()) :: similarity_score()
  def cosine_similarity(text1, text2) do
    # Build term frequency vectors
    vector1 = term_frequency_vector(text1)
    vector2 = term_frequency_vector(text2)

    # Handle edge cases
    if map_size(vector1) == 0 and map_size(vector2) == 0 do
      1.0
    else
      if map_size(vector1) == 0 or map_size(vector2) == 0 do
        0.0
      else
        # Calculate dot product and magnitudes
        dot_product = calculate_dot_product(vector1, vector2)
        magnitude1 = calculate_magnitude(vector1)
        magnitude2 = calculate_magnitude(vector2)

        if magnitude1 == 0.0 or magnitude2 == 0.0 do
          0.0
        else
          dot_product
          |> Kernel./(magnitude1 * magnitude2)
          |> Float.round(12)
          |> clamp(0.0, 1.0)
        end
      end
    end
  end

  @doc """
  Compute similarity using the specified metric.

  Convenience function that delegates to the appropriate similarity function.

  ## Examples

      iex> CrucibleEnsemble.Similarity.compute("hello", "hallo", :levenshtein)
      0.8

      iex> Float.round(CrucibleEnsemble.Similarity.compute("hello world", "hello there", :jaccard), 3)
      0.333

  """
  @spec compute(String.t(), String.t(), similarity_metric()) :: similarity_score()
  def compute(text1, text2, metric \\ :levenshtein) do
    case metric do
      :levenshtein -> levenshtein_similarity(text1, text2, normalize: true)
      :jaccard -> jaccard_similarity(text1, text2)
      :cosine -> cosine_similarity(text1, text2)
      _ -> levenshtein_similarity(text1, text2, normalize: true)
    end
  end

  @doc """
  Create a similarity matrix for a list of texts.

  Returns a 2D list (list of lists) where element [i][j] contains
  the similarity between texts[i] and texts[j].

  ## Examples

      iex> texts = ["hello", "hallo", "world"]
      iex> matrix = CrucibleEnsemble.Similarity.similarity_matrix(texts, :levenshtein)
      iex> Enum.at(Enum.at(matrix, 0), 0)
      1.0
      iex> Enum.at(Enum.at(matrix, 0), 1) == Enum.at(Enum.at(matrix, 1), 0)
      true

  """
  @spec similarity_matrix([String.t()], similarity_metric()) :: [[similarity_score()]]
  def similarity_matrix(texts, metric \\ :levenshtein) do
    if Enum.empty?(texts) do
      []
    else
      n = length(texts)
      texts_list = Enum.to_list(texts)

      for i <- 0..(n - 1) do
        for j <- 0..(n - 1) do
          if i == j do
            1.0
          else
            text1 = Enum.at(texts_list, i)
            text2 = Enum.at(texts_list, j)
            compute(text1, text2, metric)
          end
        end
      end
    end
  end

  @doc """
  Cluster texts by similarity threshold.

  Groups texts where similarity exceeds the threshold using a simple
  agglomerative approach.

  Returns a list of clusters, where each cluster is a list of indices
  into the original texts list.

  ## Parameters

    * `texts` - List of strings to cluster
    * `threshold` - Minimum similarity to group together (0.0-1.0)
    * `metric` - Similarity metric to use

  ## Examples

      iex> texts = ["hello", "hallo", "hullo", "world"]
      iex> CrucibleEnsemble.Similarity.cluster_by_threshold(texts, 0.7, :levenshtein)
      [[0, 1, 2], [3]]

  """
  @spec cluster_by_threshold([String.t()], float(), similarity_metric()) :: [cluster()]
  def cluster_by_threshold(texts, threshold, metric \\ :levenshtein) do
    if Enum.empty?(texts) do
      []
    else
      n = length(texts)
      matrix = similarity_matrix(texts, metric)

      # Start with each text in its own cluster
      initial_clusters = Enum.map(0..(n - 1), fn i -> [i] end)

      # Merge clusters iteratively
      merge_clusters(initial_clusters, matrix, threshold)
    end
  end

  @doc """
  Find the most representative text from a cluster.

  Returns the text that has the highest average similarity to all
  other texts in the cluster (the centroid).

  ## Examples

      iex> texts = ["hello", "hallo", "hullo"]
      iex> CrucibleEnsemble.Similarity.find_representative(texts, [0, 1, 2])
      "hello"  # Or whichever is most central

  """
  @spec find_representative([String.t()], cluster()) :: String.t()
  def find_representative(texts, cluster, metric \\ :levenshtein) do
    if length(cluster) == 1 do
      Enum.at(texts, hd(cluster))
    else
      # Calculate average similarity for each text in cluster
      scores =
        Enum.map(cluster, fn i ->
          text_i = Enum.at(texts, i)

          # Calculate average similarity to all other texts in cluster
          avg_similarity =
            cluster
            |> Enum.reject(&(&1 == i))
            |> Enum.map(fn j ->
              text_j = Enum.at(texts, j)
              compute(text_i, text_j, metric)
            end)
            |> then(fn similarities ->
              if Enum.empty?(similarities) do
                1.0
              else
                Enum.sum(similarities) / length(similarities)
              end
            end)

          {i, avg_similarity}
        end)

      # Find text with highest average similarity
      {best_index, _score} = Enum.max_by(scores, fn {_i, score} -> score end)
      Enum.at(texts, best_index)
    end
  end

  # Private helper functions

  defp levenshtein_distance(string1, string2) do
    # Use the algorithm from CrucibleEnsemble.Normalize
    string1_chars = String.graphemes(string1)
    string2_chars = String.graphemes(string2)
    length2 = length(string2_chars)

    # Initialize distance matrix
    initial_row = Enum.to_list(0..length2)
    initial_matrix = [initial_row]

    # Calculate distance using dynamic programming
    {final_matrix, _} =
      Enum.reduce(string1_chars, {initial_matrix, 0}, fn char1, {matrix, i} ->
        prev_row = hd(matrix)

        new_row =
          Enum.reduce(string2_chars, [i + 1], fn char2, acc ->
            j = length(acc) - 1
            cost = if char1 == char2, do: 0, else: 1

            deletion = Enum.at(prev_row, j + 1) + 1
            insertion = hd(acc) + 1
            substitution = Enum.at(prev_row, j) + cost

            [Enum.min([deletion, insertion, substitution]) | acc]
          end)
          |> Enum.reverse()

        {[new_row | matrix], i + 1}
      end)

    final_matrix
    |> hd()
    |> List.last()
  end

  defp tokenize(text) do
    text
    |> String.downcase()
    |> String.split(~r/\W+/, trim: true)
    |> MapSet.new()
  end

  defp term_frequency_vector(text) do
    text
    |> String.downcase()
    |> String.split(~r/\W+/, trim: true)
    |> Enum.frequencies()
  end

  defp calculate_dot_product(vector1, vector2) do
    # Sum of products of common terms
    vector1
    |> Enum.reduce(0.0, fn {term, freq1}, acc ->
      freq2 = Map.get(vector2, term, 0)
      acc + freq1 * freq2
    end)
  end

  defp calculate_magnitude(vector) do
    vector
    |> Map.values()
    |> Enum.reduce(0.0, fn freq, acc -> acc + freq * freq end)
    |> :math.sqrt()
  end

  defp merge_clusters(clusters, similarity_matrix, threshold) do
    # Nothing to merge when fewer than two clusters remain
    if length(clusters) < 2 do
      clusters
    else
      # Find clusters that can be merged
      case find_mergeable_clusters(clusters, similarity_matrix, threshold) do
        nil ->
          # No more clusters can be merged
          clusters

        {cluster1_idx, cluster2_idx} ->
          # Merge the two clusters
          cluster1 = Enum.at(clusters, cluster1_idx)
          cluster2 = Enum.at(clusters, cluster2_idx)
          merged = cluster1 ++ cluster2

          # Remove old clusters and add merged one
          new_clusters =
            clusters
            |> Enum.with_index()
            |> Enum.reject(fn {_cluster, idx} ->
              idx == cluster1_idx or idx == cluster2_idx
            end)
            |> Enum.map(fn {cluster, _idx} -> cluster end)
            |> then(fn remaining -> [merged | remaining] end)

          # Continue merging
          merge_clusters(new_clusters, similarity_matrix, threshold)
      end
    end
  end

  defp find_mergeable_clusters(clusters, similarity_matrix, threshold) do
    if length(clusters) < 2 do
      nil
    else
      # Find first pair of clusters that should be merged
      max_index = length(clusters) - 1

      cluster_pairs =
        for i <- 0..(max_index - 1), j <- (i + 1)..max_index do
          {i, j}
        end

      Enum.find_value(cluster_pairs, fn {i, j} ->
        cluster1 = Enum.at(clusters, i)
        cluster2 = Enum.at(clusters, j)

        if clusters_should_merge?(cluster1, cluster2, similarity_matrix, threshold) do
          {i, j}
        else
          nil
        end
      end)
    end
  end

  defp clusters_should_merge?(cluster1, cluster2, similarity_matrix, threshold) do
    # Check if any pair of texts between clusters exceeds threshold
    Enum.any?(cluster1, fn i ->
      Enum.any?(cluster2, fn j ->
        similarity = Enum.at(Enum.at(similarity_matrix, i), j)
        similarity >= threshold
      end)
    end)
  end

  defp clamp(value, min_value, max_value) do
    value
    |> max(min_value)
    |> min(max_value)
  end
end
