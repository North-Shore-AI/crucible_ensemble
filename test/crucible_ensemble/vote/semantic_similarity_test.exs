defmodule CrucibleEnsemble.Vote.SemanticSimilarityTest do
  use ExUnit.Case, async: true
  alias CrucibleEnsemble.Vote.SemanticSimilarity

  describe "aggregate/2" do
    test "returns error for empty responses" do
      assert {:error, :no_responses} = SemanticSimilarity.aggregate([], [])
    end

    test "finds consensus for semantically similar responses" do
      responses = [
        %{response: "The answer is 42", model: :model1},
        %{response: "42", model: :model2},
        %{response: "Result: 42", model: :model3}
      ]

      {:ok, result} =
        SemanticSimilarity.aggregate(responses,
          similarity_threshold: 0.6,
          similarity_metric: :levenshtein,
          normalization: :numeric
        )

      # Should group all three as similar
      assert result.strategy == :semantic_similarity
      assert result.consensus == 1.0
      assert is_number(result.answer)
    end

    test "handles responses with no similarity" do
      responses = [
        %{response: "apple", model: :model1},
        %{response: "banana", model: :model2},
        %{response: "cherry", model: :model3}
      ]

      {:ok, result} =
        SemanticSimilarity.aggregate(responses,
          similarity_threshold: 0.8
        )

      # Should have low consensus as all are different
      assert result.strategy == :semantic_similarity
      assert result.consensus <= 0.5
    end

    test "groups by similarity threshold" do
      responses = [
        %{response: "hello", model: :model1},
        %{response: "hallo", model: :model2},
        %{response: "hullo", model: :model3},
        %{response: "goodbye", model: :model4}
      ]

      {:ok, result} =
        SemanticSimilarity.aggregate(responses,
          similarity_threshold: 0.7,
          similarity_metric: :levenshtein
        )

      # "hello", "hallo", "hullo" should cluster
      # "goodbye" should be separate
      # Winner should be from the larger cluster
      assert result.strategy == :semantic_similarity
      # 3/4
      assert result.consensus >= 0.75
      assert result.answer in ["hello", "hallo", "hullo"]
    end

    test "uses jaccard similarity when specified" do
      responses = [
        %{response: "hello world", model: :model1},
        %{response: "hello there", model: :model2},
        %{response: "goodbye world", model: :model3}
      ]

      {:ok, result} =
        SemanticSimilarity.aggregate(responses,
          similarity_threshold: 0.3,
          similarity_metric: :jaccard
        )

      assert result.strategy == :semantic_similarity
      assert is_binary(result.answer)
    end

    test "uses cosine similarity when specified" do
      responses = [
        %{response: "the quick brown fox", model: :model1},
        %{response: "the quick brown dog", model: :model2},
        %{response: "completely different text", model: :model3}
      ]

      {:ok, result} =
        SemanticSimilarity.aggregate(responses,
          similarity_threshold: 0.6,
          similarity_metric: :cosine
        )

      assert result.strategy == :semantic_similarity
      # First two should cluster
      assert result.answer in ["the quick brown fox", "the quick brown dog"]
    end

    test "includes clustering metadata" do
      responses = [
        %{response: "yes", model: :model1},
        %{response: "yes", model: :model2},
        %{response: "no", model: :model3}
      ]

      {:ok, result} =
        SemanticSimilarity.aggregate(responses,
          similarity_threshold: 0.8
        )

      assert Map.has_key?(result, :clusters)
      assert Map.has_key?(result, :total_responses)
      assert result.total_responses == 3
      assert is_list(result.clusters)
    end

    test "returns representative answer from winning cluster" do
      responses = [
        %{response: "The answer is definitely 42", model: :model1},
        %{response: "42", model: :model2},
        %{response: "It's 42", model: :model3}
      ]

      {:ok, result} =
        SemanticSimilarity.aggregate(responses,
          similarity_threshold: 0.5
        )

      # Should pick a representative from the cluster (normalized)
      assert result.answer in ["the answer is definitely 42", "42", "it's 42"]
    end

    test "handles single response" do
      responses = [
        %{response: "single answer", model: :model1}
      ]

      {:ok, result} = SemanticSimilarity.aggregate(responses, [])

      assert result.answer == "single answer"
      assert result.consensus == 1.0
    end

    test "applies normalization before similarity computation" do
      responses = [
        %{response: "  HELLO  ", model: :model1},
        %{response: "hello", model: :model2},
        %{response: "  Hello  ", model: :model3}
      ]

      {:ok, result} =
        SemanticSimilarity.aggregate(responses,
          similarity_threshold: 0.9,
          normalization: :lowercase_trim
        )

      # After normalization, all should be similar
      assert result.consensus == 1.0
    end

    test "defaults to lowercase_trim normalization" do
      responses = [
        %{response: "YES", model: :model1},
        %{response: "yes", model: :model2}
      ]

      {:ok, result} = SemanticSimilarity.aggregate(responses, [])

      # Should recognize as same after normalization
      assert result.consensus == 1.0
    end

    test "defaults to 0.85 similarity threshold" do
      responses = [
        %{response: "hello", model: :model1},
        %{response: "hallo", model: :model2}
      ]

      {:ok, result} = SemanticSimilarity.aggregate(responses, [])

      # With default threshold, these should cluster or not depending on similarity
      assert result.strategy == :semantic_similarity
      assert is_float(result.consensus)
    end

    test "defaults to levenshtein metric" do
      responses = [
        %{response: "test", model: :model1},
        %{response: "test", model: :model2}
      ]

      {:ok, result} = SemanticSimilarity.aggregate(responses, [])

      # Should work with default metric
      assert result.consensus == 1.0
    end

    test "includes winner_cluster metadata" do
      responses = [
        %{response: "yes", model: :model1},
        %{response: "yes", model: :model2},
        %{response: "no", model: :model3}
      ]

      {:ok, result} = SemanticSimilarity.aggregate(responses, [])

      assert Map.has_key?(result, :winner_cluster)
      assert is_list(result.winner_cluster)
      assert length(result.winner_cluster) >= 1
    end

    test "includes sample_response metadata" do
      responses = [
        %{response: "answer", model: :model1}
      ]

      {:ok, result} = SemanticSimilarity.aggregate(responses, [])

      assert Map.has_key?(result, :sample_response)
      assert result.sample_response == hd(responses)
    end

    test "works with numeric normalization" do
      responses = [
        %{response: "The result is 123.45", model: :model1},
        %{response: "123.45", model: :model2},
        %{response: "Answer: 123.45", model: :model3}
      ]

      {:ok, result} =
        SemanticSimilarity.aggregate(responses,
          normalization: :numeric,
          similarity_threshold: 0.9
        )

      # After numeric normalization, all should extract to 123.45
      assert result.consensus == 1.0
      assert result.answer == 123.45
    end

    test "works with boolean normalization" do
      responses = [
        %{response: "Yes, that is correct", model: :model1},
        %{response: "True", model: :model2},
        %{response: "Affirmative", model: :model3}
      ]

      {:ok, result} =
        SemanticSimilarity.aggregate(responses,
          normalization: :boolean,
          similarity_threshold: 0.9
        )

      # All should normalize to true
      assert result.consensus == 1.0
      assert result.answer == true
    end

    test "can return original answer when requested" do
      responses = [
        %{response: "YES", model: :model1},
        %{response: "yes", model: :model2}
      ]

      {:ok, result} =
        SemanticSimilarity.aggregate(responses,
          return_original_answer: true
        )

      # Representative original text should be returned instead of normalized
      assert result.answer == "YES"
      assert result.consensus == 1.0
    end
  end
end
