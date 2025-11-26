defmodule CrucibleEnsemble.SimilarityTest do
  use ExUnit.Case, async: true
  alias CrucibleEnsemble.Similarity

  doctest CrucibleEnsemble.Similarity

  describe "levenshtein_similarity/2" do
    test "returns 1.0 for identical strings" do
      assert Similarity.levenshtein_similarity("hello", "hello") == 1.0
    end

    test "returns 0.0 for completely different strings of same length" do
      similarity = Similarity.levenshtein_similarity("abc", "xyz")
      assert similarity == 0.0
    end

    test "returns value between 0 and 1 for similar strings" do
      similarity = Similarity.levenshtein_similarity("hello", "hallo")
      assert similarity > 0.5
      assert similarity < 1.0
    end

    test "is case-sensitive by default" do
      similarity1 = Similarity.levenshtein_similarity("Hello", "hello")
      similarity2 = Similarity.levenshtein_similarity("hello", "hello")
      assert similarity1 < similarity2
    end

    test "handles empty strings" do
      assert Similarity.levenshtein_similarity("", "") == 1.0
      assert Similarity.levenshtein_similarity("abc", "") == 0.0
      assert Similarity.levenshtein_similarity("", "abc") == 0.0
    end

    test "normalizes to lowercase when option provided" do
      similarity = Similarity.levenshtein_similarity("Hello", "hello", normalize: true)
      assert similarity == 1.0
    end
  end

  describe "jaccard_similarity/2" do
    test "returns 1.0 for identical sets" do
      assert Similarity.jaccard_similarity("hello world", "hello world") == 1.0
    end

    test "returns 0.0 for disjoint sets" do
      assert Similarity.jaccard_similarity("abc", "xyz") == 0.0
    end

    test "calculates correct similarity for overlapping sets" do
      # "hello world" tokens: [hello, world]
      # "hello there" tokens: [hello, there]
      # intersection: [hello] = 1
      # union: [hello, world, there] = 3
      # jaccard = 1/3 ≈ 0.333
      similarity = Similarity.jaccard_similarity("hello world", "hello there")
      assert_in_delta similarity, 0.333, 0.01
    end

    test "is case-insensitive by default" do
      assert Similarity.jaccard_similarity("Hello", "hello") == 1.0
    end

    test "handles empty strings" do
      assert Similarity.jaccard_similarity("", "") == 1.0
      assert Similarity.jaccard_similarity("abc", "") == 0.0
    end
  end

  describe "cosine_similarity/2" do
    test "returns 1.0 for identical strings" do
      assert Similarity.cosine_similarity("hello world", "hello world") == 1.0
    end

    test "returns 0.0 for disjoint strings" do
      similarity = Similarity.cosine_similarity("abc", "xyz")
      assert_in_delta similarity, 0.0, 0.01
    end

    test "calculates similarity based on term frequency" do
      # Repeated terms should affect similarity
      similarity = Similarity.cosine_similarity("hello hello world", "hello world world")
      assert similarity > 0.0
      assert similarity < 1.0
    end

    test "is case-insensitive by default" do
      similarity = Similarity.cosine_similarity("Hello World", "hello world")
      assert_in_delta similarity, 1.0, 0.01
    end

    test "handles empty strings" do
      assert Similarity.cosine_similarity("", "") == 1.0
      assert Similarity.cosine_similarity("abc", "") == 0.0
    end
  end

  describe "compute/3" do
    test "delegates to levenshtein with :levenshtein metric" do
      result = Similarity.compute("hello", "hallo", :levenshtein)
      expected = Similarity.levenshtein_similarity("hello", "hallo")
      assert result == expected
    end

    test "delegates to jaccard with :jaccard metric" do
      result = Similarity.compute("hello world", "hello there", :jaccard)
      expected = Similarity.jaccard_similarity("hello world", "hello there")
      assert result == expected
    end

    test "delegates to cosine with :cosine metric" do
      result = Similarity.compute("hello world", "hello there", :cosine)
      expected = Similarity.cosine_similarity("hello world", "hello there")
      assert result == expected
    end

    test "defaults to levenshtein when invalid metric provided" do
      result = Similarity.compute("hello", "hallo", :invalid)
      expected = Similarity.levenshtein_similarity("hello", "hallo")
      assert result == expected
    end
  end

  describe "similarity_matrix/2" do
    test "creates symmetric matrix" do
      texts = ["hello", "hallo", "world"]
      matrix = Similarity.similarity_matrix(texts, :levenshtein)

      # Check symmetry
      assert Enum.at(Enum.at(matrix, 0), 1) == Enum.at(Enum.at(matrix, 1), 0)
      assert Enum.at(Enum.at(matrix, 0), 2) == Enum.at(Enum.at(matrix, 2), 0)
      assert Enum.at(Enum.at(matrix, 1), 2) == Enum.at(Enum.at(matrix, 2), 1)
    end

    test "has 1.0 on diagonal" do
      texts = ["hello", "world", "test"]
      matrix = Similarity.similarity_matrix(texts, :levenshtein)

      assert Enum.at(Enum.at(matrix, 0), 0) == 1.0
      assert Enum.at(Enum.at(matrix, 1), 1) == 1.0
      assert Enum.at(Enum.at(matrix, 2), 2) == 1.0
    end

    test "returns empty list for empty input" do
      assert Similarity.similarity_matrix([], :levenshtein) == []
    end

    test "works with single element" do
      matrix = Similarity.similarity_matrix(["hello"], :levenshtein)
      assert matrix == [[1.0]]
    end
  end

  describe "cluster_by_threshold/2" do
    test "groups identical texts into single cluster" do
      texts = ["hello", "hello", "hello"]
      clusters = Similarity.cluster_by_threshold(texts, 1.0, :levenshtein)

      assert length(clusters) == 1
      assert length(hd(clusters)) == 3
    end

    test "separates completely different texts" do
      texts = ["abc", "xyz", "123"]
      clusters = Similarity.cluster_by_threshold(texts, 0.9, :levenshtein)

      # Each should be in its own cluster
      assert length(clusters) == 3
    end

    test "groups similar texts with appropriate threshold" do
      texts = ["hello", "hallo", "hullo", "world"]
      # "hello", "hallo", "hullo" should cluster together
      # "world" should be separate
      clusters = Similarity.cluster_by_threshold(texts, 0.7, :levenshtein)

      assert length(clusters) == 2

      # Find the cluster sizes
      sizes = Enum.map(clusters, &length/1) |> Enum.sort()
      assert sizes == [1, 3]
    end

    test "returns single cluster for very low threshold" do
      texts = ["abc", "xyz", "123"]
      clusters = Similarity.cluster_by_threshold(texts, 0.0, :levenshtein)

      # All should be in one cluster at 0.0 threshold
      assert length(clusters) == 1
    end

    test "preserves original text indices" do
      texts = ["hello", "hallo", "world"]
      clusters = Similarity.cluster_by_threshold(texts, 0.7, :levenshtein)

      # Flatten all clusters and check all indices present
      all_indices = List.flatten(clusters)
      assert length(all_indices) == 3
      assert Enum.all?([0, 1, 2], fn i -> i in all_indices end)
    end

    test "handles empty input" do
      assert Similarity.cluster_by_threshold([], 0.8, :levenshtein) == []
    end
  end

  describe "find_representative/2" do
    test "returns first element for single-element cluster" do
      texts = ["hello"]
      assert Similarity.find_representative(texts, [0]) == "hello"
    end

    test "returns centroid element for multi-element cluster" do
      texts = ["hello", "hallo", "hullo", "world"]
      # The similar "h_llo" words
      cluster = [0, 1, 2]

      # Should return one of the three similar words (most central)
      representative = Similarity.find_representative(texts, cluster)
      assert representative in ["hello", "hallo", "hullo"]
    end

    test "handles cluster with single index" do
      texts = ["hello", "world"]
      assert Similarity.find_representative(texts, [1]) == "world"
    end
  end
end
