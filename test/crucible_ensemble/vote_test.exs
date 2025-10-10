defmodule CrucibleEnsemble.VoteTest do
  use ExUnit.Case, async: true
  alias CrucibleEnsemble.Vote

  describe "apply_strategy/3 with :majority" do
    test "selects most common response" do
      responses = [
        %{response: "4", model: :model1},
        %{response: "4", model: :model2},
        %{response: "5", model: :model3}
      ]

      {:ok, result} = Vote.apply_strategy(responses, :majority)

      assert result.answer == "4"
      assert result.strategy == :majority
      assert result.consensus == 2 / 3
      assert result.votes["4"] == 2
      assert result.votes["5"] == 1
    end

    test "achieves 100% consensus when all agree" do
      responses = [
        %{response: "Paris", model: :model1},
        %{response: "Paris", model: :model2},
        %{response: "Paris", model: :model3}
      ]

      {:ok, result} = Vote.apply_strategy(responses, :majority)

      # normalized to lowercase
      assert result.answer == "paris"
      assert result.consensus == 1.0
    end

    test "handles case-insensitive voting" do
      responses = [
        %{response: "YES", model: :model1},
        %{response: "yes", model: :model2},
        %{response: "Yes", model: :model3}
      ]

      {:ok, result} = Vote.apply_strategy(responses, :majority, normalization: :lowercase_trim)

      assert result.answer == "yes"
      assert result.consensus == 1.0
    end

    test "returns error for empty responses" do
      assert {:error, :no_responses} = Vote.apply_strategy([], :majority)
    end
  end

  describe "apply_strategy/3 with :weighted" do
    test "weighs responses by confidence" do
      responses = [
        %{response: "A", model: :model1, confidence: 0.9},
        %{response: "B", model: :model2, confidence: 0.6},
        %{response: "B", model: :model3, confidence: 0.5}
      ]

      {:ok, result} = Vote.apply_strategy(responses, :weighted)

      # A has weight 0.9, B has weight 1.1 (0.6 + 0.5)
      assert result.answer == "b"
      assert result.strategy == :weighted
      assert result.scores["a"] == 0.9
      assert result.scores["b"] == 1.1
    end

    test "defaults to confidence 1.0 when not provided" do
      responses = [
        %{response: "A", model: :model1},
        %{response: "B", model: :model2}
      ]

      {:ok, result} = Vote.apply_strategy(responses, :weighted)

      assert result.scores["a"] == 1.0
      assert result.scores["b"] == 1.0
    end

    test "extracts confidence from metadata" do
      responses = [
        %{response: "A", model: :model1, metadata: %{confidence: 0.8}},
        %{response: "B", model: :model2, confidence: 0.5}
      ]

      {:ok, result} = Vote.apply_strategy(responses, :weighted)

      assert result.scores["a"] == 0.8
      assert result.scores["b"] == 0.5
    end
  end

  describe "apply_strategy/3 with :best_confidence" do
    test "selects highest confidence response" do
      responses = [
        %{response: "A", model: :model1, confidence: 0.6},
        %{response: "B", model: :model2, confidence: 0.9},
        %{response: "C", model: :model3, confidence: 0.7}
      ]

      {:ok, result} = Vote.apply_strategy(responses, :best_confidence)

      assert result.answer == "b"
      assert result.strategy == :best_confidence
      assert result.confidence == 0.9
      assert result.selected_model == :model2
    end

    test "works with single response" do
      responses = [
        %{response: "Only", model: :model1, confidence: 0.8}
      ]

      {:ok, result} = Vote.apply_strategy(responses, :best_confidence)

      assert result.answer == "only"
      assert result.confidence == 0.8
    end
  end

  describe "apply_strategy/3 with :unanimous" do
    test "succeeds when all models agree" do
      responses = [
        %{response: "4", model: :model1},
        %{response: "4", model: :model2},
        %{response: "4", model: :model3}
      ]

      {:ok, result} = Vote.apply_strategy(responses, :unanimous)

      assert result.answer == "4"
      assert result.strategy == :unanimous
      assert result.consensus == 1.0
    end

    test "fails when models disagree" do
      responses = [
        %{response: "4", model: :model1},
        %{response: "5", model: :model2},
        %{response: "4", model: :model3}
      ]

      {:error, error} = Vote.apply_strategy(responses, :unanimous)

      assert error.reason == :no_unanimous_consensus
      assert error.unique_answers == 2
      assert error.frequencies["4"] == 2
      assert error.frequencies["5"] == 1
    end

    test "handles normalization for unanimous check" do
      responses = [
        %{response: "YES", model: :model1},
        %{response: "yes", model: :model2},
        %{response: "Yes", model: :model3}
      ]

      {:ok, result} = Vote.apply_strategy(responses, :unanimous, normalization: :lowercase_trim)

      assert result.answer == "yes"
      assert result.consensus == 1.0
    end
  end

  describe "consensus_strength/1" do
    test "calculates consensus for vote distribution" do
      votes = %{"A" => 3, "B" => 1, "C" => 1}
      consensus = Vote.consensus_strength(votes)
      # 3/5
      assert consensus == 0.6
    end

    test "returns 1.0 for unanimous votes" do
      votes = %{"A" => 5}
      consensus = Vote.consensus_strength(votes)
      assert consensus == 1.0
    end

    test "returns 0.0 for empty distribution" do
      consensus = Vote.consensus_strength(%{})
      assert consensus == 0.0
    end

    test "handles tied votes" do
      votes = %{"A" => 2, "B" => 2}
      consensus = Vote.consensus_strength(votes)
      # 2/4
      assert consensus == 0.5
    end
  end

  describe "sufficient_consensus?/2" do
    test "returns true when consensus meets threshold" do
      result = %{consensus: 0.75}
      assert Vote.sufficient_consensus?(result, 0.5) == true
      assert Vote.sufficient_consensus?(result, 0.75) == true
    end

    test "returns false when consensus below threshold" do
      result = %{consensus: 0.4}
      assert Vote.sufficient_consensus?(result, 0.5) == false
    end

    test "handles exact threshold match" do
      result = %{consensus: 0.5}
      assert Vote.sufficient_consensus?(result, 0.5) == true
    end
  end
end
