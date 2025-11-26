defmodule CrucibleEnsemble.Vote.RankedChoiceTest do
  use ExUnit.Case, async: true
  alias CrucibleEnsemble.Vote.RankedChoice

  describe "aggregate/2 with instant runoff" do
    test "returns error for empty responses" do
      assert {:error, :no_responses} = RankedChoice.aggregate([], [])
    end

    test "returns single response when only one provided" do
      responses = [
        %{response: "Option A", model: :model1}
      ]

      {:ok, result} = RankedChoice.aggregate(responses, ranking_method: :instant_runoff)

      assert result.answer == "option a"
      assert result.consensus == 1.0
      assert result.strategy == :ranked_choice
    end

    test "returns immediate winner with majority first-choice votes" do
      responses = [
        %{response: "Option A", ranked_choices: ["Option A", "Option B"], model: :model1},
        %{response: "Option A", ranked_choices: ["Option A", "Option B"], model: :model2},
        %{response: "Option B", ranked_choices: ["Option B", "Option A"], model: :model3}
      ]

      {:ok, result} = RankedChoice.aggregate(responses, ranking_method: :instant_runoff)

      assert result.answer == "option a"
      # Has majority
      assert result.consensus > 0.5
      # No runoff needed
      assert result.rounds == 0
    end

    test "performs instant runoff when no majority" do
      responses = [
        %{
          response: "Option A",
          ranked_choices: ["Option A", "Option B", "Option C"],
          model: :model1
        },
        %{
          response: "Option B",
          ranked_choices: ["Option B", "Option C", "Option A"],
          model: :model2
        },
        %{
          response: "Option C",
          ranked_choices: ["Option C", "Option B", "Option A"],
          model: :model3
        },
        %{
          response: "Option A",
          ranked_choices: ["Option A", "Option C", "Option B"],
          model: :model4
        }
      ]

      {:ok, result} = RankedChoice.aggregate(responses, ranking_method: :instant_runoff)

      # Should eliminate weakest and redistribute
      assert result.strategy == :ranked_choice
      assert is_binary(result.answer)
      assert result.rounds >= 1
      assert Map.has_key?(result, :eliminated)
    end

    test "handles responses without ranked_choices field" do
      responses = [
        %{response: "Option A", model: :model1},
        %{response: "Option B", model: :model2},
        %{response: "Option A", model: :model3}
      ]

      {:ok, result} = RankedChoice.aggregate(responses, ranking_method: :instant_runoff)

      # Should treat first choice as only choice
      assert result.answer == "option a"
      assert result.consensus > 0.5
    end

    test "includes elimination history" do
      responses = [
        %{response: "A", ranked_choices: ["A", "B"], model: :model1},
        %{response: "B", ranked_choices: ["B", "A"], model: :model2},
        %{response: "C", ranked_choices: ["C", "A"], model: :model3}
      ]

      {:ok, result} = RankedChoice.aggregate(responses, ranking_method: :instant_runoff)

      assert Map.has_key?(result, :eliminated)
      assert is_list(result.eliminated)
    end

    test "includes round-by-round tallies" do
      responses = [
        %{response: "A", ranked_choices: ["A", "B", "C"], model: :model1},
        %{response: "B", ranked_choices: ["B", "C", "A"], model: :model2},
        %{response: "C", ranked_choices: ["C", "A", "B"], model: :model3}
      ]

      {:ok, result} = RankedChoice.aggregate(responses, ranking_method: :instant_runoff)

      assert Map.has_key?(result, :round_tallies)
      assert is_list(result.round_tallies)
      assert length(result.round_tallies) == result.rounds
    end
  end

  describe "aggregate/2 with Borda count" do
    test "calculates Borda scores correctly" do
      responses = [
        %{response: "A", ranked_choices: ["A", "B", "C"], model: :model1},
        %{response: "B", ranked_choices: ["B", "A", "C"], model: :model2},
        %{response: "A", ranked_choices: ["A", "C", "B"], model: :model3}
      ]

      {:ok, result} = RankedChoice.aggregate(responses, ranking_method: :borda_count)

      # A gets: (2 + 1) + (1 + 0) + (2 + 1) = 7 points
      # B gets: (1 + 0) + (2 + 1) + (0 + 0) = 4 points
      # C gets: (0 + 0) + (0 + 0) + (1 + 0) = 1 point
      # A should win
      assert result.answer == "a"
      assert result.strategy == :ranked_choice
      assert result.method == :borda_count
      assert Map.has_key?(result, :scores)
    end

    test "handles partial rankings in Borda" do
      responses = [
        %{response: "A", ranked_choices: ["A", "B"], model: :model1},
        %{response: "B", ranked_choices: ["B"], model: :model2},
        %{response: "A", model: :model3}
      ]

      {:ok, result} = RankedChoice.aggregate(responses, ranking_method: :borda_count)

      # Should handle missing rankings gracefully
      assert result.answer in ["a", "b"]
      assert Map.has_key?(result, :scores)
    end
  end

  describe "normalization" do
    test "applies normalization to choices" do
      responses = [
        %{response: "  OPTION A  ", ranked_choices: ["  OPTION A  ", "Option B"], model: :model1},
        %{response: "option a", ranked_choices: ["option a", "option b"], model: :model2}
      ]

      {:ok, result} =
        RankedChoice.aggregate(responses,
          ranking_method: :instant_runoff,
          normalization: :lowercase_trim
        )

      # Should recognize as same option after normalization
      assert result.consensus == 1.0
    end
  end

  describe "consensus calculation" do
    test "calculates consensus based on final round" do
      responses = [
        %{response: "A", ranked_choices: ["A", "B"], model: :model1},
        %{response: "A", ranked_choices: ["A", "B"], model: :model2},
        %{response: "B", ranked_choices: ["B", "A"], model: :model3}
      ]

      {:ok, result} = RankedChoice.aggregate(responses, [])

      # 2 out of 3 voted for A
      assert_in_delta result.consensus, 0.666, 0.01
    end
  end

  describe "edge cases" do
    test "handles tie with equal first-choice votes" do
      responses = [
        %{response: "A", ranked_choices: ["A", "B"], model: :model1},
        %{response: "B", ranked_choices: ["B", "A"], model: :model2}
      ]

      {:ok, result} = RankedChoice.aggregate(responses, [])

      # Should pick one (deterministically)
      assert result.answer in ["a", "b"]
      assert result.consensus == 0.5
    end

    test "handles all different first choices" do
      responses = [
        %{response: "A", model: :model1},
        %{response: "B", model: :model2},
        %{response: "C", model: :model3},
        %{response: "D", model: :model4}
      ]

      {:ok, result} = RankedChoice.aggregate(responses, [])

      # Should complete runoff and pick winner
      assert result.answer in ["a", "b", "c", "d"]
      assert result.consensus <= 0.5
    end

    test "handles duplicate choices in rankings" do
      responses = [
        %{response: "A", ranked_choices: ["A", "A", "B"], model: :model1}
      ]

      {:ok, result} = RankedChoice.aggregate(responses, [])

      # Should handle gracefully (deduplicate or ignore)
      assert result.answer in ["a", "b"]
    end
  end

  describe "defaults" do
    test "defaults to instant_runoff method" do
      responses = [
        %{response: "A", model: :model1},
        %{response: "B", model: :model2}
      ]

      {:ok, result} = RankedChoice.aggregate(responses, [])

      assert result.method == :instant_runoff
    end

    test "defaults to lowercase_trim normalization" do
      responses = [
        %{response: "  OPTION A  ", model: :model1}
      ]

      {:ok, result} = RankedChoice.aggregate(responses, [])

      # Should normalize by default
      assert is_binary(result.answer)
    end
  end
end
