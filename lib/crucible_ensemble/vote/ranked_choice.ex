defmodule CrucibleEnsemble.Vote.RankedChoice do
  @moduledoc """
  Ranked choice voting: aggregate preferences using ranked ballots.

  Enables models to provide ranked preferences when multiple valid answers exist,
  then aggregates using instant-runoff voting or Borda count methods.

  This strategy is particularly effective for:
  - Multiple valid approaches to a problem
  - Design decisions with tradeoffs
  - Prioritization tasks
  - Scenarios where second-best choices matter

  ## Response Format

  Responses should include a `ranked_choices` field with an ordered list:

      %{
        response: "Option A",  # First choice
        ranked_choices: ["Option A", "Option B", "Option C"],
        model: :gemini_flash
      }

  If `ranked_choices` is not provided, only the first choice (`response`) is used.

  ## Ranking Methods

  ### Instant Runoff (IRV)

  1. Count first-choice votes for each candidate
  2. If any candidate has >50%, they win
  3. Otherwise, eliminate candidate with fewest votes
  4. Redistribute their votes to next choice
  5. Repeat until winner found

  ### Borda Count

  Points assigned by ranking position:
  - 1st choice: n-1 points
  - 2nd choice: n-2 points
  - ...
  - Last choice: 0 points

  Candidate with most total points wins.

  ## Options

    * `:ranking_method` - Algorithm: :instant_runoff or :borda_count (default: :instant_runoff)
    * `:normalization` - Pre-processing strategy (default: :lowercase_trim)
    * `:require_rankings` - Fail if models don't provide rankings (default: false)

  ## Examples

      responses = [
        %{response: "quicksort", ranked_choices: ["quicksort", "mergesort"], model: :model1},
        %{response: "mergesort", ranked_choices: ["mergesort", "quicksort"], model: :model2}
      ]

      {:ok, result} = RankedChoice.aggregate(responses, ranking_method: :instant_runoff)

      result.answer
      # => "quicksort" or "mergesort" (whichever wins runoff)

      result.rounds
      # => 2 (number of runoff rounds)

  """

  alias CrucibleEnsemble.Normalize

  @default_method :instant_runoff

  @doc """
  Aggregate responses using ranked choice voting.

  Supports instant-runoff and Borda count methods.
  """
  @spec aggregate([map()], keyword()) :: {:ok, map()} | {:error, term()}
  def aggregate([], _opts), do: {:error, :no_responses}

  def aggregate(responses, opts) do
    normalization = Keyword.get(opts, :normalization, :lowercase_trim)
    method = Keyword.get(opts, :ranking_method, @default_method)

    # Extract and normalize rankings from responses
    ballots = extract_ballots(responses, normalization)

    # Apply the chosen ranking method
    result =
      case method do
        :instant_runoff -> instant_runoff_voting(ballots)
        :borda_count -> borda_count_voting(ballots)
        _ -> instant_runoff_voting(ballots)
      end

    # Find sample response for winner
    winner = result.answer

    sample_response =
      Enum.find(responses, fn resp ->
        normalized = Normalize.normalize_result(resp, normalization)
        to_string(normalized) == winner
      end) || hd(responses)

    # Add metadata and return
    enhanced_result =
      Map.merge(result, %{
        strategy: :ranked_choice,
        method: method,
        total_responses: length(responses),
        sample_response: sample_response
      })
      |> maybe_return_original_answer(opts, sample_response)

    # Emit telemetry
    :telemetry.execute(
      [:crucible_ensemble, :vote, :complete],
      %{rounds: Map.get(result, :rounds, 1)},
      %{
        strategy: :ranked_choice,
        method: method,
        consensus: result.consensus
      }
    )

    {:ok, enhanced_result}
  end

  # Private helper functions

  defp extract_ballots(responses, normalization) do
    Enum.map(responses, fn resp ->
      # Normalize first choice
      first_choice =
        Normalize.normalize_result(resp, normalization)
        |> to_string()

      # Extract ranked choices if available
      ranked_choices =
        case Map.get(resp, :ranked_choices) do
          nil ->
            # No rankings provided, use only first choice
            [first_choice]

          choices when is_list(choices) ->
            # Normalize all choices
            Enum.map(choices, fn choice ->
              Normalize.normalize(to_string(choice), normalization)
              |> to_string()
            end)
            # Remove duplicates
            |> Enum.uniq()

          _ ->
            [first_choice]
        end

      %{
        choices: ranked_choices,
        original: resp
      }
    end)
  end

  defp instant_runoff_voting(ballots) do
    # Start with all choices
    candidates = get_all_candidates(ballots)

    # Run instant runoff rounds
    {winner, history, final_tally} = runoff_rounds(ballots, candidates, [], [])

    # Calculate consensus using the tally from the last competitive round
    consensus_tally =
      case history do
        [] -> final_tally
        [latest | _] -> latest.tally
      end

    winner_votes = Map.get(consensus_tally, winner, 0)
    total_votes = length(ballots)
    consensus = if total_votes > 0, do: winner_votes / total_votes, else: 1.0

    %{
      answer: winner,
      consensus: consensus,
      rounds: length(history),
      eliminated: Enum.reverse(history),
      round_tallies: get_round_tallies(history)
    }
  end

  defp runoff_rounds(ballots, candidates, eliminated, history) do
    # Count current round votes
    tally = count_votes(ballots, candidates)

    # Check for winner (>50% of votes)
    total_votes = length(ballots)
    majority = total_votes / 2

    winner =
      Enum.find(candidates, fn candidate ->
        Map.get(tally, candidate, 0) > majority
      end)

    cond do
      winner != nil ->
        # We have a winner!
        {winner, history, tally}

      length(candidates) == 1 ->
        # Only one candidate left
        {hd(candidates), history, tally}

      true ->
        # Eliminate candidate with fewest votes
        {loser, _votes} = Enum.min_by(tally, fn {_candidate, votes} -> votes end)

        new_candidates = List.delete(candidates, loser)
        new_eliminated = [loser | eliminated]
        new_history = [%{round: length(history) + 1, tally: tally, eliminated: loser} | history]

        # Continue to next round
        runoff_rounds(ballots, new_candidates, new_eliminated, new_history)
    end
  end

  defp count_votes(ballots, active_candidates) do
    # Count first-choice votes among active candidates
    ballots
    |> Enum.reduce(%{}, fn ballot, acc ->
      # Find first choice that's still active
      first_active =
        Enum.find(ballot.choices, fn choice ->
          choice in active_candidates
        end)

      if first_active do
        Map.update(acc, first_active, 1, &(&1 + 1))
      else
        acc
      end
    end)
    |> then(fn tally ->
      # Ensure all active candidates are in tally (even with 0 votes)
      Enum.reduce(active_candidates, tally, fn candidate, acc ->
        Map.put_new(acc, candidate, 0)
      end)
    end)
  end

  defp borda_count_voting(ballots) do
    # Calculate Borda scores
    scores =
      ballots
      |> Enum.reduce(%{}, fn ballot, acc ->
        # Assign points based on position
        ballot.choices
        |> Enum.with_index()
        |> Enum.reduce(acc, fn {choice, index}, acc2 ->
          points = length(ballot.choices) - index - 1
          Map.update(acc2, choice, points, &(&1 + points))
        end)
      end)

    # Find winner
    {winner, winner_score} = Enum.max_by(scores, fn {_choice, score} -> score end)

    # Calculate consensus (winner's score / max possible score)
    max_possible = length(ballots) * (length(Map.keys(scores)) - 1)
    consensus = if max_possible > 0, do: winner_score / max_possible, else: 1.0

    %{
      answer: winner,
      consensus: consensus,
      scores: scores,
      rounds: 1
    }
  end

  defp get_all_candidates(ballots) do
    ballots
    |> Enum.flat_map(fn ballot -> ballot.choices end)
    |> Enum.uniq()
  end

  defp get_round_tallies(history) do
    history
    |> Enum.reverse()
    |> Enum.map(fn round ->
      %{round: round.round, tally: round.tally, eliminated: round.eliminated}
    end)
  end

  defp maybe_return_original_answer(result, opts, sample_response) do
    if Keyword.get(opts, :return_original_answer, false) do
      Map.put(result, :answer, Normalize.extract_response_text(sample_response))
    else
      result
    end
  end
end
