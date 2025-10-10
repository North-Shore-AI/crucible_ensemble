defmodule CrucibleEnsemble.PricingTest do
  use ExUnit.Case, async: true
  alias CrucibleEnsemble.Pricing

  describe "calculate_cost/2" do
    test "calculates cost for standard usage format" do
      response = %{usage: %{input_tokens: 100, output_tokens: 50}}
      cost = Pricing.calculate_cost(:gemini_flash, response)

      # gemini_flash: input $0.10/1M, output $0.30/1M
      # input: 100/1_000_000 * 0.10 = 0.00001
      # output: 50/1_000_000 * 0.30 = 0.000015
      # total: 0.000025
      assert_in_delta cost, 0.000025, 0.0000001
    end

    test "calculates cost for OpenAI format" do
      response = %{usage: %{prompt_tokens: 100, completion_tokens: 50}}
      cost = Pricing.calculate_cost(:openai_gpt4o_mini, response)

      # openai_gpt4o_mini: input $0.15/1M, output $0.60/1M
      # input: 100/1_000_000 * 0.15 = 0.000015
      # output: 50/1_000_000 * 0.60 = 0.00003
      # total: 0.000045
      assert_in_delta cost, 0.000045, 0.0000001
    end

    test "handles zero tokens" do
      response = %{usage: %{input_tokens: 0, output_tokens: 0}}
      cost = Pricing.calculate_cost(:gemini_flash, response)
      assert cost == 0.0
    end

    test "handles missing usage data" do
      response = %{}
      cost = Pricing.calculate_cost(:gemini_flash, response)
      assert cost == 0.0
    end

    test "handles unknown model with default pricing" do
      response = %{usage: %{input_tokens: 100, output_tokens: 50}}
      cost = Pricing.calculate_cost(:unknown_model, response)
      assert cost == 0.0
    end
  end

  describe "calculate_cost_breakdown/2" do
    test "returns detailed cost breakdown" do
      response = %{usage: %{input_tokens: 100, output_tokens: 50}}
      breakdown = Pricing.calculate_cost_breakdown(:gemini_flash, response)

      assert breakdown.input_tokens == 100
      assert breakdown.output_tokens == 50
      assert_in_delta breakdown.input_cost, 0.00001, 0.0000001
      assert_in_delta breakdown.output_cost, 0.000015, 0.0000001
      assert breakdown.request_cost == 0.0
      assert_in_delta breakdown.total_usd, 0.000025, 0.0000001
    end
  end

  describe "get_prices/1" do
    test "returns pricing for known model" do
      prices = Pricing.get_prices(:gemini_flash)
      assert prices.input_per_1m == 0.10
      assert prices.output_per_1m == 0.30
      assert prices.per_request == 0.0
    end

    test "returns default pricing for unknown model" do
      prices = Pricing.get_prices(:unknown_model)
      assert prices.input_per_1m == 0.0
      assert prices.output_per_1m == 0.0
      assert prices.per_request == 0.0
    end
  end

  describe "aggregate_costs/1" do
    test "aggregates costs from multiple results" do
      results = [
        {:ok, %{model: :gemini_flash, cost: 0.0001}},
        {:ok, %{model: :openai_gpt4o_mini, cost: 0.0002}},
        {:ok, %{model: :anthropic_haiku, cost: 0.00015}}
      ]

      aggregate = Pricing.aggregate_costs(results)

      assert_in_delta aggregate.total_usd, 0.00045, 0.0000001
      assert aggregate.successful_responses == 3
      assert aggregate.failed_responses == 0
      assert_in_delta aggregate.cost_per_response, 0.00015, 0.0000001
      assert aggregate.per_model[:gemini_flash] == 0.0001
      assert aggregate.per_model[:openai_gpt4o_mini] == 0.0002
    end

    test "handles failed results" do
      results = [
        {:ok, %{model: :gemini_flash, cost: 0.0001}},
        {:error, %{model: :openai_gpt4o_mini}},
        {:ok, %{model: :anthropic_haiku, cost: 0.00015}}
      ]

      aggregate = Pricing.aggregate_costs(results)

      assert_in_delta aggregate.total_usd, 0.00025, 0.0000001
      assert aggregate.successful_responses == 2
      assert aggregate.failed_responses == 1
    end

    test "handles empty results" do
      aggregate = Pricing.aggregate_costs([])

      assert aggregate.total_usd == 0.0
      assert aggregate.successful_responses == 0
      assert aggregate.cost_per_response == 0.0
    end
  end

  describe "estimate_cost/3" do
    test "estimates cost for multiple models" do
      estimate =
        Pricing.estimate_cost(
          [:gemini_flash, :openai_gpt4o_mini],
          100,
          50
        )

      assert_in_delta estimate.total_usd, 0.00007, 0.0000001
      assert map_size(estimate.per_model) == 2
      assert_in_delta estimate.per_model[:gemini_flash], 0.000025, 0.0000001
      assert_in_delta estimate.per_model[:openai_gpt4o_mini], 0.000045, 0.0000001
    end

    test "handles single model" do
      estimate = Pricing.estimate_cost([:gemini_flash], 1000, 500)

      assert_in_delta estimate.total_usd, 0.00025, 0.0000001
      assert map_size(estimate.per_model) == 1
    end
  end
end
