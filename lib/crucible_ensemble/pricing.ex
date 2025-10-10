defmodule CrucibleEnsemble.Pricing do
  @moduledoc """
  Cost calculation and tracking for ensemble predictions.

  Provides accurate cost tracking based on token usage and model-specific pricing.
  Pricing data is current as of 2025-10.
  """

  @type model :: atom()
  @type cost_usd :: float()

  @type cost_breakdown :: %{
          total_usd: cost_usd(),
          input_cost: cost_usd(),
          output_cost: cost_usd(),
          request_cost: cost_usd(),
          input_tokens: non_neg_integer(),
          output_tokens: non_neg_integer()
        }

  @doc """
  Current pricing data per 1M tokens (as of 2025-10).

  Returns a map with model pricing information including:
  - `:input_per_1m` - Cost per 1M input tokens in USD
  - `:output_per_1m` - Cost per 1M output tokens in USD
  - `:per_request` - Per-request cost in USD (usually 0)
  """
  @prices %{
    # Google Gemini models
    gemini_flash: %{
      input_per_1m: 0.10,
      output_per_1m: 0.30,
      per_request: 0.0
    },
    gemini_pro: %{
      input_per_1m: 0.50,
      output_per_1m: 1.50,
      per_request: 0.0
    },

    # OpenAI models
    openai_gpt4o_mini: %{
      input_per_1m: 0.150,
      output_per_1m: 0.600,
      per_request: 0.0
    },
    openai_gpt4o: %{
      input_per_1m: 2.50,
      output_per_1m: 10.00,
      per_request: 0.0
    },
    openai_gpt4: %{
      input_per_1m: 30.00,
      output_per_1m: 60.00,
      per_request: 0.0
    },

    # Anthropic models
    anthropic_haiku: %{
      input_per_1m: 0.25,
      output_per_1m: 1.25,
      per_request: 0.0
    },
    anthropic_sonnet: %{
      input_per_1m: 3.00,
      output_per_1m: 15.00,
      per_request: 0.0
    },
    anthropic_opus: %{
      input_per_1m: 15.00,
      output_per_1m: 75.00,
      per_request: 0.0
    }
  }

  def prices, do: @prices

  @doc """
  Calculate cost for a single model response.

  ## Parameters

    * `model` - The model identifier (e.g., `:gemini_flash`)
    * `response` - The response map containing usage information

  ## Response Format

  The response should contain usage information in one of these formats:
  - `%{usage: %{input_tokens: N, output_tokens: M}}`
  - `%{usage: %{prompt_tokens: N, completion_tokens: M}}`
  - `%{metadata: %{usage: %{...}}}`

  ## Returns

  A float representing the total cost in USD.

  ## Examples

      iex> response = %{usage: %{input_tokens: 100, output_tokens: 50}}
      iex> CrucibleEnsemble.Pricing.calculate_cost(:gemini_flash, response)
      0.000025

  """
  @spec calculate_cost(model(), map()) :: cost_usd()
  def calculate_cost(model, response) do
    prices = get_prices(model)
    {input_tokens, output_tokens} = extract_tokens(response)

    input_cost = input_tokens / 1_000_000 * prices.input_per_1m
    output_cost = output_tokens / 1_000_000 * prices.output_per_1m
    request_cost = prices.per_request

    input_cost + output_cost + request_cost
  end

  @doc """
  Calculate detailed cost breakdown for a model response.

  Returns a map with detailed cost information including per-token costs
  and token counts.

  ## Examples

      iex> response = %{usage: %{input_tokens: 100, output_tokens: 50}}
      iex> CrucibleEnsemble.Pricing.calculate_cost_breakdown(:gemini_flash, response)
      %{
        total_usd: 0.000025,
        input_cost: 0.00001,
        output_cost: 0.000015,
        request_cost: 0.0,
        input_tokens: 100,
        output_tokens: 50
      }

  """
  @spec calculate_cost_breakdown(model(), map()) :: cost_breakdown()
  def calculate_cost_breakdown(model, response) do
    prices = get_prices(model)
    {input_tokens, output_tokens} = extract_tokens(response)

    input_cost = input_tokens / 1_000_000 * prices.input_per_1m
    output_cost = output_tokens / 1_000_000 * prices.output_per_1m
    request_cost = prices.per_request

    %{
      total_usd: input_cost + output_cost + request_cost,
      input_cost: input_cost,
      output_cost: output_cost,
      request_cost: request_cost,
      input_tokens: input_tokens,
      output_tokens: output_tokens
    }
  end

  @doc """
  Get pricing information for a specific model.

  ## Examples

      iex> CrucibleEnsemble.Pricing.get_prices(:gemini_flash)
      %{input_per_1m: 0.10, output_per_1m: 0.30, per_request: 0.0}

  """
  @spec get_prices(model()) :: map()
  def get_prices(model) do
    Map.get(@prices, model, %{
      input_per_1m: 0.0,
      output_per_1m: 0.0,
      per_request: 0.0
    })
  end

  @doc """
  Aggregate costs from multiple model results.

  ## Examples

      iex> results = [
      ...>   {:ok, %{model: :gemini_flash, cost: 0.0001}},
      ...>   {:ok, %{model: :openai_gpt4o_mini, cost: 0.0002}},
      ...>   {:error, %{model: :anthropic_haiku}}
      ...> ]
      iex> CrucibleEnsemble.Pricing.aggregate_costs(results)
      %{
        total_usd: 0.0003,
        per_model: %{gemini_flash: 0.0001, openai_gpt4o_mini: 0.0002},
        successful_responses: 2,
        failed_responses: 1,
        cost_per_response: 0.00015
      }

  """
  @spec aggregate_costs(list()) :: map()
  def aggregate_costs(results) do
    successes =
      Enum.filter(results, fn
        {:ok, _} -> true
        _ -> false
      end)

    failures = length(results) - length(successes)

    costs_by_model =
      successes
      |> Enum.map(fn {:ok, result} -> {result.model, result.cost} end)
      |> Enum.reduce(%{}, fn {model, cost}, acc ->
        Map.update(acc, model, cost, &(&1 + cost))
      end)

    total_cost =
      costs_by_model
      |> Map.values()
      |> Enum.sum()

    success_count = length(successes)

    %{
      total_usd: total_cost,
      per_model: costs_by_model,
      successful_responses: success_count,
      failed_responses: failures,
      cost_per_response: if(success_count > 0, do: total_cost / success_count, else: 0.0)
    }
  end

  @doc """
  Estimate cost for a query before execution.

  Provides rough cost estimation based on expected token counts.

  ## Parameters

    * `models` - List of models to use
    * `estimated_input_tokens` - Expected number of input tokens
    * `estimated_output_tokens` - Expected number of output tokens

  ## Examples

      iex> CrucibleEnsemble.Pricing.estimate_cost(
      ...>   [:gemini_flash, :openai_gpt4o_mini],
      ...>   100,
      ...>   50
      ...> )
      %{
        total_usd: 0.0000475,
        per_model: %{gemini_flash: 0.000025, openai_gpt4o_mini: 0.0000225}
      }

  """
  @spec estimate_cost([model()], non_neg_integer(), non_neg_integer()) :: map()
  def estimate_cost(models, estimated_input_tokens, estimated_output_tokens) do
    costs_by_model =
      models
      |> Enum.map(fn model ->
        prices = get_prices(model)
        input_cost = estimated_input_tokens / 1_000_000 * prices.input_per_1m
        output_cost = estimated_output_tokens / 1_000_000 * prices.output_per_1m
        request_cost = prices.per_request
        {model, input_cost + output_cost + request_cost}
      end)
      |> Map.new()

    total_cost =
      costs_by_model
      |> Map.values()
      |> Enum.sum()

    %{
      total_usd: total_cost,
      per_model: costs_by_model
    }
  end

  # Private helper functions

  defp extract_tokens(response) do
    # Try multiple common response formats
    cond do
      # Standard format: %{usage: %{input_tokens: N, output_tokens: M}}
      match?(%{usage: %{input_tokens: _, output_tokens: _}}, response) ->
        {response.usage.input_tokens, response.usage.output_tokens}

      # OpenAI format: %{usage: %{prompt_tokens: N, completion_tokens: M}}
      match?(%{usage: %{prompt_tokens: _, completion_tokens: _}}, response) ->
        {response.usage.prompt_tokens, response.usage.completion_tokens}

      # Nested in metadata: %{metadata: %{usage: %{...}}}
      match?(%{metadata: %{usage: %{input_tokens: _, output_tokens: _}}}, response) ->
        {response.metadata.usage.input_tokens, response.metadata.usage.output_tokens}

      match?(%{metadata: %{usage: %{prompt_tokens: _, completion_tokens: _}}}, response) ->
        {response.metadata.usage.prompt_tokens, response.metadata.usage.completion_tokens}

      # Default: no token information available
      true ->
        {0, 0}
    end
  end
end
