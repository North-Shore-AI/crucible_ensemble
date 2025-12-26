defmodule EnsembleTest do
  use ExUnit.Case
  doctest CrucibleEnsemble

  # Note: Executor uses a mock request but still requires API keys to be present.

  @moduletag :integration
  @default_models [:gemini_flash, :openai_gpt4o_mini, :anthropic_haiku]

  defp api_keys_for(models) do
    Map.new(models, fn model -> {model, "test-key"} end)
  end

  defp expected_answer(query) do
    String.downcase("Mock response: #{query}")
  end

  describe "predict/2" do
    test "basic prediction with default settings" do
      query = "What is 2+2?"
      api_keys = api_keys_for(@default_models)

      {:ok, result} = CrucibleEnsemble.predict(query, api_keys: api_keys)

      assert result.answer == expected_answer(query)
      assert result.metadata.consensus == 1.0
      assert result.metadata.successes == length(@default_models)
      assert result.metadata.failures == 0
      assert result.metadata.models_used == @default_models
    end

    test "prediction with custom models" do
      query = "What is the capital of France?"
      models = [:gemini_flash, :openai_gpt4o_mini]
      api_keys = api_keys_for(models)

      {:ok, result} = CrucibleEnsemble.predict(query, models: models, api_keys: api_keys)

      assert result.answer == expected_answer(query)
      assert result.metadata.models_used == models
      assert result.metadata.successes == length(models)
    end

    test "prediction with weighted voting" do
      query = "Explain quantum computing in one sentence"
      models = [:openai_gpt4o, :anthropic_sonnet]
      api_keys = api_keys_for(models)

      {:ok, result} =
        CrucibleEnsemble.predict(query,
          strategy: :weighted,
          models: models,
          api_keys: api_keys
        )

      assert result.answer == expected_answer(query)
      assert result.metadata.strategy == :weighted
      assert result.metadata.consensus == 1.0
    end
  end

  describe "predict_async/2" do
    test "asynchronous prediction" do
      query = "What is 5+5?"
      api_keys = api_keys_for(@default_models)

      task = CrucibleEnsemble.predict_async(query, api_keys: api_keys)
      {:ok, result} = Task.await(task, 10_000)

      assert result.answer == expected_answer(query)
      assert result.metadata.successes == length(@default_models)
    end
  end

  describe "predict_stream/2" do
    test "streaming prediction results" do
      query = "What is the meaning of life?"
      models = [:gemini_flash, :openai_gpt4o_mini]
      api_keys = api_keys_for(models)

      events =
        CrucibleEnsemble.predict_stream(query, models: models, api_keys: api_keys)
        |> Enum.to_list()

      response_events = Enum.filter(events, &match?({:response, _, _}, &1))
      assert length(response_events) == length(models)

      assert {:complete, final_result} = List.last(events)
      assert final_result.answer == expected_answer(query)
      assert final_result.strategy == :majority
    end
  end
end
