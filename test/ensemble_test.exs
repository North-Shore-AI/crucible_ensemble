defmodule EnsembleTest do
  use ExUnit.Case
  doctest CrucibleEnsemble

  # Note: These are integration tests that would require actual API keys
  # For now, they serve as documentation of expected behavior

  @moduletag :integration

  describe "predict/2" do
    @tag :skip
    test "basic prediction with default settings" do
      # This would require actual API keys to run
      # {:ok, result} = CrucibleEnsemble.predict("What is 2+2?")
      # assert result.answer =~ "4"
      # assert result.metadata.consensus > 0.5
      # assert result.metadata.successes > 0
    end

    @tag :skip
    test "prediction with custom models" do
      # {:ok, result} = CrucibleEnsemble.predict(
      #   "What is the capital of France?",
      #   models: [:gemini_flash, :openai_gpt4o_mini]
      # )
      # assert result.answer =~ "Paris"
    end

    @tag :skip
    test "prediction with weighted voting" do
      # {:ok, result} = CrucibleEnsemble.predict(
      #   "Explain quantum computing in one sentence",
      #   strategy: :weighted,
      #   models: [:openai_gpt4o, :anthropic_sonnet]
      # )
      # assert is_binary(result.answer)
      # assert result.metadata.strategy == :weighted
    end
  end

  describe "predict_async/2" do
    @tag :skip
    test "asynchronous prediction" do
      # task = CrucibleEnsemble.predict_async("What is 5+5?")
      # {:ok, result} = Task.await(task, 10_000)
      # assert result.answer =~ "10"
    end
  end

  describe "predict_stream/2" do
    @tag :skip
    test "streaming prediction results" do
      # stream = CrucibleEnsemble.predict_stream("What is the meaning of life?")
      # results = Enum.to_list(stream)
      # assert length(results) > 0
    end
  end
end
