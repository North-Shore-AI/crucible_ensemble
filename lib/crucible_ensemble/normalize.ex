defmodule CrucibleEnsemble.Normalize do
  @moduledoc """
  Response normalization for ensemble voting.

  Provides strategies to normalize LLM responses into comparable forms,
  enabling accurate voting and consensus detection across different models.
  """

  @type strategy :: :lowercase_trim | :numeric | :json | :boolean | :custom
  @type normalized :: String.t() | number() | map() | boolean()

  @doc """
  Normalize a response using the specified strategy.

  ## Strategies

    * `:lowercase_trim` - Convert to lowercase and trim whitespace (default)
    * `:numeric` - Extract and parse numeric values
    * `:json` - Parse JSON responses
    * `:boolean` - Extract boolean yes/no answers
    * `{:custom, function}` - Use custom normalization function

  ## Examples

      iex> CrucibleEnsemble.Normalize.normalize("  Hello World  ", :lowercase_trim)
      "hello world"

      iex> CrucibleEnsemble.Normalize.normalize("The answer is 42.", :numeric)
      42.0

      iex> CrucibleEnsemble.Normalize.normalize(~s({"result": "success"}), :json)
      %{"result" => "success"}

      iex> CrucibleEnsemble.Normalize.normalize("Yes, that is correct.", :boolean)
      true

  """
  @spec normalize(String.t(), strategy() | {atom(), function()}) :: normalized()
  def normalize(response, strategy \\ :lowercase_trim)

  def normalize(response, :lowercase_trim) do
    response
    |> to_string()
    |> String.trim()
    |> String.downcase()
  end

  def normalize(response, :numeric) do
    extract_numeric(response)
  end

  def normalize(response, :json) do
    parse_json(response)
  end

  def normalize(response, :boolean) do
    extract_boolean(response)
  end

  def normalize(response, {:custom, fun}) when is_function(fun, 1) do
    fun.(response)
  end

  @doc """
  Extract numeric value from text response.

  Supports integers, floats, scientific notation, and common number formats.

  ## Examples

      iex> CrucibleEnsemble.Normalize.extract_numeric("The answer is 42")
      42.0

      iex> CrucibleEnsemble.Normalize.extract_numeric("Price: $1,234.56")
      1234.56

      iex> CrucibleEnsemble.Normalize.extract_numeric("approximately 3.14159")
      3.14159

      iex> CrucibleEnsemble.Normalize.extract_numeric("No numbers here")
      nil

  """
  @spec extract_numeric(String.t()) :: float() | nil
  def extract_numeric(text) when is_binary(text) do
    # Remove common currency symbols and thousands separators
    cleaned =
      text
      |> String.replace(~r/[$,€£¥]/, "")
      |> String.trim()

    # Try to match number patterns
    case Regex.run(~r/-?\d+\.?\d*(?:[eE][+-]?\d+)?/, cleaned) do
      [number_str] ->
        case Float.parse(number_str) do
          {num, _} -> num
          :error -> nil
        end

      nil ->
        nil
    end
  end

  def extract_numeric(_), do: nil

  @doc """
  Parse JSON response.

  Attempts to parse the response as JSON. If parsing fails, returns
  the original response.

  ## Examples

      iex> CrucibleEnsemble.Normalize.parse_json(~s({"status": "ok", "value": 123}))
      %{"status" => "ok", "value" => 123}

      iex> CrucibleEnsemble.Normalize.parse_json("not json")
      "not json"

  """
  @spec parse_json(String.t()) :: map() | list() | String.t()
  def parse_json(text) when is_binary(text) do
    # Try to find JSON in text (might be wrapped in markdown code blocks)
    json_text =
      case Regex.run(~r/```(?:json)?\s*(\{.*\}|\[.*\])\s*```/s, text) do
        [_, json] -> json
        nil -> text
      end

    case Jason.decode(json_text) do
      {:ok, decoded} -> decoded
      {:error, _} -> text
    end
  end

  def parse_json(data), do: data

  @doc """
  Extract boolean value from text response.

  Recognizes common affirmative and negative patterns.

  ## Examples

      iex> CrucibleEnsemble.Normalize.extract_boolean("Yes, that's correct.")
      true

      iex> CrucibleEnsemble.Normalize.extract_boolean("No, that's wrong.")
      false

      iex> CrucibleEnsemble.Normalize.extract_boolean("The answer is true.")
      true

      iex> CrucibleEnsemble.Normalize.extract_boolean("Maybe")
      nil

  """
  @spec extract_boolean(String.t()) :: boolean() | nil
  def extract_boolean(text) when is_binary(text) do
    normalized = String.downcase(String.trim(text))

    cond do
      # Uncertain patterns (check first)
      Regex.match?(~r/(maybe|perhaps|unsure|don't know|i don't|uncertain)/, normalized) ->
        nil

      # Affirmative patterns
      Regex.match?(~r/^(yes|true|correct|right|affirmative|1)\b/, normalized) ->
        true

      String.contains?(normalized, "yes") ->
        true

      String.contains?(normalized, "true") ->
        true

      # Negative patterns (but not part of "know", "not sure" etc.)
      Regex.match?(~r/^(no|false|incorrect|wrong|negative|0)\b/, normalized) ->
        false

      # Check for standalone "no" not in words like "know"
      Regex.match?(~r/\bno\b/, normalized) ->
        false

      String.contains?(normalized, "false") ->
        false

      # Uncertain
      true ->
        nil
    end
  end

  def extract_boolean(_), do: nil

  @doc """
  Calculate text similarity using Levenshtein distance.

  Returns a similarity score between 0.0 (completely different) and 1.0 (identical).

  ## Examples

      iex> CrucibleEnsemble.Normalize.text_similarity("hello", "hello")
      1.0

      iex> CrucibleEnsemble.Normalize.text_similarity("hello", "hallo")
      0.8

      iex> CrucibleEnsemble.Normalize.text_similarity("abc", "xyz")
      0.0

  """
  @spec text_similarity(String.t(), String.t()) :: float()
  def text_similarity(text1, text2) do
    text1 = String.downcase(String.trim(text1))
    text2 = String.downcase(String.trim(text2))

    if text1 == text2 do
      1.0
    else
      distance = levenshtein_distance(text1, text2)
      max_length = max(String.length(text1), String.length(text2))

      if max_length == 0 do
        1.0
      else
        1.0 - distance / max_length
      end
    end
  end

  @doc """
  Normalize response from a model result map.

  Extracts the response text from common model result formats.

  ## Examples

      iex> result = %{response: "Hello World", model: :gemini_flash}
      iex> CrucibleEnsemble.Normalize.normalize_result(result, :lowercase_trim)
      "hello world"

      iex> result = %{text: "The answer is 42", model: :openai}
      iex> CrucibleEnsemble.Normalize.normalize_result(result, :numeric)
      42.0

  """
  @spec normalize_result(map(), strategy()) :: normalized()
  def normalize_result(result, strategy \\ :lowercase_trim) do
    text = extract_response_text(result)
    normalize(text, strategy)
  end

  @doc """
  Extract response text from various model result formats.

  ## Examples

      iex> CrucibleEnsemble.Normalize.extract_response_text(%{response: "Hello"})
      "Hello"

      iex> CrucibleEnsemble.Normalize.extract_response_text(%{text: "World"})
      "World"

      iex> CrucibleEnsemble.Normalize.extract_response_text(%{content: "Test"})
      "Test"

  """
  @spec extract_response_text(map()) :: String.t()
  def extract_response_text(result) when is_map(result) do
    cond do
      Map.has_key?(result, :response) -> to_string(result.response)
      Map.has_key?(result, :text) -> to_string(result.text)
      Map.has_key?(result, :content) -> to_string(result.content)
      Map.has_key?(result, "response") -> to_string(result["response"])
      Map.has_key?(result, "text") -> to_string(result["text"])
      Map.has_key?(result, "content") -> to_string(result["content"])
      true -> ""
    end
  end

  def extract_response_text(_), do: ""

  # Private helper functions

  defp levenshtein_distance(string1, string2) do
    string1_chars = String.graphemes(string1)
    string2_chars = String.graphemes(string2)
    _length1 = length(string1_chars)
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
end
