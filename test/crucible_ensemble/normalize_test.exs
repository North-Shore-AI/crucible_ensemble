defmodule CrucibleEnsemble.NormalizeTest do
  use ExUnit.Case, async: true
  alias CrucibleEnsemble.Normalize

  describe "normalize/2 with :lowercase_trim" do
    test "converts to lowercase and trims whitespace" do
      assert Normalize.normalize("  Hello World  ", :lowercase_trim) == "hello world"
      assert Normalize.normalize("UPPERCASE", :lowercase_trim) == "uppercase"
      assert Normalize.normalize("  spaces  ", :lowercase_trim) == "spaces"
    end

    test "handles empty strings" do
      assert Normalize.normalize("", :lowercase_trim) == ""
      assert Normalize.normalize("   ", :lowercase_trim) == ""
    end
  end

  describe "normalize/2 with :numeric" do
    test "extracts numeric values" do
      assert Normalize.normalize("The answer is 42", :numeric) == 42.0
      assert Normalize.normalize("Price: $1,234.56", :numeric) == 1234.56
      assert Normalize.normalize("3.14159", :numeric) == 3.14159
    end

    test "handles negative numbers" do
      assert Normalize.normalize("-42", :numeric) == -42.0
    end

    test "handles scientific notation" do
      assert Normalize.normalize("1.5e10", :numeric) == 1.5e10
    end

    test "returns nil when no number found" do
      assert Normalize.normalize("no numbers here", :numeric) == nil
    end
  end

  describe "normalize/2 with :json" do
    test "parses valid JSON" do
      result = Normalize.normalize(~s({"status": "ok", "value": 123}), :json)
      assert result == %{"status" => "ok", "value" => 123}
    end

    test "handles JSON in markdown code blocks" do
      markdown = """
      ```json
      {"result": "success"}
      ```
      """

      result = Normalize.normalize(markdown, :json)
      assert result == %{"result" => "success"}
    end

    test "returns original text for invalid JSON" do
      assert Normalize.normalize("not json", :json) == "not json"
    end
  end

  describe "normalize/2 with :boolean" do
    test "recognizes affirmative patterns" do
      assert Normalize.normalize("Yes", :boolean) == true
      assert Normalize.normalize("yes, that's correct", :boolean) == true
      assert Normalize.normalize("True", :boolean) == true
      assert Normalize.normalize("The answer is true", :boolean) == true
    end

    test "recognizes negative patterns" do
      assert Normalize.normalize("No", :boolean) == false
      assert Normalize.normalize("no, that's wrong", :boolean) == false
      assert Normalize.normalize("False", :boolean) == false
      assert Normalize.normalize("The answer is false", :boolean) == false
    end

    test "returns nil for uncertain responses" do
      assert Normalize.normalize("Maybe", :boolean) == nil
      assert Normalize.normalize("I don't know", :boolean) == nil
    end
  end

  describe "normalize/2 with custom function" do
    test "applies custom normalization function" do
      custom_fn = fn text -> String.upcase(text) end
      assert Normalize.normalize("hello", {:custom, custom_fn}) == "HELLO"
    end

    test "custom function can return any type" do
      custom_fn = fn _text -> 42 end
      assert Normalize.normalize("anything", {:custom, custom_fn}) == 42
    end
  end

  describe "extract_numeric/1" do
    test "extracts first number from text" do
      assert Normalize.extract_numeric("The answer is 42") == 42.0
      assert Normalize.extract_numeric("approximately 3.14159") == 3.14159
    end

    test "handles currency and formatting" do
      assert Normalize.extract_numeric("$1,234.56") == 1234.56
      assert Normalize.extract_numeric("€999.99") == 999.99
    end

    test "returns nil for non-numeric text" do
      assert Normalize.extract_numeric("no numbers") == nil
      assert Normalize.extract_numeric("") == nil
    end
  end

  describe "parse_json/1" do
    test "parses valid JSON objects" do
      result = Normalize.parse_json(~s({"key": "value"}))
      assert result == %{"key" => "value"}
    end

    test "parses valid JSON arrays" do
      result = Normalize.parse_json(~s([1, 2, 3]))
      assert result == [1, 2, 3]
    end

    test "extracts JSON from markdown" do
      markdown = "```\n{\"test\": true}\n```"
      result = Normalize.parse_json(markdown)
      assert result == %{"test" => true}
    end
  end

  describe "extract_boolean/1" do
    test "extracts boolean from various formats" do
      assert Normalize.extract_boolean("Yes") == true
      assert Normalize.extract_boolean("No") == false
      assert Normalize.extract_boolean("correct") == true
      assert Normalize.extract_boolean("incorrect") == false
    end
  end

  describe "text_similarity/2" do
    test "returns 1.0 for identical strings" do
      assert Normalize.text_similarity("hello", "hello") == 1.0
    end

    test "returns 0.0 for completely different strings" do
      assert Normalize.text_similarity("abc", "xyz") == 0.0
    end

    test "returns value between 0 and 1 for similar strings" do
      similarity = Normalize.text_similarity("hello", "hallo")
      assert similarity > 0.0 and similarity < 1.0
    end

    test "is case-insensitive" do
      assert Normalize.text_similarity("Hello", "hello") == 1.0
    end

    test "ignores leading/trailing whitespace" do
      assert Normalize.text_similarity("  hello  ", "hello") == 1.0
    end
  end

  describe "normalize_result/2" do
    test "extracts response from various result formats" do
      result1 = %{response: "Hello World"}
      assert Normalize.normalize_result(result1, :lowercase_trim) == "hello world"

      result2 = %{text: "Hello World"}
      assert Normalize.normalize_result(result2, :lowercase_trim) == "hello world"

      result3 = %{content: "Hello World"}
      assert Normalize.normalize_result(result3, :lowercase_trim) == "hello world"
    end

    test "applies normalization strategy" do
      result = %{response: "The answer is 42"}
      assert Normalize.normalize_result(result, :numeric) == 42.0
    end
  end

  describe "extract_response_text/1" do
    test "extracts text from :response key" do
      assert Normalize.extract_response_text(%{response: "test"}) == "test"
    end

    test "extracts text from :text key" do
      assert Normalize.extract_response_text(%{text: "test"}) == "test"
    end

    test "extracts text from :content key" do
      assert Normalize.extract_response_text(%{content: "test"}) == "test"
    end

    test "handles string keys" do
      assert Normalize.extract_response_text(%{"response" => "test"}) == "test"
      assert Normalize.extract_response_text(%{"text" => "test"}) == "test"
    end

    test "returns empty string for unknown format" do
      assert Normalize.extract_response_text(%{unknown: "test"}) == ""
      assert Normalize.extract_response_text(%{}) == ""
    end
  end
end
