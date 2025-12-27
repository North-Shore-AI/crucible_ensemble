defmodule CrucibleEnsemble.ConformanceTest do
  @moduledoc """
  Conformance tests for CrucibleEnsemble.Stage describe/1 contract.

  These tests ensure the stage implements the canonical schema format
  as defined in the Crucible.Stage behaviour contract.
  """
  use ExUnit.Case, async: true

  alias CrucibleEnsemble.Stage

  describe "stage conformance" do
    test "implements Crucible.Stage behaviour" do
      assert function_exported?(Stage, :run, 2)
      assert function_exported?(Stage, :describe, 1)
    end

    test "describe/1 returns valid canonical schema" do
      schema = Stage.describe(%{})

      # Name must be atom
      assert is_atom(schema.name)
      assert schema.name == :ensemble_voting

      # Required core fields
      assert Map.has_key?(schema, :description)
      assert is_binary(schema.description)
      assert Map.has_key?(schema, :required)
      assert is_list(schema.required)
      assert Map.has_key?(schema, :optional)
      assert is_list(schema.optional)
      assert Map.has_key?(schema, :types)
      assert is_map(schema.types)

      # No overlap between required and optional
      overlap =
        MapSet.intersection(
          MapSet.new(schema.required),
          MapSet.new(schema.optional)
        )

      assert MapSet.size(overlap) == 0,
             "Required and optional fields must not overlap, found: #{inspect(MapSet.to_list(overlap))}"

      # All required fields have types
      for key <- schema.required do
        assert Map.has_key?(schema.types, key),
               "Required field #{inspect(key)} missing from types"
      end

      # All optional fields have types
      for key <- schema.optional do
        assert Map.has_key?(schema.types, key),
               "Optional field #{inspect(key)} missing from types"
      end
    end

    test "describe/1 has schema version marker" do
      schema = Stage.describe(%{})

      assert Map.has_key?(schema, :__schema_version__)
      assert is_binary(schema.__schema_version__)
      assert schema.__schema_version__ == "1.0.0"
    end

    test "describe/1 has extensions for ensemble-specific metadata" do
      schema = Stage.describe(%{})

      assert Map.has_key?(schema, :__extensions__)
      assert is_map(schema.__extensions__)
      assert Map.has_key?(schema.__extensions__, :ensemble)

      ensemble_ext = schema.__extensions__.ensemble

      # Verify ensemble extension contains expected metadata
      assert Map.has_key?(ensemble_ext, :strategies)
      assert is_list(ensemble_ext.strategies)
      assert :majority in ensemble_ext.strategies
      assert :weighted in ensemble_ext.strategies

      assert Map.has_key?(ensemble_ext, :execution_modes)
      assert is_list(ensemble_ext.execution_modes)
      assert :parallel in ensemble_ext.execution_modes

      assert Map.has_key?(ensemble_ext, :config_type)
      assert ensemble_ext.config_type == CrucibleIR.Reliability.Ensemble

      assert Map.has_key?(ensemble_ext, :inputs)
      assert is_list(ensemble_ext.inputs)

      assert Map.has_key?(ensemble_ext, :outputs)
      assert is_list(ensemble_ext.outputs)
    end

    test "describe/1 defaults are valid" do
      schema = Stage.describe(%{})

      if Map.has_key?(schema, :defaults) do
        # All defaults must be for optional fields
        for key <- Map.keys(schema.defaults) do
          assert key in schema.optional,
                 "Default for #{inspect(key)} but #{inspect(key)} is not in optional"
        end
      end
    end

    test "describe/1 version field is present" do
      schema = Stage.describe(%{})

      assert Map.has_key?(schema, :version)
      assert is_binary(schema.version)
    end

    test "describe/1 types are valid type specs" do
      schema = Stage.describe(%{})

      for {key, type_spec} <- schema.types do
        assert valid_type_spec?(type_spec),
               "Invalid type spec for #{inspect(key)}: #{inspect(type_spec)}"
      end
    end
  end

  describe "describe/1 canonical format" do
    test "returns canonical schema format" do
      schema = Stage.describe(%{})

      # Core fields
      assert schema.name == :ensemble_voting
      assert is_binary(schema.description)
      assert is_list(schema.required)
      assert is_list(schema.optional)
      assert is_map(schema.types)

      # Optional fields moved to extensions
      assert Map.has_key?(schema, :__extensions__)
      assert Map.has_key?(schema.__extensions__, :ensemble)

      assert schema.__extensions__.ensemble.strategies == [
               :majority,
               :weighted,
               :best_confidence,
               :unanimous,
               :semantic_similarity,
               :ranked_choice
             ]
    end

    test "name is an atom" do
      schema = Stage.describe(%{})
      assert is_atom(schema.name)
    end
  end

  # Helper function to validate type specs
  defp valid_type_spec?(type_spec) when is_atom(type_spec) do
    type_spec in [:string, :integer, :float, :boolean, :atom, :map, :list, :module, :any]
  end

  defp valid_type_spec?({:struct, mod}) when is_atom(mod), do: true
  defp valid_type_spec?({:enum, values}) when is_list(values), do: true
  defp valid_type_spec?({:list, inner}), do: valid_type_spec?(inner)

  defp valid_type_spec?({:map, key_type, val_type}),
    do: valid_type_spec?(key_type) and valid_type_spec?(val_type)

  defp valid_type_spec?({:tuple, types}) when is_list(types),
    do: Enum.all?(types, &valid_type_spec?/1)

  defp valid_type_spec?({:function, arity}) when is_integer(arity) and arity >= 0, do: true

  defp valid_type_spec?({:union, types}) when is_list(types),
    do: Enum.all?(types, &valid_type_spec?/1)

  defp valid_type_spec?(_), do: false
end
