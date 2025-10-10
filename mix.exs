defmodule CrucibleEnsemble.MixProject do
  use Mix.Project

  @version "0.1.0"
  @source_url "https://github.com/North-Shore-AI/crucible_ensemble"

  def project do
    [
      app: :crucible_ensemble,
      version: @version,
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description: description(),
      package: package(),
      docs: docs(),
      source_url: @source_url,
      homepage_url: @source_url,
      name: "CrucibleEnsemble"
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      # Core dependencies
      # Note: req_llm is optional - the library provides a mock implementation for testing
      # {:req_llm, git: "https://github.com/calebjcourtney/req_llm.git", branch: "main"},
      {:jason, "~> 1.4"},
      {:telemetry, "~> 1.2"},

      # Development and testing
      {:ex_doc, "~> 0.31", only: :dev, runtime: false},
      {:mox, "~> 1.1", only: :test}
    ]
  end

  defp description do
    "Multi-model ensemble prediction with voting strategies for AI reliability. Leverages BEAM parallelism for massively concurrent LLM queries."
  end

  defp package do
    [
      name: "crucible_ensemble",
      description: description(),
      files: ~w(lib mix.exs README.md CHANGELOG.md LICENSE),
      licenses: ["MIT"],
      links: %{
        "GitHub" => @source_url,
        "Online documentation" => "https://hexdocs.pm/crucible_ensemble"
      },
      maintainers: ["nshkrdotcom"]
    ]
  end

  defp docs do
    [
      main: "readme",
      name: "CrucibleEnsemble",
      source_ref: "v#{@version}",
      source_url: @source_url,
      homepage_url: @source_url,
      extras: ["README.md", "CHANGELOG.md"],
      assets: %{"assets" => "assets"},
      logo: "assets/crucible_ensemble.svg",
      before_closing_head_tag: &mermaid_config/1
    ]
  end

  defp mermaid_config(:html) do
    """
    <script defer src="https://cdn.jsdelivr.net/npm/mermaid@10.2.3/dist/mermaid.min.js"></script>
    <script>
      let initialized = false;

      window.addEventListener("exdoc:loaded", () => {
        if (!initialized) {
          mermaid.initialize({
            startOnLoad: false,
            theme: document.body.className.includes("dark") ? "dark" : "default"
          });
          initialized = true;
        }

        let id = 0;
        for (const codeEl of document.querySelectorAll("pre code.mermaid")) {
          const preEl = codeEl.parentElement;
          const graphDefinition = codeEl.textContent;
          const graphEl = document.createElement("div");
          const graphId = "mermaid-graph-" + id++;
          mermaid.render(graphId, graphDefinition).then(({svg, bindFunctions}) => {
            graphEl.innerHTML = svg;
            bindFunctions?.(graphEl);
            preEl.insertAdjacentElement("afterend", graphEl);
            preEl.remove();
          });
        }
      });
    </script>
    """
  end

  defp mermaid_config(_), do: ""
end
