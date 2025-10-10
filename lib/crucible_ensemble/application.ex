defmodule CrucibleEnsemble.Application do
  # See https://hexdocs.pm/elixir/Application.html
  # for more information on OTP Applications
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    # Attach telemetry handlers for metrics collection
    CrucibleEnsemble.Metrics.attach_handlers()

    children = [
      # Future: Add workers for connection pooling, circuit breakers, etc.
      # {CrucibleEnsemble.Worker, arg}
    ]

    # See https://hexdocs.pm/elixir/Supervisor.html
    # for other strategies and supported options
    opts = [strategy: :one_for_one, name: CrucibleEnsemble.Supervisor]
    Supervisor.start_link(children, opts)
  end
end
