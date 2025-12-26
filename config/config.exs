import Config

# Disable the crucible_framework Repo (requires postgres) for crucible_ensemble
# since we only need the Stage behaviour and Context struct
config :crucible_framework, enable_repo: false

# Configure Logger with metadata keys used by CrucibleEnsemble.Metrics
config :logger, :default_formatter,
  metadata: [
    :model,
    :models,
    :query_length,
    :duration_ms,
    :consensus,
    :total_cost,
    :successes,
    :failures,
    :error,
    :strategy,
    :threshold,
    :cost
  ]
