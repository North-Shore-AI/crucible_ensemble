defmodule CrucibleEnsemble.MetricsTest do
  use ExUnit.Case, async: true
  alias CrucibleEnsemble.Metrics

  describe "aggregate_stats/1" do
    test "calculates statistics from prediction data" do
      predictions = [
        %{duration_us: 1_000_000, total_cost: 0.001, consensus: 0.8},
        %{duration_us: 1_500_000, total_cost: 0.0015, consensus: 1.0},
        %{duration_us: 800_000, total_cost: 0.0008, consensus: 0.9}
      ]

      stats = Metrics.aggregate_stats(predictions)

      assert stats.total_predictions == 3
      assert_in_delta stats.avg_latency_ms, 1100.0, 1.0
      assert_in_delta stats.p50_latency_ms, 1000.0, 1.0
      assert_in_delta stats.total_cost, 0.0033, 0.0001
      assert_in_delta stats.avg_cost, 0.0011, 0.0001
      assert_in_delta stats.avg_consensus, 0.9, 0.01
      assert stats.success_rate == 100.0
    end

    test "handles empty prediction list" do
      stats = Metrics.aggregate_stats([])

      assert stats.total_predictions == 0
      assert stats.avg_latency_ms == 0.0
      assert stats.total_cost == 0.0
    end

    test "calculates percentiles correctly" do
      # Create 100 predictions with latencies from 1ms to 100ms
      predictions =
        Enum.map(1..100, fn i ->
          %{duration_us: i * 1_000, total_cost: 0.001, consensus: 1.0}
        end)

      stats = Metrics.aggregate_stats(predictions)

      assert_in_delta stats.p50_latency_ms, 50.0, 1.0
      assert_in_delta stats.p95_latency_ms, 95.0, 1.0
      assert_in_delta stats.p99_latency_ms, 99.0, 1.0
    end
  end

  describe "export_to_csv/2" do
    @tag :tmp_dir
    test "exports predictions to CSV format", %{tmp_dir: tmp_dir} do
      predictions = [
        %{
          timestamp: "2025-10-07T10:00:00Z",
          query: "What is 2+2?",
          models: [:model1, :model2],
          strategy: :majority,
          answer: "4",
          consensus: 1.0,
          duration_us: 1_000_000,
          total_cost: 0.001,
          successes: 2,
          failures: 0
        }
      ]

      path = Path.join(tmp_dir, "metrics.csv")
      assert :ok = Metrics.export_to_csv(predictions, path)
      assert File.exists?(path)

      content = File.read!(path)

      assert content =~
               "timestamp,query,models,strategy,answer,consensus,duration_ms,total_cost,successes,failures"

      assert content =~ "2025-10-07T10:00:00Z"
      assert content =~ "What is 2+2?"
      assert content =~ "model1;model2"
      assert content =~ "majority,4"
      assert content =~ "1.0"
      assert content =~ "0.001"
      assert content =~ "2,0"
    end
  end

  describe "summary_report/1" do
    test "generates human-readable summary" do
      predictions = [
        %{duration_us: 1_000_000, total_cost: 0.001, consensus: 0.8},
        %{duration_us: 1_500_000, total_cost: 0.0015, consensus: 1.0}
      ]

      report = Metrics.summary_report(predictions)

      assert report =~ "Ensemble Performance Summary"
      assert report =~ "Total Predictions: 2"
      assert report =~ "Average:"
      assert report =~ "P50"
      assert report =~ "P95"
      assert report =~ "P99"
      assert report =~ "Total Cost:"
      assert report =~ "Average Cost:"
      assert report =~ "Average Consensus:"
    end
  end

  describe "record_prediction/1" do
    test "emits telemetry event" do
      # Attach a test handler
      ref = make_ref()
      self_pid = self()

      :telemetry.attach(
        "test-prediction-handler",
        [:crucible_ensemble, :predict, :stop],
        fn _event, measurements, metadata, _config ->
          send(self_pid, {ref, measurements, metadata})
        end,
        nil
      )

      metadata = %{
        query: "test",
        models: [:model1],
        strategy: :majority,
        consensus: 0.8,
        duration_us: 1_000_000,
        total_cost: 0.001,
        successes: 1,
        failures: 0
      }

      Metrics.record_prediction(metadata)

      assert_receive {^ref, measurements, received_metadata}
      assert measurements.duration == 1_000_000
      assert received_metadata.consensus == 0.8

      :telemetry.detach("test-prediction-handler")
    end
  end

  describe "record_model_response/4" do
    test "emits success event" do
      ref = make_ref()
      self_pid = self()

      :telemetry.attach(
        "test-model-success-handler",
        [:crucible_ensemble, :model, :stop],
        fn _event, measurements, metadata, _config ->
          send(self_pid, {ref, :success, measurements, metadata})
        end,
        nil
      )

      Metrics.record_model_response(:model1, 500_000, 0.0005, true)

      assert_receive {^ref, :success, measurements, metadata}
      assert measurements.duration == 500_000
      assert metadata.model == :model1
      assert metadata.cost == 0.0005

      :telemetry.detach("test-model-success-handler")
    end

    test "emits failure event" do
      ref = make_ref()
      self_pid = self()

      :telemetry.attach(
        "test-model-failure-handler",
        [:crucible_ensemble, :model, :exception],
        fn _event, measurements, metadata, _config ->
          send(self_pid, {ref, :failure, measurements, metadata})
        end,
        nil
      )

      Metrics.record_model_response(:model1, 500_000, 0.0, false)

      assert_receive {^ref, :failure, measurements, metadata}
      assert measurements.duration == 500_000
      assert metadata.model == :model1
      assert metadata.success == false

      :telemetry.detach("test-model-failure-handler")
    end
  end
end
