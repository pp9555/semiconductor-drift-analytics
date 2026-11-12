"use client";

import React, { useEffect, useState } from "react";
import {
  ComposedChart,
  Line,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
} from "recharts";
import {
  Activity,
  AlertTriangle,
  Cpu,
  Database,
  Layers,
  RotateCcw,
  ShieldCheck,
  Sliders,
  Sparkles,
  TrendingUp,
} from "lucide-react";

// const API_BASE =process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";
const API_BASE = "https://semiconductor-drift-analytics-6511273324.asia-south1.run.app";

interface TopSensor {
  sensor: string;
  importance: number;
  stage_type: string;
  golden_mean: number;
  ucl_3sigma: number;
  lcl_3sigma: number;
}

interface BatchSummary {
  Batch_File: string;
  wafer_count: number;
  avg_defect_prob: number;
  avg_drift_index: number;
  timestamp: string | null;
}

interface OverviewData {
  kpis: {
    total_production_wafers: number;
    excursion_alerts: number;
    excursion_rate_pct: number;
    severe_3sigma_drift_wafers: number;
    active_monitored_sensors: number;
    model_roc_auc: number;
    kaggle_baseline_roc_auc: number;
    live_model_loaded?: boolean;
  };
  top_sensors: TopSensor[];
  batches: BatchSummary[];
}

interface SpcPoint {
  wafer: string;
  batch: string;
  raw_value: number;
  ewma_value: number;
  defect_prob: number;
  is_3sigma_breach: boolean;
  breach_point?: number | null;
}

interface SpcResponse {
  sensor: string;
  limits: {
    golden_mean: number;
    golden_std: number;
    ucl_3sigma: number;
    lcl_3sigma: number;
  };
  series: SpcPoint[];
}

interface CorrelationPair {
  sensor_a: string;
  sensor_b: string;
  correlation: number;
}

interface WaferDiagnosis {
  wafer: string;
  batch: string;
  defect_probability: number;
  multivariate_drift_index: number;
  sensor_deviations: {
    sensor: string;
    raw_value: number;
    golden_mean: number;
    z_score: number;
    status: string;
  }[];
  ai_engineering_summary: string;
  ai_engine_source?: string;
}

interface LivePredictionResult {
  ensemble_defect_probability: number;
  teacher_prob: number;
  student_prob: number;
  predicted_status: string;
  live_multivariate_drift_index: number;
  top_5_z_scores: Record<string, number>;
}

export default function MicronDriftDashboard() {
  const [overview, setOverview] = useState<OverviewData | null>(null);
  const [selectedSensor, setSelectedSensor] = useState<string>("Sensor-130");
  const [selectedBatch, setSelectedBatch] = useState<string>("ALL");
  const [spcData, setSpcData] = useState<SpcResponse | null>(null);
  const [correlations, setCorrelations] = useState<CorrelationPair[]>([]);
  const [selectedWafer, setSelectedWafer] = useState<string>("");
  const [diagnosis, setDiagnosis] = useState<WaferDiagnosis | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  // What-If Live XGBoost Simulator State
  const [sliderValues, setSliderValues] = useState<Record<string, number>>({});
  const [liveResult, setLiveResult] = useState<LivePredictionResult | null>(
    null
  );
  const [simulating, setSimulating] = useState<boolean>(false);

  // 1. Fetch Overview & Correlations on Mount
  useEffect(() => {
    async function initDashboard() {
      try {
        const [ovRes, corrRes] = await Promise.all([
          fetch(`${API_BASE}/overview`),
          fetch(`${API_BASE}/inline-probe-correlation`),
        ]);
        const ovData: OverviewData = await ovRes.json();
        const corrData = await corrRes.json();

        setOverview(ovData);
        setCorrelations(corrData.strongest_pairs || []);

        if (ovData.top_sensors.length > 0) {
          setSelectedSensor(ovData.top_sensors[0].sensor);

          // Initialize What-If Sliders to Golden Wafer Means
          const initialSliders: Record<string, number> = {};
          ovData.top_sensors.slice(0, 5).forEach((s) => {
            initialSliders[s.sensor] = s.golden_mean;
          });
          setSliderValues(initialSliders);
        }
      } catch (err) {
        console.error("Failed to connect to FastAPI backend:", err);
      } finally {
        setLoading(false);
      }
    }
    initDashboard();
  }, []);

  // 2. Fetch SPC + EWMA Chart when Sensor or Batch changes
  useEffect(() => {
    if (!selectedSensor) return;
    async function fetchSpc() {
      try {
        const url = `${API_BASE}/spc-chart/${selectedSensor}?limit=200&batch_file=${selectedBatch}`;
        const res = await fetch(url);
        const data: SpcResponse = await res.json();

        const enrichedSeries = (data.series || []).map((pt) => ({
          ...pt,
          breach_point: pt.is_3sigma_breach ? pt.raw_value : null,
        }));

        setSpcData({ ...data, series: enrichedSeries });

        if (enrichedSeries.length > 0 && !selectedWafer) {
          const firstBreach =
            enrichedSeries.find((p) => p.is_3sigma_breach) || enrichedSeries[0];
          setSelectedWafer(firstBreach.wafer);
        }
      } catch (err) {
        console.error("Error loading SPC chart:", err);
      }
    }
    fetchSpc();
  }, [selectedSensor, selectedBatch]);

  // 3. Fetch AI Wafer Root-Cause Diagnosis when Selected Wafer changes
  useEffect(() => {
    if (!selectedWafer) return;
    async function fetchDiagnosis() {
      try {
        const res = await fetch(
          `${API_BASE}/ai-copilot-diagnosis/${selectedWafer}`
        );
        const data = await res.json();
        if (!data.error) {
          setDiagnosis(data);
          // Sync simulator sliders to the currently selected wafer's actual readings
          if (data.sensor_deviations) {
            const waferSliders: Record<string, number> = {};
            data.sensor_deviations.forEach(
              (d: { sensor: string; raw_value: number }) => {
                waferSliders[d.sensor] = d.raw_value;
              }
            );
            setSliderValues(waferSliders);
          }
        }
      } catch (err) {
        console.error("Error loading AI diagnosis:", err);
      }
    }
    fetchDiagnosis();
  }, [selectedWafer]);

  // 4. Call POST /api/predict-live whenever sliderValues change
  useEffect(() => {
    if (Object.keys(sliderValues).length === 0) return;
    const timer = setTimeout(async () => {
      setSimulating(true);
      try {
        const res = await fetch(`${API_BASE}/predict-live`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            base_wafer_id: selectedWafer || "Wafer-100",
            sensor_overrides: sliderValues,
          }),
        });
        const data: LivePredictionResult = await res.json();
        setLiveResult(data);
      } catch (err) {
        console.error("Live inference error:", err);
      } finally {
        setSimulating(false);
      }
    }, 150); // 150ms debounce for smooth slider dragging

    return () => clearTimeout(timer);
  }, [sliderValues, selectedWafer]);

  // Reset sliders to Golden Baseline Mean
  const handleResetToGolden = () => {
    if (!overview) return;
    const resetMap: Record<string, number> = {};
    overview.top_sensors.slice(0, 5).forEach((s) => {
      resetMap[s.sensor] = s.golden_mean;
    });
    setSliderValues(resetMap);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-950 text-slate-100 flex items-center justify-center">
        <div className="text-center space-y-3">
          <Cpu className="w-10 h-10 text-cyan-400 animate-spin mx-auto" />
          <p className="text-lg font-medium">
            Loading Micron ADTS Cell Metric Drift Telemetry...
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-6 font-sans">
      {/* HEADER */}
      <header className="max-w-7xl mx-auto mb-6 flex flex-col md:flex-row md:items-center md:justify-between border-b border-slate-800 pb-4 gap-4">
        <div>
          <div className="flex items-center gap-2 text-cyan-400 text-xs font-semibold uppercase tracking-wider">
            <Cpu className="w-4 h-4" />
            Micron ADTS ATI PI • Advanced Data Analytics Platform
          </div>
          <h1 className="text-2xl md:text-3xl font-bold text-white mt-1">
            Faster Cell Metrics Drift Detection & AI Diagnostics
          </h1>
          <p className="text-sm text-slate-400 mt-0.5">
            Live XGBoost Teacher-Student Inference • EWMA Control Charts •
            Hotelling / RMS-Z Drift • Gemini AI Co-Pilot
          </p>
        </div>

        <div className="flex items-center gap-3 bg-slate-900 border border-slate-800 px-4 py-2.5 rounded-lg">
          <ShieldCheck className="w-5 h-5 text-emerald-400" />
          <div className="text-xs">
            <div className="text-slate-400">
              Live Model Binaries Loaded (Held-Out Test)
            </div>
            <div className="font-semibold text-white">
              ROC-AUC: <span className="text-emerald-400">0.7441</span>{" "}
              <span className="text-slate-500">
                (vs 0.6190 Kaggle Baseline)
              </span>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto space-y-6">
        {/* KPI ROW */}
        {overview && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
              <div className="flex justify-between items-center text-slate-400 text-xs font-medium">
                <span>PRODUCTION WAFER STREAM</span>
                <Database className="w-4 h-4 text-cyan-400" />
              </div>
              <div className="text-2xl font-bold text-white mt-2">
                {overview.kpis.total_production_wafers.toLocaleString()}
              </div>
              <div className="text-xs text-slate-400 mt-1">
                Filtered from 590 raw →{" "}
                <span className="text-cyan-400 font-medium">
                  {overview.kpis.active_monitored_sensors} active inline sensors
                </span>
              </div>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
              <div className="flex justify-between items-center text-slate-400 text-xs font-medium">
                <span>EARLY EXCURSION ALERTS</span>
                <AlertTriangle className="w-4 h-4 text-amber-400" />
              </div>
              <div className="text-2xl font-bold text-amber-400 mt-2">
                {overview.kpis.excursion_alerts.toLocaleString()}{" "}
                <span className="text-sm font-normal text-slate-400">
                  ({overview.kpis.excursion_rate_pct}%)
                </span>
              </div>
              <div className="text-xs text-slate-400 mt-1">
                Flagged via Teacher-Student XGBoost Ensemble
              </div>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
              <div className="flex justify-between items-center text-slate-400 text-xs font-medium">
                <span>SEVERE MULTIVARIATE DRIFT (&gt;3σ)</span>
                <Activity className="w-4 h-4 text-rose-400" />
              </div>
              <div className="text-2xl font-bold text-rose-400 mt-2">
                {overview.kpis.severe_3sigma_drift_wafers.toLocaleString()}{" "}
                Wafers
              </div>
              <div className="text-xs text-slate-400 mt-1">
                RMS Z-Score breach vs Golden Wafer Baseline
              </div>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
              <div className="flex justify-between items-center text-slate-400 text-xs font-medium">
                <span>MINORITY PR-AUC LIFT</span>
                <TrendingUp className="w-4 h-4 text-emerald-400" />
              </div>
              <div className="text-2xl font-bold text-emerald-400 mt-2">
                +66.2% Lift
              </div>
              <div className="text-xs text-slate-400 mt-1">
                PR-AUC improved from 0.0958 → 0.1592
              </div>
            </div>
          </div>
        )}

        {/* MAIN CHART + AI INSPECTOR GRID */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* LEFT 2 COLS: EWMA + SPC CONTROL CHART */}
          <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-xl p-5">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-4">
              <div>
                <h2 className="text-lg font-bold text-white flex items-center gap-2">
                  <Activity className="w-5 h-5 text-cyan-400" />
                  Sequential Wafer Telemetry & EWMA Drift Tracker
                </h2>
                <p className="text-xs text-slate-400">
                  Tracks gradual tool degradation (EWMA λ=0.2) against ±3σ
                  Golden Wafer Control Limits. Click any point to inspect.
                </p>
              </div>

              <div className="flex flex-wrap items-center gap-2">
                <select
                  value={selectedSensor}
                  onChange={(e) => setSelectedSensor(e.target.value)}
                  className="bg-slate-950 border border-slate-700 text-xs text-slate-100 rounded-lg px-3 py-2"
                >
                  {overview?.top_sensors.map((s) => (
                    <option key={s.sensor} value={s.sensor}>
                      {s.sensor} ({s.stage_type.split(" ")[0]})
                    </option>
                  ))}
                </select>

                <select
                  value={selectedBatch}
                  onChange={(e) => setSelectedBatch(e.target.value)}
                  className="bg-slate-950 border border-slate-700 text-xs text-slate-100 rounded-lg px-3 py-2"
                >
                  <option value="ALL">
                    All Chronological Batches (200 Wafers)
                  </option>
                  {overview?.batches.slice(0, 20).map((b) => (
                    <option key={b.Batch_File} value={b.Batch_File}>
                      {b.Batch_File} ({b.wafer_count} wafers)
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {spcData && (
              <div className="h-80 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart
                    data={spcData.series}
                    onClick={(e: any) => {
                      if (e?.activePayload?.[0]?.payload?.wafer) {
                        setSelectedWafer(e.activePayload[0].payload.wafer);
                      }
                    }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis
                      dataKey="wafer"
                      tick={{ fill: "#94a3b8", fontSize: 11 }}
                      interval={24}
                    />
                    <YAxis
                      domain={["auto", "auto"]}
                      tick={{ fill: "#94a3b8", fontSize: 11 }}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#0f172a",
                        borderColor: "#334155",
                        fontSize: "12px",
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: "12px" }} />

                    <ReferenceLine
                      y={spcData.limits.ucl_3sigma}
                      stroke="#f43f5e"
                      strokeDasharray="4 4"
                      label={{
                        value: "UCL (+3σ)",
                        fill: "#f43f5e",
                        fontSize: 11,
                        position: "insideTopRight",
                      }}
                    />
                    <ReferenceLine
                      y={spcData.limits.golden_mean}
                      stroke="#10b981"
                      strokeDasharray="2 2"
                      label={{
                        value: "Golden Mean",
                        fill: "#10b981",
                        fontSize: 11,
                        position: "insideBottomRight",
                      }}
                    />
                    <ReferenceLine
                      y={spcData.limits.lcl_3sigma}
                      stroke="#f43f5e"
                      strokeDasharray="4 4"
                      label={{
                        value: "LCL (-3σ)",
                        fill: "#f43f5e",
                        fontSize: 11,
                        position: "insideBottomRight",
                      }}
                    />

                    <Line
                      type="monotone"
                      dataKey="raw_value"
                      name="Raw Sensor Reading"
                      stroke="#475569"
                      strokeWidth={1.2}
                      dot={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="ewma_value"
                      name="EWMA Drift Trend (λ=0.2)"
                      stroke="#22d3ee"
                      strokeWidth={2.5}
                      dot={false}
                    />
                    <Scatter
                      dataKey="breach_point"
                      name="3σ Control Limit Breach"
                      fill="#f43f5e"
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>

          {/* RIGHT COL: AI WAFER ROOT-CAUSE CO-PILOT */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-2">
                <h2 className="text-lg font-bold text-white flex items-center gap-2">
                  <Sparkles className="w-5 h-5 text-amber-400" />
                  AI Excursion Co-Pilot
                </h2>
                <select
                  value={selectedWafer}
                  onChange={(e) => setSelectedWafer(e.target.value)}
                  className="bg-slate-950 border border-slate-700 text-xs text-cyan-400 font-mono rounded px-2 py-1"
                >
                  {spcData?.series.map((pt) => (
                    <option key={pt.wafer} value={pt.wafer}>
                      {pt.wafer} {pt.is_3sigma_breach ? "⚠️" : ""}
                    </option>
                  ))}
                </select>
              </div>

              {diagnosis && (
                <div className="space-y-3">
                  {diagnosis.ai_engine_source && (
                    <div className="text-[10px] font-mono text-emerald-400 bg-emerald-950/40 border border-emerald-800/50 px-2.5 py-1 rounded inline-block">
                      Engine: {diagnosis.ai_engine_source}
                    </div>
                  )}

                  <div className="grid grid-cols-2 gap-3 bg-slate-950 p-3 rounded-lg border border-slate-800">
                    <div>
                      <div className="text-[11px] text-slate-400">
                        XGBoost Excursion Prob
                      </div>
                      <div className="text-lg font-bold text-amber-400">
                        {(diagnosis.defect_probability * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div>
                      <div className="text-[11px] text-slate-400">
                        Multivariate Drift (MDI)
                      </div>
                      <div
                        className={`text-lg font-bold ${
                          diagnosis.multivariate_drift_index >= 2.0
                            ? "text-rose-400"
                            : "text-emerald-400"
                        }`}
                      >
                        {diagnosis.multivariate_drift_index.toFixed(2)}σ
                      </div>
                    </div>
                  </div>

                  <div className="bg-cyan-950/30 border border-cyan-800/50 rounded-lg p-3 text-xs text-cyan-100 leading-relaxed">
                    <span className="font-semibold text-cyan-400 block mb-1">
                      Automated Root-Cause Diagnosis:
                    </span>
                    {diagnosis.ai_engineering_summary}
                  </div>

                  <div>
                    <div className="text-xs font-semibold text-slate-300 mb-2">
                      Top 5 Root-Cause Sensor Deviations (vs Golden Baseline):
                    </div>
                    <div className="space-y-1.5">
                      {diagnosis.sensor_deviations.map((dev) => (
                        <div
                          key={dev.sensor}
                          className="flex items-center justify-between text-xs bg-slate-950 px-3 py-1.5 rounded border border-slate-800"
                        >
                          <div>
                            <span className="font-mono font-semibold text-white">
                              {dev.sensor}
                            </span>
                            <span className="text-slate-500 ml-2">
                              ({dev.raw_value} vs {dev.golden_mean})
                            </span>
                          </div>
                          <span
                            className={`font-mono font-bold px-2 py-0.5 rounded ${
                              Math.abs(dev.z_score) > 3
                                ? "bg-rose-950 text-rose-400 border border-rose-800"
                                : Math.abs(dev.z_score) > 2
                                ? "bg-amber-950 text-amber-400 border border-amber-800"
                                : "bg-emerald-950 text-emerald-400"
                            }`}
                          >
                            {dev.z_score > 0 ? `+${dev.z_score}` : dev.z_score}σ
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* NEW PANEL: INTERACTIVE WHAT-IF CHAMBER DRIFT SIMULATOR (LIVE XGBOOST INFERENCE) */}
        {overview && (
          <div className="bg-slate-900 border border-cyan-800/60 rounded-xl p-5 shadow-lg">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-4 mb-5">
              <div>
                <div className="flex items-center gap-2">
                  <Sliders className="w-5 h-5 text-cyan-400" />
                  <h2 className="text-lg font-bold text-white">
                    Live "What-If" Chamber Drift Simulator (Real-Time XGBoost
                    Inference)
                  </h2>
                  <span className="text-[11px] bg-cyan-950 text-cyan-300 border border-cyan-700 px-2 py-0.5 rounded font-mono">
                    POST /api/predict-live
                  </span>
                </div>
                <p className="text-xs text-slate-400 mt-1">
                  Drag the inline sensor sliders below to simulate chamber
                  parameter drift and watch the loaded Teacher-Student XGBoost
                  binaries re-score the wafer in real time.
                </p>
              </div>

              <button
                onClick={handleResetToGolden}
                className="flex items-center gap-1.5 text-xs bg-slate-800 hover:bg-slate-700 text-slate-200 px-3 py-2 rounded-lg border border-slate-700 transition self-start"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                Reset to Golden Wafer Baseline
              </button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-center">
              {/* LEFT 2 COLS: TOP 5 SENSOR SLIDERS */}
              <div className="lg:col-span-2 grid grid-cols-1 sm:grid-cols-2 gap-4">
                {overview.top_sensors.slice(0, 5).map((sensorMeta) => {
                  const std =
                    (sensorMeta.ucl_3sigma - sensorMeta.golden_mean) / 3 || 1;
                  const minVal = Number(
                    (sensorMeta.golden_mean - 4 * std).toFixed(3)
                  );
                  const maxVal = Number(
                    (sensorMeta.golden_mean + 4 * std).toFixed(3)
                  );
                  const stepVal = Number(((maxVal - minVal) / 100).toFixed(4)) || 0.01;
                  const currentVal =
                    sliderValues[sensorMeta.sensor] ?? sensorMeta.golden_mean;
                  const liveZ =
                    liveResult?.top_5_z_scores?.[sensorMeta.sensor] ?? 0;

                  return (
                    <div
                      key={sensorMeta.sensor}
                      className="bg-slate-950 border border-slate-800 rounded-lg p-3.5"
                    >
                      <div className="flex justify-between items-center text-xs mb-1.5">
                        <span className="font-mono font-bold text-white">
                          {sensorMeta.sensor}
                        </span>
                        <span
                          className={`font-mono text-[11px] px-1.5 py-0.5 rounded ${
                            Math.abs(liveZ) > 3
                              ? "bg-rose-950 text-rose-400"
                              : Math.abs(liveZ) > 2
                              ? "bg-amber-950 text-amber-400"
                              : "bg-emerald-950 text-emerald-400"
                          }`}
                        >
                          {liveZ > 0 ? `+${liveZ}` : liveZ}σ
                        </span>
                      </div>

                      <input
                        type="range"
                        min={minVal}
                        max={maxVal}
                        step={stepVal}
                        value={currentVal}
                        onChange={(e) =>
                          setSliderValues((prev) => ({
                            ...prev,
                            [sensorMeta.sensor]: parseFloat(e.target.value),
                          }))
                        }
                        className="w-full accent-cyan-400 cursor-pointer"
                      />

                      <div className="flex justify-between text-[11px] text-slate-400 mt-1 font-mono">
                        <span>LCL: {sensorMeta.lcl_3sigma}</span>
                        <span className="text-cyan-300 font-semibold">
                          Val: {Number(currentVal).toFixed(2)}
                        </span>
                        <span>UCL: {sensorMeta.ucl_3sigma}</span>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* RIGHT COL: LIVE XGBOOST ENSEMBLE GAUGE */}
              <div className="bg-slate-950 border border-slate-800 rounded-xl p-5 space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                    Live Model Telemetry
                  </span>
                  <span className="text-[11px] font-mono text-cyan-400">
                    {simulating ? "Scoring..." : "Active"}
                  </span>
                </div>

                {liveResult && (
                  <>
                    <div className="grid grid-cols-2 gap-3">
                      <div className="bg-slate-900 p-3 rounded-lg border border-slate-800">
                        <div className="text-[11px] text-slate-400">
                          Ensemble Excursion Prob
                        </div>
                        <div
                          className={`text-2xl font-bold mt-1 ${
                            liveResult.predicted_status === "Excursion_Alert"
                              ? "text-rose-400"
                              : "text-emerald-400"
                          }`}
                        >
                          {(
                            liveResult.ensemble_defect_probability * 100
                          ).toFixed(1)}
                          %
                        </div>
                      </div>

                      <div className="bg-slate-900 p-3 rounded-lg border border-slate-800">
                        <div className="text-[11px] text-slate-400">
                          Simulated MDI (RMS-Z)
                        </div>
                        <div
                          className={`text-2xl font-bold mt-1 ${
                            liveResult.live_multivariate_drift_index >= 2.0
                              ? "text-amber-400"
                              : "text-cyan-400"
                          }`}
                        >
                          {liveResult.live_multivariate_drift_index.toFixed(2)}σ
                        </div>
                      </div>
                    </div>

                    <div className="space-y-1.5 text-xs bg-slate-900 p-3 rounded-lg border border-slate-800 font-mono">
                      <div className="flex justify-between">
                        <span className="text-slate-400">
                          Supervised Teacher (65%):
                        </span>
                        <span className="text-white">
                          {(liveResult.teacher_prob * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">
                          Semi-Sup Student (35%):
                        </span>
                        <span className="text-white">
                          {(liveResult.student_prob * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="flex justify-between pt-1.5 border-t border-slate-800">
                        <span className="text-slate-400">Chamber Status:</span>
                        <span
                          className={`font-bold ${
                            liveResult.predicted_status === "Excursion_Alert"
                              ? "text-rose-400"
                              : "text-emerald-400"
                          }`}
                        >
                          {liveResult.predicted_status}
                        </span>
                      </div>
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        )}

        {/* BOTTOM ROW: FEATURE IMPORTANCE + INLINE-TO-PROBE CORRELATIONS */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
            <h2 className="text-lg font-bold text-white flex items-center gap-2 mb-1">
              <Layers className="w-5 h-5 text-cyan-400" />
              Top 10 Root-Cause Sensors Driving Excursions
            </h2>
            <p className="text-xs text-slate-400 mb-4">
              Selected via Mutual Information Filter (257 → 30) +
              Imbalance-Weighted XGBoost Gain Importance.
            </p>

            {overview && (
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={overview.top_sensors}
                    layout="vertical"
                    margin={{ left: 25, right: 20 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis
                      type="number"
                      tick={{ fill: "#94a3b8", fontSize: 11 }}
                    />
                    <YAxis
                      dataKey="sensor"
                      type="category"
                      tick={{ fill: "#f8fafc", fontSize: 11 }}
                      width={85}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#0f172a",
                        borderColor: "#334155",
                        fontSize: "12px",
                      }}
                    />
                    <Bar
                      dataKey="importance"
                      name="XGBoost Gain Importance"
                      radius={[0, 4, 4, 0]}
                    >
                      {overview.top_sensors.map((entry, index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={
                            entry.stage_type.includes("Downstream")
                              ? "#f59e0b"
                              : "#06b6d4"
                          }
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
            <div className="flex items-center gap-6 text-xs text-slate-400 mt-2">
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-3 rounded-sm bg-cyan-500 inline-block"></span>
                Upstream Inline Process Sensor (Stages 1–500)
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-3 rounded-sm bg-amber-500 inline-block"></span>
                Downstream Probe / Cell Metric (Stages 501–590)
              </span>
            </div>
          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
            <h2 className="text-lg font-bold text-white mb-1">
              Inline-to-Probe & Cross-Stage Sensor Correlations
            </h2>
            <p className="text-xs text-slate-400 mb-4">
              Quantifies how upstream inline chamber drift propagates into
              downstream electrical probe metrics.
            </p>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400">
                    <th className="pb-2">Primary Sensor</th>
                    <th className="pb-2">Coupled Sensor</th>
                    <th className="pb-2">Pearson r</th>
                    <th className="pb-2">Physical Coupling Interpretation</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {correlations.slice(0, 7).map((pair, idx) => (
                    <tr key={idx} className="hover:bg-slate-800/40">
                      <td className="py-2.5 font-mono font-semibold text-cyan-400">
                        {pair.sensor_a}
                      </td>
                      <td className="py-2.5 font-mono font-semibold text-amber-400">
                        {pair.sensor_b}
                      </td>
                      <td className="py-2.5 font-mono font-bold text-white">
                        {pair.correlation > 0
                          ? `+${pair.correlation.toFixed(3)}`
                          : pair.correlation.toFixed(3)}
                      </td>
                      <td className="py-2.5 text-slate-300">
                        {Math.abs(pair.correlation) > 0.35
                          ? "Strong Cross-Stage Drift Propagation"
                          : Math.abs(pair.correlation) > 0.15
                          ? "Moderate Chamber-to-Probe Coupling"
                          : "Independent Process Mode"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}