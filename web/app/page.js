"use client";

import { useState } from "react";
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  ImagePlus,
  Layers,
  Loader2,
  Paperclip,
  ScanSearch,
  Upload,
  Waves,
  X,
  XCircle,
} from "lucide-react";

const fileInputClass =
  "absolute h-px w-px overflow-hidden whitespace-nowrap border-0 p-0 [-webkit-clip-path:inset(50%)] [clip-path:inset(50%)]";

export default function Home() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragging, setDragging] = useState(false);

  function onFile(f) {
    if (!f) return;
    setFile(f);
    setResult(null);
    setError(null);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(URL.createObjectURL(f));
  }

  function clearFile() {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f?.type.startsWith("image/")) onFile(f);
  }

  async function onSubmit(e) {
    e.preventDefault();
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch("/api/predict", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) {
        const msg =
          typeof data.detail === "string" ? data.detail : data.error || "Request failed";
        throw new Error(msg);
      }
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const isFake = result?.prediction === "FAKE";
  const pct = result ? Math.round(result.confidence * 100) : 0;
  const probFakePct = result
    ? (Number(result.prob_fake) * 100).toFixed(2)
    : null;
  const branches = result?.branch_contributions || {};

  const branchRows = [
    { key: "spatial", label: "Spatial", icon: Layers, pct: (branches.spatial || 0) * 100 },
    { key: "frequency", label: "Frequency", icon: Waves, pct: (branches.frequency || 0) * 100 },
    {
      key: "noise_residual",
      label: "Noise",
      icon: Activity,
      pct: (branches.noise_residual || 0) * 100,
    },
  ];

  return (
    <main className="mx-auto max-w-[550px] px-5 pb-12 pt-16">
      <header className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight text-foreground">
          Image Detector (CIFAKE Dataset)
        </h1>
        <p className="mt-1 text-sm leading-snug text-muted">
          Spatial + frequency + noise fusion model
        </p>
      </header>

      <form className="flex flex-col gap-4" onSubmit={onSubmit}>
        <section className="relative z-0" aria-label="Upload image">
          {preview ? (
            <div className="flex flex-col gap-3 rounded border border-border bg-panel p-3 shadow-sm">
              <img
                className="block max-h-60 w-full rounded border border-border bg-white object-contain"
                src={preview}
                alt="Selected preview"
              />
              <div className="flex flex-wrap gap-2">
                <label className="relative inline-flex cursor-pointer items-center gap-1.5 rounded-lg border border-border bg-white px-3.5 py-2 text-sm font-medium text-accent shadow-sm transition-colors hover:bg-gray-50">
                  <ImagePlus size={18} strokeWidth={2} />
                  Change image
                  <input
                    type="file"
                    accept="image/*"
                    className={fileInputClass}
                    onChange={(e) => onFile(e.target.files?.[0])}
                  />
                </label>

                <button
                  type="button"
                  className="inline-flex items-center gap-1.5 rounded-lg border border-red-200 bg-red-50 px-3.5 py-2 text-sm font-medium text-red-600 transition-colors hover:bg-red-100"
                  onClick={clearFile}
                >
                  <X size={18} strokeWidth={2} />
                  Remove
                </button>
              </div>
            </div>
          ) : (
            <label
              className={`relative flex min-h-[220px] cursor-pointer flex-col items-center justify-center gap-2 rounded border border-dashed border-border-strong bg-panel px-6 py-8 transition-colors ${
                dragging
                  ? "border-accent bg-blue-50"
                  : "hover:border-accent hover:bg-gray-50"
              }`}
              onDragEnter={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={(e) => {
                e.preventDefault();
                if (!e.currentTarget.contains(e.relatedTarget)) setDragging(false);
              }}
              onDrop={handleDrop}
            >
              <span
                className="mb-1 flex h-14 w-14 items-center justify-center rounded-full bg-white text-accent shadow-sm ring-1 ring-border"
                aria-hidden
              >
                <Paperclip size={32} strokeWidth={1.5} />
              </span>
              <span className="text-[0.95rem] font-medium text-foreground">
                Click or drop an image
              </span>
              <span className="text-xs text-dim">PNG, JPG, WebP</span>
              <input
                type="file"
                accept="image/*"
                className={fileInputClass}
                onChange={(e) => onFile(e.target.files?.[0])}
              />
            </label>
          )}
        </section>

        <button
          type="submit"
          className="mt-10 flex w-full items-center justify-center gap-2 rounded-md bg-accent px-4 py-3.5 text-base font-semibold text-white shadow-sm transition enabled:hover:bg-accent-light disabled:cursor-not-allowed disabled:opacity-40"
          disabled={!file || loading}
        >
          {loading ? (
            <Loader2 size={20} className="animate-spin" aria-hidden />
          ) : (
            <ScanSearch size={20} strokeWidth={2} aria-hidden />
          )}
          {loading ? "Analysing…" : "Detect"}
        </button>
      </form>

      {error && (
        <p
          className="mt-4 flex items-start gap-2 rounded-lg border border-red-200 bg-fake-bg px-4 py-3 text-sm text-fake"
          role="alert"
        >
          <AlertCircle size={18} className="mt-0.5 shrink-0" aria-hidden />
          {error}
        </p>
      )}

      {result && (
        <div
          className="mt-6 border border-gray-100 p-5 shadow-sm"
          role="status"
        >
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <span
              className={`inline-flex items-center gap-1.5 rounded-full px-3.5 py-1.5 text-2xl font-bold ${
                isFake ? "text-fake" : "text-real"
              }`}
            >
              {result.prediction} Image
            </span>
            <span className="inline-flex items-center gap-1.5 text-sm text-muted">
              {pct}% confidence
            </span>
          </div>

          <div className="mb-5 h-1.5 overflow-hidden rounded-full bg-border" aria-hidden>
            <span
              style={{ width: `${pct}%` }}
              className={`block h-full transition-[width] duration-300 ${
                isFake ? "bg-fake" : "bg-real"
              }`}
            />
          </div>

          <div className="mb-2">
            <p className="mb-2.5 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-muted">
              <Layers size={14} aria-hidden />
              Branch signal
            </p>
            <ul className="flex flex-col gap-2.5">
              {branchRows.map(({ key, label, icon: Icon, pct: branchPct }) => (
                <li
                  key={key}
                  className="grid grid-cols-[6.5rem_1fr_2.5rem] items-center gap-2 text-sm"
                >
                  <span className="inline-flex items-center gap-1.5 text-foreground">
                    <Icon size={15} aria-hidden />
                    {label}
                  </span>
                  <div className="h-1.5 overflow-hidden rounded-full bg-border">
                    <div
                      className="h-full min-w-0.5 rounded-full bg-accent transition-[width] duration-300"
                      style={{ width: `${branchPct}%` }}
                    />
                  </div>
                  <span className="text-right tabular-nums text-muted">
                    {branchPct.toFixed(0)}%
                  </span>
                </li>
              ))}
            </ul>
          </div>

          <p className="mt-3 text-base text-foreground">
            Probability of Fake ={" "}
            <span className="font-bold">{probFakePct}%</span> · Processing Time ={" "}
            <span className="font-bold">{result.processing_time_ms} ms</span>
          </p>
        </div>
      )}
    </main>
  );
}
