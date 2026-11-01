"use client";

import * as React from "react";
import confetti from "canvas-confetti";
import { Header } from "@/components/layout/header";
import { Footer } from "@/components/layout/footer";
import { ImageUploader } from "@/components/diagnosis/image-uploader";
import { SamplePicker } from "@/components/diagnosis/sample-picker";
import { DiagnosisHero } from "@/components/analysis/diagnosis-hero";
import { SeverityGauge } from "@/components/analysis/severity-gauge";
import { ClassProbabilities } from "@/components/analysis/class-probabilities";
import { AdvisoryReport } from "@/components/advisory/advisory-report";
import { RAGSourcesCard } from "@/components/advisory/rag-sources-card";
import { DiseaseAtlas } from "@/components/knowledge/disease-atlas";
import { MultiAgentSystemView } from "@/components/pipeline/multi-agent-system-view";
import { SAMPLE_LEAVES } from "@/lib/sample-data";
import { SampleLeaf, AgriGuardDiagnosisResult } from "@/lib/types";
import { ShieldCheck, Sparkles, RefreshCw, AlertCircle, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function Home() {
  const [activeTab, setActiveTab] = React.useState<string>("diagnosis");
  const [selectedFile, setSelectedFile] = React.useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = React.useState<string | null>(null);
  const [selectedSample, setSelectedSample] = React.useState<SampleLeaf | null>(null);
  const [isLoading, setIsLoading] = React.useState(false);
  const [diagnosisResult, setDiagnosisResult] = React.useState<AgriGuardDiagnosisResult | null>(null);
  const [errorMessage, setErrorMessage] = React.useState<string | null>(null);
  const resultsRef = React.useRef<HTMLDivElement>(null);

  // Initialize with a default benchmark specimen in preview, but keep state clean without auto-running
  React.useEffect(() => {
    const defaultSample = SAMPLE_LEAVES[1]; // Scab specimen
    setSelectedSample(defaultSample);
    setPreviewUrl(defaultSample.imageUrl);
  }, []);

  const handleImageSelected = (file: File | null, url: string | null) => {
    setSelectedFile(file);
    setPreviewUrl(url);
    setSelectedSample(null);
    setDiagnosisResult(null);
    setErrorMessage(null);
  };

  const handleSelectSample = (sample: SampleLeaf) => {
    setSelectedSample(sample);
    setSelectedFile(null);
    setPreviewUrl(sample.imageUrl);
    setDiagnosisResult(null);
    setErrorMessage(null);
  };

  const handleTestFromAtlas = (sample: SampleLeaf) => {
    setSelectedSample(sample);
    setSelectedFile(null);
    setPreviewUrl(sample.imageUrl);
    setActiveTab("diagnosis");
    setDiagnosisResult(null);
    setErrorMessage(null);
  };

  const runDiagnosis = async () => {
    if (!previewUrl && !selectedFile && !selectedSample) return;

    setIsLoading(true);
    setErrorMessage(null);

    try {
      let resData: AgriGuardDiagnosisResult;

      if (selectedFile) {
        const formData = new FormData();
        formData.append("file", selectedFile);
        const res = await fetch("/api/diagnose", {
          method: "POST",
          body: formData,
        });
        if (!res.ok) throw new Error("Diagnosis API failed");
        resData = await res.json();
      } else if (selectedSample) {
        const res = await fetch("/api/diagnose", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sampleFilename: selectedSample.filename,
            sampleId: selectedSample.id,
          }),
        });
        if (!res.ok) throw new Error("Diagnosis API failed");
        resData = await res.json();
      } else {
        throw new Error("No image selected");
      }

      setDiagnosisResult(resData);

      // Trigger celebratory confetti on high-confidence diagnosis
      if (resData.policy.decision === "PASS") {
        try {
          confetti({
            particleCount: 45,
            spread: 60,
            origin: { y: 0.6 },
            colors: ["#10b981", "#14b8a6", "#3b82f6"],
          });
        } catch (e) {
          // ignore in SSR/no-canvas
        }
      }

      // Smoothly scroll down to the animated diagnosis and advisory card
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 150);
    } catch (err: any) {
      console.error("Diagnosis error:", err);
      setErrorMessage(err.message || "Failed to execute backend diagnosis pipeline");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50/50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 subtle-mesh-bg selection:bg-emerald-500 selection:text-white">
      <Header activeTab={activeTab} onTabChange={setActiveTab} />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {/* Tab 1: Diagnosis Studio */}
        {activeTab === "diagnosis" && (
          <div className="space-y-8">
            {/* Studio Hero Intro */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-200/60 dark:border-slate-800/60 pb-6">
              <div>
                <div className="flex items-center gap-2 text-emerald-600 dark:text-emerald-400 font-semibold text-xs tracking-wide uppercase mb-1">
                  <ShieldCheck className="h-4 w-4" />
                  <span>Precision Agronomy Diagnostic Studio</span>
                </div>
                <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-slate-900 dark:text-white">
                  Apple Leaf Disease Detection
                </h1>
                <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400 mt-1 max-w-2xl leading-relaxed">
                  Upload an apple leaf photograph or select a benchmark specimen. Our 8-agent LangGraph pipeline validates sharpness, verifies species, extracts 848 handcrafted features, and produces actionable IPM spray advice.
                </p>
              </div>

              {diagnosisResult && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    setDiagnosisResult(null);
                    setSelectedFile(null);
                    setSelectedSample(null);
                    setPreviewUrl(null);
                  }}
                  className="w-fit text-xs gap-1.5"
                >
                  <RefreshCw className="h-3.5 w-3.5" />
                  New Scan
                </Button>
              )}
            </div>

            {/* Top Interactive Area: Uploader & Sample Picker */}
            <div className="space-y-6">
              <ImageUploader
                onImageSelected={handleImageSelected}
                onRunDiagnosis={runDiagnosis}
                selectedPreviewUrl={previewUrl}
                selectedFileName={selectedFile?.name || null}
                isLoading={isLoading}
                activeSampleTitle={selectedSample?.name}
              />

              <SamplePicker
                selectedSampleId={selectedSample?.id || null}
                onSelectSample={handleSelectSample}
                isLoading={isLoading}
              />
            </div>

            {/* Clean Minimal Loading State */}
            {isLoading && (
              <div className="flex flex-col items-center justify-center p-8 rounded-2xl border border-emerald-200/80 bg-emerald-50/40 dark:border-emerald-900/50 dark:bg-emerald-950/20 text-center animate-pulse">
                <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-emerald-100 text-emerald-700 dark:bg-emerald-900/60 dark:text-emerald-400 mb-3 shadow-inner">
                  <Loader2 className="h-6 w-6 animate-spin" />
                </div>
                <h3 className="text-sm font-bold text-slate-800 dark:text-slate-100">
                  Executing LangGraph Multi-Agent Pipeline...
                </h3>
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 max-w-md">
                  Analyzing foliage features, validating species gates, executing LightGBM classification, and synthesizing IPM advisory.
                </p>
              </div>
            )}

            {/* Error Message */}
            {errorMessage && (
              <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-xs text-red-800 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300 flex items-center gap-2">
                <AlertCircle className="h-4 w-4 shrink-0 text-red-600" />
                <span>{errorMessage}</span>
              </div>
            )}

            {/* Pop-Up Results Section with Smooth Scroll Target */}
            {diagnosisResult && (
              <div
                ref={resultsRef}
                className="space-y-6 animate-popup-in scroll-mt-6"
              >
                {/* Hero Diagnosis Banner */}
                <DiagnosisHero diagnosis={diagnosisResult} />

                {/* 2-Column Diagnostic Breakdown (Only when PASS) */}
                {diagnosisResult.policy.decision === "PASS" && (
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <SeverityGauge severity={diagnosisResult.severity} />
                    <ClassProbabilities
                      probabilities={diagnosisResult.classifier?.probabilities}
                      predictedClass={diagnosisResult.classifier?.diseaseName}
                    />
                  </div>
                )}

                {/* Advisory Report (2nd Pic) & Citations */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  <div
                    className={
                      diagnosisResult.policy.decision === "PASS" &&
                      diagnosisResult.sources &&
                      diagnosisResult.sources.length > 0
                        ? "lg:col-span-2"
                        : "lg:col-span-3"
                    }
                  >
                    <AdvisoryReport
                      advisory={diagnosisResult.advisory}
                      diseaseName={diagnosisResult.classifier?.diseaseName}
                    />
                  </div>
                  {diagnosisResult.policy.decision === "PASS" &&
                    diagnosisResult.sources &&
                    diagnosisResult.sources.length > 0 && (
                      <div>
                        <RAGSourcesCard sources={diagnosisResult.sources} />
                      </div>
                    )}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Tab 2: AppleLeaf9 Atlas */}
        {activeTab === "atlas" && (
          <DiseaseAtlas onTestDisease={handleTestFromAtlas} />
        )}

        {/* Tab 3: Multi-Agent Architecture */}
        {activeTab === "agents" && <MultiAgentSystemView />}
      </main>

      <Footer />
    </div>
  );
}
