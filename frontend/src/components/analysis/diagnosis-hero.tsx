"use client";

import * as React from "react";
import {
  CheckCircle,
  AlertTriangle,
  FileText,
  Clock,
  Layers,
  Sparkles,
  Download,
  Info,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { AgriGuardDiagnosisResult } from "@/lib/types";

interface DiagnosisHeroProps {
  diagnosis: AgriGuardDiagnosisResult;
  onExportReport?: () => void;
}

export function DiagnosisHero({ diagnosis, onExportReport }: DiagnosisHeroProps) {
  const isHealthy = diagnosis.classifier?.diseaseName === "Healthy";
  const confidencePct = Math.round((diagnosis.classifier?.confidence || 0) * 100);
  const decision = diagnosis.policy.decision;

  const downloadReportJson = () => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(diagnosis, null, 2));
    const downloadAnchor = document.createElement("a");
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `AgriGuard_Report_${diagnosis.classifier?.diseaseName || "Diagnosis"}_${Date.now()}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  return (
    <Card className="overflow-hidden border border-slate-200 bg-white shadow-sm">
      <CardContent className="p-6">
        <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-6">
          {/* Main Diagnosis Info */}
          <div className="space-y-3">
            <div className="flex flex-wrap items-center gap-2">
              <Badge
                variant={
                  decision === "REJECT" || decision === "RETAKE"
                    ? "destructive"
                    : isHealthy
                    ? "success"
                    : "destructive"
                }
                className="text-xs px-2.5 py-1 font-semibold uppercase tracking-wide"
              >
                {decision === "REJECT"
                  ? "Validation Gate: REJECT"
                  : decision === "RETAKE"
                  ? "Quality Gate: RETAKE"
                  : isHealthy
                  ? "Healthy Foliage"
                  : "Pathogen Detected"}
              </Badge>
              <Badge
                variant="outline"
                className="border-slate-300 dark:border-slate-700 text-xs px-2 py-0.5"
              >
                Gate Status: {decision}
              </Badge>
              {decision === "PASS" && (
                <Badge
                  variant="secondary"
                  className="text-xs px-2 py-0.5 text-slate-600 dark:text-slate-300"
                >
                  848 Handcrafted Features
                </Badge>
              )}
            </div>

            <div>
              <h2 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-slate-900 dark:text-white">
                {decision === "REJECT"
                  ? "Non-Apple Leaf Rejected"
                  : decision === "RETAKE"
                  ? "Blurry Image - Retake Needed"
                  : diagnosis.classifier?.diseaseName || "Diagnosed Condition"}
              </h2>
              <p className="text-xs sm:text-sm text-slate-600 dark:text-slate-400 mt-1 max-w-2xl leading-relaxed">
                {decision !== "PASS" && diagnosis.policy.reason
                  ? diagnosis.policy.reason
                  : diagnosis.advisory?.summary || "Diagnosis synthesized from multi-agent validation and LightGBM classification."}
              </p>
            </div>

            {/* Quick Metrics Strip */}
            <div className="flex flex-wrap items-center gap-4 text-xs text-slate-500 dark:text-slate-400 pt-1">
              <div className="flex items-center gap-1.5">
                <Clock className="h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400" />
                <span>Processing Latency: <strong>{diagnosis.totalProcessingTimeMs}ms</strong></span>
              </div>
              <div className="flex items-center gap-1.5">
                <Layers className="h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400" />
                <span>Validation: <strong>{decision === "PASS" ? "Full Consensus" : "Short-Circuited"}</strong></span>
              </div>
              <div className="flex items-center gap-1.5">
                <Sparkles className="h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400" />
                <span>Pipeline: <strong>Multi-Agent LangGraph</strong></span>
              </div>
            </div>
          </div>

          {/* Confidence Meter Badge & Export Button */}
          <div className="flex flex-col sm:flex-row lg:flex-col items-center sm:items-end justify-between w-full lg:w-auto gap-4 border-t lg:border-t-0 pt-4 lg:pt-0 border-slate-200/80 dark:border-slate-800">
            <div className="flex items-center gap-3 bg-white/80 dark:bg-slate-800/80 p-3 rounded-2xl border border-slate-200/60 dark:border-slate-700/60 shadow-xs">
              <div className={`relative flex h-14 w-14 items-center justify-center rounded-full font-extrabold text-base border-2 ${
                decision !== "PASS" 
                  ? "bg-rose-50 text-rose-600 border-rose-300 dark:bg-rose-950/60 dark:text-rose-400 dark:border-rose-800"
                  : "bg-emerald-50 text-emerald-600 border-emerald-500/30 dark:bg-emerald-950/60 dark:text-emerald-400"
              }`}>
                {decision !== "PASS" ? "N/A" : `${confidencePct}%`}
              </div>
              <div>
                <p className="text-[11px] font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                  {decision !== "PASS" ? "Validation Status" : "Diagnostic Confidence"}
                </p>
                <p className="text-xs font-semibold text-slate-900 dark:text-slate-100">
                  {decision === "REJECT"
                    ? "Rejected: Species Gate"
                    : decision === "RETAKE"
                    ? "Rejected: Blur Threshold"
                    : confidencePct >= 90
                    ? "High Statistical Certainty"
                    : confidencePct >= 75
                    ? "Adequate Confidence"
                    : "Marginal - Review Flag"}
                </p>
              </div>
            </div>

            <Button
              variant="outline"
              size="sm"
              onClick={downloadReportJson}
              className="w-full sm:w-auto gap-2 text-xs"
            >
              <Download className="h-3.5 w-3.5" />
              Export Agronomy Report (.JSON)
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
