"use client";

import * as React from "react";
import {
  Camera,
  Eye,
  CheckCircle2,
  AlertTriangle,
  Flame,
  Cpu,
  BookOpen,
  Sparkles,
  ArrowRight,
  ShieldCheck,
  RotateCw,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { AgriGuardDiagnosisResult } from "@/lib/types";

interface AgentStepperProps {
  currentDiagnosis: AgriGuardDiagnosisResult | null;
  isLoading: boolean;
  activeStageIndex?: number;
}

interface AgentDefinition {
  number: number;
  name: string;
  role: string;
  icon: React.ElementType;
}

const AGENTS: AgentDefinition[] = [
  { number: 1, name: "Quality Guard", role: "Laplacian Blur & Lux", icon: Camera },
  { number: 2, name: "Leaf Presence", role: "Foliage Masking", icon: Eye },
  { number: 3, name: "Apple Validator", role: "Malus Species Check", icon: ShieldCheck },
  { number: 4, name: "Policy Router", role: "PASS / REVIEW / REJECT", icon: AlertTriangle },
  { number: 5, name: "Severity Profiler", role: "Lesion Quantification", icon: Flame },
  { number: 6, name: "Disease Classifier", role: "848 Handcrafted Features", icon: Cpu },
  { number: 7, name: "RAG Retrieval", role: "Extension Sources", icon: BookOpen },
  { number: 8, name: "LLM Advisory", role: "IPM & Safety Guidance", icon: Sparkles },
];

export function AgentStepper({
  currentDiagnosis,
  isLoading,
  activeStageIndex = 0,
}: AgentStepperProps) {
  return (
    <Card className="w-full overflow-hidden border border-slate-200 bg-white shadow-xs">
      <CardHeader className="py-4 px-6 border-b border-slate-100 flex flex-row items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-emerald-100 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-400">
            <Cpu className="h-4 w-4" />
          </div>
          <div>
            <CardTitle className="text-sm font-bold text-slate-900 dark:text-white">
              LangGraph Multi-Agent Pipeline
            </CardTitle>
            <p className="text-[11px] text-slate-500 dark:text-slate-400">
              Short-circuiting validation gates & explainable diagnostic stages
            </p>
          </div>
        </div>

        {currentDiagnosis && (
          <Badge
            variant={
              currentDiagnosis.policy.decision === "PASS"
                ? "success"
                : currentDiagnosis.policy.decision === "REVIEW"
                ? "warning"
                : "destructive"
            }
            className="text-xs px-2.5 py-0.5"
          >
            Gate Status: {currentDiagnosis.policy.decision}
          </Badge>
        )}
      </CardHeader>

      <CardContent className="p-4 sm:p-6">
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-3">
          {AGENTS.map((agent, index) => {
            const Icon = agent.icon;
            let status: "completed" | "running" | "idle" | "error" = "idle";
            let metricBadge = "";

            if (isLoading) {
              if (index < activeStageIndex) status = "completed";
              else if (index === activeStageIndex) status = "running";
              else status = "idle";
            } else if (currentDiagnosis) {
              status = "completed";
              if (index === 0 && currentDiagnosis.quality) {
                metricBadge = `${Math.round(currentDiagnosis.quality.blurScore)} blur`;
              } else if (index === 1 && currentDiagnosis.leafPresence) {
                metricBadge = `${Math.round(currentDiagnosis.leafPresence.leafAreaFraction * 100)}% leaf`;
              } else if (index === 2 && currentDiagnosis.appleValidator) {
                metricBadge = `${Math.round(currentDiagnosis.appleValidator.appleConfidence * 100)}% Malus`;
              } else if (index === 3) {
                metricBadge = currentDiagnosis.policy.decision;
              } else if (index === 4 && currentDiagnosis.severity) {
                metricBadge = currentDiagnosis.severity.severityPercentage !== null
                  ? `${currentDiagnosis.severity.severityPercentage}%`
                  : "Pending";
              } else if (index === 5 && currentDiagnosis.classifier) {
                metricBadge = `${Math.round(currentDiagnosis.classifier.confidence * 100)}% conf`;
              } else if (index === 6 && currentDiagnosis.sources) {
                metricBadge = `${currentDiagnosis.sources.length} sources`;
              } else if (index === 7) {
                metricBadge = "Safe";
              }
            }

            return (
              <div
                key={agent.number}
                className={`relative flex flex-col rounded-xl border p-3 transition-all duration-300 ${
                  status === "running"
                    ? "border-emerald-500 bg-emerald-50/70 shadow-md ring-2 ring-emerald-500/20 dark:bg-emerald-950/40 dark:border-emerald-400"
                    : status === "completed"
                    ? "border-emerald-200/80 bg-white dark:border-emerald-950 dark:bg-slate-900/80"
                    : "border-slate-200/70 bg-slate-50/50 opacity-60 dark:border-slate-800 dark:bg-slate-900/30"
                }`}
              >
                {/* Header row */}
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[10px] font-mono font-bold text-slate-400 dark:text-slate-500">
                    A0{agent.number}
                  </span>
                  {status === "completed" && (
                    <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400" />
                  )}
                  {status === "running" && (
                    <RotateCw className="h-3.5 w-3.5 animate-spin text-emerald-600 dark:text-emerald-400" />
                  )}
                </div>

                {/* Icon & Name */}
                <div className="mb-2 flex items-center gap-2">
                  <div
                    className={`flex h-7 w-7 items-center justify-center rounded-lg ${
                      status === "running"
                        ? "bg-emerald-600 text-white"
                        : status === "completed"
                        ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300"
                        : "bg-slate-100 text-slate-400 dark:bg-slate-800 dark:text-slate-500"
                    }`}
                  >
                    <Icon className="h-3.5 w-3.5" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-xs font-semibold text-slate-900 dark:text-slate-100">
                      {agent.name}
                    </p>
                  </div>
                </div>

                {/* Role / Metric */}
                <p className="truncate text-[10px] text-slate-500 dark:text-slate-400 mb-2">
                  {agent.role}
                </p>

                {metricBadge && (
                  <div className="mt-auto pt-1">
                    <span className="inline-block rounded-md bg-slate-100 px-1.5 py-0.5 text-[9px] font-mono font-medium text-slate-700 dark:bg-slate-800 dark:text-slate-300">
                      {metricBadge}
                    </span>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}
