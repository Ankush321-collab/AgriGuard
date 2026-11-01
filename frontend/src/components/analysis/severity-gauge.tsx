"use client";

import * as React from "react";
import { Flame, ShieldAlert, Activity, CheckCircle } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { SeverityResult } from "@/lib/types";

interface SeverityGaugeProps {
  severity?: SeverityResult;
}

export function SeverityGauge({ severity }: SeverityGaugeProps) {
  if (!severity) {
    return null;
  }

  const pct = severity.severityPercentage;
  const tier = severity.severityTier || "Unavailable";
  const isWithheld = pct === null || tier.includes("Unavailable");

  // Determine color and status badge
  let tierBadgeVariant: "success" | "warning" | "destructive" | "secondary" = "success";
  let progressColor = "bg-emerald-500";

  if (isWithheld) {
    tierBadgeVariant = "secondary";
    progressColor = "bg-slate-300";
  } else if (tier === "Minimal" || pct === 0) {
    tierBadgeVariant = "success";
    progressColor = "bg-emerald-500";
  } else if (tier === "Low" || (pct !== null && pct < 15)) {
    tierBadgeVariant = "warning";
    progressColor = "bg-amber-400";
  } else if (tier === "Moderate" || (pct !== null && pct < 30)) {
    tierBadgeVariant = "warning";
    progressColor = "bg-amber-500";
  } else {
    tierBadgeVariant = "destructive";
    progressColor = "bg-red-500";
  }

  return (
    <Card className="border-slate-200/80 bg-white/70 dark:border-slate-800/80 dark:bg-slate-900/70 backdrop-blur-sm">
      <CardHeader className="p-4 pb-2 flex flex-row items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-amber-50 text-amber-600 dark:bg-amber-950/60 dark:text-amber-400">
            <Flame className="h-4 w-4" />
          </div>
          <CardTitle className="text-sm font-bold text-slate-900 dark:text-white">
            Agent 5: Severity Quantification
          </CardTitle>
        </div>
        <Badge variant={tierBadgeVariant} className="text-xs">
          {isWithheld ? "Withheld (Phase 4)" : `Tier: ${tier}`}
        </Badge>
      </CardHeader>

      <CardContent className="p-4 pt-2 space-y-3">
        <div className="flex items-baseline justify-between">
          <span className="text-2xl font-black text-slate-900 dark:text-white">
            {isWithheld ? "Pending" : `${pct}%`}
          </span>
          <span className="text-xs text-slate-500 dark:text-slate-400">
            {isWithheld ? "Benchmark Masks Required" : "Foliar Lesion Fraction"}
          </span>
        </div>

        {/* Progress bar */}
        <Progress
          value={isWithheld ? 0 : Math.min(pct ?? 0, 100)}
          indicatorClassName={progressColor}
          className="h-2.5"
        />

        <div className="grid grid-cols-5 text-[10px] text-slate-400 font-mono text-center pt-1 border-t border-slate-100 dark:border-slate-800/60">
          <span className={pct !== null && pct <= 5 ? "font-bold text-emerald-600" : ""}>0-5% Min</span>
          <span className={pct !== null && pct > 5 && pct <= 15 ? "font-bold text-amber-500" : ""}>5-15% Low</span>
          <span className={pct !== null && pct > 15 && pct <= 25 ? "font-bold text-amber-600" : ""}>15-25% Mod</span>
          <span className={pct !== null && pct > 25 && pct <= 45 ? "font-bold text-red-500" : ""}>25-45% Sev</span>
          <span className={pct !== null && pct > 45 ? "font-bold text-red-700" : ""}>&gt;45% Crit</span>
        </div>

        <p className="text-[11px] text-slate-500 dark:text-slate-400 leading-relaxed pt-1">
          {isWithheld
            ? (severity.severityNote || "Severity quantification is withheld: Pixel-level lesion masks have not undergone reference Dice/IoU benchmark validation on expert-annotated ground truth masks.")
            : pct === 0
            ? "Specimen demonstrates clean leaf blade with zero necrotic tissue detected."
            : `Foliar damage covers ${pct}% of blade area, demanding ${tier === "Moderate" || tier === "Severe" ? "active IPM remediation" : "targeted preventive monitoring"}.`}
        </p>
      </CardContent>
    </Card>
  );
}
