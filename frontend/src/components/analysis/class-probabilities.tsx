"use client";

import * as React from "react";
import { BarChart3, TrendingUp } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { AppleLeafClass } from "@/lib/types";

interface ClassProbabilitiesProps {
  probabilities?: Record<AppleLeafClass, number>;
  predictedClass?: AppleLeafClass;
}

export function ClassProbabilities({
  probabilities,
  predictedClass,
}: ClassProbabilitiesProps) {
  if (!probabilities) return null;

  // Sort descending by probability
  const sortedEntries = Object.entries(probabilities).sort(
    ([, a], [, b]) => b - a
  ) as [AppleLeafClass, number][];

  return (
    <Card className="border-slate-200/80 bg-white/70 dark:border-slate-800/80 dark:bg-slate-900/70 backdrop-blur-sm">
      <CardHeader className="p-4 pb-2 flex flex-row items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-emerald-50 text-emerald-600 dark:bg-emerald-950/60 dark:text-emerald-400">
            <BarChart3 className="h-4 w-4" />
          </div>
          <CardTitle className="text-sm font-bold text-slate-900 dark:text-white">
            Agent 6: 9-Class Probability Spectrum
          </CardTitle>
        </div>
        <span className="text-[11px] font-mono text-slate-400">
          Soft Voting Ensemble
        </span>
      </CardHeader>

      <CardContent className="p-4 pt-2 space-y-2.5">
        {sortedEntries.map(([className, prob]) => {
          const isWinner = className === predictedClass;
          const pct = Math.round(prob * 1000) / 10;

          return (
            <div key={className} className="space-y-1">
              <div className="flex items-center justify-between text-xs">
                <span
                  className={`truncate max-w-[160px] sm:max-w-none ${
                    isWinner
                      ? "font-bold text-emerald-700 dark:text-emerald-400 flex items-center gap-1.5"
                      : "text-slate-600 dark:text-slate-400 font-medium"
                  }`}
                >
                  {isWinner && <TrendingUp className="h-3 w-3 inline shrink-0" />}
                  {className}
                </span>
                <span
                  className={`font-mono text-[11px] ${
                    isWinner
                      ? "font-bold text-emerald-700 dark:text-emerald-400"
                      : "text-slate-400"
                  }`}
                >
                  {pct.toFixed(1)}%
                </span>
              </div>

              <Progress
                value={Math.max(pct, 1)}
                className="h-1.5"
                indicatorClassName={
                  isWinner
                    ? "bg-gradient-to-r from-emerald-500 to-teal-500"
                    : "bg-slate-300 dark:bg-slate-700"
                }
              />
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}
