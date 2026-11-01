"use client";

import * as React from "react";
import Image from "next/image";
import { SAMPLE_LEAVES } from "@/lib/sample-data";
import { SampleLeaf } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Sparkles, ArrowRight, CheckCircle2 } from "lucide-react";

interface SamplePickerProps {
  selectedSampleId: string | null;
  onSelectSample: (sample: SampleLeaf) => void;
  isLoading?: boolean;
}

export function SamplePicker({
  selectedSampleId,
  onSelectSample,
  isLoading = false,
}: SamplePickerProps) {
  return (
    <div className="w-full space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
          <h3 className="text-sm font-semibold text-slate-800 dark:text-slate-200">
            Or Test with AppleLeaf9 Dataset Samples
          </h3>
        </div>
        <span className="text-xs text-slate-500 dark:text-slate-400">
          9 Curated Class Specimens
        </span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-9 gap-2.5">
        {SAMPLE_LEAVES.map((sample) => {
          const isSelected = selectedSampleId === sample.id;
          const isHealthy = sample.className === "Healthy";

          return (
            <button
              key={sample.id}
              onClick={() => onSelectSample(sample)}
              disabled={isLoading}
              className={`group relative flex flex-col rounded-xl border p-2 text-left transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-emerald-500 cursor-pointer overflow-hidden ${
                isSelected
                  ? "border-emerald-500 bg-emerald-50/50 shadow-md ring-1 ring-emerald-500 dark:bg-emerald-950/30 dark:border-emerald-400"
                  : "border-slate-200/80 bg-white hover:border-slate-300 hover:shadow-sm dark:border-slate-800 dark:bg-slate-900 dark:hover:border-slate-700"
              }`}
            >
              {/* Image thumbnail container */}
              <div className="relative aspect-square w-full overflow-hidden rounded-lg bg-slate-100 dark:bg-slate-800 mb-2">
                <img
                  src={sample.imageUrl}
                  alt={sample.className}
                  className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-105"
                  loading="lazy"
                />
                {isSelected && (
                  <div className="absolute inset-0 bg-emerald-600/20 backdrop-blur-[1px] flex items-center justify-center">
                    <CheckCircle2 className="h-6 w-6 text-white drop-shadow-md" />
                  </div>
                )}
                <div className="absolute top-1 right-1">
                  <Badge
                    variant={isHealthy ? "success" : "secondary"}
                    className="text-[9px] px-1.5 py-0 font-medium backdrop-blur-md bg-white/80 dark:bg-slate-900/80 shadow-xs"
                  >
                    {isHealthy ? "Control" : "Sample"}
                  </Badge>
                </div>
              </div>

              {/* Title & Pathogen */}
              <div className="flex-1 min-w-0">
                <p className="truncate text-xs font-semibold text-slate-800 dark:text-slate-200 group-hover:text-emerald-600 dark:group-hover:text-emerald-400 transition-colors">
                  {sample.className}
                </p>
                <p className="truncate text-[10px] text-slate-500 dark:text-slate-400 italic">
                  {sample.pathogen.split(" ")[0]}
                </p>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
