import * as React from "react";
import { ShieldCheck, Heart, ExternalLink, Leaf } from "lucide-react";

export function Footer() {
  return (
    <footer className="border-t border-slate-200/80 bg-slate-50/60 dark:border-slate-800/80 dark:bg-slate-950/60 py-8 text-xs text-slate-500 dark:text-slate-400">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-6">
          <div className="space-y-2">
            <div className="flex items-center gap-2 font-bold text-slate-900 dark:text-white text-sm">
              <Leaf className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
              AgriGuard-ML
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
              Classical ML-only, multi-agent apple leaf disease detection framework utilizing 848 handcrafted visual features and short-circuiting validation gates.
            </p>
          </div>

          <div>
            <h4 className="font-semibold text-slate-900 dark:text-slate-200 text-xs uppercase tracking-wider mb-2">
              Multi-Agent Architecture
            </h4>
            <ul className="space-y-1">
              <li>Quality Guard (Laplacian Blur & Lux)</li>
              <li>Foliage Presence (HSV Masking)</li>
              <li>Species Verification (ResNet Embedding)</li>
              <li>Policy Router (PASS / REVIEW / REJECT)</li>
              <li>Severity Estimator & Lesion Profiler</li>
            </ul>
          </div>

          <div>
            <h4 className="font-semibold text-slate-900 dark:text-slate-200 text-xs uppercase tracking-wider mb-2">
              Agronomy Knowledge Sources
            </h4>
            <ul className="space-y-1">
              <li>Cornell Cooperative Extension Tree Fruit IPM</li>
              <li>Penn State Tree Fruit Production Guide</li>
              <li>Washington State University Plant Pathology</li>
              <li>USDA-ARS Apple Disease Bulletin</li>
            </ul>
          </div>

          <div>
            <h4 className="font-semibold text-slate-900 dark:text-slate-200 text-xs uppercase tracking-wider mb-2">
              Compliance & Safety
            </h4>
            <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
              AgriGuard-ML adheres to IPM pesticide safety protocols, Worker Protection Standards (WPS), and strict chemical abstention on uncertain diagnoses.
            </p>
          </div>
        </div>

        <div className="border-t border-slate-200/50 dark:border-slate-800/50 pt-4 flex flex-col sm:flex-row items-center justify-between gap-3 text-slate-400">
          <div className="flex items-center gap-1.5">
            <span>Built for AppleLeaf9 Precision Agriculture</span>
            <span>•</span>
            <span>Next.js 16 + shadcn/ui + Tailwind CSS</span>
          </div>
          <div className="flex items-center gap-3">
            <span>CPU-Friendly Classical ML</span>
            <span>•</span>
            <span>Explainable AI</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
