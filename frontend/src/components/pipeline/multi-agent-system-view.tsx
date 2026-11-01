"use client";

import * as React from "react";
import {
  Camera,
  Eye,
  ShieldCheck,
  AlertTriangle,
  Flame,
  Cpu,
  BookOpen,
  Sparkles,
  ArrowDown,
  CheckCircle2,
  XCircle,
  HelpCircle,
  Zap,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export function MultiAgentSystemView() {
  return (
    <div className="space-y-8">
      {/* Overview header */}
      <div>
        <h2 className="text-xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
          <Cpu className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
          LangGraph Multi-Agent Architecture
        </h2>
        <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
          Sequential short-circuiting validation gates preceding classical machine learning classification
        </p>
      </div>

      {/* Workflow Diagram */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Phase 1: Pre-classification */}
        <Card className="border-emerald-200/80 bg-emerald-50/30 dark:border-emerald-950 dark:bg-emerald-950/20">
          <CardHeader className="p-4 pb-2">
            <Badge variant="outline" className="w-fit text-[10px] border-emerald-500/40 text-emerald-700 dark:text-emerald-300">
              Phase 1: Pre-Classification
            </Badge>
            <CardTitle className="text-sm font-bold text-slate-900 dark:text-white mt-1">
              Quality & Species Guards
            </CardTitle>
          </CardHeader>
          <CardContent className="p-4 pt-1 space-y-3 text-xs">
            <div className="rounded-lg bg-white p-2.5 shadow-xs dark:bg-slate-900 border border-slate-100 dark:border-slate-800">
              <div className="flex items-center gap-2 font-semibold text-slate-800 dark:text-slate-200">
                <Camera className="h-3.5 w-3.5 text-emerald-600" />
                Agent 1: Quality Guard
              </div>
              <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-1">
                Computes Laplacian variance (sharpness &gt; 100) and lux brightness. Rejects blurry photos early.
              </p>
            </div>

            <div className="rounded-lg bg-white p-2.5 shadow-xs dark:bg-slate-900 border border-slate-100 dark:border-slate-800">
              <div className="flex items-center gap-2 font-semibold text-slate-800 dark:text-slate-200">
                <Eye className="h-3.5 w-3.5 text-emerald-600" />
                Agent 2: Leaf Presence
              </div>
              <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-1">
                Applies HSV chlorophyll masking. Requires foliage area fraction &ge; 10% of frame.
              </p>
            </div>

            <div className="rounded-lg bg-white p-2.5 shadow-xs dark:bg-slate-900 border border-slate-100 dark:border-slate-800">
              <div className="flex items-center gap-2 font-semibold text-slate-800 dark:text-slate-200">
                <ShieldCheck className="h-3.5 w-3.5 text-emerald-600" />
                Agent 3: Apple Validator
              </div>
              <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-1">
                Validates Malus domestica morphology (&ge; 75% confidence) to prevent misdiagnosing other crops.
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Phase 2: Policy Gate */}
        <Card className="border-amber-200/80 bg-amber-50/30 dark:border-amber-950 dark:bg-amber-950/20">
          <CardHeader className="p-4 pb-2">
            <Badge variant="outline" className="w-fit text-[10px] border-amber-500/40 text-amber-700 dark:text-amber-300">
              Phase 2: Short-Circuit Gate
            </Badge>
            <CardTitle className="text-sm font-bold text-slate-900 dark:text-white mt-1">
              Agent 4: Policy Router
            </CardTitle>
          </CardHeader>
          <CardContent className="p-4 pt-1 space-y-2.5 text-xs">
            <div className="rounded-lg bg-white p-2.5 shadow-xs dark:bg-slate-900 border border-emerald-200 dark:border-emerald-900/60">
              <div className="flex items-center gap-1.5 font-bold text-emerald-700 dark:text-emerald-400">
                <CheckCircle2 className="h-3.5 w-3.5" />
                PASS Outcome
              </div>
              <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">
                All pre-checks satisfied. Image advances to feature extraction and classifier.
              </p>
            </div>

            <div className="rounded-lg bg-white p-2.5 shadow-xs dark:bg-slate-900 border border-amber-200 dark:border-amber-900/60">
              <div className="flex items-center gap-1.5 font-bold text-amber-700 dark:text-amber-400">
                <HelpCircle className="h-3.5 w-3.5" />
                REVIEW Outcome
              </div>
              <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">
                Marginal confidence. Abstains from toxic chemical advice; routes to university extension.
              </p>
            </div>

            <div className="rounded-lg bg-white p-2.5 shadow-xs dark:bg-slate-900 border border-red-200 dark:border-red-900/60">
              <div className="flex items-center gap-1.5 font-bold text-red-700 dark:text-red-400">
                <XCircle className="h-3.5 w-3.5" />
                RETAKE / REJECT
              </div>
              <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">
                Image is too blurry, unlit, or non-leaf. Provides actionable retake instructions.
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Phase 3: Classical ML Engine */}
        <Card className="border-blue-200/80 bg-blue-50/30 dark:border-blue-950 dark:bg-blue-950/20">
          <CardHeader className="p-4 pb-2">
            <Badge variant="outline" className="w-fit text-[10px] border-blue-500/40 text-blue-700 dark:text-blue-300">
              Phase 3: Explainable ML Core
            </Badge>
            <CardTitle className="text-sm font-bold text-slate-900 dark:text-white mt-1">
              848 Handcrafted Features
            </CardTitle>
          </CardHeader>
          <CardContent className="p-4 pt-1 space-y-3 text-xs">
            <div className="rounded-lg bg-white p-2.5 shadow-xs dark:bg-slate-900 border border-slate-100 dark:border-slate-800">
              <div className="flex items-center gap-2 font-semibold text-slate-800 dark:text-slate-200">
                <Flame className="h-3.5 w-3.5 text-amber-500" />
                Agent 5: Severity Quantification
              </div>
              <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-1">
                Otsu adaptive thresholding calculates necrotic surface fraction (Minimal &rarr; Critical).
              </p>
            </div>

            <div className="rounded-lg bg-white p-2.5 shadow-xs dark:bg-slate-900 border border-slate-100 dark:border-slate-800">
              <div className="flex items-center gap-2 font-semibold text-slate-800 dark:text-slate-200">
                <Cpu className="h-3.5 w-3.5 text-blue-600" />
                Agent 6: Disease Classifier
              </div>
              <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-1">
                Soft-voting ensemble across LightGBM, Random Forest, SVM &amp; KNN over HOG, Color Moments, and GLCM.
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Phase 4: RAG & Safety Advisory */}
        <Card className="border-purple-200/80 bg-purple-50/30 dark:border-purple-950 dark:bg-purple-950/20">
          <CardHeader className="p-4 pb-2">
            <Badge variant="outline" className="w-fit text-[10px] border-purple-500/40 text-purple-700 dark:text-purple-300">
              Phase 4: Post-Diagnosis RAG
            </Badge>
            <CardTitle className="text-sm font-bold text-slate-900 dark:text-white mt-1">
              Agronomy Guidance
            </CardTitle>
          </CardHeader>
          <CardContent className="p-4 pt-1 space-y-3 text-xs">
            <div className="rounded-lg bg-white p-2.5 shadow-xs dark:bg-slate-900 border border-slate-100 dark:border-slate-800">
              <div className="flex items-center gap-2 font-semibold text-slate-800 dark:text-slate-200">
                <BookOpen className="h-3.5 w-3.5 text-purple-600" />
                Agent 7: Extension RAG
              </div>
              <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-1">
                Retrieves verified factsheets from Cornell, Penn State, and UC Davis IPM libraries.
              </p>
            </div>

            <div className="rounded-lg bg-white p-2.5 shadow-xs dark:bg-slate-900 border border-slate-100 dark:border-slate-800">
              <div className="flex items-center gap-2 font-semibold text-slate-800 dark:text-slate-200">
                <Sparkles className="h-3.5 w-3.5 text-teal-600" />
                Agent 8: Safety & Advisory
              </div>
              <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-1">
                Synthesizes chemical vs organic options, Pre-Harvest Intervals (PHI), and safety warnings.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Feature Engineering Breakdown Table */}
      <Card className="border-slate-200/80 bg-white/80 dark:border-slate-800/80 dark:bg-slate-900/80">
        <CardHeader className="p-5 pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Zap className="h-4 w-4 text-emerald-600" />
              <CardTitle className="text-sm font-bold text-slate-900 dark:text-white">
                Handcrafted Feature Engineering (Total: 848 Features)
              </CardTitle>
            </div>
            <Badge variant="secondary" className="text-xs">
              Zero Deep Learning Latency (CPU Native)
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="p-5 pt-0">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="border-b border-slate-200 dark:border-slate-800 text-slate-500">
                <tr>
                  <th className="py-2.5 font-semibold">Descriptor Family</th>
                  <th className="py-2.5 font-semibold">Dimensions</th>
                  <th className="py-2.5 font-semibold">Target Botanical Properties</th>
                  <th className="py-2.5 font-semibold">Biological Rationale</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800/60 text-slate-700 dark:text-slate-300">
                <tr>
                  <td className="py-2.5 font-medium text-emerald-600 dark:text-emerald-400">HOG (Histograms of Oriented Gradients)</td>
                  <td className="py-2.5 font-mono">756</td>
                  <td className="py-2.5">Edge orientations, lesion border feathery vs defined</td>
                  <td className="py-2.5 text-slate-500 dark:text-slate-400">Distinguishes feathery Scab margins from circular Frogeye spots</td>
                </tr>
                <tr>
                  <td className="py-2.5 font-medium text-emerald-600 dark:text-emerald-400">Color Moments & Histograms</td>
                  <td className="py-2.5 font-mono">48</td>
                  <td className="py-2.5">HSV &amp; Lab color distributions, chlorosis vs necrosis</td>
                  <td className="py-2.5 text-slate-500 dark:text-slate-400">Captures bright yellow halos in Rust vs velvety olive in Scab</td>
                </tr>
                <tr>
                  <td className="py-2.5 font-medium text-emerald-600 dark:text-emerald-400">GLCM &amp; LBP Texture</td>
                  <td className="py-2.5 font-mono">36</td>
                  <td className="py-2.5">Contrast, energy, homogeneity, powdery mycelium roughness</td>
                  <td className="py-2.5 text-slate-500 dark:text-slate-400">Separates talcum powdery mildew mycelium from smooth blade</td>
                </tr>
                <tr>
                  <td className="py-2.5 font-medium text-emerald-600 dark:text-emerald-400">Hu Moments &amp; Morphology</td>
                  <td className="py-2.5 font-mono">8</td>
                  <td className="py-2.5">Invariable spatial moments, aspect ratio, eccentricity</td>
                  <td className="py-2.5 text-slate-500 dark:text-slate-400">Quantifies leaf curl and lesion elongation along veins</td>
                </tr>
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
