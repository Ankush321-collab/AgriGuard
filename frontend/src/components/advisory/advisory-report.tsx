"use client";

import * as React from "react";
import {
  Sparkles,
  AlertTriangle,
  Leaf,
  FlaskConical,
  ShieldCheck,
  CheckCircle2,
  Calendar,
  Building,
  Check,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { LLMAdvisoryResult } from "@/lib/types";

interface AdvisoryReportProps {
  advisory?: LLMAdvisoryResult;
  diseaseName?: string;
}

export function AdvisoryReport({ advisory, diseaseName }: AdvisoryReportProps) {
  if (!advisory) return null;

  // Determine smart default tab: if organic controls are empty (e.g. rejection or warning), jump to immediate action
  const hasOrganic = advisory.organicControls && advisory.organicControls.length > 0;
  const hasImmediate = advisory.immediateActions && advisory.immediateActions.length > 0;
  const defaultTab = !hasOrganic && hasImmediate ? "immediate" : "organic";

  return (
    <Card className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm transition-all duration-500 animate-advisory-glow">
      <CardHeader className="p-5 pb-3 border-b border-slate-100 dark:border-slate-800 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-emerald-50 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-800">
            <Sparkles className="h-5 w-5" />
          </div>
          <div>
            <CardTitle className="text-base font-bold text-slate-900 dark:text-white">
              Agent 8: Agronomic Advisory &amp; Treatment Protocol
            </CardTitle>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Grounded strictly in backend multi-agent retrieval and university extension citations
            </p>
          </div>
        </div>

        {advisory.extensionSpecialistReferralNeeded && (
          <Badge variant="warning" className="w-fit text-xs gap-1.5 py-1 font-medium bg-amber-50 dark:bg-amber-950/40 text-amber-800 dark:text-amber-300 border-amber-200 dark:border-amber-800">
            <Building className="h-3.5 w-3.5 text-amber-600 dark:text-amber-400" />
            County Extension Referral Recommended
          </Badge>
        )}
      </CardHeader>

      <CardContent className="p-5 space-y-4">
        {/* Safety Warnings Banner */}
        {advisory.safetyWarnings && advisory.safetyWarnings.length > 0 && (
          <Alert className="rounded-xl border-amber-200 dark:border-amber-900/60 bg-amber-50/60 dark:bg-amber-950/30 text-amber-900 dark:text-amber-200">
            <AlertTriangle className="h-4 w-4 text-amber-600 dark:text-amber-400" />
            <AlertTitle className="text-xs font-bold uppercase tracking-wider text-amber-800 dark:text-amber-300">
              Mandatory Regulatory &amp; Worker Safety Precautions
            </AlertTitle>
            <AlertDescription className="text-xs space-y-1 mt-1 text-amber-900 dark:text-amber-300">
              {advisory.safetyWarnings.map((warn, i) => (
                <p key={i}>• {warn}</p>
              ))}
            </AlertDescription>
          </Alert>
        )}

        {/* Clean White Tabbed Controls */}
        <Tabs defaultValue={defaultTab} className="w-full">
          <TabsList className="grid grid-cols-2 sm:grid-cols-3 w-full bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700">
            <TabsTrigger value="organic" className="text-xs gap-1.5 font-semibold data-[state=active]:bg-white dark:data-[state=active]:bg-slate-900 data-[state=active]:text-emerald-800 dark:data-[state=active]:text-emerald-400 data-[state=active]:shadow-xs">
              <Leaf className="h-3.5 w-3.5 text-emerald-600" />
              Cultural Precautions & Curing
            </TabsTrigger>
            <TabsTrigger value="chemical" className="text-xs gap-1.5 font-semibold data-[state=active]:bg-white dark:data-[state=active]:bg-slate-900 data-[state=active]:text-blue-800 dark:data-[state=active]:text-blue-400 data-[state=active]:shadow-xs">
              <FlaskConical className="h-3.5 w-3.5 text-blue-600" />
              Chemical Regulations
            </TabsTrigger>
            <TabsTrigger value="immediate" className="text-xs gap-1.5 font-semibold data-[state=active]:bg-white dark:data-[state=active]:bg-slate-900 data-[state=active]:text-slate-800 dark:data-[state=active]:text-slate-200 data-[state=active]:shadow-xs">
              <CheckCircle2 className="h-3.5 w-3.5 text-teal-600" />
              Immediate Action Steps
            </TabsTrigger>
          </TabsList>

          {/* Cultural Precautions & Curing */}
          <TabsContent value="organic" className="space-y-3 pt-3">
            <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/40 p-4 space-y-3">
              <div className="flex items-center justify-between">
                <h4 className="text-xs font-bold text-slate-800 dark:text-slate-200 uppercase tracking-wide">
                  Backend-Validated Cultural Controls (Non-Chemical Sanitation)
                </h4>
                <Badge variant="outline" className="text-[10px] bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-400">
                  IPM Primary Tier
                </Badge>
              </div>
              {advisory.organicControls && advisory.organicControls.length > 0 ? (
                <ul className="space-y-2">
                  {advisory.organicControls.map((ctrl, i) => (
                    <li key={i} className="flex items-start gap-2.5 text-xs text-slate-700 dark:text-slate-200 leading-relaxed bg-white dark:bg-slate-800/90 p-3 rounded-lg border border-slate-200/70 dark:border-slate-700/70 shadow-xs">
                      <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-emerald-100 dark:bg-emerald-900/50 text-emerald-800 dark:text-emerald-300 font-bold text-[10px]">
                        {i + 1}
                      </span>
                      <span>{ctrl}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-xs text-slate-500 italic p-3 bg-white dark:bg-slate-800/60 rounded-lg border border-slate-200/70 dark:border-slate-800">
                  No chemical or cultural controls recommended until specimen validation passes.
                </p>
              )}
            </div>
          </TabsContent>

          {/* Chemical Regulations */}
          <TabsContent value="chemical" className="space-y-3 pt-3">
            <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/40 p-4 space-y-3">
              <div className="flex items-center justify-between">
                <h4 className="text-xs font-bold text-slate-800 dark:text-slate-200 uppercase tracking-wide">
                  Regulatory & Chemical Spray Guidance
                </h4>
                <Badge variant="outline" className="text-[10px] bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-400">
                  EPA / University Compliant
                </Badge>
              </div>
              {advisory.chemicalTreatments && advisory.chemicalTreatments.length > 0 ? (
                <ul className="space-y-2">
                  {advisory.chemicalTreatments.map((chem, i) => (
                    <li key={i} className="flex items-start gap-2.5 text-xs text-slate-700 dark:text-slate-200 leading-relaxed bg-white dark:bg-slate-800/90 p-3 rounded-lg border border-slate-200/70 dark:border-slate-700/70 shadow-xs">
                      <FlaskConical className="h-4 w-4 shrink-0 text-blue-600 dark:text-blue-400 mt-0.5" />
                      <span>{chem}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-xs text-slate-500 italic p-3 bg-white dark:bg-slate-800/60 rounded-lg border border-slate-200/70 dark:border-slate-800">
                  Chemical intervention withheld. Consult county extension office for localized spray guides.
                </p>
              )}
              <p className="text-[11px] text-slate-500 dark:text-slate-400 italic pt-1">
                Note: AgriGuard-ML abstains from prescribing arbitrary numeric dosages to avoid phytotoxic risk. Refer to product label instructions and local extension spray schedules.
              </p>
            </div>
          </TabsContent>

          {/* Immediate Action Steps */}
          <TabsContent value="immediate" className="space-y-3 pt-3">
            <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/40 p-4 space-y-3">
              <h4 className="text-xs font-bold text-slate-800 dark:text-slate-200 uppercase tracking-wide">
                Immediate Orchard Response Actions
              </h4>
              <ul className="space-y-2">
                {(advisory.immediateActions && advisory.immediateActions.length > 0
                  ? advisory.immediateActions
                  : ["Scout adjacent trees to inspect potential pathogen spread."]
                ).map((action, i) => (
                  <li key={i} className="flex items-start gap-2.5 text-xs text-slate-800 dark:text-slate-100 font-medium leading-relaxed bg-white dark:bg-slate-800/90 p-3.5 rounded-lg border border-slate-200/70 dark:border-slate-700/70 shadow-xs">
                    <Check className="h-4 w-4 shrink-0 text-emerald-600 dark:text-emerald-400 mt-0.5" />
                    <span>{action}</span>
                  </li>
                ))}
              </ul>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
