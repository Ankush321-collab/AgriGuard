"use client";

import * as React from "react";
import { BookOpen, ExternalLink, GraduationCap, Award } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { RAGSource } from "@/lib/types";

interface RAGSourcesCardProps {
  sources?: RAGSource[];
}

export function RAGSourcesCard({ sources }: RAGSourcesCardProps) {
  if (!sources || sources.length === 0) return null;

  return (
    <Card className="border-slate-200/80 bg-white/70 dark:border-slate-800/80 dark:bg-slate-900/70 backdrop-blur-sm">
      <CardHeader className="p-4 pb-2 flex flex-row items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-teal-50 text-teal-700 dark:bg-teal-950/60 dark:text-teal-400">
            <BookOpen className="h-4 w-4" />
          </div>
          <CardTitle className="text-sm font-bold text-slate-900 dark:text-white">
            Agent 7: University Extension Knowledge Citations (RAG)
          </CardTitle>
        </div>
        <Badge variant="outline" className="text-xs">
          {sources.length} Peer Sources
        </Badge>
      </CardHeader>

      <CardContent className="p-4 pt-2 space-y-3">
        {sources.map((src, i) => (
          <div
            key={i}
            className="rounded-xl border border-slate-100 bg-slate-50/60 p-3.5 dark:border-slate-800/70 dark:bg-slate-800/40 space-y-1.5 transition-all hover:border-slate-300 dark:hover:border-slate-700"
          >
            <div className="flex items-start justify-between gap-2">
              <h5 className="text-xs font-bold text-slate-900 dark:text-slate-100 flex items-center gap-1.5">
                <GraduationCap className="h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400 shrink-0" />
                {src.title}
              </h5>
              <Badge variant="secondary" className="font-mono text-[10px] shrink-0">
                {Math.round(src.relevanceScore * 100)}% Match
              </Badge>
            </div>

            <p className="text-[11px] font-medium text-emerald-800 dark:text-emerald-300 flex items-center gap-1">
              <span>{src.institution}</span>
              {src.year && <span>• {src.year}</span>}
              <span className="text-slate-400 font-normal">({src.source})</span>
            </p>

            <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
              {src.summary}
            </p>

            {src.url && (
              <a
                href={src.url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 text-[11px] font-semibold text-emerald-700 hover:text-emerald-800 hover:underline pt-1"
              >
                <span>Read Extension Guidance</span>
                <ExternalLink className="h-3 w-3" />
              </a>
            )}
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
