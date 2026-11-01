"use client";

import * as React from "react";
import { SAMPLE_LEAVES } from "@/lib/sample-data";
import { SampleLeaf } from "@/lib/types";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Search, Sparkles, Layers, ArrowUpRight, ShieldCheck, ExternalLink, Leaf, FlaskConical } from "lucide-react";

interface DiseaseAtlasProps {
  onTestDisease: (sample: SampleLeaf) => void;
}

export function DiseaseAtlas({ onTestDisease }: DiseaseAtlasProps) {
  const [searchQuery, setSearchQuery] = React.useState("");
  const [filterType, setFilterType] = React.useState<"all" | "fungal" | "viral" | "healthy">("all");
  const [backendSources, setBackendSources] = React.useState<any[]>([]);
  const [isLoadingKb, setIsLoadingKb] = React.useState(true);

  // Fetch backend knowledge base directly
  React.useEffect(() => {
    fetch("/api/knowledge")
      .then((res) => res.json())
      .then((data) => {
        if (data && data.sources) {
          setBackendSources(data.sources);
        }
      })
      .catch((err) => console.error("Could not fetch backend KB:", err))
      .finally(() => setIsLoadingKb(false));
  }, []);

  const mergedLeaves = React.useMemo(() => {
    return SAMPLE_LEAVES.map((sample) => {
      const match = backendSources.find(
        (s) =>
          s.disease_target.toLowerCase() === sample.className.toLowerCase() ||
          (s.disease_synonyms && s.disease_synonyms.some((syn: string) => syn.toLowerCase() === sample.className.toLowerCase()))
      );

      return {
        ...sample,
        backendSource: match || null,
        culturalControls: match?.cultural_controls || [],
        chemicalRegulations: match?.chemical_regulations || "Consult local agricultural extension guidelines.",
        passage: match?.retrieved_passage || sample.description,
        sourceOrg: match?.organization || "University Agricultural Extension",
        sourceUrl: match?.url || "https://ipm.ucanr.edu/",
      };
    });
  }, [backendSources]);

  const filteredLeaves = mergedLeaves.filter((leaf) => {
    const matchesSearch =
      leaf.className.toLowerCase().includes(searchQuery.toLowerCase()) ||
      leaf.pathogen.toLowerCase().includes(searchQuery.toLowerCase()) ||
      leaf.symptomsShort.toLowerCase().includes(searchQuery.toLowerCase());

    if (!matchesSearch) return false;

    if (filterType === "healthy") return leaf.className === "Healthy";
    if (filterType === "viral") return leaf.className === "Mosaic";
    if (filterType === "fungal") return leaf.className !== "Healthy" && leaf.className !== "Mosaic";
    return true;
  });

  return (
    <div className="space-y-6">
      {/* Intro and Search */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-200 pb-5">
        <div>
          <div className="flex items-center gap-2 text-emerald-700 font-semibold text-xs uppercase tracking-wider mb-1">
            <Layers className="h-4 w-4" />
            <span>Backend Curated Agronomy Database</span>
          </div>
          <h2 className="text-2xl font-bold text-slate-900">
            AppleLeaf9 Pathological Encyclopedia
          </h2>
          <p className="text-xs text-slate-600 mt-0.5">
            Real precautions, cultural sanitation protocols, and chemical regulations fetched directly from the backend knowledge base
          </p>
        </div>

        {/* Filter Pills */}
        <div className="flex items-center gap-1.5 flex-wrap">
          {(["all", "fungal", "viral", "healthy"] as const).map((type) => (
            <button
              key={type}
              onClick={() => setFilterType(type)}
              className={`rounded-full px-3 py-1 text-xs font-semibold capitalize transition-all cursor-pointer ${
                filterType === type
                  ? "bg-emerald-600 text-white shadow-sm"
                  : "bg-white border border-slate-200 text-slate-700 hover:bg-slate-50"
              }`}
            >
              {type === "all" ? "All 9 Classes" : type}
            </button>
          ))}
        </div>
      </div>

      {/* Grid of 9 Diseases */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredLeaves.map((leaf) => {
          const isHealthy = leaf.className === "Healthy";
          const isViral = leaf.className === "Mosaic";

          return (
            <Card
              key={leaf.id}
              className="clean-card group overflow-hidden rounded-xl border border-slate-200 bg-white transition-all duration-200"
            >
              <div className="relative aspect-[16/10] w-full overflow-hidden bg-slate-100 border-b border-slate-100">
                <img
                  src={leaf.imageUrl}
                  alt={leaf.className}
                  className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-105"
                  loading="lazy"
                />
                <div className="absolute top-3 left-3">
                  <Badge
                    variant={isHealthy ? "success" : isViral ? "purple" : "destructive"}
                    className="text-xs font-semibold shadow-xs"
                  >
                    {isHealthy ? "Physiological Clean" : isViral ? "Systemic Virus" : "Fungal Pathogen"}
                  </Badge>
                </div>
              </div>

              <CardContent className="p-5 space-y-4">
                <div>
                  <h3 className="text-lg font-bold text-slate-900 group-hover:text-emerald-700 transition-colors">
                    {leaf.className}
                  </h3>
                  <p className="text-xs font-medium text-emerald-700 italic">
                    {leaf.pathogen}
                  </p>
                </div>

                <div className="rounded-lg bg-slate-50 border border-slate-100 p-3 space-y-2 text-xs">
                  <p className="text-slate-700 leading-relaxed font-normal">
                    {leaf.passage}
                  </p>

                  {/* Backend Precautions & Curing Methods */}
                  {leaf.culturalControls.length > 0 && (
                    <div className="pt-2 border-t border-slate-200/60 space-y-1">
                      <span className="text-[11px] font-bold text-slate-800 uppercase tracking-wider flex items-center gap-1">
                        <Leaf className="h-3 w-3 text-emerald-600" />
                        Backend Cultural Precautions:
                      </span>
                      <ul className="space-y-1 pl-1">
                        {leaf.culturalControls.map((ctrl: string, idx: number) => (
                          <li key={idx} className="text-[11px] text-slate-600 leading-normal">
                            • {ctrl}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Chemical Regulations */}
                  <div className="pt-1.5 border-t border-slate-200/60">
                    <span className="text-[11px] font-bold text-slate-800 uppercase tracking-wider flex items-center gap-1">
                      <FlaskConical className="h-3 w-3 text-blue-600" />
                      Regulatory Notice:
                    </span>
                    <p className="text-[11px] text-slate-600 leading-normal mt-0.5">
                      {leaf.chemicalRegulations}
                    </p>
                  </div>
                </div>

                {/* Authority citation */}
                <div className="flex items-center justify-between text-[11px] text-slate-500 pt-1">
                  <span className="truncate max-w-[180px] font-medium">{leaf.sourceOrg}</span>
                  <a
                    href={leaf.sourceUrl}
                    target="_blank"
                    rel="noreferrer"
                    className="text-emerald-600 hover:text-emerald-700 flex items-center gap-1 font-semibold"
                  >
                    Source <ExternalLink className="h-3 w-3" />
                  </a>
                </div>

                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => onTestDisease(leaf)}
                  className="w-full text-xs gap-1.5 bg-white hover:bg-emerald-50 hover:text-emerald-700 hover:border-emerald-300 border-slate-200 font-semibold"
                >
                  <Sparkles className="h-3.5 w-3.5 text-emerald-600" />
                  Test Specimen In Studio
                  <ArrowUpRight className="h-3.5 w-3.5 ml-auto" />
                </Button>
              </CardContent>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
