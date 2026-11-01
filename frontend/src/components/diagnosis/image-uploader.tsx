"use client";

import * as React from "react";
import { UploadCloud, Image as ImageIcon, X, AlertCircle, Play, ShieldAlert } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface ImageUploaderProps {
  onImageSelected: (file: File | null, previewUrl: string | null) => void;
  onRunDiagnosis: () => void;
  selectedPreviewUrl: string | null;
  selectedFileName: string | null;
  isLoading: boolean;
  activeSampleTitle?: string | null;
}

export function ImageUploader({
  onImageSelected,
  onRunDiagnosis,
  selectedPreviewUrl,
  selectedFileName,
  isLoading,
  activeSampleTitle,
}: ImageUploaderProps) {
  const [isDragging, setIsDragging] = React.useState(false);
  const [errorMessage, setErrorMessage] = React.useState<string | null>(null);
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  const handleFile = (file: File) => {
    setErrorMessage(null);
    if (!file.type.startsWith("image/")) {
      setErrorMessage("Please upload a valid image file (JPG, PNG, WEBP).");
      return;
    }
    if (file.size > 15 * 1024 * 1024) {
      setErrorMessage("File size exceeds 15MB limit.");
      return;
    }

    const previewUrl = URL.createObjectURL(file);
    onImageSelected(file, previewUrl);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const clearSelection = (e: React.MouseEvent) => {
    e.stopPropagation();
    onImageSelected(null, null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <Card className="overflow-hidden border-dashed border-2 transition-all duration-300">
      <CardContent className="p-6">
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => {
            if (e.target.files && e.target.files[0]) {
              handleFile(e.target.files[0]);
            }
          }}
        />

        {!selectedPreviewUrl ? (
          <div
            onClick={() => fileInputRef.current?.click()}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            className={`group flex flex-col items-center justify-center rounded-xl py-12 px-6 text-center transition-all cursor-pointer ${
              isDragging
                ? "bg-emerald-50/70 dark:bg-emerald-950/40 border-emerald-500 scale-[0.99]"
                : "hover:bg-slate-50/80 dark:hover:bg-slate-900/60"
            }`}
          >
            <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-emerald-50 text-emerald-600 shadow-inner group-hover:scale-110 group-hover:bg-emerald-100 transition-all dark:bg-emerald-950/60 dark:text-emerald-400">
              <UploadCloud className="h-8 w-8 stroke-[1.75]" />
            </div>

            <h4 className="text-base font-semibold text-slate-900 dark:text-slate-100 mb-1">
              Upload Apple Leaf Photo
            </h4>
            <p className="max-w-md text-xs text-slate-500 dark:text-slate-400 mb-4 leading-relaxed">
              Drag and drop high-resolution foliage image here, or click to browse. Supported formats: JPG, PNG, WEBP (up to 15MB).
            </p>

            <div className="flex flex-wrap items-center justify-center gap-2">
              <Badge variant="secondary" className="text-[11px] font-normal">
                Pre-check: Blur & Lux
              </Badge>
              <Badge variant="secondary" className="text-[11px] font-normal">
                Species Validation
              </Badge>
              <Badge variant="secondary" className="text-[11px] font-normal">
                848 Handcrafted Features
              </Badge>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="relative overflow-hidden rounded-xl bg-slate-50 flex items-center justify-center max-h-[380px] group border border-slate-200 shadow-inner">
              <img
                src={selectedPreviewUrl}
                alt="Selected leaf specimen"
                className="max-h-[360px] w-auto object-contain rounded-lg shadow-sm"
              />

              {/* Top overlay metadata badge */}
              <div className="absolute top-3 left-3 flex items-center gap-2">
                <Badge className="bg-white/95 backdrop-blur-md text-slate-900 border-slate-200 shadow-xs text-xs font-semibold">
                  {activeSampleTitle ? activeSampleTitle : selectedFileName || "Uploaded Leaf"}
                </Badge>
                {activeSampleTitle && (
                  <Badge variant="success" className="text-xs shadow-xs">
                    Benchmark Specimen
                  </Badge>
                )}
              </div>

              {/* Close / clear button */}
              {!isLoading && (
                <button
                  onClick={clearSelection}
                  className="absolute top-3 right-3 rounded-full bg-white/95 border border-slate-200 p-1.5 text-slate-700 shadow-md hover:bg-red-50 hover:text-red-600 transition-colors cursor-pointer"
                  title="Remove image"
                >
                  <X className="h-4 w-4" />
                </button>
              )}
            </div>

            <div className="flex flex-col sm:flex-row items-center justify-between gap-3 pt-2">
              <div className="text-xs text-slate-500 dark:text-slate-400 flex items-center gap-1.5">
                <ImageIcon className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
                <span>Ready for multi-agent validation and LightGBM classification</span>
              </div>

              <div className="flex items-center gap-2 w-full sm:w-auto">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isLoading}
                  className="w-full sm:w-auto text-xs"
                >
                  Choose Different
                </Button>
                <Button
                  variant="glow"
                  size="default"
                  onClick={onRunDiagnosis}
                  disabled={isLoading}
                  className="w-full sm:w-auto gap-2 font-semibold"
                >
                  {isLoading ? (
                    <>
                      <span className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                      Running Multi-Agent Pipeline...
                    </>
                  ) : (
                    <>
                      <Play className="h-4 w-4 fill-white" />
                      Execute Diagnosis Pipeline
                    </>
                  )}
                </Button>
              </div>
            </div>
          </div>
        )}

        {errorMessage && (
          <div className="mt-3 flex items-center gap-2 rounded-lg bg-red-50 p-2.5 text-xs text-red-700 dark:bg-red-950/50 dark:text-red-300 border border-red-200 dark:border-red-900">
            <AlertCircle className="h-4 w-4 shrink-0" />
            <span>{errorMessage}</span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
