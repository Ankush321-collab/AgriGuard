export type AppleLeafClass =
  | "Healthy"
  | "Alternaria leaf spot"
  | "Brown spot"
  | "Frogeye leaf spot"
  | "Grey spot"
  | "Mosaic"
  | "Powdery mildew"
  | "Rust"
  | "Scab";

export type PolicyDecision = "PASS" | "RETAKE" | "REJECT" | "REVIEW";

export type SeverityTier = "Minimal" | "Low" | "Moderate" | "Severe" | "Critical";

export interface QualityAgentResult {
  passed: boolean;
  blurScore: number;
  brightness: number;
  resolution: [number, number];
  reason?: string;
}

export interface LeafPresenceResult {
  passed: boolean;
  leafAreaFraction: number;
  leafDetected: boolean;
  reason?: string;
}

export interface AppleValidatorResult {
  passed: boolean;
  appleConfidence: number;
  isAppleLeaf: boolean;
  reason?: string;
}

export interface PolicyRouterResult {
  decision: PolicyDecision;
  reason: string;
  suggestedAction: string;
}

export interface SeverityResult {
  severityPercentage: number | null;
  severityTier: SeverityTier | string;
  lesionAreaPixels?: number;
  segmentationReliable: boolean;
  lesionMapUrl?: string;
  severityStatus?: string;
  severityNote?: string;
}

export interface DiseaseClassifierResult {
  diseaseName: AppleLeafClass;
  confidence: number;
  modelName: string;
  isUncertain: boolean;
  probabilities: Record<AppleLeafClass, number>;
}

export interface RAGSource {
  title: string;
  source: string;
  institution: string;
  year?: string;
  summary: string;
  relevanceScore: number;
  url?: string;
}

export interface LLMAdvisoryResult {
  title: string;
  summary: string;
  immediateActions: string[];
  organicControls: string[];
  chemicalTreatments: string[];
  safetyWarnings: string[];
  preventivePractices: string[];
  extensionSpecialistReferralNeeded: boolean;
}

export interface AgentStage {
  id: string;
  name: string;
  number: number;
  role: string;
  status: "idle" | "running" | "completed" | "skipped" | "failed";
  durationMs?: number;
  summary?: string;
}

export interface AgriGuardDiagnosisResult {
  id: string;
  timestamp: string;
  imageUrl: string;
  imageName: string;
  quality: QualityAgentResult;
  leafPresence: LeafPresenceResult;
  appleValidator: AppleValidatorResult;
  policy: PolicyRouterResult;
  severity?: SeverityResult;
  classifier?: DiseaseClassifierResult;
  sources?: RAGSource[];
  advisory?: LLMAdvisoryResult;
  executionTrace: string[];
  totalProcessingTimeMs: number;
  isSimulation?: boolean;
}

export interface SampleLeaf {
  id: string;
  name: string;
  className: AppleLeafClass;
  filename: string;
  imageUrl: string;
  thumbnailUrl: string;
  expectedSeverity: string;
  symptomsShort: string;
  description: string;
  pathogen: string;
}
