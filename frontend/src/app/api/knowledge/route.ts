import { NextResponse } from "next/server";
import fs from "fs/promises";
import path from "path";

export async function GET() {
  try {
    const projectRoot = path.resolve(process.cwd(), "..");
    const kbPath = path.join(projectRoot, "src", "knowledge_base", "curated_extension_sources.json");

    const content = await fs.readFile(kbPath, "utf-8");
    const parsed = JSON.parse(content);

    return NextResponse.json(parsed);
  } catch (error) {
    console.error("Failed to load knowledge base:", error);
    return NextResponse.json(
      { error: "Could not load knowledge base", details: String(error) },
      { status: 500 }
    );
  }
}
