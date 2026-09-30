import type { DetectionResult, PlateDetectionResponse } from "@/types/detection";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export function resolveAssetUrl(path: string | null | undefined): string {
  if (!path) return "";
  if (/^https?:\/\//i.test(path)) return path;
  return `${API_BASE_URL}${path.startsWith("/") ? path : `/${path}`}`;
}

export async function detectLicensePlate(file: File): Promise<DetectionResult> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/detect`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw new Error(error?.detail ?? "Failed to detect license plate.");
  }

  return response.json();
}

export async function detectLicensePlateDetails(file: File): Promise<PlateDetectionResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/detect/details`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw new Error(error?.detail ?? "Failed to analyze license plate evidence.");
  }

  return response.json();
}

export async function detectManualPlateCrop(
  file: File,
  coordinates: { x1: number; y1: number; x2: number; y2: number },
): Promise<PlateDetectionResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("x1", String(coordinates.x1));
  formData.append("y1", String(coordinates.y1));
  formData.append("x2", String(coordinates.x2));
  formData.append("y2", String(coordinates.y2));

  const response = await fetch(`${API_BASE_URL}/detect/manual-crop`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw new Error(error?.detail ?? "Failed to analyze selected plate region.");
  }

  return response.json();
}
