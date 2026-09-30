"use client";

import { ChangeEvent, CSSProperties, DragEvent, PointerEvent, useEffect, useMemo, useRef, useState } from "react";
import { detectLicensePlateDetails, detectManualPlateCrop, resolveAssetUrl } from "@/services/api";
import type { PlateDetection, PlateDetectionResponse } from "@/types/detection";

type ExportDetection = {
  index: number;
  plate: string;
  final_confidence: number;
  detector_confidence: number;
  ocr_confidence: number;
  coordinates: PlateDetection["coordinates"];
  cropped_plate_url: string;
  candidates: { plate: string; confidence: number }[];
};

type InvestigationExport = {
  exported_at: string;
  processed_at: string | null;
  uploaded_file_name: string | null;
  stored_filename: string;
  original_image_url: string;
  annotated_image_url: string | null;
  detector: string;
  model_configured: boolean;
  plates_detected: number;
  quality_score: number;
  processing_time_ms: number | null;
  detections: ExportDetection[];
};

type QualityStatus = {
  label: string;
  tone: "strong" | "review" | "weak" | "empty";
};

type ManualSelection = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

export default function Home() {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const manualImageRef = useRef<HTMLImageElement | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [isDetecting, setIsDetecting] = useState(false);
  const [isManualAnalyzing, setIsManualAnalyzing] = useState(false);
  const [isManualMode, setIsManualMode] = useState(false);
  const [isSelectingManualRegion, setIsSelectingManualRegion] = useState(false);
  const [manualSelection, setManualSelection] = useState<ManualSelection | null>(null);
  const [result, setResult] = useState<PlateDetectionResponse | null>(null);
  const [processingTimeMs, setProcessingTimeMs] = useState<number | null>(null);
  const [processedAt, setProcessedAt] = useState<string | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!file) {
      setPreviewUrl("");
      return;
    }

    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);

    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  const bestDetection = result?.detections[0] ?? null;
  const qualityScore = useMemo(() => (result ? calculateQualityScore(result) : 0), [result]);
  const qualityStatus = getQualityStatus(qualityScore, result?.plates_detected ?? 0);
  const annotatedImageUrl = resolveAssetUrl(result?.annotated_image_url);
  const storedOriginalUrl = result ? resolveAssetUrl(`/uploads/${result.filename}`) : "";
  const originalImageUrl = previewUrl || storedOriginalUrl;
  const candidateRows = useMemo(() => buildCandidateRows(bestDetection), [bestDetection]);
  const shouldShowManualReview = Boolean(
    file && (isManualMode || (result && (result.plates_detected === 0 || !bestDetection?.text))),
  );
  const manualSelectionBox = normalizeSelection(manualSelection);
  const manualSelectionStyle = manualSelectionBox
    ? ({
        left: `${manualSelectionBox.x1 * 100}%`,
        top: `${manualSelectionBox.y1 * 100}%`,
        width: `${(manualSelectionBox.x2 - manualSelectionBox.x1) * 100}%`,
        height: `${(manualSelectionBox.y2 - manualSelectionBox.y1) * 100}%`,
      } as CSSProperties)
    : undefined;
  const manualCropCoordinates = getManualCropCoordinates(manualSelectionBox, manualImageRef.current);
  const hasManualSelection = Boolean(
    manualCropCoordinates &&
      manualCropCoordinates.x2 - manualCropCoordinates.x1 >= 8 &&
      manualCropCoordinates.y2 - manualCropCoordinates.y1 >= 4,
  );

  function handleFile(selectedFile?: File) {
    if (!selectedFile) return;

    if (!selectedFile.type.startsWith("image/")) {
      setError("Unsupported evidence file. Select an image.");
      return;
    }

    setFile(selectedFile);
    setResult(null);
    setProcessingTimeMs(null);
    setProcessedAt(null);
    setIsManualMode(false);
    setManualSelection(null);
    setError("");
  }

  function handleInputChange(event: ChangeEvent<HTMLInputElement>) {
    handleFile(event.target.files?.[0]);
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragging(false);
    handleFile(event.dataTransfer.files?.[0]);
  }

  async function handleDetect() {
    if (!file) {
      setError("Select an evidence image first.");
      return;
    }

    setIsDetecting(true);
    setError("");
    setResult(null);
    setProcessingTimeMs(null);
    setProcessedAt(null);
    setManualSelection(null);

    const startTime = performance.now();
    try {
      const detection = await detectLicensePlateDetails(file);
      setProcessingTimeMs(Math.round(performance.now() - startTime));
      setProcessedAt(new Date().toISOString());
      setResult(detection);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Evidence analysis failed.");
    } finally {
      setIsDetecting(false);
    }
  }

  async function handleManualDetect() {
    if (!file) {
      setError("Select an evidence image first.");
      return;
    }

    const coordinates = getManualCropCoordinates(normalizeSelection(manualSelection), manualImageRef.current);
    if (!coordinates || coordinates.x2 - coordinates.x1 < 8 || coordinates.y2 - coordinates.y1 < 4) {
      setError("Select a larger plate region.");
      return;
    }

    setIsManualAnalyzing(true);
    setError("");
    const startTime = performance.now();
    try {
      const detection = await detectManualPlateCrop(file, coordinates);
      setProcessingTimeMs(Math.round(performance.now() - startTime));
      setProcessedAt(new Date().toISOString());
      setResult(detection);
      setIsManualMode(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Manual plate analysis failed.");
    } finally {
      setIsManualAnalyzing(false);
    }
  }

  function handleManualPointerDown(event: PointerEvent<HTMLDivElement>) {
    if (!file || isManualAnalyzing) return;

    const point = getManualPoint(event, manualImageRef.current);
    if (!point) return;

    event.currentTarget.setPointerCapture(event.pointerId);
    setIsSelectingManualRegion(true);
    setManualSelection({ x1: point.x, y1: point.y, x2: point.x, y2: point.y });
  }

  function handleManualPointerMove(event: PointerEvent<HTMLDivElement>) {
    if (!isSelectingManualRegion) return;

    const point = getManualPoint(event, manualImageRef.current);
    if (!point) return;

    setManualSelection((selection) => (selection ? { ...selection, x2: point.x, y2: point.y } : selection));
  }

  function handleManualPointerUp(event: PointerEvent<HTMLDivElement>) {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    setIsSelectingManualRegion(false);
  }

  function clearCase() {
    setFile(null);
    setResult(null);
    setProcessingTimeMs(null);
    setProcessedAt(null);
    setIsManualMode(false);
    setManualSelection(null);
    setError("");
    if (inputRef.current) inputRef.current.value = "";
  }

  function downloadJson() {
    if (!result) return;
    const payload = createExportPayload(result, file, processingTimeMs, qualityScore, processedAt);
    downloadBlob(
      new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }),
      buildFilename(result.filename, "json"),
    );
  }

  function downloadCsv() {
    if (!result) return;
    const payload = createExportPayload(result, file, processingTimeMs, qualityScore, processedAt);
    downloadBlob(new Blob([createCsv(payload)], { type: "text/csv" }), buildFilename(result.filename, "csv"));
  }

  function downloadPdf() {
    if (!result) return;
    const payload = createExportPayload(result, file, processingTimeMs, qualityScore, processedAt);
    downloadBlob(createPdf(payload), buildFilename(result.filename, "pdf"));
  }

  return (
    <main className="investigation-shell">
      <header className="command-bar">
        <div>
          <p className="eyebrow">LPR Investigation Console</p>
          <h1>Vehicle Plate Evidence Review</h1>
        </div>
        <div className="case-status" aria-label="Case status">
          <span>{result ? "Analyzed" : file ? "Queued" : "No Evidence"}</span>
          <strong>{formatDuration(processingTimeMs)}</strong>
        </div>
      </header>

      <section className="case-toolbar" aria-label="Evidence controls">
        <div
          className={`upload-target ${isDragging ? "is-dragging" : ""}`}
          onClick={() => inputRef.current?.click()}
          onDragOver={(event) => {
            event.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === " ") inputRef.current?.click();
          }}
          role="button"
          tabIndex={0}
        >
          <input ref={inputRef} type="file" accept="image/*" onChange={handleInputChange} hidden />
          <span className="upload-label">Evidence Image</span>
          <strong>{file?.name ?? "Drop image or browse"}</strong>
        </div>

        <div className="action-group">
          <button className="primary-button" type="button" disabled={!file || isDetecting} onClick={handleDetect}>
            {isDetecting ? "Analyzing" : "Analyze"}
          </button>
          <button
            className="secondary-button"
            type="button"
            disabled={!file}
            onClick={() => setIsManualMode((current) => !current)}
          >
            Manual
          </button>
          <button className="secondary-button" type="button" disabled={!result} onClick={downloadPdf}>
            PDF
          </button>
          <button className="secondary-button" type="button" disabled={!result} onClick={downloadJson}>
            JSON
          </button>
          <button className="secondary-button" type="button" disabled={!result} onClick={downloadCsv}>
            CSV
          </button>
          <button className="ghost-button" type="button" disabled={!file && !result} onClick={clearCase}>
            Clear
          </button>
        </div>
      </section>

      {error && <div className="system-message">{error}</div>}

      <div className="dashboard-grid">
        <div className="visual-column">
          <section className="panel evidence-panel" aria-label="Evidence imagery">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">Evidence Imagery</p>
                <h2>Original and Annotated Views</h2>
              </div>
              <span className="panel-count">{result?.plates_detected ?? 0} plates</span>
            </div>

            <div className="image-grid">
              <figure className="image-viewer">
                <figcaption>Original Image</figcaption>
                {originalImageUrl ? (
                  <img src={originalImageUrl} alt="Original vehicle evidence" />
                ) : (
                  <div className="empty-image">Awaiting image</div>
                )}
              </figure>

              <figure className="image-viewer">
                <figcaption>Annotated Image</figcaption>
                {annotatedImageUrl ? (
                  <img src={annotatedImageUrl} alt="Annotated plate detection result" />
                ) : (
                  <div className="empty-image">{result ? "No annotation generated" : "Awaiting analysis"}</div>
                )}
              </figure>
            </div>
          </section>

          {shouldShowManualReview && (
            <section className="panel manual-panel" aria-label="Manual plate region selection">
              <div className="panel-heading">
                <div>
                  <p className="eyebrow">Manual Review</p>
                  <h2>Selected Plate Region</h2>
                </div>
                <span className="panel-count">{formatManualSelection(manualCropCoordinates)}</span>
              </div>

              <div className={`manual-crop-stage ${isSelectingManualRegion ? "is-selecting" : ""}`}>
                {originalImageUrl ? (
                  <div
                    className="manual-image-wrap"
                    onPointerDown={handleManualPointerDown}
                    onPointerMove={handleManualPointerMove}
                    onPointerUp={handleManualPointerUp}
                    onPointerCancel={handleManualPointerUp}
                  >
                    <img ref={manualImageRef} src={originalImageUrl} alt="Manual plate region source" draggable={false} />
                    {manualSelectionStyle && <span className="manual-selection-box" style={manualSelectionStyle} />}
                  </div>
                ) : (
                  <div className="empty-section">Awaiting image</div>
                )}
              </div>

              <div className="manual-actions">
                <button
                  className="primary-button"
                  type="button"
                  disabled={!hasManualSelection || isManualAnalyzing}
                  onClick={handleManualDetect}
                >
                  {isManualAnalyzing ? "Reading" : "Manual OCR"}
                </button>
                <button className="secondary-button" type="button" onClick={() => setManualSelection(null)}>
                  Reset
                </button>
              </div>
            </section>
          )}

          <section className="panel crop-panel" aria-label="Plate crop previews">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">Plate Crops</p>
                <h2>Detected Plate Evidence</h2>
              </div>
            </div>

            {result && result.detections.length > 0 ? (
              <div className="crop-grid">
                {result.detections.map((detection, index) => (
                  <article className="crop-item" key={`${detection.cropped_plate_url}-${index}`}>
                    <div className="crop-image-frame">
                      <img src={resolveAssetUrl(detection.cropped_plate_url)} alt={`Plate crop ${index + 1}`} />
                    </div>
                    <div className="crop-details">
                      <div>
                        <span>Plate {index + 1}</span>
                        <strong>{detection.text || "Unread"}</strong>
                      </div>
                      <div className="mini-metrics">
                        <span>OCR {formatPercent(detection.ocr_confidence)}</span>
                        <span>Final {formatPercent(detection.final_confidence)}</span>
                      </div>
                      <code>{formatBox(detection)}</code>
                    </div>
                  </article>
                ))}
              </div>
            ) : (
              <div className="empty-section">{result ? "No plates detected" : "Plate crops will appear after analysis"}</div>
            )}
          </section>
        </div>

        <aside className="analysis-column" aria-label="Analysis summary">
          <section className="panel summary-panel">
            <div className="quality-row">
              <div
                className={`quality-meter ${qualityStatus.tone}`}
                style={{ "--score": `${qualityScore}%` } as CSSProperties}
                aria-label={`Quality score ${qualityScore}%`}
              >
                <strong>{qualityScore}</strong>
                <span>Quality</span>
              </div>
              <div className="finding-primary">
                <span className={`status-pill ${qualityStatus.tone}`}>{qualityStatus.label}</span>
                <p>Best Candidate</p>
                <strong>{bestDetection?.text || "No plate"}</strong>
              </div>
            </div>

            <div className="confidence-stack">
              <ConfidenceMeter label="Final Confidence" value={bestDetection?.final_confidence ?? 0} tone="green" />
              <ConfidenceMeter label="OCR Confidence" value={bestDetection?.ocr_confidence ?? 0} tone="amber" />
              <ConfidenceMeter label="Detection Confidence" value={bestDetection?.confidence ?? 0} tone="blue" />
            </div>
          </section>

          <section className="panel candidate-panel">
            <div className="panel-heading compact">
              <div>
                <p className="eyebrow">Candidates</p>
                <h2>Ranked Plate Reads</h2>
              </div>
            </div>
            {candidateRows.length > 0 ? (
              <ol className="candidate-list">
                {candidateRows.map((candidate, index) => (
                  <li key={`${candidate.plate}-${candidate.confidence}-${index}`}>
                    <span>{String(index + 1).padStart(2, "0")}</span>
                    <strong>{candidate.plate}</strong>
                    <em>{formatPercent(candidate.confidence)}</em>
                  </li>
                ))}
              </ol>
            ) : (
              <div className="empty-section">No OCR candidates available</div>
            )}
          </section>

          <section className="panel metadata-panel">
            <div className="panel-heading compact">
              <div>
                <p className="eyebrow">Metadata</p>
                <h2>Detection Record</h2>
              </div>
            </div>
            <dl className="metadata-list">
              <div>
                <dt>Detector</dt>
                <dd>{result?.detector ?? "Pending"}</dd>
              </div>
              <div>
                <dt>Model</dt>
                <dd>{result ? (result.model_configured ? "Configured" : "Fallback") : "Pending"}</dd>
              </div>
              <div>
                <dt>Stored File</dt>
                <dd>{result?.filename ?? "Pending"}</dd>
              </div>
              <div>
                <dt>Processing Time</dt>
                <dd>{formatDuration(processingTimeMs)}</dd>
              </div>
              <div>
                <dt>Processed At</dt>
                <dd>{processedAt ? formatTimestamp(processedAt) : "Pending"}</dd>
              </div>
              <div>
                <dt>Best Box</dt>
                <dd>{bestDetection ? formatBox(bestDetection) : "Pending"}</dd>
              </div>
            </dl>
          </section>
        </aside>
      </div>
    </main>
  );
}

function ConfidenceMeter({
  label,
  value,
  tone,
}: {
  label: string;
  value: number;
  tone: "green" | "amber" | "blue";
}) {
  const percent = Math.round(clamp(value, 0, 1) * 100);
  return (
    <div className="confidence-row">
      <div>
        <span>{label}</span>
        <strong>{percent}%</strong>
      </div>
      <div className="confidence-track" role="meter" aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent}>
        <span className={tone} style={{ width: `${percent}%` }} />
      </div>
    </div>
  );
}

function calculateQualityScore(result: PlateDetectionResponse): number {
  const best = result.detections[0];
  if (!best) return 0;

  const textSignal = best.text ? Math.min(best.text.length / 10, 1) : 0;
  const score = best.final_confidence * 0.55 + best.ocr_confidence * 0.25 + best.confidence * 0.15 + textSignal * 0.05;
  return Math.round(clamp(score, 0, 1) * 100);
}

function getQualityStatus(score: number, platesDetected: number): QualityStatus {
  if (platesDetected === 0) return { label: "No Signal", tone: "empty" };
  if (score >= 82) return { label: "High Confidence", tone: "strong" };
  if (score >= 62) return { label: "Review", tone: "review" };
  return { label: "Low Confidence", tone: "weak" };
}

function getManualPoint(event: PointerEvent<HTMLDivElement>, image: HTMLImageElement | null) {
  if (!image) return null;

  const rect = image.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return null;

  return {
    x: clamp((event.clientX - rect.left) / rect.width, 0, 1),
    y: clamp((event.clientY - rect.top) / rect.height, 0, 1),
  };
}

function normalizeSelection(selection: ManualSelection | null): ManualSelection | null {
  if (!selection) return null;

  return {
    x1: Math.min(selection.x1, selection.x2),
    y1: Math.min(selection.y1, selection.y2),
    x2: Math.max(selection.x1, selection.x2),
    y2: Math.max(selection.y1, selection.y2),
  };
}

function getManualCropCoordinates(selection: ManualSelection | null, image: HTMLImageElement | null) {
  if (!selection || !image || image.naturalWidth <= 0 || image.naturalHeight <= 0) return null;

  return {
    x1: Math.round(selection.x1 * image.naturalWidth),
    y1: Math.round(selection.y1 * image.naturalHeight),
    x2: Math.round(selection.x2 * image.naturalWidth),
    y2: Math.round(selection.y2 * image.naturalHeight),
  };
}

function formatManualSelection(coordinates: { x1: number; y1: number; x2: number; y2: number } | null): string {
  if (!coordinates) return "No selection";

  return `${Math.max(0, coordinates.x2 - coordinates.x1)} x ${Math.max(0, coordinates.y2 - coordinates.y1)} px`;
}

function buildCandidateRows(bestDetection: PlateDetection | null) {
  if (!bestDetection) return [];

  const rows = [
    {
      plate: bestDetection.text || "Unread",
      confidence: bestDetection.final_confidence,
    },
    ...bestDetection.candidates,
  ];
  const seen = new Set<string>();

  return rows
    .filter((candidate) => {
      const key = candidate.plate.toUpperCase();
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    })
    .slice(0, 10);
}

function createExportPayload(
  result: PlateDetectionResponse,
  file: File | null,
  processingTimeMs: number | null,
  qualityScore: number,
  processedAt: string | null,
): InvestigationExport {
  return {
    exported_at: new Date().toISOString(),
    processed_at: processedAt,
    uploaded_file_name: file?.name ?? null,
    stored_filename: result.filename,
    original_image_url: resolveAssetUrl(`/uploads/${result.filename}`),
    annotated_image_url: result.annotated_image_url ? resolveAssetUrl(result.annotated_image_url) : null,
    detector: result.detector,
    model_configured: result.model_configured,
    plates_detected: result.plates_detected,
    quality_score: qualityScore,
    processing_time_ms: processingTimeMs,
    detections: result.detections.map((detection, index) => ({
      index: index + 1,
      plate: detection.text,
      final_confidence: detection.final_confidence,
      detector_confidence: detection.confidence,
      ocr_confidence: detection.ocr_confidence,
      coordinates: detection.coordinates,
      cropped_plate_url: resolveAssetUrl(detection.cropped_plate_url),
      candidates: detection.candidates,
    })),
  };
}

function createCsv(payload: InvestigationExport): string {
  const headers = [
    "stored_filename",
    "uploaded_file_name",
    "detector",
    "model_configured",
    "plates_detected",
    "quality_score",
    "processing_time_ms",
    "detection_index",
    "plate",
    "final_confidence",
    "detector_confidence",
    "ocr_confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "cropped_plate_url",
    "candidates",
  ];

  const detections = payload.detections.length > 0 ? payload.detections : [null];
  const rows = detections.map((detection) => {
    const base = [
      payload.stored_filename,
      payload.uploaded_file_name ?? "",
      payload.detector,
      String(payload.model_configured),
      String(payload.plates_detected),
      String(payload.quality_score),
      payload.processing_time_ms?.toString() ?? "",
    ];

    if (!detection) {
      return [...base, "", "", "", "", "", "", "", "", "", ""].map(csvEscape).join(",");
    }

    return [
      ...base,
      String(detection.index),
      detection.plate,
      String(detection.final_confidence),
      String(detection.detector_confidence),
      String(detection.ocr_confidence),
      String(detection.coordinates.x1),
      String(detection.coordinates.y1),
      String(detection.coordinates.x2),
      String(detection.coordinates.y2),
      detection.cropped_plate_url,
      detection.candidates.map((candidate) => `${candidate.plate}:${candidate.confidence}`).join(" | "),
    ]
      .map(csvEscape)
      .join(",");
  });

  return [headers.join(","), ...rows].join("\n");
}

function createPdf(payload: InvestigationExport): Blob {
  const lines = [
    "LPR Investigation Report",
    `Exported: ${formatTimestamp(payload.exported_at)}`,
    `Processed: ${payload.processed_at ? formatTimestamp(payload.processed_at) : "Pending"}`,
    `Uploaded file: ${payload.uploaded_file_name ?? "Unknown"}`,
    `Stored file: ${payload.stored_filename}`,
    `Detector: ${payload.detector}`,
    `Model configured: ${payload.model_configured ? "yes" : "no"}`,
    `Plates detected: ${payload.plates_detected}`,
    `Quality score: ${payload.quality_score}/100`,
    `Processing time: ${formatDuration(payload.processing_time_ms)}`,
    `Annotated image: ${payload.annotated_image_url ?? "Not generated"}`,
    "",
    "Detections",
    ...payload.detections.flatMap((detection) => [
      `#${detection.index} Plate: ${detection.plate || "Unread"}`,
      `  Final: ${formatPercent(detection.final_confidence)} | OCR: ${formatPercent(
        detection.ocr_confidence,
      )} | Detector: ${formatPercent(detection.detector_confidence)}`,
      `  Box: x1 ${detection.coordinates.x1}, y1 ${detection.coordinates.y1}, x2 ${detection.coordinates.x2}, y2 ${detection.coordinates.y2}`,
      `  Crop: ${detection.cropped_plate_url}`,
      `  Candidates: ${
        detection.candidates.length > 0
          ? detection.candidates.map((candidate) => `${candidate.plate} ${formatPercent(candidate.confidence)}`).join(", ")
          : "None"
      }`,
      "",
    ]),
  ];

  const wrappedLines = lines.flatMap((line) => wrapLine(line, 92)).slice(0, 52);
  const content = buildPdfContent(wrappedLines);
  const objects = [
    "<< /Type /Catalog /Pages 2 0 R >>",
    "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
    "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
    `<< /Length ${content.length} >>\nstream\n${content}\nendstream`,
    "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
  ];

  let pdf = "%PDF-1.4\n";
  const offsets = [0];
  objects.forEach((object, index) => {
    offsets.push(pdf.length);
    pdf += `${index + 1} 0 obj\n${object}\nendobj\n`;
  });

  const xrefStart = pdf.length;
  pdf += `xref\n0 ${objects.length + 1}\n`;
  pdf += "0000000000 65535 f \n";
  offsets.slice(1).forEach((offset) => {
    pdf += `${String(offset).padStart(10, "0")} 00000 n \n`;
  });
  pdf += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xrefStart}\n%%EOF`;

  return new Blob([pdf], { type: "application/pdf" });
}

function buildPdfContent(lines: string): string;
function buildPdfContent(lines: string[]): string;
function buildPdfContent(lines: string | string[]): string {
  const pageLines = Array.isArray(lines) ? lines : [lines];
  const operations = ["BT", "/F1 16 Tf", `1 0 0 1 50 758 Tm`, `(${escapePdfText(pageLines[0] ?? "")}) Tj`, "/F1 10 Tf"];

  pageLines.slice(1).forEach((line, index) => {
    operations.push(`1 0 0 1 50 ${734 - index * 14} Tm`, `(${escapePdfText(line)}) Tj`);
  });
  operations.push("ET");
  return operations.join("\n");
}

function wrapLine(line: string, maxLength: number): string[] {
  if (line.length <= maxLength) return [line];

  const words = line.split(" ");
  const wrapped: string[] = [];
  let current = "";

  words.forEach((word) => {
    if (`${current} ${word}`.trim().length > maxLength) {
      if (current) wrapped.push(current);
      current = word;
      return;
    }
    current = `${current} ${word}`.trim();
  });

  if (current) wrapped.push(current);
  return wrapped;
}

function downloadBlob(blob: Blob, filename: string) {
  const objectUrl = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = objectUrl;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(objectUrl);
}

function buildFilename(filename: string, extension: "json" | "csv" | "pdf"): string {
  const base = filename.replace(/\.[^.]+$/, "").replace(/[^a-z0-9_-]+/gi, "_");
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  return `${base || "lpr_result"}_${stamp}.${extension}`;
}

function csvEscape(value: string): string {
  if (/[",\n]/.test(value)) return `"${value.replace(/"/g, '""')}"`;
  return value;
}

function escapePdfText(value: string): string {
  return value.replace(/\\/g, "\\\\").replace(/\(/g, "\\(").replace(/\)/g, "\\)");
}

function formatPercent(value: number): string {
  return `${Math.round(clamp(value, 0, 1) * 100)}%`;
}

function formatDuration(value: number | null): string {
  if (value === null) return "Pending";
  if (value < 1000) return `${value} ms`;
  return `${(value / 1000).toFixed(2)} s`;
}

function formatTimestamp(value: string): string {
  return new Intl.DateTimeFormat("en-IN", {
    dateStyle: "medium",
    timeStyle: "medium",
  }).format(new Date(value));
}

function formatBox(detection: PlateDetection): string {
  const { x1, y1, x2, y2 } = detection.coordinates;
  return `x1 ${x1} / y1 ${y1} / x2 ${x2} / y2 ${y2}`;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
