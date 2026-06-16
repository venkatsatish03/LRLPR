"use client";

import { ChangeEvent, DragEvent, useEffect, useRef, useState } from "react";
import { detectLicensePlate } from "@/services/api";
import type { DetectionResult } from "@/types/detection";

export default function Home() {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>("");
  const [isDragging, setIsDragging] = useState(false);
  const [isDetecting, setIsDetecting] = useState(false);
  const [result, setResult] = useState<DetectionResult | null>(null);
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

  function handleFile(selectedFile?: File) {
    if (!selectedFile) return;

    if (!selectedFile.type.startsWith("image/")) {
      setError("Please upload an image file.");
      return;
    }

    setFile(selectedFile);
    setResult(null);
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
      setError("Upload a vehicle image first.");
      return;
    }

    setIsDetecting(true);
    setError("");
    setResult(null);

    try {
      const detection = await detectLicensePlate(file);
      setResult(detection);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Detection failed.");
    } finally {
      setIsDetecting(false);
    }
  }

  return (
    <main className="page-shell">
      <section className="workspace">
        <div className="intro">
          <p className="eyebrow">LPR System</p>
          <h1>License Plate Recognition</h1>
          <p className="subtitle">Upload a vehicle image, preview it, then run plate detection and OCR.</p>
        </div>

        <div className="panel-grid">
          <section className="upload-panel" aria-label="Image upload">
            <div
              className={`drop-zone ${isDragging ? "is-dragging" : ""} ${previewUrl ? "has-preview" : ""}`}
              onClick={() => inputRef.current?.click()}
              onDragOver={(event) => {
                event.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              role="button"
              tabIndex={0}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") inputRef.current?.click();
              }}
            >
              <input ref={inputRef} type="file" accept="image/*" onChange={handleInputChange} hidden />

              {previewUrl ? (
                <img className="preview-image" src={previewUrl} alt="Uploaded vehicle preview" />
              ) : (
                <div className="empty-preview">
                  <div className="upload-icon">+</div>
                  <p>Drag and drop vehicle image</p>
                  <span>or click to browse</span>
                </div>
              )}
            </div>

            <div className="file-row">
              <span>{file ? file.name : "No image selected"}</span>
              {file && (
                <button
                  className="text-button"
                  type="button"
                  onClick={() => {
                    setFile(null);
                    setResult(null);
                    setError("");
                    if (inputRef.current) inputRef.current.value = "";
                  }}
                >
                  Clear
                </button>
              )}
            </div>
          </section>

          <section className="result-panel" aria-label="Detection result">
            <div className="result-header">
              <div>
                <p className="eyebrow">Detection</p>
                <h2>Plate Result</h2>
              </div>
              <button className="detect-button" type="button" disabled={!file || isDetecting} onClick={handleDetect}>
                {isDetecting ? "Detecting..." : "Detect"}
              </button>
            </div>

            {error && <div className="message error">{error}</div>}

            {result ? (
              <div className="result-card">
                <div>
                  <span className="label">Detected Plate</span>
                  <strong className="plate-text">{result.plate || "Not detected"}</strong>
                </div>
                <div>
                  <span className="label">Confidence Score</span>
                  <strong className="confidence">{Math.round(result.confidence * 100)}%</strong>
                </div>
                {result.confidence < 0.8 && (
                  <div>
                    <span className="label">Possible Outcomes</span>
                    <p className="candidate-note">Generated using Indian plate format AA00AA0000.</p>
                    {result.candidates.length > 0 ? (
                      <ol className="candidate-list">
                        {result.candidates.map((candidate) => (
                          <li key={`${candidate.plate}-${candidate.confidence}`}>
                            <span>{candidate.plate}</span>
                            <strong>{Math.round(candidate.confidence * 100)}%</strong>
                          </li>
                        ))}
                      </ol>
                    ) : (
                      <p className="muted-text">No alternate plate candidates were generated.</p>
                    )}
                  </div>
                )}
              </div>
            ) : (
              <div className="empty-result">
                <span className="label">Waiting for image</span>
                <p>Your plate number and confidence score will appear here.</p>
              </div>
            )}
          </section>
        </div>
      </section>
    </main>
  );
}
