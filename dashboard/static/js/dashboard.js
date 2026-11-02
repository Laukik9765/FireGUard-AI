/**
 * dashboard/static/js/dashboard.js
 * ──────────────────────────────────
 * FireGuard AI — Cyber-Situational Awareness System
 *
 * Responsibilities:
 * - 9-page view switcher (Live Monitor, Analysis, File Analysis, LocateAnything,
 *   Events, Incident History, Reports, Cameras, Settings)
 * - Live clock & date updater
 * - Fullscreen toggle
 * - Incident History interactive table & preview updater
 * - LocateAnything view switcher (Localization Result vs Mask View) & JSON copy/export
 * - File Analysis interactive upload & analyzer
 * - Real-time Socket.IO streams & live camera inference
 */

"use strict";

// ── Utility: HTML Escaping ────────────────────────────────────
function escHtml(str) {
  if (str == null) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// ── Socket.IO ─────────────────────────────────────────────────
const socket = io({ transports: ["websocket", "polling"] });

// ── DOM Helper ────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const pages    = document.querySelectorAll(".page");
const navItems = document.querySelectorAll(".nav-item");

// Live Monitor Elements
const cameraImg         = $("camera-img");
const cameraPlaceholder = $("camera-placeholder");
const camBadgeLive      = $("cam-badge-live");
const camBadgeFps       = $("cam-badge-fps");
const statFire          = $("stat-fire");
const statFireSub       = $("stat-fire-sub");
const statSmoke         = $("stat-smoke");
const statSmokeDensity  = $("stat-smoke-density");
const statSmokeCoverage = $("stat-smoke-coverage");
const statSmokeTrend    = $("stat-smoke-trend");
const statPeople        = $("stat-people");
const statLivestock     = $("stat-livestock");
const hazardValue       = $("hazard-value");
const hazardTrend       = $("hazard-trend");
const btnStartCam       = $("btn-start-cam");
const btnStopCam        = $("btn-stop-cam");
const cameraSelect      = $("camera-select");

// Alert Elements
const cardAlert         = $("card-alert");
const statAlertTitle    = $("stat-alert-title");
const statAlertMsg      = $("stat-alert-msg");
const statAlertTime     = $("stat-alert-time");

// Header Elements
const clockTimeEl       = $("clock-time");
const clockDateEl       = $("clock-date");
const sysStatusPill     = $("sys-status-pill");
const btnFullscreen     = $("btn-fullscreen");

// ── State ─────────────────────────────────────────────────────
let cameraRunning = false;
let frameCount    = 0;
let fpsTimer      = null;
let lastFpsTs     = Date.now();
let currentPage   = "live";

// ══════════════════════════════════════════════════════════════
// 1. NAVIGATION SYSTEM
// ══════════════════════════════════════════════════════════════
navItems.forEach((item) => {
  item.addEventListener("click", () => {
    const page = item.dataset.page;
    if (page) navigateTo(page);
  });
});

function navigateTo(page) {
  currentPage = page;

  pages.forEach((p) => p.classList.remove("active"));
  navItems.forEach((n) => {
    n.classList.remove("active");
    n.removeAttribute("aria-current");
  });

  const targetPage = $("page-" + page);
  const targetNav  = $("nav-" + page);
  if (targetPage) targetPage.classList.add("active");
  if (targetNav) {
    targetNav.classList.add("active");
    targetNav.setAttribute("aria-current", "page");
  }

  if (page === "history") renderIncidentsTable();
  if (page === "events") loadEvents();
}

// ══════════════════════════════════════════════════════════════
// 2. LIVE CLOCK & SYSTEM HEADER
// ══════════════════════════════════════════════════════════════
function updateClock() {
  const now = new Date();
  if (clockTimeEl) {
    const h = String(now.getHours()).padStart(2, "0");
    const m = String(now.getMinutes()).padStart(2, "0");
    const s = String(now.getSeconds()).padStart(2, "0");
    clockTimeEl.textContent = `${h}:${m}:${s}`;
  }
  if (clockDateEl) {
    const days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
    const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    const day = days[now.getDay()];
    const date = String(now.getDate()).padStart(2, "0");
    const month = months[now.getMonth()];
    const year = now.getFullYear();
    clockDateEl.textContent = `${day}, ${date} ${month} ${year}`;
  }
}
setInterval(updateClock, 1000);
updateClock();

// Fullscreen toggle
if (btnFullscreen) {
  btnFullscreen.addEventListener("click", () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(() => {});
    } else {
      if (document.exitFullscreen) document.exitFullscreen();
    }
  });
}

// ══════════════════════════════════════════════════════════════
// 3. INCIDENT HISTORY DATASET & INTERACTION (Image 1)
// ══════════════════════════════════════════════════════════════
const incidentsData = [
  {
    id: "INC-0142",
    datetime: "31 Oct 2026 13:37:41",
    camera: "Camera 1 – Main Hall",
    type: "Smoke",
    severity: "High",
    smokeScore: "0.88 (Heavy)",
    people: 2,
    hazards: "1 (LPG Cylinder)",
    image: "/static/img/yolo_detection.jpg",
    boxes: [
      { tag: "Smoke (0.88)", cls: "box-smoke tag-smoke", top: "22%", left: "48%", w: "34%", h: "42%" },
      { tag: "Person (0.94)", cls: "box-person tag-person", top: "40%", left: "27%", w: "9%", h: "38%" },
      { tag: "Person (0.91)", cls: "box-person tag-person", top: "42%", left: "38%", w: "8%", h: "32%" },
      { tag: "LPG Cylinder (0.93)", cls: "box-lpg tag-lpg", top: "58%", left: "85%", w: "10%", h: "28%" }
    ]
  },
  {
    id: "INC-0141",
    datetime: "31 Oct 2026 12:08:17",
    camera: "Camera 2 – Lab",
    type: "Fire",
    severity: "High",
    smokeScore: "0.72 (Moderate)",
    people: 0,
    hazards: "1 (Open Flame)",
    image: "/static/img/yolo_detection.jpg",
    boxes: [
      { tag: "Fire (0.97)", cls: "box-fire tag-smoke", top: "48%", left: "20%", w: "18%", h: "34%" }
    ]
  },
  {
    id: "INC-0140",
    datetime: "30 Oct 2026 19:45:03",
    camera: "Camera 1 – Main Hall",
    type: "LPG Cylinder",
    severity: "Medium",
    smokeScore: "0.12 (Normal)",
    people: 1,
    hazards: "1 (LPG Cylinder)",
    image: "/static/img/yolo_detection.jpg",
    boxes: [
      { tag: "LPG Cylinder (0.92)", cls: "box-lpg tag-lpg", top: "60%", left: "48%", w: "12%", h: "26%" }
    ]
  },
  {
    id: "INC-0139",
    datetime: "30 Oct 2026 15:22:11",
    camera: "Camera 3 – Store",
    type: "Smoke",
    severity: "Medium",
    smokeScore: "0.54 (Moderate)",
    people: 0,
    hazards: "None",
    image: "/static/img/yolo_detection.jpg",
    boxes: [
      { tag: "Smoke (0.54)", cls: "box-smoke tag-smoke", top: "25%", left: "35%", w: "30%", h: "35%" }
    ]
  },
  {
    id: "INC-0138",
    datetime: "29 Oct 2026 11:07:36",
    camera: "Camera 2 – Lab",
    type: "Smoke",
    severity: "Low",
    smokeScore: "0.31 (Low)",
    people: 1,
    hazards: "None",
    image: "/static/img/yolo_detection.jpg",
    boxes: [
      { tag: "Smoke (0.31)", cls: "box-smoke tag-smoke", top: "30%", left: "40%", w: "22%", h: "25%" }
    ]
  },
  {
    id: "INC-0137",
    datetime: "28 Oct 2026 17:59:20",
    camera: "Camera 1 – Main Hall",
    type: "Fire",
    severity: "High",
    smokeScore: "0.81 (Heavy)",
    people: 2,
    hazards: "1 (Open Flame)",
    image: "/static/img/yolo_detection.jpg",
    boxes: [
      { tag: "Fire (0.91)", cls: "box-fire tag-smoke", top: "50%", left: "22%", w: "16%", h: "30%" }
    ]
  },
  {
    id: "INC-0136",
    datetime: "27 Oct 2026 10:41:08",
    camera: "Camera 3 – Store",
    type: "Electrical",
    severity: "Medium",
    smokeScore: "0.45 (Moderate)",
    people: 0,
    hazards: "1 (Electrical Arc)",
    image: "/static/img/yolo_detection.jpg",
    boxes: [
      { tag: "Electrical Hazard", cls: "box-lpg tag-lpg", top: "45%", left: "55%", w: "14%", h: "20%" }
    ]
  },
  {
    id: "INC-0135",
    datetime: "26 Oct 2026 16:20:50",
    camera: "Camera 1 – Entrance",
    type: "Smoke",
    severity: "Low",
    smokeScore: "0.28 (Low)",
    people: 3,
    hazards: "None",
    image: "/static/img/yolo_detection.jpg",
    boxes: [
      { tag: "Smoke (0.28)", cls: "box-smoke tag-smoke", top: "32%", left: "38%", w: "25%", h: "28%" }
    ]
  }
];

let selectedIncidentId = "INC-0142";

function renderIncidentsTable(filterText = "") {
  const tbody = $("incidents-tbody");
  if (!tbody) return;

  tbody.innerHTML = "";
  const filter = filterText.toLowerCase();

  const filtered = incidentsData.filter((inc) => {
    return (
      inc.id.toLowerCase().includes(filter) ||
      inc.camera.toLowerCase().includes(filter) ||
      inc.type.toLowerCase().includes(filter) ||
      inc.severity.toLowerCase().includes(filter)
    );
  });

  filtered.forEach((inc) => {
    const tr = document.createElement("tr");
    tr.className = `incident-row ${inc.id === selectedIncidentId ? "active" : ""}`;
    tr.dataset.id = inc.id;

    // Severity badge class
    let badgeClass = "badge-low";
    if (inc.severity === "High") badgeClass = "badge-high";
    else if (inc.severity === "Medium") badgeClass = "badge-medium";

    // Incident type icon
    let typeIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:14px;height:14px;margin-right:6px;vertical-align:-2px;"><path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"/></svg>`;
    if (inc.type === "Fire") {
      typeIcon = `<svg viewBox="0 0 24 24" fill="currentColor" style="width:14px;height:14px;margin-right:6px;vertical-align:-2px;color:#ef233c;"><path d="M12 2c0 0-3 4-3 7a3 3 0 0 0 6 0c0-3-3-7-3-7z"/></svg>`;
    } else if (inc.type.includes("LPG") || inc.type === "Electrical") {
      typeIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:14px;height:14px;margin-right:6px;vertical-align:-2px;color:#ffb703;"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/></svg>`;
    }

    tr.innerHTML = `
      <td class="mono-cell">${inc.id}</td>
      <td>${inc.datetime}</td>
      <td>${inc.camera}</td>
      <td>${typeIcon} ${inc.type}</td>
      <td><span class="badge ${badgeClass}">${inc.severity}</span></td>
      <td class="action-icons-cell">
        <button class="table-action-icon btn-view-inc" title="View Incident" aria-label="View">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
        </button>
        <button class="table-action-icon btn-dl-inc" title="Download Report" aria-label="Download">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
        </button>
      </td>
    `;

    tr.addEventListener("click", () => {
      selectIncident(inc.id);
    });

    tbody.appendChild(tr);
  });
}

function selectIncident(id) {
  selectedIncidentId = id;
  const inc = incidentsData.find((x) => x.id === id);
  if (!inc) return;

  // Highlight active row
  document.querySelectorAll(".incident-row").forEach((row) => {
    row.classList.toggle("active", row.dataset.id === id);
  });

  // Update detail panel
  if ($("detail-inc-id")) $("detail-inc-id").textContent = `ID: ${inc.id}`;
  if ($("detail-cam-tag")) $("detail-cam-tag").textContent = inc.camera;
  if ($("detail-datetime")) $("detail-datetime").textContent = inc.datetime;
  if ($("detail-location")) $("detail-location").textContent = inc.camera;
  if ($("detail-type")) $("detail-type").textContent = inc.type;

  const sevBadge = $("detail-severity");
  if (sevBadge) {
    sevBadge.textContent = inc.severity;
    sevBadge.className = `badge ${inc.severity === "High" ? "badge-high" : inc.severity === "Medium" ? "badge-medium" : "badge-low"}`;
  }

  if ($("detail-smoke-score")) $("detail-smoke-score").textContent = inc.smokeScore;
  if ($("detail-people-count")) $("detail-people-count").textContent = inc.people;
  if ($("detail-hazard-count")) $("detail-hazard-count").textContent = inc.hazards;

  // Update preview boxes
  const previewBox = document.querySelector(".preview-media-box");
  if (previewBox && inc.boxes) {
    const existingBoxes = previewBox.querySelectorAll(".overlay-box");
    existingBoxes.forEach((b) => b.remove());

    inc.boxes.forEach((b) => {
      const boxEl = document.createElement("div");
      const [boxCls, tagCls] = b.cls.split(" ");
      boxEl.className = `overlay-box ${boxCls}`;
      boxEl.style.top = b.top;
      boxEl.style.left = b.left;
      boxEl.style.width = b.w;
      boxEl.style.height = b.h;
      boxEl.innerHTML = `<span class="box-tag ${tagCls}">${b.tag}</span>`;
      previewBox.appendChild(boxEl);
    });
  }
}

// Search input listener
const incidentSearch = $("incident-search");
if (incidentSearch) {
  incidentSearch.addEventListener("input", (e) => {
    renderIncidentsTable(e.target.value);
  });
}

// Initial table render
renderIncidentsTable();

// ══════════════════════════════════════════════════════════════
// 4. LOCATEANYTHING VIEWS & JSON COPY/EXPORT (Images 4 & 5)
// ══════════════════════════════════════════════════════════════
const tabLocalization = $("tab-localization-result");
const tabMaskView     = $("tab-mask-view");
const locateDisplay   = $("locate-stream-img");
const btnCopyJson     = $("btn-copy-json");
const btnExportJson   = $("btn-export-json");

if (tabLocalization && tabMaskView) {
  tabLocalization.addEventListener("click", () => {
    tabLocalization.classList.add("active");
    tabMaskView.classList.remove("active");
    if (locateDisplay) locateDisplay.src = "/static/img/yolo_detection.jpg";
    document.querySelectorAll(".locate-video-container .overlay-box").forEach(b => b.classList.remove("hidden"));
  });

  tabMaskView.addEventListener("click", () => {
    tabMaskView.classList.add("active");
    tabLocalization.classList.remove("active");
    if (locateDisplay) locateDisplay.src = "/static/img/segmentation_mask.png";
    document.querySelectorAll(".locate-video-container .overlay-box").forEach(b => b.classList.add("hidden"));
  });
}

// Copy JSON
if (btnCopyJson) {
  btnCopyJson.addEventListener("click", () => {
    const jsonCode = $("locate-json-display") || $("locate-json-code");
    if (jsonCode) {
      navigator.clipboard.writeText(jsonCode.textContent).then(() => {
        const origTitle = btnCopyJson.getAttribute("title");
        btnCopyJson.setAttribute("title", "Copied to clipboard!");
        btnCopyJson.style.color = "var(--accent-green)";
        setTimeout(() => {
          btnCopyJson.setAttribute("title", origTitle);
          btnCopyJson.style.color = "";
        }, 1500);
      });
    }
  });
}

// Export JSON
if (btnExportJson) {
  btnExportJson.addEventListener("click", () => {
    const jsonEl = $("locate-json-display") || $("locate-json-code");
    const jsonCode = jsonEl ? jsonEl.textContent : "{}";
    const blob = new Blob([jsonCode], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "locate_anything_output.json";
    a.click();
    URL.revokeObjectURL(url);
  });
}

// ══════════════════════════════════════════════════════════════
// 5. FILE ANALYSIS UPLOAD & REAL INFERENCE
// ══════════════════════════════════════════════════════════════
let currentUploadedFile = null;
let latestAnalysisReport = null;
let isAnalyzingFile = false;

const fileDropzone     = $("file-dropzone");
const fileInputCustom  = $("file-analysis-input");
const btnChooseFile    = $("btn-choose-file");
const btnAnalyzeFile   = $("btn-analyze-file");
const uploadedChip     = $("uploaded-chip");
const btnRemoveFile    = $("btn-remove-file");
const analysisStatus   = $("file-analysis-status");
const btnDlReport      = $("btn-dl-file-report");

if (btnChooseFile && fileInputCustom) {
  btnChooseFile.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInputCustom.click();
  });
}

if (fileDropzone && fileInputCustom) {
  fileDropzone.addEventListener("click", () => fileInputCustom.click());

  fileDropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    fileDropzone.style.borderColor = "var(--accent-red)";
  });

  fileDropzone.addEventListener("dragleave", () => {
    fileDropzone.style.borderColor = "";
  });

  fileDropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    fileDropzone.style.borderColor = "";
    if (e.dataTransfer.files.length) {
      selectUploadedFile(e.dataTransfer.files[0]);
    }
  });

  fileInputCustom.addEventListener("change", (e) => {
    if (e.target.files.length) {
      selectUploadedFile(e.target.files[0]);
    }
  });
}

function selectUploadedFile(file) {
  currentUploadedFile = file;
  if (uploadedChip) {
    uploadedChip.classList.remove("hidden");
    const filenameEl = $("chip-filename");
    const detailsEl  = $("chip-details");
    if (filenameEl) filenameEl.textContent = file.name;
    if (detailsEl) {
      const isVid = /\.(mp4|avi|mov|mkv)$/i.test(file.name);
      const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
      detailsEl.textContent = `${isVid ? "Video" : "Image"} • ${sizeMb} MB`;
    }
  }
}

if (btnRemoveFile && uploadedChip) {
  btnRemoveFile.addEventListener("click", (e) => {
    e.stopPropagation();
    currentUploadedFile = null;
    uploadedChip.classList.add("hidden");
    if (fileInputCustom) fileInputCustom.value = "";
  });
}

// Socket progress for video analysis - ignore if not actively analyzing
socket.on("video_progress", (data) => {
  if (isAnalyzingFile && analysisStatus) {
    analysisStatus.innerHTML = `<span class="status-pulse-dot" style="background:#ffb703;box-shadow:0 0 8px #ffb703;"></span> Processing frame ${data.frame}/${data.total} (${data.timestamp})…`;
    analysisStatus.style.color = "var(--accent-amber)";
    analysisStatus.style.borderColor = "rgba(255,183,3,0.3)";
  }
});

if (btnAnalyzeFile) {
  btnAnalyzeFile.addEventListener("click", async () => {
    if (!currentUploadedFile) {
      if (fileInputCustom) fileInputCustom.click();
      return;
    }

    const isVideo = /\.(mp4|avi|mov|mkv)$/i.test(currentUploadedFile.name);
    const endpoint = isVideo ? "/api/analyse/video" : "/api/analyse/image";
    isAnalyzingFile = true;

    if (analysisStatus) {
      analysisStatus.innerHTML = `<span class="status-pulse-dot" style="background:#ffb703;box-shadow:0 0 8px #ffb703;"></span> Processing AI Inference…`;
      analysisStatus.style.color = "var(--accent-amber)";
      analysisStatus.style.borderColor = "rgba(255,183,3,0.3)";
    }
    btnAnalyzeFile.disabled = true;

    try {
      const formData = new FormData();
      formData.append("file", currentUploadedFile);

      const res = await fetch(endpoint, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (!data.ok) throw new Error(data.error || "Analysis failed");

      isAnalyzingFile = false;
      latestAnalysisReport = data.report;
      renderAnalysisReport(data.report, isVideo);

      if (analysisStatus) {
        analysisStatus.innerHTML = `<span class="pill-dot"></span> Analysis Completed`;
        analysisStatus.style.color = "var(--accent-green)";
        analysisStatus.style.borderColor = "rgba(0,230,118,0.3)";
      }
    } catch (err) {
      isAnalyzingFile = false;
      console.error("Analysis error:", err);
      if (analysisStatus) {
        analysisStatus.innerHTML = `<span class="status-pulse-dot" style="background:#ef233c;box-shadow:0 0 8px #ef233c;"></span> Analysis Failed: ${escHtml(err.message)}`;
        analysisStatus.style.color = "var(--accent-red)";
        analysisStatus.style.borderColor = "rgba(239,35,60,0.3)";
      }
    } finally {
      btnAnalyzeFile.disabled = false;
    }
  });
}

function renderAnalysisReport(report, isVideo) {
  if (!report) return;

  // 1. Header meta
  try {
    const metaEl = $("player-header-meta");
    if (metaEl) {
      const dur = report.duration_formatted || "00:00:01";
      const procTime = report.processing_time || 0.5;
      metaEl.textContent = `${report.source} | Duration: ${dur} | Processed in: ${procTime} s`;
    }
  } catch (e) {
    console.warn("Meta update:", e);
  }

  // 2. Video / image preview frame
  try {
    const previewImg = $("analysis-preview-frame");
    const videoBox   = $("analysis-video-box");

    if (previewImg) {
      const previewUrl = report.preview_url || report.annotated_url;
      if (previewUrl) {
        previewImg.src = previewUrl + "?t=" + Date.now();
      }
    }

    // Clear static placeholder overlay boxes
    if (videoBox) {
      videoBox.querySelectorAll(".overlay-box").forEach(b => b.remove());
    }
  } catch (e) {
    console.warn("Preview frame update:", e);
  }

  // 3. Scrubber time
  try {
    const timeDisplay = $("player-time-display");
    if (timeDisplay) {
      timeDisplay.textContent = `00:00 / ${report.duration_formatted || "00:00"}`;
    }
    const sliderFilled = document.querySelector(".player-timeline-slider .slider-filled");
    const sliderThumb  = document.querySelector(".player-timeline-slider .slider-thumb");
    if (sliderFilled) sliderFilled.style.width = "100%";
    if (sliderThumb) sliderThumb.style.left = "100%";
  } catch (e) {
    console.warn("Scrubber update:", e);
  }

  // 4. Detection Summary 4-card row
  try {
    const smokeLvlEl = $("summary-smoke-level");
    if (smokeLvlEl) {
      const sHazard = report.smoke?.hazard || "None";
      smokeLvlEl.textContent = sHazard;
      smokeLvlEl.className = `stat-cell-val ${sHazard === "Heavy" ? "red" : sHazard === "Moderate" ? "amber" : sHazard === "Low" ? "cyan" : "state-normal"}`;
    }

    const peopleEl = $("summary-people-count");
    if (peopleEl) {
      peopleEl.textContent = report.people?.count ?? 0;
      peopleEl.className = "stat-cell-val cyan";
    }

    const hazardEl = $("summary-hazard-count");
    if (hazardEl) {
      const hCount = report.livestock?.count ?? 0;
      hazardEl.textContent = hCount;
      hazardEl.className = `stat-cell-val ${hCount > 0 ? "amber" : "state-normal"}`;
    }

    const fireEl = $("summary-fire-status");
    if (fireEl) {
      const isFire = Boolean(report.fire?.detected);
      fireEl.textContent = isFire ? "Detected" : "None";
      fireEl.className = `stat-cell-val ${isFire ? "red" : "state-normal"}`;
    }
  } catch (e) {
    console.warn("Summary row update:", e);
  }

  // 5. Circular Smoke Gauge
  try {
    const smokeScore = parseFloat(report.smoke?.smoke_score ?? (((report.smoke?.density ?? 0) / 100).toFixed(2))) || 0.0;
    const scoreNumEl = $("analysis-gauge-score");
    if (scoreNumEl) {
      scoreNumEl.textContent = smokeScore.toFixed(2);
    }

    const gaugePath = $("analysis-gauge-path");
    if (gaugePath) {
      const val = Math.max(0.01, Math.min(1.0, smokeScore));
      const theta = Math.PI - (val * Math.PI);
      const endX = (70 + 55 * Math.cos(theta)).toFixed(1);
      const endY = (90 - 55 * Math.sin(theta)).toFixed(1);
      const largeArc = val > 0.5 ? 1 : 0;
      gaugePath.setAttribute("d", `M 15 90 A 55 55 0 ${largeArc} 1 ${endX} ${endY}`);
      if (smokeScore === 0) {
        gaugePath.style.stroke = "rgba(0, 230, 118, 0.6)";
      } else if (smokeScore < 0.4) {
        gaugePath.style.stroke = "#ffbe0b";
      } else {
        gaugePath.style.stroke = "#ef233c";
      }
    }

    const badgeEl = $("analysis-smoke-level-badge");
    if (badgeEl) {
      const lvl = report.smoke?.hazard || "None";
      badgeEl.textContent = lvl;
      badgeEl.className = `badge ${lvl === "Heavy" ? "badge-high" : lvl === "Moderate" ? "badge-medium" : lvl === "Low" ? "badge-medium" : "badge-low"}`;
    }

    const densityEl = $("analysis-smoke-density");
    if (densityEl) {
      densityEl.textContent = `~ ${Math.round(report.smoke?.density ?? 0)}%`;
    }

    const coverageEl = $("analysis-smoke-coverage");
    if (coverageEl) {
      coverageEl.textContent = `~ ${Math.round(report.smoke?.coverage ?? 0)}%`;
    }

    const trendEl = $("analysis-smoke-trend");
    if (trendEl) {
      const tr = report.smoke?.trend || "Stable";
      trendEl.textContent = tr === "Increasing" ? "↗ Increasing" : tr === "Decreasing" ? "↘ Decreasing" : "→ Stable";
      trendEl.className = `breakdown-val ${tr === "Increasing" ? "red" : ""}`;
    }
  } catch (e) {
    console.warn("Gauge update:", e);
  }

  // 6. Detected Hazards Card
  try {
    const hazTitle = $("analysis-hazard-title");
    const hazConf  = $("analysis-hazard-conf");
    const hazIcon  = $("analysis-hazard-icon");

    if (hazTitle && hazConf) {
      const hazCount = report.livestock?.count ?? 0;
      if (hazCount > 0) {
        hazTitle.textContent = "Combustible / LPG Hazard";
        hazConf.textContent  = `Detected in scene (${hazCount} item${hazCount > 1 ? "s" : ""})`;
        if (hazIcon) {
          hazIcon.style.color = "#ffb703";
          hazIcon.style.background = "rgba(255, 183, 3, 0.15)";
        }
      } else if (report.fire && report.fire.detected) {
        hazTitle.textContent = "Open Flame / Fire Hazard";
        hazConf.textContent  = "High Thermal Combustion Detected";
        if (hazIcon) {
          hazIcon.style.color = "#ef233c";
          hazIcon.style.background = "rgba(239, 35, 60, 0.15)";
        }
      } else {
        hazTitle.textContent = "No Combustible Hazards";
        hazConf.textContent  = "Thresholds safe (0 detected)";
        if (hazIcon) {
          hazIcon.style.color = "#00e676";
          hazIcon.style.background = "rgba(0, 230, 118, 0.15)";
        }
      }
    }
  } catch (e) {
    console.warn("Hazard card update:", e);
  }

  // 7. Frame Timeline Chart
  try {
    const polyline = $("timeline-polyline");
    const dotsG    = $("timeline-dots");
    const xLabels  = $("timeline-x-labels");

    if (report.timeline && report.timeline.length > 0) {
      const pts = report.timeline;
      const len = pts.length;
      const polyCoords = [];
      if (dotsG) dotsG.innerHTML = "";

      pts.forEach((pt, idx) => {
        const x = (35 + (idx / Math.max(1, len - 1)) * 210).toFixed(1);
        let density = pt.smoke_density != null ? pt.smoke_density : (pt.smoke !== "Normal" ? 50 : 0);
        density = Math.min(100, Math.max(0, density));
        const y = (85 - (density / 100) * 67).toFixed(1);
        polyCoords.push(`${x},${y}`);

        if (dotsG) {
          const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
          dot.setAttribute("cx", x);
          dot.setAttribute("cy", y);
          dot.setAttribute("r", "3");
          dot.setAttribute("fill", density > 40 ? "#ef233c" : density > 0 ? "#ffb703" : "#00e676");
          dotsG.appendChild(dot);
        }
      });

      if (polyline) polyline.setAttribute("points", polyCoords.join(" "));

      if (xLabels && pts.length >= 2) {
        xLabels.innerHTML = `
          <span>${pts[0].time}</span>
          <span>${pts[Math.floor(len * 0.25)].time}</span>
          <span>${pts[Math.floor(len * 0.5)].time}</span>
          <span>${pts[Math.floor(len * 0.75)].time}</span>
          <span>${pts[len - 1].time}</span>
        `;
      }
    }
  } catch (e) {
    console.warn("Timeline update:", e);
  }
}

// Download Report button (PDF format)
if (btnDlReport) {
  btnDlReport.addEventListener("click", async () => {
    if (!latestAnalysisReport) {
      alert("Please upload and analyze an image or video file first.");
      return;
    }

    const origHtml = btnDlReport.innerHTML;
    btnDlReport.disabled = true;
    btnDlReport.innerHTML = `<span class="status-pulse-dot" style="background:#00e676;box-shadow:0 0 6px #00e676;display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;"></span> Generating PDF...`;

    try {
      const res = await fetch("/api/report/pdf", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ report: latestAnalysisReport }),
      });

      if (!res.ok) {
        throw new Error(`PDF endpoint returned HTTP ${res.status}`);
      }

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      const srcName = String(latestAnalysisReport.source || "analysis").replace(/\.[^/.]+$/, "");
      a.download = `${srcName}_FireGuard_Report.pdf`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.warn("PDF generation error, falling back to JSON:", err);
      // Fallback to formatted JSON download
      const reportStr = JSON.stringify(latestAnalysisReport, null, 2);
      const blob = new Blob([reportStr], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${latestAnalysisReport.source || "file"}_analysis_report.json`;
      a.click();
      URL.revokeObjectURL(url);
    } finally {
      btnDlReport.disabled = false;
      btnDlReport.innerHTML = origHtml;
    }
  });
}

// ══════════════════════════════════════════════════════════════
// 6. LIVE CAMERA STREAM & SOCKET INFERENCE
// ══════════════════════════════════════════════════════════════
socket.on("connect", () => {
  console.log("[FireGuard AI] WebSocket connected.");
  if (sysStatusPill) {
    sysStatusPill.className = "system-status-pill online";
    sysStatusPill.innerHTML = `<span class="status-pulse-dot"></span><span id="status-label">System Online</span>`;
  }
});

socket.on("disconnect", () => {
  console.warn("[FireGuard AI] WebSocket disconnected.");
  if (sysStatusPill) {
    sysStatusPill.className = "system-status-pill offline";
    sysStatusPill.innerHTML = `<span class="status-pulse-dot" style="background:#ef233c;box-shadow:0 0 8px #ef233c;"></span><span>System Offline</span>`;
  }
});

// Live camera frame
socket.on("frame", ({ image }) => {
  if (cameraPlaceholder && !cameraPlaceholder.classList.contains("hidden")) {
    setCameraRunning(true);
  }
  if (cameraImg) {
    cameraImg.classList.remove("hidden");
    cameraImg.src = "data:image/jpeg;base64," + image;
  }

  frameCount++;
  const now = Date.now();
  if (now - lastFpsTs >= 1000) {
    const fps = (frameCount / ((now - lastFpsTs) / 1000)).toFixed(1);
    if (camBadgeFps) camBadgeFps.textContent = `${fps} fps`;
    frameCount = 0;
    lastFpsTs  = now;
  }
});

// Detection snapshot
socket.on("snapshot", (data) => {
  updateStatusPanel(data);
  if (data.alert) showAlert(data.alert);
});

// Camera Start / Stop
if (btnStartCam) {
  btnStartCam.addEventListener("click", () => startCamera());
}

if (btnStopCam) {
  btnStopCam.addEventListener("click", async () => {
    try {
      await fetch("/api/camera/stop", { method: "POST" });
    } catch (_) {}
    setCameraRunning(false);
  });
}

async function startCamera(camIdx) {
  let cameraIndex = camIdx != null ? camIdx : 1;
  if (camIdx == null && cameraSelect && cameraSelect.value !== "") {
    cameraIndex = parseInt(cameraSelect.value, 10);
  }

  if (btnStartCam) btnStartCam.disabled = true;
  try {
    const res = await fetch("/api/camera/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ camera_index: cameraIndex }),
    });
    const data = await res.json();
    if (data.ok) {
      setCameraRunning(true);
      return true;
    } else {
      console.warn("Camera start:", data.error);
      if (btnStartCam) btnStartCam.disabled = false;
      return false;
    }
  } catch (e) {
    console.warn("Failed to start camera:", e);
    if (btnStartCam) btnStartCam.disabled = false;
    return false;
  }
}

function setCameraRunning(running) {
  cameraRunning = running;
  if (running) {
    if (cameraImg) {
      cameraImg.src = "/video_feed?t=" + Date.now();
      cameraImg.classList.remove("hidden");
    }
    if (cameraPlaceholder) cameraPlaceholder.classList.add("hidden");
    if (camBadgeLive) {
      camBadgeLive.textContent = "LIVE";
      camBadgeLive.classList.add("active");
    }
    if (btnStartCam) {
      btnStartCam.classList.add("hidden");
      btnStartCam.disabled = false;
    }
    if (btnStopCam) btnStopCam.classList.remove("hidden");
  } else {
    if (cameraImg) {
      cameraImg.src = "";
      cameraImg.classList.add("hidden");
    }
    if (cameraPlaceholder) cameraPlaceholder.classList.remove("hidden");
    if (camBadgeLive) {
      camBadgeLive.textContent = "STANDBY";
      camBadgeLive.classList.remove("active");
    }
    if (camBadgeFps) camBadgeFps.textContent = "";
    if (btnStartCam) btnStartCam.classList.remove("hidden");
    if (btnStopCam) btnStopCam.classList.add("hidden");
    resetStatusPanel();
  }
}

function updateStatusPanel(data) {
  // Fire
  const fireState = data.fire_state || "Normal";
  if (statFire) {
    statFire.textContent = fireState.toUpperCase();
    statFire.className = "telemetry-card-value " + (fireState.toLowerCase() === "normal" ? "state-normal" : "red");
  }

  // Smoke
  const smokeState = data.smoke_state || "Normal";
  if (statSmoke) {
    statSmoke.textContent = smokeState.toUpperCase();
    statSmoke.className = "telemetry-card-value " + (smokeState.toLowerCase() === "normal" ? "state-normal" : "red");
  }

  if (data.smoke && data.smoke.detected) {
    const s = data.smoke;
    if (statSmokeDensity) statSmokeDensity.textContent   = `Density: ${s.density.toFixed(0)}/100`;
    if (statSmokeCoverage) statSmokeCoverage.textContent = `Coverage: ${s.coverage.toFixed(0)}%`;
    if (statSmokeTrend) statSmokeTrend.textContent       = `Trend: ${s.trend_arrow} ${s.trend}`;
    if (hazardTrend) hazardTrend.textContent             = `Smoke trend: ${s.trend_arrow} ${s.trend}`;
  } else {
    if (statSmokeDensity) statSmokeDensity.textContent   = "Density: —";
    if (statSmokeCoverage) statSmokeCoverage.textContent = "Coverage: —";
    if (statSmokeTrend) statSmokeTrend.textContent       = "Trend: —";
    if (hazardTrend) hazardTrend.textContent             = "Smoke trend: —";
  }

  if (statPeople) statPeople.textContent       = data.people_count ?? 0;
  if (statLivestock) statLivestock.textContent = data.livestock_count ?? 0;

  const level = (data.hazard_level || "NORMAL").toUpperCase();
  if (hazardValue) {
    hazardValue.textContent = level;
    hazardValue.className = "hazard-pill hazard-" + level;
  }
  updateHazardGauge(level);

  if (level === "NORMAL" && statAlertTitle && statAlertTitle.textContent !== "NO ACTIVE ALERTS") {
    statAlertTitle.textContent = "NO ACTIVE ALERTS";
    statAlertTitle.className = "alert-title state-normal";
    if (statAlertMsg) statAlertMsg.textContent = "All parameters within normal thresholds.";
    if (statAlertTime) statAlertTime.textContent = "";
    if (cardAlert) cardAlert.classList.remove("alert-critical");
  }
}

const GAUGE_LEVELS = { NORMAL: 1, LOW: 2, MODERATE: 3, HIGH: 4, CRITICAL: 5 };
function updateHazardGauge(lvl) {
  const count = GAUGE_LEVELS[(lvl || "NORMAL").toUpperCase()] || 1;
  document.querySelectorAll("#hazard-gauge .gauge-step").forEach((step, idx) => {
    step.classList.toggle("active", idx < count);
  });
}

function resetStatusPanel() {
  if (statFire) {
    statFire.textContent = "—";
    statFire.className = "telemetry-card-value state-normal";
  }
  if (statSmoke) {
    statSmoke.textContent = "—";
    statSmoke.className = "telemetry-card-value state-normal";
  }
  if (statSmokeDensity) statSmokeDensity.textContent   = "Density: —";
  if (statSmokeCoverage) statSmokeCoverage.textContent = "Coverage: —";
  if (statSmokeTrend) statSmokeTrend.textContent       = "Trend: —";
  if (statPeople) statPeople.textContent               = "0";
  if (statLivestock) statLivestock.textContent         = "0";
  if (hazardValue) {
    hazardValue.textContent = "NORMAL";
    hazardValue.className = "hazard-pill hazard-NORMAL";
  }
  if (hazardTrend) hazardTrend.textContent             = "Smoke trend: —";
  updateHazardGauge("NORMAL");
  if (statAlertTitle) {
    statAlertTitle.textContent = "NO ACTIVE ALERTS";
    statAlertTitle.className = "alert-title state-normal";
  }
  if (statAlertMsg) statAlertMsg.textContent = "All parameters within normal thresholds.";
  if (statAlertTime) statAlertTime.textContent = "";
  if (cardAlert) cardAlert.classList.remove("alert-critical");
}

function showAlert(alert) {
  if (!alert) return;
  const time = new Date(alert.timestamp || Date.now()).toLocaleTimeString();
  const lvl  = (alert.level || "warning").toLowerCase();

  if (statAlertTitle) {
    statAlertTitle.textContent = (alert.title || "ALERT").toUpperCase();
    statAlertTitle.className = "alert-title " + (lvl === "critical" ? "state-critical" : "state-normal");
  }
  if (statAlertMsg) statAlertMsg.textContent = alert.message || "";
  if (statAlertTime) statAlertTime.textContent = time;

  if (cardAlert) {
    cardAlert.classList.toggle("alert-critical", lvl === "critical");
  }
}

async function loadCameras() {
  if (!cameraSelect) return;
  try {
    const res = await fetch("/api/cameras");
    const list = await res.json();
    if (Array.isArray(list) && list.length > 0) {
      cameraSelect.innerHTML = "";
      list.forEach((cam) => {
        const opt = document.createElement("option");
        opt.value = cam.index;
        opt.textContent = `${cam.name} (${cam.resolution})`;
        if (cam.index === 1 || (cam.name && cam.name.toLowerCase().includes("iriun"))) {
          opt.selected = true;
        }
        cameraSelect.appendChild(opt);
      });
    }
  } catch (e) {
    console.warn("Could not load camera list:", e);
  }
}

async function loadEvents() {
  const eventsList  = $("events-list");
  const eventsEmpty = $("events-empty");
  if (!eventsList) return;

  try {
    const res    = await fetch("/api/events");
    const events = await res.json();
    if (!events.length) {
      if (eventsEmpty) eventsEmpty.classList.remove("hidden");
      eventsList.innerHTML = "";
      return;
    }
    if (eventsEmpty) eventsEmpty.classList.add("hidden");
    eventsList.innerHTML = "";
    events.forEach((ev) => {
      const card = document.createElement("div");
      card.className = "card";
      card.style.marginBottom = "10px";
      const start = ev.started_at ? new Date(ev.started_at).toLocaleString() : "—";
      card.innerHTML = `
        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
          <strong>Event #${ev.event_id}</strong>
          <span class="badge ${ev.peak_hazard === 'CRITICAL' ? 'badge-high' : 'badge-low'}">${ev.peak_hazard}</span>
        </div>
        <div style="font-size:11px;color:var(--text-muted);">${start}</div>
      `;
      eventsList.appendChild(card);
    });
  } catch (_) {}
}

// ── Initialise ────────────────────────────────────────────────
async function init() {
  await loadCameras();
  setTimeout(() => {
    startCamera(1);
  }, 300);
}

init();
