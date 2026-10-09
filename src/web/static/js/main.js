const fileInput = document.getElementById("file-input");
const dropZone = document.getElementById("drop-zone");
const fileSelected = document.getElementById("file-selected");
const filename = document.getElementById("filename");
const uploadContent = document.querySelector(".upload-content");
const uploadForm = document.getElementById("upload-form");
const loading = document.getElementById("loading");
const submitBtn = document.getElementById("submit-btn");

function showToast(message) {
    const toast = document.getElementById("toast");
    if (!toast) return;

    toast.textContent = message;
    toast.classList.add("show");

    setTimeout(() => {
        toast.classList.remove("show");
    }, 2400);
}

document.addEventListener("click", function (event) {
    const closeBtn = event.target.closest(".flash-close-btn");
    if (!closeBtn) return;

    const flash = closeBtn.closest(".flash-message, .flash, .alert");
    if (flash) flash.remove();

    const container = document.querySelector(".flash-container");
    if (container && !container.querySelector(".flash-message, .flash, .alert")) {
        container.remove();
    }
});

function applyTheme(theme) {
    const t = theme === "dark" ? "dark" : "light";
    document.documentElement.dataset.theme = t;
    localStorage.setItem("theme", t);

    const btn = document.querySelector(".theme-toggle i");
    if (btn) {
        btn.className = t === "dark" ? "fa-solid fa-sun" : "fa-solid fa-moon";
    }
}

function toggleTheme() {
    const current = document.documentElement.dataset.theme || localStorage.getItem("theme") || "light";
    applyTheme(current === "dark" ? "light" : "dark");
}

const _savedTheme = localStorage.getItem("theme");
if (_savedTheme === "dark" || _savedTheme === "light") {
    applyTheme(_savedTheme);
}

function showComingSoon(feature) {
    showToast(`${feature} will be added in the next version.`);
}

function buildTranscriptHeader() {
    const meta = window.transcriptMeta || {};
    const lines = [
        `Title: ${window.transcriptTitle || "Transcript"}`,
    ];

    if (meta.course) lines.push(`Course: ${meta.course}`);
    if (meta.speaker) lines.push(`Speaker: ${meta.speaker}`);
    if (meta.duration) lines.push(`Duration: ${meta.duration}`);
    if (meta.wordCount) lines.push(`Words: ${meta.wordCount}`);
    if (meta.readingTime) lines.push(`Reading time: ${meta.readingTime}`);
    if (window.transcriptDecodeMode) lines.push(`Decoding: ${window.transcriptDecodeMode}`);
    if (meta.processedAt) lines.push(`Processed: ${meta.processedAt}`);
    lines.push("");

    return lines.join("\n");
}

function buildTimestampedText() {
    const segments = window.transcriptSegments || [];
    if (!segments.length) {
        return window.fullTranscript || "";
    }

    return segments
        .map(seg => `[${seg.timestamp || "00:00"}] ${seg.text || ""}`.trim())
        .join("\n");
}

function getTranscriptText(mode) {
    if (mode === "timestamped") {
        return buildTimestampedText();
    }
    return window.fullTranscript || "";
}

if (fileInput) {
    fileInput.addEventListener("change", function () {
        if (this.files.length > 0) {
            showSelectedFile(this.files[0]);
        }
    });
}

if (dropZone) {
    ["dragenter", "dragover"].forEach(eventName => {
        dropZone.addEventListener(eventName, function (e) {
            e.preventDefault();
            dropZone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropZone.addEventListener(eventName, function (e) {
            e.preventDefault();
            dropZone.classList.remove("dragover");
        });
    });

    dropZone.addEventListener("drop", function (e) {
        const files = e.dataTransfer.files;

        if (files.length > 0 && fileInput) {
            fileInput.files = files;
            showSelectedFile(files[0]);
        }
    });
}

function showSelectedFile(file) {
    if (!filename || !fileSelected || !uploadContent) return;

    filename.textContent = file.name;
    fileSelected.style.display = "flex";
    uploadContent.style.display = "none";
}

function clearFile() {
    if (!fileInput || !filename || !fileSelected || !uploadContent) return;

    fileInput.value = "";
    filename.textContent = "";
    fileSelected.style.display = "none";
    uploadContent.style.display = "block";
}

function updateTranscribeProgress(label, percent) {
    void label;
    if (!submitBtn) return;

    const pct = Math.max(0, Math.min(100, Math.round(percent || 0)));
    submitBtn.disabled = true;
    submitBtn.classList.add("is-processing");
    submitBtn.style.setProperty("--progress", `${pct}%`);

    if (!submitBtn.querySelector(".btn-label") || !submitBtn.querySelector(".btn-pct")) {
        submitBtn.innerHTML = '<span class="btn-label">Processing...</span><span class="btn-pct">0%</span>';
    }

    const pctNode = submitBtn.querySelector(".btn-pct");
    if (pctNode) {
        pctNode.textContent = `${pct}%`;
    }
}

function resetSubmitButton() {
    if (!submitBtn) return;
    submitBtn.disabled = false;
    submitBtn.classList.remove("is-processing");
    submitBtn.style.removeProperty("--progress");
    submitBtn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Start Transcription';
}

async function submitTranscription(event) {
    event.preventDefault();
    if (!uploadForm || !fileInput?.files?.length) return;

    const file = fileInput.files[0];
    const formData = new FormData(uploadForm);

    if (loading) loading.style.display = "block";
    updateTranscribeProgress("Preparing upload", 0);

    const durationSec = await getMediaDurationSeconds(file);
    const estimateMs = estimateTranscribeMs(durationSec);
    let processPercent = 35;
    let processSim = null;
    let processingStarted = false;

    function startProcessingProgress() {
        if (processingStarted) return;
        processingStarted = true;
        updateTranscribeProgress("Upload complete — transcribing", 35);
        const start = Date.now();
        processSim = setInterval(() => {
            const elapsed = Date.now() - start;
            const ratio = Math.min(1, elapsed / estimateMs);
            processPercent = 35 + Math.round(ratio * 58);
            const label =
                processPercent < 55
                    ? "Converting audio"
                    : processPercent < 75
                      ? "Running speech recognition"
                      : "Building transcript";
            updateTranscribeProgress(label, processPercent);
        }, 500);
    }

    updateTranscribeProgress("Preparing upload", 2);

    await new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open("POST", uploadForm.action);

        xhr.upload.addEventListener("progress", (ev) => {
            if (ev.lengthComputable && ev.total > 0) {
                const uploadPct = Math.round((ev.loaded / ev.total) * 32);
                updateTranscribeProgress("Uploading file", uploadPct);
                if (ev.loaded >= ev.total) startProcessingProgress();
            }
        });

        xhr.upload.addEventListener("loadend", startProcessingProgress);

        xhr.addEventListener("load", () => {
            if (processSim) clearInterval(processSim);
            if (xhr.status >= 200 && xhr.status < 300) {
                updateTranscribeProgress("Finishing", 98);
                if (xhr.responseURL && (xhr.responseURL.includes("/transcript/") || xhr.responseURL !== window.location.href)) {
                    window.location.href = xhr.responseURL;
                    resolve();
                    return;
                }
                setTimeout(() => {
                    document.open();
                    document.write(xhr.responseText);
                    document.close();
                }, 200);
                resolve();
            } else {
                reject(new Error("Transcription request failed. Check server logs."));
            }
        });

        xhr.addEventListener("error", () => {
            if (processSim) clearInterval(processSim);
            reject(new Error("Network error during upload."));
        });

        xhr.send(formData);
    }).catch((err) => {
        if (processSim) clearInterval(processSim);
        updateTranscribeProgress("Failed", 0);
        showToast(err.message || "Transcription failed.");
        if (loading) loading.style.display = "none";
        resetSubmitButton();
    });
}

if (uploadForm) {
    uploadForm.addEventListener("submit", submitTranscription);
}

/* RESULT PAGE FUNCTIONS */
function copyToClipboard(mode = "plain") {
    const body = getTranscriptText(mode);
    const text = buildTranscriptHeader() + body;

    navigator.clipboard.writeText(text).then(() => {
        const label = mode === "timestamped" ? "Timestamped transcript" : "Transcript";
        showToast(`${label} copied to clipboard.`);
    }).catch(() => {
        showToast("Could not copy to clipboard.");
    });
}

function downloadTXT(mode = "plain") {
    const body = getTranscriptText(mode);
    const text = buildTranscriptHeader() + body;
    const title = window.transcriptTitle || "transcript";
    const suffix = mode === "timestamped" ? "_timestamped" : "";

    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);

    const link = document.createElement("a");
    link.href = url;
    link.download = `${title.replace(/[^a-z0-9]/gi, "_").toLowerCase()}${suffix}.txt`;
    link.click();

    URL.revokeObjectURL(url);
    showToast("Transcript downloaded.");
}

function toggleAnalysisPanel(panelId) {
    const panel = document.getElementById(panelId);
    if (!panel) return;

    const card = panel.closest(".ai-card");
    const isOpen = panel.classList.contains("open");

    if (card) {
        card.querySelectorAll(".analysis-panel.open").forEach(p => p.classList.remove("open"));
    } else {
        document.querySelectorAll(".analysis-panel.open").forEach(p => p.classList.remove("open"));
    }

    if (!isOpen) {
        panel.classList.add("open");
    }
}

function searchKeyword(keyword) {
    const input = document.getElementById("search-input");
    if (!input) return;

    input.value = keyword;
    showLayout("timestamped");
    searchTranscript();
}

function showLayout(type) {
    const timestamped = document.getElementById("timestamped-layout");
    const paragraph = document.getElementById("paragraph-layout");
    const btnTimestamped = document.getElementById("btn-timestamped");
    const btnParagraph = document.getElementById("btn-paragraph");

    if (!timestamped || !paragraph) return;

    if (type === "paragraph") {
        timestamped.style.display = "none";
        paragraph.style.display = "block";
        btnTimestamped?.classList.remove("active");
        btnParagraph?.classList.add("active");
    } else {
        timestamped.style.display = "block";
        paragraph.style.display = "none";
        btnTimestamped?.classList.add("active");
        btnParagraph?.classList.remove("active");
    }
}

function escapeHtml(str) {
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
}

function highlightText(text, query) {
    if (!query) return escapeHtml(text);

    const escapedQuery = query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const re = new RegExp(`(${escapedQuery})`, "gi");
    const parts = escapeHtml(text).split(re);

    return parts
        .map((part, i) => (i % 2 === 1 ? `<mark class="search-hit">${part}</mark>` : part))
        .join("");
}

function searchTranscript() {
    const input = document.getElementById("search-input");
    const noResults = document.getElementById("no-results");
    const segments = document.querySelectorAll(".segment");
    const paragraph = document.getElementById("paragraph-text");

    if (!input) return;

    const query = input.value.trim();
    const queryLower = query.toLowerCase();
    let found = 0;

    segments.forEach(segment => {
        const textEl = segment.querySelector(".segment-text");
        const original = textEl?.dataset.original || textEl?.textContent || "";
        const matches = !queryLower || original.toLowerCase().includes(queryLower);

        if (textEl) {
            textEl.innerHTML = highlightText(original, query);
        }

        if (matches) {
            segment.style.display = "grid";
            found++;
        } else {
            segment.style.display = "none";
        }
    });

    if (paragraph) {
        const original = paragraph.dataset.original || paragraph.textContent || "";
        const matches = !queryLower || original.toLowerCase().includes(queryLower);

        paragraph.innerHTML = highlightText(original, query);

        if (query && document.getElementById("paragraph-layout")?.style.display !== "none") {
            found = matches ? 1 : 0;
        }
    }

    if (noResults) {
        noResults.style.display = query && found === 0 ? "block" : "none";
    }
}

/* ==========================================================
   SPOTIFY-STYLE REAL-TIME TRANSCRIPT SYNC & PLAYBACK ENGINE
   ========================================================== */

let userScrolledManually = false;
let userScrollTimer = null;
let isProgrammaticScroll = false;
let programmaticScrollTimeout = null;
let currentActiveSegment = null;

function setProgrammaticScroll() {
    isProgrammaticScroll = true;
    if (programmaticScrollTimeout) clearTimeout(programmaticScrollTimeout);
    programmaticScrollTimeout = setTimeout(() => {
        isProgrammaticScroll = false;
    }, 500);
}

function showLyricsSyncPill() {
    const container = document.getElementById("lyricsSyncContainer");
    if (container) {
        container.style.display = "flex";
    }
}

function hideLyricsSyncPill() {
    const container = document.getElementById("lyricsSyncContainer");
    if (container) {
        container.style.display = "none";
    }
}

function resumeLyricsSync() {
    userScrolledManually = false;
    if (userScrollTimer) {
        clearTimeout(userScrollTimer);
        userScrollTimer = null;
    }
    hideLyricsSyncPill();

    if (currentActiveSegment) {
        scrollSegmentToCenter(currentActiveSegment, "smooth");
    }
}

function scrollSegmentToCenter(segmentEl, behavior = "smooth") {
    const container = document.getElementById("transcriptScrollArea") || document.querySelector(".transcript-scroll-area");
    if (!container || !segmentEl) return;

    setProgrammaticScroll();

    const containerRect = container.getBoundingClientRect();
    const segmentRect = segmentEl.getBoundingClientRect();

    const currentScrollTop = container.scrollTop;
    const relativeTop = segmentRect.top - containerRect.top;
    const targetScrollTop = currentScrollTop + relativeTop - (containerRect.height / 2) + (segmentRect.height / 2);

    container.scrollTo({
        top: Math.max(0, targetScrollTop),
        behavior: behavior
    });
}

function initLyricsScrollDetection() {
    const scrollArea = document.getElementById("transcriptScrollArea") || document.querySelector(".transcript-scroll-area");
    if (!scrollArea) return;

    const onUserScroll = () => {
        if (isProgrammaticScroll) return;
        userScrolledManually = true;
        showLyricsSyncPill();

        if (userScrollTimer) clearTimeout(userScrollTimer);
        // Auto-resume after 4 seconds of idle inactivity
        userScrollTimer = setTimeout(() => {
            userScrolledManually = false;
            hideLyricsSyncPill();
            if (currentActiveSegment) {
                scrollSegmentToCenter(currentActiveSegment, "smooth");
            }
        }, 4000);
    };

    scrollArea.addEventListener("wheel", onUserScroll, { passive: true });
    scrollArea.addEventListener("touchmove", onUserScroll, { passive: true });
    scrollArea.addEventListener("mousedown", (e) => {
        // If clicking scrollbar track
        if (e.offsetX > scrollArea.clientWidth) {
            onUserScroll();
        }
    });
}

function handleSegmentClick(event, seconds) {
    if (event.target.closest(".note-btn, .segment-note-wrap, .note-save-btn, .segment-note-input")) {
        return;
    }
    jumpToTime(seconds);
}

function jumpToTime(seconds) {
    const audio = document.getElementById("audioPlayer");
    if (!audio) return;

    audio.currentTime = seconds;
    audio.play().catch(() => {});

    // Reset manual scroll lock and instantly align
    userScrolledManually = false;
    hideLyricsSyncPill();
    highlightActiveSegment(seconds, { forceScroll: true, immediate: false });
}

function formatTime(seconds) {
    seconds = Math.max(0, seconds || 0);
    const mm = Math.floor(seconds / 60);
    const ss = Math.floor(seconds % 60);
    return `${String(mm).padStart(2, "0")}:${String(ss).padStart(2, "0")}`;
}

function highlightActiveSegment(currentTime, options = {}) {
    const segments = document.querySelectorAll(".segment");
    if (!segments.length) return;

    const audio = document.getElementById("audioPlayer");
    const isPlaying = audio && !audio.paused && !audio.ended;

    let active = null;

    // Fast range matching: find segment where start <= currentTime < next_start
    for (let i = 0; i < segments.length; i++) {
        const seg = segments[i];
        const start = parseFloat(seg.dataset.time || "0");
        const nextSeg = segments[i + 1];
        const end = nextSeg ? parseFloat(nextSeg.dataset.time || "Infinity") : Infinity;

        if (currentTime >= start && currentTime < end) {
            active = seg;
            break;
        } else if (start <= currentTime) {
            active = seg;
        }
    }

    if (!active && segments.length > 0 && currentTime < parseFloat(segments[0].dataset.time || "0")) {
        active = segments[0];
    }

    // Update active and playing states on all segments
    segments.forEach(segment => {
        const isCurrent = segment === active;

        if (isCurrent) {
            segment.classList.add("active");
            if (isPlaying) {
                segment.classList.add("is-playing");
            } else {
                segment.classList.remove("is-playing");
            }

            const icon = segment.querySelector(".ts-icon");
            if (icon) {
                icon.className = isPlaying
                    ? "fa-solid fa-volume-high ts-icon"
                    : "fa-solid fa-play ts-icon";
            }
        } else {
            segment.classList.remove("active", "is-playing");
            const icon = segment.querySelector(".ts-icon");
            if (icon) {
                icon.className = "fa-solid fa-play ts-icon";
            }
        }
    });

    // Also support sub-sentence highlighting if single-segment spans exist
    if (active) {
        const sentenceSpans = active.querySelectorAll(".lyric-sentence");
        if (sentenceSpans.length > 0) {
            sentenceSpans.forEach(span => {
                const sStart = parseFloat(span.dataset.start || "0");
                const sEnd = parseFloat(span.dataset.end || "Infinity");
                if (currentTime >= sStart && currentTime < sEnd) {
                    span.classList.add("active");
                } else {
                    span.classList.remove("active");
                }
            });
        }
    }

    const segmentChanged = active !== currentActiveSegment;
    currentActiveSegment = active;

    // Auto-scroll logic: only when segment changes or forceScroll is true
    if (active) {
        if (options.forceScroll) {
            scrollSegmentToCenter(active, options.immediate ? "auto" : "smooth");
        } else if (segmentChanged && !userScrolledManually) {
            scrollSegmentToCenter(active, "smooth");
        }
    }
}

// Fallback sentence breakdown for single-segment transcripts
function initSingleSegmentLyricSpans() {
    const segments = document.querySelectorAll(".segment");
    const audio = document.getElementById("audioPlayer");
    if (segments.length !== 1 || !audio) return;

    const singleSeg = segments[0];
    const textEl = singleSeg.querySelector(".segment-text");
    if (!textEl) return;

    const fullText = (textEl.dataset.original || textEl.textContent || "").trim();
    if (!fullText) return;

    // Split into sentences / clauses
    const sentences = fullText.match(/[^.!?]+[.!?]+|[^.!?]+$/g) || [fullText];
    if (sentences.length <= 1) {
        const words = fullText.split(/\s+/);
        if (words.length <= 10) return;
        const chunks = [];
        for (let i = 0; i < words.length; i += 10) {
            chunks.push(words.slice(i, i + 10).join(" "));
        }
        setupSpans(chunks);
    } else {
        setupSpans(sentences);
    }

    function setupSpans(parts) {
        const duration = audio.duration || parseFloat(singleSeg.dataset.duration || "10") || 10;
        const totalLen = parts.reduce((acc, p) => acc + p.trim().length, 0);
        let curTime = 0;

        const html = parts.map(part => {
            const pTrim = part.trim();
            const segDur = (pTrim.length / Math.max(1, totalLen)) * duration;
            const start = curTime;
            const end = curTime + segDur;
            curTime = end;
            return `<span class="lyric-sentence" data-start="${start.toFixed(1)}" data-end="${end.toFixed(1)}">${escapeHtml(part)} </span>`;
        }).join("");

        textEl.innerHTML = html;
    }
}

// Attach player listeners
const audio = document.getElementById("audioPlayer");

if (audio) {
    audio.addEventListener("timeupdate", function () {
        const currentTime = document.getElementById("currentTime");
        const progressFill = document.getElementById("audioProgressFill");

        if (currentTime) {
            currentTime.textContent = formatTime(audio.currentTime);
        }

        if (progressFill && audio.duration) {
            const percent = (audio.currentTime / audio.duration) * 100;
            progressFill.style.width = `${percent}%`;
        }

        highlightActiveSegment(audio.currentTime);
    });

    // Real-time Scrubbing / Seeking: instantly jump transcript to matching minute
    audio.addEventListener("seeking", function () {
        highlightActiveSegment(audio.currentTime, { forceScroll: false });
    });

    audio.addEventListener("seeked", function () {
        highlightActiveSegment(audio.currentTime, { forceScroll: !userScrolledManually });
    });

    audio.addEventListener("play", function () {
        document.querySelector(".audio-artwork-card")?.classList.add("is-playing");
        highlightActiveSegment(audio.currentTime, { forceScroll: !userScrolledManually });
    });

    audio.addEventListener("pause", function () {
        document.querySelector(".audio-artwork-card")?.classList.remove("is-playing");
        highlightActiveSegment(audio.currentTime, { forceScroll: false });
    });

    audio.addEventListener("ended", function () {
        document.querySelector(".audio-artwork-card")?.classList.remove("is-playing");
        highlightActiveSegment(audio.currentTime, { forceScroll: false });
    });

    audio.addEventListener("loadedmetadata", function () {
        initSingleSegmentLyricSpans();
    });
}

// Initialize on document ready
document.addEventListener("DOMContentLoaded", function () {
    initLyricsScrollDetection();
    initSingleSegmentLyricSpans();
    if (audio) {
        highlightActiveSegment(audio.currentTime, { forceScroll: false });
    }
});
