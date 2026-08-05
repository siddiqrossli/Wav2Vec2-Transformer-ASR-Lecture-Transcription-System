/**
 * Percentage progress UI for transcription and AI tasks.
 */

function setProgressUI(container, label, percent, options = {}) {
    if (!container) return;
    const pct = Math.min(100, Math.max(0, Math.round(percent)));
    const bannerLabel = options.bannerLabel || label;
    const detailLabel = options.detailLabel ?? (bannerLabel === label ? "" : label);
    container.innerHTML = `
        <div class="progress-status">
            <div class="progress-banner" style="--progress:${pct}%" role="progressbar" aria-valuenow="${pct}" aria-valuemin="0" aria-valuemax="100">
                <span class="progress-banner-label">${escapeHtml(bannerLabel)}</span>
                <span class="progress-pct">${pct}%</span>
            </div>
            ${detailLabel ? `<div class="progress-detail">${escapeHtml(detailLabel)}</div>` : ""}
        </div>
    `;
}

function startSimulatedProgress(container, label, options = {}) {
    const max = options.maxPercent ?? 92;
    const intervalMs = options.intervalMs ?? 450;
    let percent = options.startPercent ?? 5;

    setProgressUI(container, label, percent);

    const timer = setInterval(() => {
        const step = options.step ?? (1.5 + Math.random() * 2.5);
        percent = Math.min(max, percent + step);
        let text = label;
        if (percent >= 70) text = `${label} — finishing up…`;
        else if (percent >= 40) text = `${label} — processing…`;
        setProgressUI(container, text, percent);
    }, intervalMs);

    return {
        stop() {
            clearInterval(timer);
        },
        complete(finalLabel = "Complete") {
            clearInterval(timer);
            setProgressUI(container, finalLabel, 100);
        },
        fail(message = "Failed") {
            clearInterval(timer);
            setProgressUI(container, message, 0);
        },
    };
}

async function runWithProgress(container, label, taskFn, options = {}) {
    const sim = startSimulatedProgress(container, label, options);
    try {
        const result = await taskFn();
        sim.complete(options.completeLabel || "Complete");
        return result;
    } catch (err) {
        sim.fail(options.failLabel || "Failed");
        throw err;
    }
}

function formatAiMarkdown(text) {
    let safe = escapeHtml(text || "");
    safe = safe.replace(/^\+ /gm, "- ");
    safe = safe.replace(/^\* /gm, "- ");
    return safe
        .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
        .replace(/^- (.+)$/gm, "− $1")
        .replace(/\n/g, "<br>");
}

async function getMediaDurationSeconds(file) {
    if (!file) return 120;
    const url = URL.createObjectURL(file);
    try {
        if (file.type.startsWith("video/") || file.type.startsWith("audio/")) {
            const el = document.createElement(file.type.startsWith("video/") ? "video" : "audio");
            el.preload = "metadata";
            const duration = await new Promise((resolve) => {
                el.onloadedmetadata = () => resolve(el.duration || 120);
                el.onerror = () => resolve(120);
                el.src = url;
            });
            return Math.max(10, duration);
        }
    } finally {
        URL.revokeObjectURL(url);
    }
    const sizeMb = file.size / (1024 * 1024);
    return Math.max(30, Math.min(3600, sizeMb * 45));
}

function estimateTranscribeMs(durationSeconds) {
    return Math.max(15000, Math.min(600000, durationSeconds * 1200));
}

/** Open one panel in the AI sidebar (summary vs chat). */
function openAiPanel(panelId) {
    const root = document.getElementById("side-panel-ai");
    if (!root) return;
    root.querySelectorAll(".ai-switch-panel").forEach((p) => p.classList.remove("open"));
    document.getElementById(panelId)?.classList.add("open");
}
