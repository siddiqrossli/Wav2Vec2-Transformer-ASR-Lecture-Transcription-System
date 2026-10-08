/**
 * Study tools, export, notes, library RAG — no ASR code.
 */

function getStudyPayload() {
    return {
        transcript: window.fullTranscript || "",
        title: window.transcriptTitle || "",
        speaker: window.transcriptMeta?.speaker || "",
        course: window.transcriptMeta?.course || "",
    };
}

function getTranscriptObject() {
    return {
        job_id: window.transcriptJobId,
        title: window.transcriptTitle,
        speaker: window.transcriptMeta?.speaker,
        course_code: window.transcriptMeta?.course,
        duration: window.transcriptMeta?.duration,
        processed_at: window.transcriptMeta?.processedAt,
        decode_mode: window.transcriptDecodeMode,
        full_text: window.fullTranscript,
        segments: window.transcriptSegments,
        word_count: window.transcriptMeta?.wordCount,
        reading_time: window.transcriptMeta?.readingTime,
    };
}

async function postJson(url, body, timeoutMs = 180000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    try {
        const res = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
            signal: controller.signal,
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
        return data;
    } catch (err) {
        if (err.name === "AbortError") {
            throw new Error("Request timed out. For long lectures, try again or translate in smaller parts.");
        }
        throw err;
    } finally {
        clearTimeout(timer);
    }
}

const AI_TOOL_LABELS = {
    "/api/ai/quiz": "Generating quiz",
    "/api/ai/glossary": "Generating glossary",
    "/api/ai/flashcards": "Generating flashcards",
};

async function runAiTool(endpoint, panelId, outputId, renderFn, extra = {}) {
    const panel = document.getElementById(panelId);
    const el = document.getElementById(outputId);
    if (panel) panel.classList.add("open");

    const label = AI_TOOL_LABELS[endpoint] || "Generating";

    try {
        const data = await runWithProgress(el, label, () =>
            postJson(endpoint, { ...getStudyPayload(), ...extra })
        );
        if (el) el.innerHTML = renderFn(data);
        showToast("Done.");
    } catch (err) {
        if (el) el.innerHTML = `<p class="muted-text">${escapeHtml(err.message)}</p>`;
        showToast("Request failed.");
    }
}

function parseQuizOption(opt, index) {
    const letters = ["A", "B", "C", "D"];
    const raw = String(opt || "").trim();
    const match = raw.match(/^([A-D])\)\s*(.*)$/i);
    if (match) {
        return { letter: match[1].toUpperCase(), text: match[2] || raw };
    }
    return { letter: letters[index] || "?", text: raw };
}

function recommendedStudyCount() {
    const words = Number(window.transcriptMeta?.wordCount || 0);
    if (words >= 4000) return 20;
    if (words >= 2500) return 15;
    if (words >= 1200) return 12;
    return 8;
}

window.quizState = { questions: [], index: 0, answered: {} };

function renderQuiz(data) {
    const qs = data.questions || [];
    window.quizState = { questions: qs, index: 0, answered: {} };
    if (!qs.length) return "<p class='muted-text'>No questions generated.</p>";
    return buildQuizView();
}

function buildQuizView() {
    const { questions, index, answered } = window.quizState;
    const q = questions[index];
    if (!q) return "<p class='muted-text'>No question.</p>";
    const options = (q.options || []).slice(0, 4).map(parseQuizOption);
    const isAnswered = Boolean(answered[index]);
    const selected = answered[index]?.selected || "";
    const correct = String(q.answer || "A")
        .trim()
        .toUpperCase()
        .charAt(0);
    const optsHtml = options
        .map((o) => {
            let cls = "quiz-option";
            if (isAnswered && o.letter === correct) cls += " quiz-correct";
            else if (isAnswered && o.letter === selected) cls += " quiz-wrong";
            return `
            <button type="button" class="${cls}" ${isAnswered ? "disabled" : ""}
                onclick="selectQuizOption('${o.letter}')">
                <span class="quiz-letter">${o.letter}</span>
                <span class="quiz-opt-text">${escapeHtml(o.text)}</span>
            </button>`;
        })
        .join("");
    const fb = isAnswered
        ? `<div class="quiz-feedback">
            <p class="${selected === correct ? "quiz-result-ok" : "quiz-result-no"}">
                ${selected === correct ? "Correct!" : `Incorrect. The answer is ${correct}.`}
            </p>
            <p class="muted-text">${escapeHtml(q.explanation || "")}</p>
           </div>`
        : "";

    return `
      <div class="study-item quiz-question">
        <div class="quiz-head">
            <strong>Q${index + 1} / ${questions.length}</strong>
        </div>
        <p class="quiz-question-text">${escapeHtml(q.question || "")}</p>
        <div class="quiz-options">${optsHtml}</div>
        ${fb}
        <div class="study-nav">
          <button type="button" class="chat-clear-btn" onclick="prevQuizQuestion()" ${index === 0 ? "disabled" : ""}>Previous</button>
          <button type="button" class="chat-send-btn" onclick="nextQuizQuestion()" ${index === questions.length - 1 ? "disabled" : ""}>Next</button>
        </div>
      </div>`;
}

function updateQuizUI() {
    const container = document.getElementById("quiz-output");
    if (container) container.innerHTML = buildQuizView();
}

function selectQuizOption(selectedLetter) {
    const { questions, index, answered } = window.quizState;
    if (!questions[index] || answered[index]) return;
    window.quizState.answered[index] = { selected: selectedLetter };
    updateQuizUI();
}

function nextQuizQuestion() {
    const { questions, index } = window.quizState;
    if (index < questions.length - 1) {
        window.quizState.index += 1;
        updateQuizUI();
    }
}

function prevQuizQuestion() {
    if (window.quizState.index > 0) {
        window.quizState.index -= 1;
        updateQuizUI();
    }
}

function renderGlossary(data) {
    const terms = data.terms || [];
    if (!terms.length) return "<p class='muted-text'>No terms found.</p>";
    return terms
        .map(
            t => `
        <div class="study-item">
            <strong>${escapeHtml(t.term || "")}</strong>
            <p>${escapeHtml(t.definition || "")}</p>
        </div>`
        )
        .join("");
}

function renderFlashcards(data) {
    const cards = data.cards || [];
    if (!cards.length) return "<p class='muted-text'>No cards generated.</p>";
    window.flashState = { cards, index: 0, flipped: false };
    return buildFlashcardView();
}

function buildFlashcardView() {
    const st = window.flashState || { cards: [], index: 0, flipped: false };
    const card = st.cards[st.index];
    if (!card) return "<p class='muted-text'>No flashcard.</p>";
    return `
      <div class="flashcard-deck">
        <div class="flashcard-flip ${st.flipped ? "flipped" : ""}" onclick="flipFlashcard()">
          <div class="flash-face flash-front">
            <div class="fc-label">Question ${st.index + 1} / ${st.cards.length}</div>
            <p>${escapeHtml(card.front || "")}</p>
            <small>Click card to reveal answer</small>
          </div>
          <div class="flash-face flash-back">
            <div class="fc-label">Answer</div>
            <p>${escapeHtml(card.back || "")}</p>
            <small>Click card to go back</small>
          </div>
        </div>
        <div class="study-nav">
          <button type="button" class="chat-clear-btn" onclick="prevFlashcard()" ${st.index === 0 ? "disabled" : ""}>Previous</button>
          <button type="button" class="chat-send-btn" onclick="nextFlashcard()" ${st.index === st.cards.length - 1 ? "disabled" : ""}>Next</button>
        </div>
      </div>`;
}

function updateFlashUI() {
    const container = document.getElementById("flashcards-output");
    if (container) container.innerHTML = buildFlashcardView();
}

function flipFlashcard() {
    if (!window.flashState) return;
    window.flashState.flipped = !window.flashState.flipped;
    updateFlashUI();
}

function nextFlashcard() {
    if (!window.flashState) return;
    if (window.flashState.index < window.flashState.cards.length - 1) {
        window.flashState.index += 1;
        window.flashState.flipped = false;
        updateFlashUI();
    }
}

function prevFlashcard() {
    if (!window.flashState) return;
    if (window.flashState.index > 0) {
        window.flashState.index -= 1;
        window.flashState.flipped = false;
        updateFlashUI();
    }
}

function generateQuiz() {
    runAiTool("/api/ai/quiz", "quiz-panel", "quiz-output", renderQuiz, { count: recommendedStudyCount() });
}

function generateGlossary() {
    runAiTool("/api/ai/glossary", "glossary-panel", "glossary-output", renderGlossary);
}

function generateFlashcards() {
    runAiTool("/api/ai/flashcards", "flashcards-panel", "flashcards-output", renderFlashcards, { count: recommendedStudyCount() });
}

function setProgressButtonState(button, label, percent) {
    if (!button) return;
    const pct = Math.max(0, Math.min(100, Math.round(percent || 0)));
    button.disabled = true;
    button.classList.add("is-processing");
    button.style.setProperty("--progress", `${pct}%`);

    if (!button.querySelector(".btn-label") || !button.querySelector(".btn-pct")) {
        button.innerHTML = `<span class="btn-label">${escapeHtml(label)}</span><span class="btn-pct">0%</span>`;
    }

    const labelNode = button.querySelector(".btn-label");
    if (labelNode) labelNode.textContent = label;

    const pctNode = button.querySelector(".btn-pct");
    if (pctNode) pctNode.textContent = `${pct}%`;
}

function resetProgressButtonState(button, originalLabel) {
    if (!button) return;
    button.disabled = false;
    button.classList.remove("is-processing");
    button.style.removeProperty("--progress");
    button.textContent = originalLabel;
}

function startButtonProgress(button, label, options = {}) {
    let percent = options.startPercent ?? 5;
    const max = options.maxPercent ?? 92;
    const intervalMs = options.intervalMs ?? 450;

    setProgressButtonState(button, label, percent);

    const timer = setInterval(() => {
        const step = options.step ?? (1.4 + Math.random() * 2.4);
        percent = Math.min(max, percent + step);
        setProgressButtonState(button, label, percent);
    }, intervalMs);

    return {
        complete() {
            clearInterval(timer);
            setProgressButtonState(button, label, 100);
        },
        stop() {
            clearInterval(timer);
        },
    };
}

function cloneSegments(segments) {
    return (segments || []).map((seg) => ({ ...seg }));
}

function ensureTranscriptState() {
    if (window.currentTranscriptState) return window.currentTranscriptState;

    window.currentTranscriptState = {
        originalText: window.fullTranscript || "",
        originalSegments: cloneSegments(window.transcriptSegments || []),
        translatedText: "",
        translatedSegments: [],
        currentLanguage: "Original",
        isTranslated: false,
    };

    return window.currentTranscriptState;
}

function applyTranscriptState() {
    const state = ensureTranscriptState();
    const paragraph = document.getElementById("paragraph-text");
    const segmentTextNodes = document.querySelectorAll(".segment .segment-text");

    const activeText = state.isTranslated ? (state.translatedText || state.originalText) : state.originalText;
    const activeSegments = state.isTranslated
        ? (state.translatedSegments.length ? state.translatedSegments : state.originalSegments)
        : state.originalSegments;

    if (paragraph) {
        paragraph.dataset.original = activeText;
        paragraph.textContent = activeText;
    }

    segmentTextNodes.forEach((el, idx) => {
        const txt = activeSegments[idx]?.text || state.originalSegments[idx]?.text || "";
        el.dataset.original = txt;
        el.textContent = txt;
    });
}

function normalizeTranslationText(text) {
    return String(text || "")
        .replace(/\s+/g, " ")
        .trim()
        .toLowerCase();
}

function looksLikeProviderErrorMessage(text) {
    const t = normalizeTranslationText(text);
    if (!t) return false;

    const errorHints = [
        "text length need to be between 0 and 5000",
        "request timed out",
        "quota exceeded",
        "rate limit",
        "too many requests",
        "translation provider",
        "invalid api key",
        "service unavailable",
    ];

    return errorHints.some((hint) => t.includes(hint));
}

function validateTranslatedOutput(source, translated, targetLanguage) {
    const sourceTrimmed = String(source || "").trim();
    const translatedTrimmed = String(translated || "").trim();
    const target = String(targetLanguage || "").trim().toLowerCase();
    const isEnglishTarget = target === "english";

    if (!translatedTrimmed) {
        throw new Error("Translation failed: translator returned empty output.");
    }

    if (looksLikeProviderErrorMessage(translatedTrimmed)) {
        throw new Error("Translation failed: provider returned an error message instead of translated text.");
    }

    if (!sourceTrimmed) return;

    if (normalizeTranslationText(sourceTrimmed) === normalizeTranslationText(translatedTrimmed)) {
        throw new Error("Translation failed: output is identical to the source text.");
    }

    if (isEnglishTarget) {
        return;
    }

    const srcWords = sourceTrimmed.toLowerCase().split(/\s+/).filter(Boolean);
    const outWords = translatedTrimmed.toLowerCase().split(/\s+/).filter(Boolean);
    const srcWordSet = new Set(srcWords);
    const overlap = outWords.filter((w) => srcWordSet.has(w)).length;
    const overlapRatio = outWords.length ? overlap / outWords.length : 1;

    if (sourceTrimmed && overlapRatio > 0.82) {
        throw new Error(`Translation validation failed for ${targetLanguage}: output is too similar to the source.`);
    }
}

async function translateLectureMain() {
    const state = ensureTranscriptState();
    const lang = document.getElementById("translate-lang")?.value || "Urdu";
    const out = document.getElementById("paragraph-text");
    const btn = document.getElementById("btn-main-translate");
    const restoreBtn = document.getElementById("btn-restore-original");
    const progressSlot = document.getElementById("translate-progress-slot");
    if (btn) btn.disabled = true;
    if (progressSlot) progressSlot.style.display = "block";

    try {
        const payload = {
            ...getStudyPayload(),
            transcript: state.originalText,
            target_language: lang,
            segments: state.originalSegments.map(seg => ({
                timestamp: seg.timestamp,
                timestamp_seconds: seg.timestamp_seconds,
                text: seg.text || "",
            })),
        };

        const data = await runWithProgress(
            progressSlot || out,
            `Translating to ${lang}`,
            () =>
                postJson(
                    "/api/ai/translate",
                    payload,
                    600000
                ),
            { maxPercent: 94, intervalMs: 700 }
        );
        const translated = data.translated_text || data.translation || "";
        const source = (state.originalText || "").trim();
        validateTranslatedOutput(source, translated, lang);

        state.translatedText = translated;
        state.translatedSegments = Array.isArray(data.translated_segments)
            ? cloneSegments(data.translated_segments)
            : [];
        state.currentLanguage = lang;
        state.isTranslated = true;

        window.translatedTranscript = translated;
        applyTranscriptState();
        if (restoreBtn) restoreBtn.style.display = "inline-flex";
        showToast(`Translated to ${lang}.`);
    } catch (err) {
        showToast(err.message || "Translation failed.");
    } finally {
        if (btn) btn.disabled = false;
        if (progressSlot) progressSlot.style.display = "none";
    }
}

function restoreOriginalTranscript() {
    const state = ensureTranscriptState();
    const restoreBtn = document.getElementById("btn-restore-original");

    state.translatedText = "";
    state.translatedSegments = [];
    state.currentLanguage = "Original";
    state.isTranslated = false;

    window.translatedTranscript = "";
    applyTranscriptState();

    if (restoreBtn) restoreBtn.style.display = "none";
    showToast("Original transcript restored.");
}

async function exportDocx() {
    try {
        const extras = {};
        const summaryEl = document.getElementById("ai-summary-output");
        if (summaryEl?.textContent && !summaryEl.classList.contains("muted-text")) {
            extras.ai_summary = summaryEl.innerText;
        }
        const res = await fetch("/api/export/docx", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ transcript: getTranscriptObject(), extras }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.error || "Export failed");
        }
        const blob = await res.blob();
        downloadBlob(blob, `${(window.transcriptTitle || "lecture").replace(/[^a-z0-9]/gi, "_")}.docx`);
        showToast("DOCX downloaded.");
    } catch (e) {
        showToast(e.message || "DOCX export failed.");
    }
}

async function exportPdfFile() {
    try {
        const res = await fetch("/api/export/pdf", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ transcript: getTranscriptObject() }),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.error || "Export failed");
        }
        const blob = await res.blob();
        downloadBlob(blob, `${(window.transcriptTitle || "lecture").replace(/[^a-z0-9]/gi, "_")}.pdf`);
        showToast("PDF downloaded.");
    } catch (e) {
        showToast(e.message || "PDF export failed.");
    }
}

function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

let lectureNotes = {};

function setNoteSaveStatus(idx, message, type = "") {
    const el = document.getElementById(`note-save-status-${idx}`);
    if (!el) return;
    el.textContent = message || "";
    el.classList.remove("saved", "error");
    if (type) el.classList.add(type);
}

function updateNotesOverview() {
    const box = document.getElementById("notes-overview");
    if (!box) return;

    const entries = Object.keys(lectureNotes)
        .filter((idx) => (lectureNotes[idx] || "").trim())
        .sort((a, b) => Number(a) - Number(b));

    if (!entries.length) {
        box.innerHTML = '<p class="muted-text small-hint">No saved notes yet.</p>';
        return;
    }

    box.innerHTML = entries
        .map((idx) => {
            const ts = window.transcriptSegments?.[idx]?.timestamp || `Segment ${Number(idx) + 1}`;
            const note = lectureNotes[idx] || "";
            return `
                <div class="study-item">
                    <strong>${escapeHtml(ts)}</strong>
                    <p>${escapeHtml(note)}</p>
                </div>`;
        })
        .join("");
}

async function loadLectureNotes() {
    const jobId = window.transcriptJobId;
    if (!jobId) return;
    try {
        const res = await fetch(`/api/notes/${jobId}`);
        const data = await res.json();
        lectureNotes = data.notes || {};
        applyNotesToSegments();
        updateNotesOverview();
    } catch (_) {
        lectureNotes = {};
        updateNotesOverview();
    }
}

function applyNotesToSegments() {
    document.querySelectorAll(".segment").forEach(seg => {
        const idx = seg.dataset.segmentIdx;
        const note = lectureNotes[idx];
        const badge = seg.querySelector(".note-badge");
        if (badge) badge.style.display = note ? "inline" : "none";
        const input = seg.querySelector(".segment-note-input");
        if (input && note) input.value = note;
    });
}

async function saveSegmentNote(idx) {
    const jobId = window.transcriptJobId;
    if (!jobId) return;

    const input = document.querySelector(`.segment-note-input[data-idx="${idx}"]`);
    if (!input) return;

    const btn = input.parentElement?.querySelector(".note-save-btn");
    const value = input.value.trim();

    if (value) lectureNotes[idx] = value;
    else delete lectureNotes[idx];

    if (btn) btn.disabled = true;
    try {
        await postJson(`/api/notes/${jobId}`, { notes: lectureNotes });
        applyNotesToSegments();
        updateNotesOverview();
        setNoteSaveStatus(idx, value ? "Saved ✓" : "", value ? "saved" : "");
    } catch (err) {
        setNoteSaveStatus(idx, "Save failed", "error");
        showToast(err.message || "Could not save note.");
    } finally {
        if (btn) btn.disabled = false;
    }
}

async function saveAllNotes() {
    const jobId = window.transcriptJobId;
    if (!jobId) return;
    document.querySelectorAll(".segment-note-input").forEach(input => {
        const idx = input.dataset.idx;
        const val = input.value.trim();
        if (val) lectureNotes[idx] = val;
        else delete lectureNotes[idx];
    });
    await postJson(`/api/notes/${jobId}`, { notes: lectureNotes });
    applyNotesToSegments();
    updateNotesOverview();
    showToast("Notes saved.");
}

function toggleSegmentNote(idx) {
    const wrap = document.getElementById(`note-wrap-${idx}`);
    if (wrap) wrap.classList.toggle("open");
}

function initLibrarySelection() {
    const card = document.getElementById("library-card-list");
    const list = document.getElementById("library-list");
    const toggleBtn = document.getElementById("library-select-toggle");
    const actions = document.getElementById("library-selection-actions");
    const selectAllBtn = document.getElementById("library-select-all-btn");
    const cancelBtn = document.getElementById("library-cancel-select-btn");
    const deleteBtn = document.getElementById("library-delete-selected-btn");

    if (!card || !list || !toggleBtn || !actions || !selectAllBtn || !cancelBtn || !deleteBtn) {
        return;
    }

    let selectionMode = false;
    const selected = new Set();

    function getRows() {
        return Array.from(list.querySelectorAll("li[data-job-id]"));
    }

    function renderSelectionState() {
        card.classList.toggle("selection-mode", selectionMode);
        actions.style.display = selectionMode ? "flex" : "none";
        toggleBtn.textContent = selectionMode ? "Selecting" : "Select";

        getRows().forEach((row) => {
            const id = row.dataset.jobId || "";
            const checkbox = row.querySelector(".library-select-checkbox");
            if (checkbox) checkbox.checked = selected.has(id);
        });
    }

    function updateActionState() {
        renderSelectionState();
    }

    function bindLibraryCheckboxes() {
        list.querySelectorAll(".library-select-checkbox").forEach((checkbox) => {
            if (checkbox.dataset.bound === "1") return;

            checkbox.dataset.bound = "1";

            checkbox.addEventListener("click", (event) => {
                event.stopPropagation();
            });

            checkbox.addEventListener("change", (event) => {
                event.stopPropagation();

                const row = event.currentTarget.closest("li[data-job-id]");
                const id = row?.dataset.jobId;
                if (!id) return;

                if (event.currentTarget.checked) {
                    selected.add(id);
                } else {
                    selected.delete(id);
                }

                updateActionState();
            });
        });
    }

    function clearSelection() {
        selected.clear();
    }

    bindLibraryCheckboxes();

    toggleBtn.addEventListener("click", () => {
        selectionMode = !selectionMode;
        if (!selectionMode) clearSelection();
        renderSelectionState();
    });

    selectAllBtn.addEventListener("click", () => {
        getRows().forEach((row) => {
            const id = row.dataset.jobId || "";
            if (id) selected.add(id);
        });
        renderSelectionState();
    });

    cancelBtn.addEventListener("click", () => {
        selectionMode = false;
        clearSelection();
        renderSelectionState();
    });

    list.addEventListener("click", (event) => {
        if (event.target.closest(".library-select-checkbox")) {
            return;
        }

        if (selectionMode) {
            event.preventDefault();
            event.stopPropagation();
            return;
        }
    });

    deleteBtn.addEventListener("click", async () => {
        const ids = Array.from(selected);
        if (!ids.length) {
            showToast("Select at least one lecture to delete.");
            return;
        }

        if (!confirm(`Delete ${ids.length} selected lecture(s)? This cannot be undone.`)) {
            return;
        }

        deleteBtn.disabled = true;
        try {
            const result = await postJson("/api/library/bulk-delete", { job_ids: ids });
            const failed = new Set([...(result.failed || []), ...(result.missing || [])]);

            ids.forEach((id) => {
                if (failed.has(id)) return;
                const row = list.querySelector(`li[data-job-id="${id}"]`);
                row?.remove();
                selected.delete(id);
            });

            selectionMode = false;
            clearSelection();
            renderSelectionState();

            const deletedCount = Number(result.deleted || 0);
            if (deletedCount > 0) {
                showToast(`${deletedCount} lecture(s) deleted.`);
            } else {
                showToast("No lectures were deleted.");
            }
        } catch (err) {
            showToast(err.message || "Could not delete selected lectures.");
        } finally {
            deleteBtn.disabled = false;
        }
    });

    renderSelectionState();
}

async function runRagSearch() {
    const q = document.getElementById("rag-search-input")?.value.trim();
    const box = document.getElementById("rag-search-results");
    if (!q || !box) return;
    box.innerHTML = "<p class='muted-text'>Searching…</p>";
    try {
        const data = await postJson("/api/rag/search", { query: q });
        const hits = data.results || [];
        if (!hits.length) {
            box.innerHTML = "<p class='muted-text'>No matches.</p>";
            return;
        }
        box.innerHTML = hits
            .map(
                h => `
            <div class="rag-hit">
                <strong>${escapeHtml(h.title)}</strong> <span class="muted-text">[${escapeHtml(h.timestamp)}]</span>
                <p>${escapeHtml(h.text)}</p>
            </div>`
            )
            .join("");
    } catch (e) {
        box.innerHTML = `<p class="muted-text">${escapeHtml(e.message)}</p>`;
    }
}

async function runRagAsk() {
    const q = document.getElementById("rag-ask-input")?.value.trim();
    const ans = document.getElementById("rag-ask-answer");
    const src = document.getElementById("rag-ask-sources");
    const askBtn = document.getElementById("ask-library-btn");
    if (!q || !ans) return;
    if (src) src.innerHTML = "";
    const progress = startButtonProgress(askBtn, "Searching library...");
    try {
        const data = await postJson("/api/rag/ask", { question: q });
        progress.complete();
        await new Promise((resolve) => setTimeout(resolve, 200));
        resetProgressButtonState(askBtn, "Ask library");
        ans.innerHTML = formatAiMarkdown(data.answer || "");
        if (src && data.sources?.length) {
            src.innerHTML = data.sources
                .map(
                    s => `<div class="rag-hit"><strong>${escapeHtml(s.title)}</strong> [${escapeHtml(s.timestamp)}]<p>${escapeHtml(s.text)}</p></div>`
                )
                .join("");
        }
    } catch (e) {
        progress.stop();
        resetProgressButtonState(askBtn, "Ask library");
        ans.textContent = e.message;
    }
}

function initSideTabs() {
    const tabs = document.querySelectorAll(".side-tab");
    if (!tabs.length) return;

    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            const panelId = tab.dataset.sidePanel;
            tabs.forEach(t => t.classList.remove("active"));
            tab.classList.add("active");
            document.querySelectorAll(".side-panel").forEach(p => p.classList.remove("active"));
            document.getElementById(panelId)?.classList.add("active");
        });
    });
}

function toggleSidePanelExpand(btn) {
    const layout = document.querySelector(".result-layout");
    if (!layout) return;
    const isExpanded = layout.classList.toggle("side-expanded");
    document.querySelectorAll(".btn-card-expand").forEach(b => {
        const icon = b.querySelector("i");
        if (icon) {
            icon.className = isExpanded ? "fa-solid fa-compress" : "fa-solid fa-expand";
        }
        b.title = isExpanded ? "Collapse panel" : "Expand panel";
        b.setAttribute("aria-label", isExpanded ? "Collapse panel" : "Expand panel");
    });
}
window.toggleSidePanelExpand = toggleSidePanelExpand;

document.addEventListener("DOMContentLoaded", () => {
    initSideTabs();
    if (window.transcriptJobId) loadLectureNotes();
    initLibrarySelection();
});
