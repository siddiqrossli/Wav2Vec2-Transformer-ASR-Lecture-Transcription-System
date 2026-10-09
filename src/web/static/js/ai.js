/**
 * Optional AI features — calls /api/ai/* only; never touches ASR.
 */

const aiChatHistory = [];

function getTranscriptPayload() {
    return {
        transcript: window.fullTranscript || "",
        title: window.transcriptTitle || "",
        speaker: window.transcriptMeta?.speaker || "",
        course: window.transcriptMeta?.course || "",
    };
}

async function fetchAi(endpoint, body) {
    const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });

    const data = await res.json().catch(() => ({}));

    if (!res.ok) {
        throw new Error(data.error || `Request failed (${res.status})`);
    }

    return data;
}

function setAiPanelLoading(panelId, loading) {
    const panel = document.getElementById(panelId);
    if (!panel) return;
    panel.classList.toggle("is-loading", loading);
}

function appendChatMessage(role, text) {
    const log = document.getElementById("ai-chat-log");
    if (!log) return;

    const div = document.createElement("div");
    div.className = `chat-msg chat-msg-${role}`;

    const label = role === "user" ? "You" : "Assistant";
    div.innerHTML = `
        <span class="chat-msg-label">${label}</span>
        <div class="chat-msg-body">${escapeHtml(text).replace(/\n/g, "<br>")}</div>
    `;

    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
}

window.aiSummaryState = {
    isGenerating: false,
    html: "",
};

function toggleAiSummary() {
    const panel = document.getElementById("ai-summary-panel");
    const output = document.getElementById("ai-summary-output");
    if (!panel || !output) return;

    // True toggle: if already open, collapse it
    if (panel.classList.contains("open")) {
        panel.classList.remove("open");
        return;
    }

    // Open panel
    panel.classList.add("open");

    // If summary is already cached, show it immediately without re-fetching
    if (window.aiSummaryState.html) {
        output.classList.remove("muted-text");
        output.innerHTML = window.aiSummaryState.html;
        return;
    }

    // If already generating in background, keep displaying the active progress bar
    if (window.aiSummaryState.isGenerating) {
        return;
    }

    // Otherwise start generating
    generateAiSummary();
}

async function generateAiSummary(force = false) {
    const output = document.getElementById("ai-summary-output");
    const panel = document.getElementById("ai-summary-panel");
    const btn = document.getElementById("btn-ai-summary");

    if (!output || !panel) return;

    if (!force && window.aiSummaryState.html) {
        panel.classList.add("open");
        output.classList.remove("muted-text");
        output.innerHTML = window.aiSummaryState.html;
        return;
    }

    if (window.aiSummaryState.isGenerating) {
        panel.classList.add("open");
        return;
    }

    window.aiSummaryState.isGenerating = true;
    panel.classList.add("open");
    setAiPanelLoading("ai-summary-panel", true);

    if (btn) {
        btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> AI Summary`;
    }

    try {
        const data = await runWithProgress(output, "Generating AI summary", () =>
            fetchAi("/api/ai/summarize", getTranscriptPayload())
        );
        const regenerateBtn = `<div style="margin-top: 14px; text-align: right;"><button type="button" class="chat-clear-btn" onclick="generateAiSummary(true)" title="Regenerate summary"><i class="fa-solid fa-arrows-rotate"></i> Regenerate</button></div>`;
        const formatted = formatAiMarkdown(data.summary || "") + regenerateBtn;
        window.aiSummaryState.html = formatted;
        output.classList.remove("muted-text");
        output.innerHTML = formatted;
        showToast("AI summary ready.");
    } catch (err) {
        window.aiSummaryState.html = "";
        output.innerHTML = `<p class="muted-text">${escapeHtml(err.message || "Could not generate summary.")}</p><div style="margin-top: 10px;"><button type="button" class="chat-clear-btn" onclick="generateAiSummary(true)">Try again</button></div>`;
        showToast("AI summary failed.");
    } finally {
        window.aiSummaryState.isGenerating = false;
        setAiPanelLoading("ai-summary-panel", false);
        if (btn) {
            btn.innerHTML = `<i class="fa-solid fa-sparkles"></i> AI Summary`;
        }
    }
}

function toggleAiChat() {
    const panel = document.getElementById("ai-chat-panel");
    if (!panel) return;
    const isNowOpen = panel.classList.toggle("open");
    if (isNowOpen) {
        const input = document.getElementById("ai-chat-input");
        input?.focus();
        const log = document.getElementById("ai-chat-log");
        if (log) log.scrollTop = log.scrollHeight;
    }
}

window.currentChatStreamTimer = null;

function streamAssistantMessage(fullText, onComplete) {
    if (window.currentChatStreamTimer) {
        clearInterval(window.currentChatStreamTimer);
        window.currentChatStreamTimer = null;
    }

    const log = document.getElementById("ai-chat-log");
    if (!log) {
        appendChatMessage("assistant", fullText);
        if (onComplete) onComplete();
        return;
    }

    const div = document.createElement("div");
    div.className = "chat-msg chat-msg-assistant";
    div.innerHTML = `
        <span class="chat-msg-label">Assistant</span>
        <div class="chat-msg-body"></div>
    `;
    log.appendChild(div);
    const body = div.querySelector(".chat-msg-body");

    // Split text into tokens while preserving whitespace & newlines
    const tokens = fullText.split(/(\s+)/);
    let index = 0;
    let accumulated = "";

    // Dynamic speed: ~50 words/sec (2 tokens per 20ms) for snappy, natural ChatGPT-style streaming
    const stepInterval = tokens.length > 300 ? 16 : 20;
    const tokensPerStep = tokens.length > 300 ? 3 : 2;

    window.currentChatStreamTimer = setInterval(() => {
        for (let i = 0; i < tokensPerStep && index < tokens.length; i++) {
            accumulated += tokens[index];
            index++;
        }

        body.innerHTML = formatAiMarkdown(accumulated) + '<span class="chat-typing-cursor"></span>';

        // Auto-scroll to follow new text smoothly
        const isNearBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 140;
        if (isNearBottom) {
            log.scrollTop = log.scrollHeight;
        }

        if (index >= tokens.length) {
            clearInterval(window.currentChatStreamTimer);
            window.currentChatStreamTimer = null;
            body.innerHTML = formatAiMarkdown(fullText);
            log.scrollTop = log.scrollHeight;
            if (onComplete) onComplete();
        }
    }, stepInterval);
}

async function sendAiChat() {
    const input = document.getElementById("ai-chat-input");
    const btn = document.getElementById("btn-ai-chat-send");

    if (!input) return;

    const question = input.value.trim();
    if (!question) return;

    input.value = "";
    appendChatMessage("user", question);
    aiChatHistory.push({ role: "user", content: question });

    if (btn) btn.disabled = true;

    const typing = document.createElement("div");
    typing.className = "chat-msg chat-msg-assistant chat-typing";
    typing.innerHTML = `
        <span class="chat-msg-label">Assistant</span>
        <div class="chat-msg-body">
            <div class="chat-typing-bubble" aria-label="Assistant is typing">
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
            </div>
        </div>
    `;
    const log = document.getElementById("ai-chat-log");
    log?.appendChild(typing);
    if (log) log.scrollTop = log.scrollHeight;

    try {
        const payload = {
            ...getTranscriptPayload(),
            question,
            history: aiChatHistory.slice(0, -1),
        };
        const data = await fetchAi("/api/ai/chat", payload);
        const answer = data.answer || "";

        typing.remove();
        streamAssistantMessage(answer, () => {
            aiChatHistory.push({ role: "assistant", content: answer });
            if (btn) btn.disabled = false;
            input.focus();
        });
    } catch (err) {
        typing.remove();
        appendChatMessage("assistant", err.message || "Sorry, something went wrong.");
        if (btn) btn.disabled = false;
        input.focus();
    }
}

window.toggleAiSummary = toggleAiSummary;
window.toggleAiChat = toggleAiChat;
window.generateAiSummary = generateAiSummary;

function clearAiChat() {
    if (window.currentChatStreamTimer) {
        clearInterval(window.currentChatStreamTimer);
        window.currentChatStreamTimer = null;
    }
    aiChatHistory.length = 0;
    const log = document.getElementById("ai-chat-log");
    if (log) {
        log.innerHTML = `
            <div class="chat-msg chat-msg-assistant">
                <span class="chat-msg-label">Assistant</span>
                <div class="chat-msg-body">Ask anything about this lecture — definitions, main ideas, or what was said at a topic.</div>
            </div>
        `;
    }
}

function initAiChat() {
    const input = document.getElementById("ai-chat-input");
    if (!input) return;

    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendAiChat();
        }
    });
}

function initAiOnReady() {
    initAiChat();
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initAiOnReady);
} else {
    initAiOnReady();
}
