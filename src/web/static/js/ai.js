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

async function generateAiSummary() {
    const output = document.getElementById("ai-summary-output");
    const btn = document.getElementById("btn-ai-summary");

    if (!output) return;

    openAiPanel("ai-summary-panel");
    setAiPanelLoading("ai-summary-panel", true);
    if (btn) btn.disabled = true;

    try {
        const data = await runWithProgress(output, "Generating AI summary", () =>
            fetchAi("/api/ai/summarize", getTranscriptPayload())
        );
        output.classList.remove("muted-text");
        output.innerHTML = formatAiMarkdown(data.summary || "");
        showToast("AI summary ready.");
    } catch (err) {
        output.innerHTML = `<p class="muted-text">${escapeHtml(err.message || "Could not generate summary.")}</p>`;
        showToast("AI summary failed.");
    } finally {
        setAiPanelLoading("ai-summary-panel", false);
        if (btn) btn.disabled = false;
    }
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

    let chatSim = null;
    const typing = document.createElement("div");
    typing.className = "chat-msg chat-msg-assistant chat-typing";
    typing.innerHTML = `<span class="chat-msg-label">Assistant</span><div class="chat-msg-body" id="chat-progress-slot"></div>`;
    const log = document.getElementById("ai-chat-log");
    log?.appendChild(typing);
    const progressSlot = document.getElementById("chat-progress-slot");
    chatSim = startSimulatedProgress(progressSlot, "Thinking", { maxPercent: 90 });

    try {
        const payload = {
            ...getTranscriptPayload(),
            question,
            history: aiChatHistory.slice(0, -1),
        };
        const data = await fetchAi("/api/ai/chat", payload);
        const answer = data.answer || "";

        chatSim.stop();
        typing.remove();
        appendChatMessage("assistant", answer);
        aiChatHistory.push({ role: "assistant", content: answer });
    } catch (err) {
        chatSim?.stop();
        typing.remove();
        appendChatMessage("assistant", err.message || "Sorry, something went wrong.");
    } finally {
        if (btn) btn.disabled = false;
        input.focus();
    }
}

function clearAiChat() {
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
    if (document.getElementById("side-panel-ai")) {
        openAiPanel("ai-summary-panel");
    }
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initAiOnReady);
} else {
    initAiOnReady();
}
