/**
 * Dr. Pearl - ChatGPT-Style Dental AI Interface
 * Features: Multi-session Conversation History, Memory, Token-by-Token Streaming, Source Citations.
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const chatSidebar = document.getElementById("chatSidebar");
    const openSidebarBtn = document.getElementById("openSidebarBtn");
    const closeSidebarBtn = document.getElementById("closeSidebarBtn");
    const newChatSidebarBtn = document.getElementById("newChatSidebarBtn");
    const newChatTopbarBtn = document.getElementById("newChatTopbarBtn");
    const historyList = document.getElementById("historyList");
    const historySearchInput = document.getElementById("historySearchInput");
    const clearAllHistoryBtn = document.getElementById("clearAllHistoryBtn");

    const chatScrollContainer = document.getElementById("chatScrollContainer");
    const welcomeHero = document.getElementById("welcomeHero");
    const messagesList = document.getElementById("messagesList");
    const chatForm = document.getElementById("chatForm");
    const userInput = document.getElementById("userInput");
    const sendBtn = document.getElementById("sendBtn");
    const stopBtn = document.getElementById("stopBtn");
    const micBtn = document.getElementById("micBtn");

    const sourceModal = document.getElementById("sourceModal");
    const closeSourceModal = document.getElementById("closeSourceModal");
    const modalSourceContent = document.getElementById("modalSourceContent");

    // State Variables
    const STORAGE_KEY = "dental_chat_sessions_v1";
    let sessions = loadSessions();
    let currentSessionId = null;
    let abortController = null;
    let isStreaming = false;

    // Initialize UI
    initSessions();

    // -------------------------------------------------------------------------
    // 1. SESSION & STORAGE MANAGEMENT
    // -------------------------------------------------------------------------
    function loadSessions() {
        try {
            const data = localStorage.getItem(STORAGE_KEY);
            return data ? JSON.parse(data) : [];
        } catch (e) {
            console.error("Failed to load sessions from localStorage:", e);
            return [];
        }
    }

    function saveSessions() {
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
        } catch (e) {
            console.error("Failed to save sessions to localStorage:", e);
        }
    }

    function getCurrentSession() {
        return sessions.find(s => s.id === currentSessionId);
    }

    function initSessions() {
        if (sessions.length > 0) {
            switchSession(sessions[0].id);
        } else {
            createNewSession();
        }
        renderHistorySidebar();
    }

    function createNewSession() {
        if (isStreaming) stopGeneration();

        const newId = "chat_" + Date.now();
        const newSession = {
            id: newId,
            title: "New Dental Consultation",
            createdAt: Date.now(),
            updatedAt: Date.now(),
            messages: []
        };
        sessions.unshift(newSession);
        saveSessions();
        switchSession(newId);
        renderHistorySidebar();

        if (window.innerWidth <= 768) {
            chatSidebar.classList.add("collapsed");
        }
    }

    function switchSession(sessionId) {
        if (isStreaming) stopGeneration();

        currentSessionId = sessionId;
        const session = getCurrentSession();
        if (!session) return;

        messagesList.innerHTML = "";
        if (session.messages.length === 0) {
            welcomeHero.style.display = "block";
        } else {
            welcomeHero.style.display = "none";
            session.messages.forEach(msg => {
                renderStaticMessage(msg.role, msg.content, msg.sources || []);
            });
        }

        renderHistorySidebar();
        scrollToBottom();
        userInput.focus();
    }

    function deleteSession(e, sessionId) {
        e.stopPropagation();
        sessions = sessions.filter(s => s.id !== sessionId);
        saveSessions();

        if (currentSessionId === sessionId) {
            if (sessions.length > 0) {
                switchSession(sessions[0].id);
            } else {
                createNewSession();
            }
        } else {
            renderHistorySidebar();
        }
    }

    function renderHistorySidebar(filterQuery = "") {
        historyList.innerHTML = "";
        const query = filterQuery.toLowerCase().trim();

        const filtered = sessions.filter(s => 
            !query || s.title.toLowerCase().includes(query)
        );

        if (filtered.length === 0) {
            historyList.innerHTML = `
                <div style="padding:16px 8px; text-align:center; color:var(--text-muted); font-size:12px;">
                    ${query ? "No matching conversations" : "No consultations yet"}
                </div>
            `;
            return;
        }

        filtered.forEach(session => {
            const item = document.createElement("div");
            item.className = `history-item ${session.id === currentSessionId ? "active" : ""}`;
            item.innerHTML = `
                <div class="history-item-title">
                    <i class="fa-regular fa-message"></i>
                    <span>${escapeHtml(session.title)}</span>
                </div>
                <button class="history-delete-btn" title="Delete consultation">
                    <i class="fa-solid fa-trash-can"></i>
                </button>
            `;

            item.addEventListener("click", () => switchSession(session.id));
            item.querySelector(".history-delete-btn").addEventListener("click", (e) => deleteSession(e, session.id));

            historyList.appendChild(item);
        });
    }

    historySearchInput.addEventListener("input", (e) => {
        renderHistorySidebar(e.target.value);
    });

    clearAllHistoryBtn.addEventListener("click", () => {
        if (confirm("Are you sure you want to clear all conversation history?")) {
            sessions = [];
            saveSessions();
            createNewSession();
        }
    });

    // -------------------------------------------------------------------------
    // 2. SIDEBAR TOGGLE EVENTS
    // -------------------------------------------------------------------------
    openSidebarBtn.addEventListener("click", () => {
        chatSidebar.classList.remove("collapsed");
    });

    closeSidebarBtn.addEventListener("click", () => {
        chatSidebar.classList.add("collapsed");
    });

    newChatSidebarBtn.addEventListener("click", createNewSession);
    newChatTopbarBtn.addEventListener("click", createNewSession);

    // Hero prompt cards click
    document.querySelectorAll(".hero-card").forEach(card => {
        card.addEventListener("click", () => {
            const prompt = card.getAttribute("data-prompt");
            if (prompt) {
                userInput.value = prompt;
                submitMessage();
            }
        });
    });

    // -------------------------------------------------------------------------
    // 3. AUTO-RESIZING & KEYBOARD SUBMISSION
    // -------------------------------------------------------------------------
    userInput.addEventListener("input", () => {
        userInput.style.height = "auto";
        userInput.style.height = Math.min(userInput.scrollHeight, 160) + "px";
        sendBtn.disabled = !userInput.value.trim() || isStreaming;
    });

    userInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            if (!isStreaming && userInput.value.trim()) {
                submitMessage();
            }
        }
    });

    chatForm.addEventListener("submit", (e) => {
        e.preventDefault();
        submitMessage();
    });

    // -------------------------------------------------------------------------
    // 4. STREAMING GENERATION & MULTI-TURN MEMORY
    // -------------------------------------------------------------------------
    async function submitMessage() {
        const text = userInput.value.trim();
        if (!text || isStreaming) return;

        const session = getCurrentSession();
        if (!session) return;

        // Auto-update session title on first user query
        if (session.messages.length === 0) {
            session.title = text.length > 32 ? text.substring(0, 32) + "..." : text;
            renderHistorySidebar();
        }

        // Hide welcome hero
        welcomeHero.style.display = "none";

        // Append user message
        session.messages.push({ role: "user", content: text });
        session.updatedAt = Date.now();
        saveSessions();
        renderStaticMessage("user", text);

        // Reset input box
        userInput.value = "";
        userInput.style.height = "auto";
        sendBtn.disabled = true;

        // Toggle buttons to streaming state
        startStreamingUI();

        // Create empty assistant message with streaming cursor
        const assistantRow = createStreamingMessageRow();
        messagesList.appendChild(assistantRow);
        const textContainer = assistantRow.querySelector(".msg-text");
        scrollToBottom();

        let accumulatedContent = "";
        let sources = [];
        abortController = new AbortController();

        try {
            // Build conversation history memory (pass prior turns to endpoint)
            const historyMemory = session.messages.slice(0, -1).map(m => ({
                role: m.role,
                content: m.content
            }));

            const response = await fetch("/api/chat/stream", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    message: text,
                    history: historyMemory
                }),
                signal: abortController.signal
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.error || `Server returned error ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let buffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n\n");
                buffer = lines.pop(); // keep remainder

                for (const line of lines) {
                    const trimmed = line.trim();
                    if (trimmed.startsWith("data: ")) {
                        const jsonStr = trimmed.slice(6);
                        try {
                            const eventData = JSON.parse(jsonStr);

                            if (eventData.type === "meta") {
                                sources = eventData.sources || [];
                            } else if (eventData.type === "token") {
                                accumulatedContent += eventData.token;
                                renderStreamingMarkdown(textContainer, accumulatedContent);
                                scrollToBottom();
                            } else if (eventData.type === "error") {
                                throw new Error(eventData.error);
                            }
                        } catch (parseErr) {
                            // Non-json or partial line
                        }
                    }
                }
            }

            // Finalize message
            finishStreamingMessage(assistantRow, textContainer, accumulatedContent, sources);
            session.messages.push({
                role: "assistant",
                content: accumulatedContent,
                sources: sources
            });
            session.updatedAt = Date.now();
            saveSessions();
            renderHistorySidebar();

        } catch (err) {
            if (err.name === "AbortError") {
                // User clicked stop
                accumulatedContent += "\n\n*(Generation stopped by user)*";
                finishStreamingMessage(assistantRow, textContainer, accumulatedContent, sources);
                session.messages.push({
                    role: "assistant",
                    content: accumulatedContent,
                    sources: sources
                });
                saveSessions();
            } else {
                textContainer.innerHTML = `
                    <div style="color:var(--error); padding:10px; background:rgba(239,68,68,0.1); border-radius:8px;">
                        <i class="fa-solid fa-triangle-exclamation"></i> <strong>Notice:</strong> ${escapeHtml(err.message)}
                        <br><a href="/admin" style="color:var(--accent); text-decoration:none; font-size:12px; margin-top:6px; display:inline-block;">Open Admin Knowledge Hub</a>
                    </div>
                `;
            }
        } finally {
            stopStreamingUI();
            scrollToBottom();
        }
    }

    function stopGeneration() {
        if (abortController) {
            abortController.abort();
            abortController = null;
        }
        stopStreamingUI();
    }

    stopBtn.addEventListener("click", stopGeneration);

    function startStreamingUI() {
        isStreaming = true;
        sendBtn.style.display = "none";
        stopBtn.style.display = "flex";
        micBtn.style.display = "none";
    }

    function stopStreamingUI() {
        isStreaming = false;
        sendBtn.style.display = "flex";
        stopBtn.style.display = "none";
        micBtn.style.display = "flex";
        sendBtn.disabled = !userInput.value.trim();
    }

    // -------------------------------------------------------------------------
    // 5. MESSAGE RENDERING HELPERS
    // -------------------------------------------------------------------------
    function createStreamingMessageRow() {
        const row = document.createElement("div");
        row.className = "chat-message-row assistant";
        row.innerHTML = `
            <div class="msg-avatar"><i class="fa-solid fa-tooth"></i></div>
            <div class="msg-content-wrapper">
                <div class="msg-text"><span class="streaming-cursor"></span></div>
            </div>
        `;
        return row;
    }

    function renderStreamingMarkdown(container, rawText) {
        if (window.marked) {
            container.innerHTML = marked.parse(rawText) + '<span class="streaming-cursor"></span>';
        } else {
            container.textContent = rawText;
        }
    }

    function finishStreamingMessage(row, textContainer, fullContent, sources) {
        // Remove blinking cursor and parse final markdown
        if (window.marked) {
            textContainer.innerHTML = marked.parse(fullContent);
        } else {
            textContainer.textContent = fullContent;
        }

        // Add action toolbar
        const wrapper = row.querySelector(".msg-content-wrapper");
        const toolbar = document.createElement("div");
        toolbar.className = "msg-toolbar";

        if (sources && sources.length > 0) {
            const sourceBtn = document.createElement("button");
            sourceBtn.className = "btn-pill-source";
            sourceBtn.innerHTML = `<i class="fa-solid fa-book-bookmark"></i> ${sources.length} Verified Sources`;
            sourceBtn.addEventListener("click", () => openSourceModal(sources));
            toolbar.appendChild(sourceBtn);
        }

        const copyBtn = document.createElement("button");
        copyBtn.className = "btn-copy-action";
        copyBtn.innerHTML = '<i class="fa-regular fa-copy"></i> Copy';
        copyBtn.addEventListener("click", () => {
            navigator.clipboard.writeText(fullContent).then(() => {
                copyBtn.innerHTML = '<i class="fa-solid fa-check"></i> Copied';
                setTimeout(() => {
                    copyBtn.innerHTML = '<i class="fa-regular fa-copy"></i> Copy';
                }, 2000);
            });
        });
        toolbar.appendChild(copyBtn);

        wrapper.appendChild(toolbar);
    }

    function renderStaticMessage(role, content, sources = []) {
        const row = document.createElement("div");
        row.className = `chat-message-row ${role}`;

        if (role === "user") {
            row.innerHTML = `
                <div class="msg-content-wrapper">
                    <div class="msg-text">${escapeHtml(content)}</div>
                </div>
            `;
        } else {
            const parsedHtml = window.marked ? marked.parse(content) : escapeHtml(content);
            row.innerHTML = `
                <div class="msg-avatar"><i class="fa-solid fa-tooth"></i></div>
                <div class="msg-content-wrapper">
                    <div class="msg-text">${parsedHtml}</div>
                </div>
            `;

            const wrapper = row.querySelector(".msg-content-wrapper");
            const toolbar = document.createElement("div");
            toolbar.className = "msg-toolbar";

            if (sources && sources.length > 0) {
                const sourceBtn = document.createElement("button");
                sourceBtn.className = "btn-pill-source";
                sourceBtn.innerHTML = `<i class="fa-solid fa-book-bookmark"></i> ${sources.length} Verified Sources`;
                sourceBtn.addEventListener("click", () => openSourceModal(sources));
                toolbar.appendChild(sourceBtn);
            }

            const copyBtn = document.createElement("button");
            copyBtn.className = "btn-copy-action";
            copyBtn.innerHTML = '<i class="fa-regular fa-copy"></i> Copy';
            copyBtn.addEventListener("click", () => {
                navigator.clipboard.writeText(content).then(() => {
                    copyBtn.innerHTML = '<i class="fa-solid fa-check"></i> Copied';
                    setTimeout(() => {
                        copyBtn.innerHTML = '<i class="fa-regular fa-copy"></i> Copy';
                    }, 2000);
                });
            });
            toolbar.appendChild(copyBtn);
            wrapper.appendChild(toolbar);
        }

        messagesList.appendChild(row);
    }

    function scrollToBottom() {
        chatScrollContainer.scrollTop = chatScrollContainer.scrollHeight;
    }

    function escapeHtml(text) {
        const div = document.createElement("div");
        div.textContent = text;
        return div.innerHTML;
    }

    // -------------------------------------------------------------------------
    // 6. SOURCE CITATIONS MODAL
    // -------------------------------------------------------------------------
    function openSourceModal(sources) {
        modalSourceContent.innerHTML = "";
        sources.forEach((s, idx) => {
            const item = document.createElement("div");
            item.className = "source-item";
            item.innerHTML = `
                <div class="source-item-header">
                    <span class="source-book"><i class="fa-solid fa-book"></i> ${escapeHtml(s.source)} (Page ${s.page})</span>
                    <span class="source-score">Relevance: ${s.score}</span>
                </div>
                <div class="source-snippet">"${escapeHtml(s.snippet)}"</div>
            `;
            modalSourceContent.appendChild(item);
        });
        sourceModal.classList.add("active");
    }

    closeSourceModal.addEventListener("click", () => sourceModal.classList.remove("active"));
    sourceModal.addEventListener("click", (e) => {
        if (e.target === sourceModal) sourceModal.classList.remove("active");
    });

    // -------------------------------------------------------------------------
    // 7. SPEECH-TO-TEXT (MICROPHONE)
    // -------------------------------------------------------------------------
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
        const recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = "en-US";

        recognition.onstart = () => micBtn.classList.add("recording");
        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            userInput.value = transcript;
            userInput.dispatchEvent(new Event("input"));
        };
        recognition.onerror = () => micBtn.classList.remove("recording");
        recognition.onend = () => micBtn.classList.remove("recording");

        micBtn.addEventListener("click", () => {
            if (micBtn.classList.contains("recording")) {
                recognition.stop();
            } else {
                recognition.start();
            }
        });
    } else {
        micBtn.style.display = "none";
    }
});
