/**
 * Admin Knowledge Hub & Ingestion Pipeline Logic
 */

document.addEventListener("DOMContentLoaded", () => {
    // Elements
    const statVectorCount = document.getElementById("statVectorCount");
    const statIndexName = document.getElementById("statIndexName");
    const statDocCount = document.getElementById("statDocCount");
    const statLlmModel = document.getElementById("statLlmModel");
    const statEmbedModel = document.getElementById("statEmbedModel");
    const statApiStatus = document.getElementById("statApiStatus");
    const statKeyStatus = document.getElementById("statKeyStatus");

    const documentSelect = document.getElementById("documentSelect");
    const runPipelineBtn = document.getElementById("runPipelineBtn");
    const currentStageBadge = document.getElementById("currentStageBadge");
    const progressMessage = document.getElementById("progressMessage");
    const progressPercentage = document.getElementById("progressPercentage");
    const progressFill = document.getElementById("progressFill");
    const terminalLogs = document.getElementById("terminalLogs");
    const clearLogsBtn = document.getElementById("clearLogsBtn");

    // Stepper elements
    const steps = {
        ingestion: document.getElementById("step-ingestion"),
        preprocessing: document.getElementById("step-preprocessing"),
        chunking: document.getElementById("step-chunking"),
        embeddings: document.getElementById("step-embeddings"),
        indexing: document.getElementById("step-indexing")
    };
    const connectors = [
        document.getElementById("conn-1"),
        document.getElementById("conn-2"),
        document.getElementById("conn-3"),
        document.getElementById("conn-4")
    ];

    // Upload elements
    const dropZone = document.getElementById("dropZone");
    const pdfFileInput = document.getElementById("pdfFileInput");
    const uploadStatus = document.getElementById("uploadStatus");
    const docList = document.getElementById("docList");
    const refreshDocsBtn = document.getElementById("refreshDocsBtn");

    // Settings elements
    const openSettingsBtn = document.getElementById("openSettingsBtn");
    const closeSettingsModal = document.getElementById("closeSettingsModal");
    const cancelSettingsBtn = document.getElementById("cancelSettingsBtn");
    const settingsModal = document.getElementById("settingsModal");
    const settingsForm = document.getElementById("settingsForm");
    const openrouterKey = document.getElementById("openrouterKey");
    const pineconeKey = document.getElementById("pineconeKey");
    const pineconeIndex = document.getElementById("pineconeIndex");
    const pineconeRegion = document.getElementById("pineconeRegion");
    const llmModel = document.getElementById("llmModel");
    const embeddingModel = document.getElementById("embeddingModel");

    let pollingInterval = null;

    // Initialize
    fetchStats();
    fetchDocuments();

    // 1. Fetch Stats & Metrics
    async function fetchStats() {
        try {
            const res = await fetch("/api/admin/stats");
            const data = await res.json();
            if (data.success) {
                const pc = data.pinecone;
                const cfg = data.config;

                statVectorCount.textContent = pc.vector_count ? pc.vector_count.toLocaleString() : "0";
                statIndexName.textContent = `Index: ${cfg.index_name || "dental-ortho-kb"}`;
                statLlmModel.textContent = cfg.llm_model.split("/").pop();
                statEmbedModel.textContent = `Embed: ${cfg.embedding_model}`;

                if (cfg.is_openrouter_set && cfg.is_pinecone_set) {
                    statApiStatus.textContent = "Ready & Configured";
                    statApiStatus.style.color = "var(--success)";
                    statKeyStatus.textContent = "Keys Active";
                } else if (!cfg.is_openrouter_set && !cfg.is_pinecone_set) {
                    statApiStatus.textContent = "Keys Missing";
                    statApiStatus.style.color = "var(--error)";
                    statKeyStatus.textContent = "Configure in Settings";
                } else {
                    statApiStatus.textContent = "Partial Setup";
                    statApiStatus.style.color = "var(--warning)";
                    statKeyStatus.textContent = cfg.is_openrouter_set ? "Pinecone Key Needed" : "OpenRouter Key Needed";
                }

                // Pre-populate settings form
                pineconeIndex.value = cfg.index_name || "dental-ortho-kb";
                if (cfg.llm_model) llmModel.value = cfg.llm_model;
                if (cfg.embedding_model) embeddingModel.value = cfg.embedding_model;
            }
        } catch (err) {
            console.error("Failed to load stats:", err);
        }
    }

    // 2. Fetch Available Documents
    async function fetchDocuments() {
        try {
            const res = await fetch("/api/admin/documents");
            const data = await res.json();
            if (data.success) {
                const docs = data.documents || [];
                statDocCount.textContent = docs.length;

                // Populate dropdown
                documentSelect.innerHTML = "";
                docList.innerHTML = "";

                docs.forEach(doc => {
                    const opt = document.createElement("option");
                    opt.value = doc.filename;
                    opt.textContent = `${doc.filename} (${doc.size_mb} MB)`;
                    if (doc.is_default) opt.selected = true;
                    documentSelect.appendChild(opt);

                    // Populate sidebar list
                    const li = document.createElement("li");
                    li.className = "doc-item";
                    li.innerHTML = `
                        <div class="doc-icon"><i class="fa-solid fa-file-pdf"></i></div>
                        <div class="doc-details">
                            <strong>${doc.filename}</strong>
                            <span>${doc.size_mb} MB • ${doc.is_default ? "Primary Guide" : "Uploaded Book"}</span>
                        </div>
                    `;
                    docList.appendChild(li);
                });
            }
        } catch (err) {
            console.error("Failed to load documents:", err);
        }
    }

    refreshDocsBtn.addEventListener("click", fetchDocuments);

    // 3. Drag & Drop File Upload
    dropZone.addEventListener("click", () => pdfFileInput.click());

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    pdfFileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    async function handleFileUpload(file) {
        if (!file.name.toLowerCase().endsWith(".pdf")) {
            uploadStatus.innerHTML = '<span style="color:var(--error);">Please upload a valid PDF document.</span>';
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        uploadStatus.innerHTML = `<span style="color:var(--accent);"><i class="fa-solid fa-spinner fa-spin"></i> Uploading ${file.name}...</span>`;

        try {
            const res = await fetch("/api/admin/upload", {
                method: "POST",
                body: formData
            });
            const data = await res.json();
            if (data.success) {
                uploadStatus.innerHTML = `<span style="color:var(--success);"><i class="fa-solid fa-check"></i> ${data.message}</span>`;
                fetchDocuments();
            } else {
                uploadStatus.innerHTML = `<span style="color:var(--error);">${data.error}</span>`;
            }
        } catch (err) {
            uploadStatus.innerHTML = '<span style="color:var(--error);">Failed to upload file.</span>';
        }
    }

    // 4. Run Pipeline Execution
    runPipelineBtn.addEventListener("click", async () => {
        const selectedDoc = documentSelect.value;
        if (!selectedDoc) {
            alert("Please select a document from the dropdown.");
            return;
        }

        runPipelineBtn.disabled = true;
        resetStepper();
        addTerminalLog(`[INIT] Requesting pipeline execution for: ${selectedDoc}...`);

        try {
            const res = await fetch("/api/admin/run-pipeline", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ filename: selectedDoc })
            });

            const data = await res.json();
            if (data.success) {
                startPollingStatus();
            } else {
                addTerminalLog(`[ERROR] ${data.error}`);
                alert(`Pipeline Error: ${data.error}`);
                runPipelineBtn.disabled = false;
            }
        } catch (err) {
            addTerminalLog(`[ERROR] Network failure: ${err.message}`);
            runPipelineBtn.disabled = false;
        }
    });

    // 5. Polling Pipeline Status
    function startPollingStatus() {
        if (pollingInterval) clearInterval(pollingInterval);

        pollingInterval = setInterval(async () => {
            try {
                const res = await fetch("/api/admin/pipeline-status");
                const state = await res.json();

                updateUIWithState(state);

                if (!state.is_running) {
                    clearInterval(pollingInterval);
                    pollingInterval = null;
                    runPipelineBtn.disabled = false;
                    fetchStats(); // update vector count
                }
            } catch (err) {
                console.error("Polling error:", err);
            }
        }, 900);
    }

    function updateUIWithState(state) {
        // Progress bar
        progressPercentage.textContent = `${state.percentage}%`;
        progressFill.style.width = `${state.percentage}%`;
        progressMessage.textContent = state.message;

        // Stage badge
        currentStageBadge.textContent = state.current_stage.toUpperCase();
        currentStageBadge.className = `badge-stage ${state.current_stage === 'error' ? 'error' : state.is_running ? 'running' : 'complete'}`;

        // Terminal logs
        if (state.logs && state.logs.length > 0) {
            terminalLogs.innerHTML = "";
            state.logs.forEach(log => {
                const line = document.createElement("div");
                line.className = "log-line";
                line.textContent = log;
                terminalLogs.appendChild(line);
            });
            terminalLogs.scrollTop = terminalLogs.scrollHeight;
        }

        // Stepper Visuals
        updateStepper(state.current_stage);
    }

    function updateStepper(stage) {
        const order = ["ingestion", "preprocessing", "chunking", "embeddings", "indexing"];
        const curIdx = order.indexOf(stage);

        order.forEach((s, idx) => {
            const el = steps[s];
            if (!el) return;

            if (stage === "complete") {
                el.className = "step-item done";
            } else if (idx < curIdx) {
                el.className = "step-item done";
            } else if (idx === curIdx) {
                el.className = "step-item active";
            } else {
                el.className = "step-item";
            }
        });

        // Connectors
        connectors.forEach((conn, idx) => {
            if (stage === "complete" || idx < curIdx) {
                conn.className = "step-connector done";
            } else {
                conn.className = "step-connector";
            }
        });
    }

    function resetStepper() {
        Object.values(steps).forEach(s => s.className = "step-item");
        connectors.forEach(c => c.className = "step-connector");
        progressFill.style.width = "0%";
        progressPercentage.textContent = "0%";
        currentStageBadge.className = "badge-stage running";
        currentStageBadge.textContent = "STARTING";
    }

    function addTerminalLog(msg) {
        const line = document.createElement("div");
        line.className = "log-line";
        line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
        terminalLogs.appendChild(line);
        terminalLogs.scrollTop = terminalLogs.scrollHeight;
    }

    clearLogsBtn.addEventListener("click", () => {
        terminalLogs.innerHTML = '<div class="log-line">[SYSTEM] Console cleared.</div>';
    });

    // 6. Settings Modal
    openSettingsBtn.addEventListener("click", () => settingsModal.classList.add("active"));
    closeSettingsModal.addEventListener("click", () => settingsModal.classList.remove("active"));
    cancelSettingsBtn.addEventListener("click", () => settingsModal.classList.remove("active"));

    settingsForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const payload = {
            PINECONE_INDEX_NAME: pineconeIndex.value.trim(),
            PINECONE_REGION: pineconeRegion.value.trim(),
            LLM_MODEL: llmModel.value,
            EMBEDDING_MODEL: embeddingModel.value
        };

        if (openrouterKey.value.trim()) {
            payload["OPENROUTER_API_KEY"] = openrouterKey.value.trim();
        }
        if (pineconeKey.value.trim()) {
            payload["PINECONE_API_KEY"] = pineconeKey.value.trim();
        }

        try {
            const res = await fetch("/api/admin/settings", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            if (data.success) {
                alert("Settings saved successfully!");
                settingsModal.classList.remove("active");
                fetchStats();
            } else {
                alert("Failed to save settings: " + data.error);
            }
        } catch (err) {
            alert("Network error saving settings.");
        }
    });
});
