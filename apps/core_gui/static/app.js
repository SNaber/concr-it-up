(function () {
  "use strict";

  const DEFAULT_PREVIEW_LIMIT = 50;
  const DEFAULT_RUNTIME_OUTPUT_DIR = "data/generated_scores/und/prediction_run";
  const DEFAULT_POS_COLUMN = "Dom_Pos";
  const DEFAULT_POS_TAGS = ["Noun"];
  const DEFAULT_POS_MATCH_MODE = "exact";
  const DEFAULT_POS_TOKEN_PATTERN = "[,;/| ]+";
  const DEFAULT_WORD_COLUMN = "Word";
  const DEFAULT_SCORE_COLUMN = "Conc.M";
  const CSRF_TOKEN = document.querySelector('meta[name="csrf-token"]')?.content || "";

  const FIELD_HELP = {
    dataset_gold: "Add gold file (CSV/TSV); set word/score columns.",
    dataset_lowercase: "If enabled, words are normalized to lowercase before matching.",
    dataset_word_column: "Word column name from the gold file.",
    dataset_score_column: "Numeric score column name from the gold file.",
    pos_enabled: "Enable POS-based filtering before model training.",
    pos_column: "Column containing POS labels when POS filtering is enabled.",
    pos_tags: "Comma-separated POS tags to keep (for example: Noun,NN).",
    pos_match_mode: "exact = whole-cell match; token_contains = tokenized tag inclusion.",
    pos_token_pattern: "Regex split pattern used for token_contains POS matching.",
    space_id: "Unique identifier for the embedding space (must not repeat).",
    space_kind: "Embedding backend kind: ft_bin (fastText .bin) or vec (word2vec text .vec).",
    space_path: "Select embedding file or type a path.",
    runtime_output_dir: "Output directory for generated artifacts.",
    runtime_seed: "Random seed for train/test split and CV shuffling.",
    runtime_n_jobs: "Parallel jobs for sklearn backend. Use -1 for all cores; 0 is invalid.",
    prediction_test_size: "Outer test split fraction in (0,1).",
    prediction_cv_folds: "Cross-validation folds for k selection (>=2).",
    prediction_topn_neighbors: "Number of neighbors written to vocab diagnostics.",
    prediction_k_min: "Minimum k candidate (>=1).",
    prediction_k_max: "Maximum k candidate (>=k_min).",
    prediction_k_step: "Step size for k grid (>=1).",
    prediction_target: "Add targets file (one token per line).",
    reports_level: "core = essential artifacts, full = includes OOV and vocab diagnostics.",
  };

  const state = {
    editorOpen: false,
    currentView: "results",
    rawConfig: null,
    dirty: false,
    currentJobId: null,
    pollTimer: null,
    elapsedTimer: null,
    activeJob: null,
    results: {
      latestJob: null,
      artifactKey: "",
      offset: 0,
      limit: DEFAULT_PREVIEW_LIMIT,
      query: "",
    },
  };

  const els = {
    flashMessage: document.getElementById("flashMessage"),
    mainLayout: document.getElementById("mainLayout"),
    runPanel: document.getElementById("runPanel"),
    editorSessionState: document.getElementById("editorSessionState"),
    configPath: document.getElementById("configPath"),
    browsePath: document.getElementById("browsePath"),
    browseListing: document.getElementById("browseListing"),
    uploadConfigInput: document.getElementById("uploadConfigInput"),
    btnNewConfig: document.getElementById("btnNewConfig"),
    btnCloseConfig: document.getElementById("btnCloseConfig"),
    btnLoadConfig: document.getElementById("btnLoadConfig"),
    btnSaveConfig: document.getElementById("btnSaveConfig"),
    btnDownloadConfig: document.getElementById("btnDownloadConfig"),
    btnBrowse: document.getElementById("btnBrowse"),

    editorSection: document.getElementById("editorSection"),
    commandControls: document.getElementById("commandControls"),
    guidedPanel: document.getElementById("guidedPanel"),
    btnUploadGoldCsv: document.getElementById("btnUploadGoldCsv"),
    goldCsvUploadInput: document.getElementById("goldCsvUploadInput"),
    btnUploadPredictVocab: document.getElementById("btnUploadPredictVocab"),
    predictVocabUploadInput: document.getElementById("predictVocabUploadInput"),
    embExistingSelect: document.getElementById("embExistingSelect"),
    btnRefreshEmbeddings: document.getElementById("btnRefreshEmbeddings"),
    btnRunPrediction: document.getElementById("btnRunPrediction"),
    btnShowEditor: document.getElementById("btnShowEditor"),
    btnShowResults: document.getElementById("btnShowResults"),
    workingIndicator: document.getElementById("workingIndicator"),
    workingBadge: document.getElementById("workingBadge"),
    workingSpinner: document.getElementById("workingSpinner"),
    workingElapsed: document.getElementById("workingElapsed"),
    workingLatestLog: document.getElementById("workingLatestLog"),
    jobStatus: document.getElementById("jobStatus"),

    btnRefreshResults: document.getElementById("btnRefreshResults"),
    resultsMeta: document.getElementById("resultsMeta"),
    resultArtifactSelect: document.getElementById("resultArtifactSelect"),
    resultSearchRow: document.getElementById("resultSearchRow"),
    resultSearchQuery: document.getElementById("resultSearchQuery"),
    btnSearchArtifact: document.getElementById("btnSearchArtifact"),
    btnPrevPage: document.getElementById("btnPrevPage"),
    btnNextPage: document.getElementById("btnNextPage"),
    resultPagingInfo: document.getElementById("resultPagingInfo"),
    btnDownloadArtifact: document.getElementById("btnDownloadArtifact"),
    btnCopyArtifactPath: document.getElementById("btnCopyArtifactPath"),
    resultPath: document.getElementById("resultPath"),
    resultPreview: document.getElementById("resultPreview"),

    helpPopover: document.getElementById("helpPopover"),
  };

  function deepClone(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function hasActiveJob() {
    return !!(state.activeJob && ["submitting", "queued", "running"].includes(state.activeJob.status));
  }

  function setMessage(text, isError = false) {
    els.flashMessage.textContent = text;
    els.flashMessage.classList.toggle("error", isError);
  }

  async function fetchJSON(url, options = {}, { allowPayloadError = false } = {}) {
    const requestOptions = { ...options };
    const method = String(requestOptions.method || "GET").toUpperCase();
    if (!["GET", "HEAD", "OPTIONS"].includes(method) && CSRF_TOKEN) {
      const headers = new Headers(requestOptions.headers || {});
      headers.set("X-CSRF-Token", CSRF_TOKEN);
      requestOptions.headers = headers;
    }
    const response = await fetch(url, requestOptions);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok || (!allowPayloadError && payload.error)) {
      const error = new Error(payload.error || `Request failed (${response.status}).`);
      error.status = response.status;
      error.payload = payload;
      throw error;
    }
    return payload;
  }

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function parseIso(value) {
    if (!value) {
      return null;
    }
    const parsed = Date.parse(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  function formatElapsed(totalSeconds) {
    const seconds = Math.max(0, Math.floor(totalSeconds));
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    if (h > 0) {
      return `${h}h ${m}m ${s}s`;
    }
    if (m > 0) {
      return `${m}m ${s}s`;
    }
    return `${s}s`;
  }

  function formatTimestamp(value) {
    const ms = parseIso(value);
    if (!ms) {
      return "n/a";
    }
    return new Date(ms).toLocaleString();
  }

  function nowIso() {
    return new Date().toISOString();
  }

  function setPath(obj, path, value) {
    const parts = path.split(".");
    let cursor = obj;
    for (const part of parts.slice(0, -1)) {
      if (!cursor[part] || typeof cursor[part] !== "object") {
        cursor[part] = {};
      }
      cursor = cursor[part];
    }
    cursor[parts[parts.length - 1]] = value;
  }

  function getPath(obj, path, fallback) {
    const parts = path.split(".");
    let cursor = obj;
    for (const part of parts) {
      if (!cursor || typeof cursor !== "object" || !(part in cursor)) {
        return fallback;
      }
      cursor = cursor[part];
    }
    return cursor;
  }

  function setText(id, value) {
    const element = document.getElementById(id);
    element.value = value === null || value === undefined ? "" : String(value);
  }

  function getText(id) {
    return document.getElementById(id).value.trim();
  }

  function setChecked(id, value) {
    document.getElementById(id).checked = !!value;
  }

  function getChecked(id) {
    return !!document.getElementById(id).checked;
  }

  function toInt(id, fallback) {
    const parsed = Number.parseInt(getText(id), 10);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function toFloat(id, fallback) {
    const parsed = Number.parseFloat(getText(id));
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function valueOrNull(raw) {
    const value = String(raw || "").trim();
    return value ? value : null;
  }

  function markDirty() {
    if (!state.editorOpen) {
      return;
    }
    if (!state.dirty) {
      state.dirty = true;
      updateSessionState();
    }
    updateRunButtons();
  }

  function markClean() {
    state.dirty = false;
    updateSessionState();
  }

  function setEditorOpen(open) {
    state.editorOpen = !!open;

    els.btnCloseConfig.disabled = !state.editorOpen;
    els.btnSaveConfig.disabled = !state.editorOpen;
    els.btnDownloadConfig.disabled = !state.editorOpen;

    if (!state.editorOpen) {
      if (state.currentView === "editor") {
        state.currentView = "results";
      }
    }

    updateViewMode();
    updateRunButtons();
    updateSessionState();
  }

  function setView(view) {
    const normalized = view === "editor" ? "editor" : "results";
    if (normalized === "editor" && !state.editorOpen) {
      setMessage("Open or create a config before switching to Editor view.", true);
      return;
    }
    state.currentView = normalized;
    updateViewMode();
  }

  function updateViewMode() {
    const showEditor = state.currentView === "editor" && state.editorOpen;
    els.editorSection.classList.toggle("hidden", !showEditor);
    els.runPanel.classList.toggle("hidden", state.currentView !== "results");
    els.mainLayout.classList.toggle("editor-focus", showEditor);
    els.mainLayout.classList.toggle("results-focus", state.currentView === "results");

    els.btnShowEditor.classList.toggle("active-view", showEditor);
    els.btnShowResults.classList.toggle("active-view", state.currentView === "results");
  }

  function clearEditorSession() {
    state.rawConfig = null;
    state.dirty = false;
    setEditorOpen(false);
  }

  function updateSessionState() {
    if (!state.editorOpen) {
      els.editorSessionState.textContent = "No config is open.";
      return;
    }
    const path = els.configPath.value.trim();
    const dirtyText = state.dirty ? "unsaved changes" : "saved/clean";
    els.editorSessionState.textContent = path
      ? `Open config: ${path} (${dirtyText})`
      : `Open in-memory config (${dirtyText})`;
  }

  function updateRunButtons() {
    const disabled = !state.editorOpen || hasActiveJob() || !hasRequiredPredictionInputs();
    els.btnRunPrediction.disabled = disabled;
    els.btnShowEditor.disabled = !state.editorOpen;
  }

  function hasRequiredPredictionInputs() {
    if (!state.editorOpen) {
      return false;
    }
    return [
      getText("dataset_gold"),
      getText("dataset_word_column"),
      getText("dataset_score_column"),
      getText("emb_single_path"),
      getText("prediction_target"),
    ].every((value) => String(value || "").trim().length > 0);
  }

  function assertPredictionRunRequirements(config) {
    const requiredPaths = [
      "dataset.gold",
      "dataset.word_column",
      "dataset.score_column",
      "embeddings.spaces.0.path",
      "prediction.target",
    ];
    const missing = requiredPaths.filter((path) => String(getPath(config, path, "") || "").trim() === "");
    if (missing.length > 0) {
      throw new Error(`Fill required fields before running prediction: ${missing.join(", ")}.`);
    }
  }

  function pickGuidedSingleSpace(cfg) {
    const embeddings = getPath(cfg, "embeddings", {}) || {};
    const spaces = Array.isArray(embeddings.spaces) ? embeddings.spaces.filter((space) => space && typeof space === "object") : [];
    const singleSpaceId = String(embeddings.active_space || "").trim();

    let chosen = null;
    if (singleSpaceId) {
      chosen = spaces.find((space) => String(space.id || "").trim() === singleSpaceId) || null;
    }
    if (!chosen && spaces.length > 0) {
      chosen = spaces[0];
    }
    if (!chosen) {
      chosen = { id: "default", kind: "ft_bin", path: "", label: "default_embedding" };
    }

    const chosenId = String(chosen.id || "").trim() || singleSpaceId || "default";
    return {
      id: chosenId,
      kind: String(chosen.kind || "").trim() || "ft_bin",
      path: String(chosen.path || "").trim(),
      label: String(chosen.label || "").trim() || chosenId,
    };
  }

  function collectSingleSpace() {
    const current = pickGuidedSingleSpace(state.rawConfig || {});
    const id = String(current.id || "").trim() || "default";
    const kind = getText("emb_single_kind") || current.kind || "ft_bin";
    const path = getText("emb_single_path");
    const label = String(current.label || "").trim() || id;
    return { id, kind, path, label };
  }

  function syncEmbeddingSelectFromPath(pathValue) {
    const select = els.embExistingSelect;
    if (!select) {
      return;
    }
    const value = String(pathValue || "").trim();
    if (!value) {
      select.value = "";
      return;
    }
    const option = Array.from(select.options).find((candidate) => candidate.value === value);
    select.value = option ? value : "";
  }

  async function refreshEmbeddingsPicker(preferredPath = "") {
    if (!els.embExistingSelect) {
      return;
    }
    const payload = await fetchJSON("/api/fs/embeddings");
    const options = Array.isArray(payload.candidates) ? payload.candidates : [];
    const previousValue = String(preferredPath || els.embExistingSelect.value || getText("emb_single_path") || "").trim();

    els.embExistingSelect.innerHTML = "";
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Select existing embedding...";
    els.embExistingSelect.appendChild(placeholder);

    for (const item of options) {
      if (!item || typeof item !== "object") {
        continue;
      }
      const option = document.createElement("option");
      option.value = String(item.path || "").trim();
      option.textContent = String(item.label || item.name || item.path || "").trim();
      if (!option.value || !option.textContent) {
        continue;
      }
      els.embExistingSelect.appendChild(option);
    }

    syncEmbeddingSelectFromPath(previousValue);
  }

  async function uploadGuidedFile(file, fieldTarget) {
    if (!file) {
      return null;
    }
    const formData = new FormData();
    formData.append("file", file);
    formData.append("field_target", String(fieldTarget || ""));
    const payload = await fetchJSON("/api/fs/upload", {
      method: "POST",
      body: formData,
    });
    return payload;
  }

  function populateGuided(cfg) {
    setText("dataset_gold", getPath(cfg, "dataset.gold", ""));
    setText("dataset_word_column", getPath(cfg, "dataset.word_column", DEFAULT_WORD_COLUMN));
    setText("dataset_score_column", getPath(cfg, "dataset.score_column", DEFAULT_SCORE_COLUMN));
    setChecked("dataset_lowercase", !!getPath(cfg, "dataset.lowercase", true));

    setChecked("pos_enabled", !!getPath(cfg, "dataset.pos_filter.enabled", false));
    setText("pos_column", getPath(cfg, "dataset.pos_filter.pos_column", DEFAULT_POS_COLUMN));
    setText("pos_tags", (getPath(cfg, "dataset.pos_filter.tags", DEFAULT_POS_TAGS) || []).join(", "));
    setText("pos_match_mode", getPath(cfg, "dataset.pos_filter.match_mode", DEFAULT_POS_MATCH_MODE));
    setText("pos_token_pattern", getPath(cfg, "dataset.pos_filter.token_pattern", DEFAULT_POS_TOKEN_PATTERN));

    const singleSpace = pickGuidedSingleSpace(cfg);
    setText("emb_single_kind", singleSpace.kind);
    setText("emb_single_path", singleSpace.path);
    syncEmbeddingSelectFromPath(singleSpace.path);

    setText("runtime_output_dir", getPath(cfg, "runtime.output_dir", DEFAULT_RUNTIME_OUTPUT_DIR));
    setText("runtime_seed", getPath(cfg, "runtime.seed", 13));
    setText("runtime_n_jobs", getPath(cfg, "runtime.n_jobs", -1));

    setText("prediction_test_size", getPath(cfg, "prediction.test_size", 0.2));
    setText("prediction_cv_folds", getPath(cfg, "prediction.cv_folds", 5));
    setText("prediction_k_min", getPath(cfg, "prediction.k_min", 5));
    setText("prediction_k_max", getPath(cfg, "prediction.k_max", 100));
    setText("prediction_k_step", getPath(cfg, "prediction.k_step", 5));
    setText("prediction_topn_neighbors", getPath(cfg, "prediction.topn_neighbors", 10));
    setText("prediction_target", getPath(cfg, "prediction.target", "") || "");

    setText("reports_level", getPath(cfg, "reports.level", "core"));
  }

  function buildConfigFromGuided() {
    const base = state.rawConfig ? deepClone(state.rawConfig) : {};
    setPath(base, "dataset.gold", getText("dataset_gold"));
    setPath(base, "dataset.word_column", getText("dataset_word_column") || DEFAULT_WORD_COLUMN);
    setPath(base, "dataset.score_column", getText("dataset_score_column") || DEFAULT_SCORE_COLUMN);
    setPath(base, "dataset.lowercase", getChecked("dataset_lowercase"));

    setPath(base, "dataset.pos_filter.enabled", getChecked("pos_enabled"));
    setPath(base, "dataset.pos_filter.pos_column", getText("pos_column") || DEFAULT_POS_COLUMN);
    const parsedPosTags = getText("pos_tags")
      .split(",")
      .map((item) => item.trim())
      .filter((item) => item);
    setPath(
      base,
      "dataset.pos_filter.tags",
      parsedPosTags.length > 0 ? parsedPosTags : [...DEFAULT_POS_TAGS]
    );
    setPath(base, "dataset.pos_filter.match_mode", getText("pos_match_mode") || DEFAULT_POS_MATCH_MODE);
    setPath(base, "dataset.pos_filter.token_pattern", getText("pos_token_pattern") || DEFAULT_POS_TOKEN_PATTERN);

    const singleSpace = collectSingleSpace();
    setPath(base, "embeddings.mode", "single");
    setPath(base, "embeddings.active_space", singleSpace.id);
    setPath(base, "embeddings.spaces", [singleSpace]);

    setPath(base, "runtime.output_dir", getText("runtime_output_dir") || DEFAULT_RUNTIME_OUTPUT_DIR);
    setPath(base, "runtime.seed", toInt("runtime_seed", 13));
    setPath(base, "runtime.n_jobs", toInt("runtime_n_jobs", -1));

    setPath(base, "prediction.test_size", toFloat("prediction_test_size", 0.2));
    setPath(base, "prediction.cv_folds", toInt("prediction_cv_folds", 5));
    setPath(base, "prediction.k_min", toInt("prediction_k_min", 5));
    setPath(base, "prediction.k_max", toInt("prediction_k_max", 100));
    setPath(base, "prediction.k_step", toInt("prediction_k_step", 5));
    setPath(base, "prediction.topn_neighbors", toInt("prediction_topn_neighbors", 10));
    setPath(base, "prediction.target", valueOrNull(getText("prediction_target")));

    setPath(base, "reports.level", getText("reports_level") || "core");
    return base;
  }

  function syncGuidedToRaw() {
    const config = buildConfigFromGuided();
    state.rawConfig = deepClone(config);
    return config;
  }

  function getLiveConfigForAction() {
    if (!state.editorOpen) {
      throw new Error("Open or create a config before running commands.");
    }
    return syncGuidedToRaw();
  }

  function setRawConfig(config, sourceLabel, markCleanFlag) {
    state.rawConfig = deepClone(config);
    populateGuided(state.rawConfig);
    refreshEmbeddingsPicker(getText("emb_single_path")).catch((_error) => {});
    setEditorOpen(true);
    setView("editor");
    if (markCleanFlag) {
      markClean();
    } else {
      state.dirty = true;
      updateSessionState();
    }
    setMessage(`Opened ${sourceLabel}.`);
  }

  function renderListing(listing) {
    const root = els.browseListing;
    root.innerHTML = "";

    if (listing.parent_path) {
      const up = document.createElement("div");
      up.className = "listing-item";
      up.innerHTML = `<span>[UP]</span><code>${escapeHtml(listing.parent_path)}</code><button type="button">Open</button>`;
      up.querySelector("button").addEventListener("click", () => {
        els.browsePath.value = listing.parent_path;
        listDirectory();
      });
      root.appendChild(up);
    }

    for (const entry of listing.entries || []) {
      const line = document.createElement("div");
      line.className = "listing-item";
      line.innerHTML = `
        <span>${entry.is_dir ? "[DIR]" : "[FILE]"}</span>
        <code title="${escapeHtml(entry.path)}">${escapeHtml(entry.path)}</code>
        <div>
          <button type="button" class="secondary">${entry.is_dir ? "Open" : "Use"}</button>
        </div>
      `;

      line.querySelector("button").addEventListener("click", async () => {
        if (entry.is_dir) {
          els.browsePath.value = entry.path;
          listDirectory();
          return;
        }
        if (entry.path.toLowerCase().endsWith(".json")) {
          try {
            await loadConfigPath(entry.path);
          } catch (error) {
            setMessage(error.message, true);
          }
        }
      });

      root.appendChild(line);
    }
  }

  async function listDirectory() {
    const data = await fetchJSON(`/api/fs/list?path=${encodeURIComponent(els.browsePath.value || ".")}`);
    renderListing(data);
  }

  function prettyPrintJob(job) {
    return JSON.stringify(job, null, 2);
  }

  function stopElapsedTimer() {
    if (state.elapsedTimer) {
      clearInterval(state.elapsedTimer);
      state.elapsedTimer = null;
    }
  }

  function startElapsedTimer(startIso) {
    stopElapsedTimer();
    const startMs = parseIso(startIso) || Date.now();
    const tick = () => {
      const elapsed = Math.floor((Date.now() - startMs) / 1000);
      els.workingElapsed.textContent = formatElapsed(elapsed);
    };
    tick();
    state.elapsedTimer = setInterval(tick, 1000);
  }

  function updateWorkingIndicator(job) {
    if (!job) {
      els.workingIndicator.classList.add("hidden");
      els.workingBadge.textContent = "IDLE";
      els.workingSpinner.classList.add("hidden");
      els.workingElapsed.textContent = "0s";
      els.workingLatestLog.textContent = "No active job.";
      stopElapsedTimer();
      return;
    }

    els.workingIndicator.classList.remove("hidden");
    els.workingBadge.textContent = String(job.status || "unknown").toUpperCase();

    const isWorking = ["submitting", "queued", "running"].includes(job.status);
    els.workingSpinner.classList.toggle("hidden", !isWorking);
    if (isWorking) {
      startElapsedTimer(job.created_at);
    } else {
      stopElapsedTimer();
      const start = parseIso(job.created_at);
      const end = parseIso(job.updated_at);
      if (start && end && end >= start) {
        els.workingElapsed.textContent = formatElapsed((end - start) / 1000);
      }
    }

    const logs = Array.isArray(job.logs) ? job.logs : [];
    const latest = logs.length > 0 ? logs[logs.length - 1] : "No logs yet.";
    els.workingLatestLog.textContent = latest.split("\n")[0].slice(0, 160);
  }

  async function pollJob(jobId) {
    if (state.pollTimer) {
      clearTimeout(state.pollTimer);
      state.pollTimer = null;
    }

    try {
      const job = await fetchJSON(
        `/api/jobs/${encodeURIComponent(jobId)}`,
        {},
        { allowPayloadError: true }
      );
      state.activeJob = job;
      updateRunButtons();
      updateWorkingIndicator(job);
      els.jobStatus.textContent = prettyPrintJob(job);

      if (["queued", "running"].includes(job.status)) {
        state.pollTimer = setTimeout(() => pollJob(jobId), 800);
      } else {
        state.currentJobId = null;
        if (job.status === "done") {
          await refreshLatestResults();
        } else if (job.status === "error") {
          const detail = String(job.error || "").trim();
          const reason = detail || "Unknown error.";
          setMessage(`Job ${job.job_id} failed: ${reason}`, true);
        }
      }
    } catch (error) {
      setMessage(error.message, true);
    }
  }

  async function submitJob(endpoint, label) {
    const config = getLiveConfigForAction();
    if (label === "prediction-run") {
      assertPredictionRunRequirements(config);
    }
    setView("results");
    const payload = { config };
    const maybePath = els.configPath.value.trim();
    if (maybePath) {
      payload.config_path = maybePath;
    }

    const transient = {
      job_id: "<pending>",
      command: label,
      status: "submitting",
      created_at: nowIso(),
      updated_at: nowIso(),
      logs: ["Submitting job request..."],
    };
    state.activeJob = transient;
    state.currentJobId = null;
    updateRunButtons();
    updateWorkingIndicator(transient);
    els.jobStatus.textContent = prettyPrintJob(transient);

    try {
      const data = await fetchJSON(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      state.currentJobId = data.job_id;
      const queued = {
        job_id: data.job_id,
        command: label,
        status: data.status || "queued",
        created_at: nowIso(),
        updated_at: nowIso(),
        logs: ["Job accepted by server queue."],
      };
      state.activeJob = queued;
      updateRunButtons();
      updateWorkingIndicator(queued);
      els.jobStatus.textContent = prettyPrintJob(queued);
      setMessage(`Started ${label}: ${data.job_id}`);
      pollJob(data.job_id);
    } catch (error) {
      if (error.status === 409 && error.payload && error.payload.active_job) {
        const active = error.payload.active_job;
        state.activeJob = active;
        updateRunButtons();
        updateWorkingIndicator(active);
        els.jobStatus.textContent = prettyPrintJob(active);
        setMessage(
          `Another job is already ${active.status}. Tracking existing job ${active.job_id}.`,
          true
        );
        pollJob(active.job_id);
        return;
      }
      state.currentJobId = null;
      state.activeJob = null;
      updateRunButtons();
      updateWorkingIndicator(null);
      els.jobStatus.textContent = "No jobs submitted yet.";
      throw error;
    }
  }

  function setResultControlsEnabled(enabled) {
    els.resultArtifactSelect.disabled = !enabled;
    els.resultSearchQuery.disabled = !enabled;
    els.btnSearchArtifact.disabled = !enabled;
    els.btnDownloadArtifact.disabled = !enabled;
    els.btnCopyArtifactPath.disabled = !enabled;
    if (!enabled) {
      els.btnPrevPage.disabled = true;
      els.btnNextPage.disabled = true;
      els.resultPagingInfo.textContent = "0-0";
    }
  }

  function isSummaryArtifactSelected() {
    const key = String(state.results.artifactKey || "");
    return key === "summary" || key === "holdout_summary";
  }

  function isSearchDisabledArtifactSelected() {
    const key = String(state.results.artifactKey || "");
    return isSummaryArtifactSelected() || key === "cv_results";
  }

  function updateArtifactModeControls() {
    const searchDisabled = isSearchDisabledArtifactSelected();
    const summaryMode = isSummaryArtifactSelected();
    const enabled = !!state.results.latestJob && !!state.results.artifactKey;
    const searchEnabled = enabled && !searchDisabled;
    if (els.resultSearchRow) {
      els.resultSearchRow.classList.toggle("hidden", searchDisabled);
    }
    els.resultSearchQuery.disabled = !searchEnabled;
    els.btnSearchArtifact.disabled = !searchEnabled;

    if (summaryMode) {
      els.btnPrevPage.disabled = true;
      els.btnNextPage.disabled = true;
      els.resultPagingInfo.textContent = "summary";
    }
  }

  function sortArtifactKeys(keys) {
    const ordered = [...keys];
    ordered.sort((left, right) => left.localeCompare(right));
    return ordered;
  }

  function pickDefaultArtifactKey(artifactKeys) {
    if (artifactKeys.includes("vocab_predictions")) {
      return "vocab_predictions";
    }
    return artifactKeys[0] || "";
  }

  function isSpaceSpecificColumn(columnName) {
    const value = String(columnName || "").trim();
    if (!value) {
      return false;
    }
    return (
      value.startsWith("pred__") ||
      value.startsWith("neighbors__") ||
      value.startsWith("neighbor_gold_scores__") ||
      value.startsWith("neighbor_cosine_distances__")
    );
  }

  function isCountLikeField(fieldName) {
    const text = String(fieldName || "").trim();
    if (!text) {
      return false;
    }
    const segments = text
      .split(".")
      .map((segment) => segment.trim().toLowerCase())
      .filter((segment) => segment.length > 0);
    return segments.some((segment) => segment.startsWith("n_") || segment === "_row_index");
  }

  function formatSummaryValue(key, value) {
    if (value === null || value === undefined) {
      return "";
    }
    if (typeof value === "number") {
      if (isCountLikeField(key)) {
        return String(value);
      }
      return formatNumberToSigFigs(value, 3);
    }
    if (typeof value === "boolean") {
      return value ? "true" : "false";
    }
    if (typeof value === "string") {
      const numeric = parseNumericLike(value);
      if (numeric !== null) {
        if (isCountLikeField(key)) {
          return value.trim();
        }
        return formatNumberToSigFigs(numeric, 3);
      }
      return value;
    }
    return JSON.stringify(value);
  }

  function formatNumberToSigFigs(value, sigFigs = 3) {
    if (!Number.isFinite(value)) {
      return String(value);
    }
    if (value === 0) {
      return "0.00";
    }
    return Number(value).toPrecision(sigFigs);
  }

  function parseNumericLike(value) {
    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }
    if (typeof value !== "string") {
      return null;
    }
    const trimmed = value.trim();
    if (!trimmed) {
      return null;
    }
    if (!/^[-+]?((\d+(\.\d*)?)|(\.\d+))(e[-+]?\d+)?$/i.test(trimmed)) {
      return null;
    }
    const parsed = Number.parseFloat(trimmed);
    if (!Number.isFinite(parsed)) {
      return null;
    }
    return parsed;
  }

  function formatNumericToken(token, fieldName) {
    const numeric = parseNumericLike(token);
    if (numeric === null) {
      return String(token);
    }
    if (isCountLikeField(fieldName)) {
      return String(token).trim();
    }
    return formatNumberToSigFigs(numeric, 3);
  }

  function formatCsvCellValue(value, columnName) {
    if (value === null || value === undefined) {
      return "";
    }

    const text = String(value);
    if (text.includes("|")) {
      return text
        .split("|")
        .map((token) => token.trim())
        .map((token) => formatNumericToken(token, columnName))
        .join(" | ");
    }
    return formatNumericToken(text, columnName);
  }

  function flattenObject(obj, prefix = "") {
    const out = [];
    for (const [key, value] of Object.entries(obj || {})) {
      const nextKey = prefix ? `${prefix}.${key}` : key;
      if (value && typeof value === "object" && !Array.isArray(value)) {
        out.push(...flattenObject(value, nextKey));
      } else {
        out.push([nextKey, value]);
      }
    }
    return out;
  }

  function createKeyValueTable(entries) {
    const ordered = [...entries].sort(([left], [right]) => String(left).localeCompare(String(right)));
    const table = document.createElement("table");
    table.className = "result-table";
    const thead = document.createElement("thead");
    thead.innerHTML = "<tr><th>Key</th><th>Value</th></tr>";
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    for (const [key, value] of ordered) {
      const tr = document.createElement("tr");
      const keyCell = document.createElement("td");
      const valueCell = document.createElement("td");
      keyCell.textContent = key;
      valueCell.textContent = formatSummaryValue(key, value);
      tr.appendChild(keyCell);
      tr.appendChild(valueCell);
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    return table;
  }

  function renderSummaryView(summaryData, rawText, artifactPath) {
    els.resultPath.textContent = artifactPath || "";
    els.resultPreview.innerHTML = "";

    const root = document.createElement("div");
    root.className = "summary-view";

    const kpiSection = document.createElement("section");
    const kpiTitle = document.createElement("h4");
    kpiTitle.textContent = "Key Metrics";
    kpiSection.appendChild(kpiTitle);
    const cards = document.createElement("div");
    cards.className = "summary-kpi-grid";

    const preferredKpis = [
      "test_spearman",
      "test_rmse",
      "best_k",
      "gold_coverage",
      "n_gold_words",
      "n_test",
      "best_weights",
    ];
    const topEntries = Object.entries(summaryData || {}).sort(([left], [right]) => String(left).localeCompare(String(right)));
    const chosen = [];
    for (const key of preferredKpis) {
      if (key in summaryData) {
        chosen.push([key, summaryData[key]]);
      }
    }

    for (const [key, value] of chosen) {
      const card = document.createElement("article");
      card.className = "summary-kpi-card";
      const label = document.createElement("div");
      label.className = "summary-kpi-label";
      label.textContent = key;
      const val = document.createElement("div");
      val.className = "summary-kpi-value";
      val.textContent = formatSummaryValue(key, value);
      card.appendChild(label);
      card.appendChild(val);
      cards.appendChild(card);
    }

    if (chosen.length > 0) {
      kpiSection.appendChild(cards);
      root.appendChild(kpiSection);
    }

    const used = new Set(chosen.map(([key]) => key));
    const scalarOverview = [];
    const structuredSections = [];

    for (const [key, value] of topEntries) {
      if (used.has(key)) {
        continue;
      }
      if (value && typeof value === "object" && !Array.isArray(value)) {
        const flattened = flattenObject(value).sort(([left], [right]) => String(left).localeCompare(String(right)));
        structuredSections.push({ key, entries: flattened });
      } else if (Array.isArray(value)) {
        structuredSections.push({ key, entries: [[key, value]] });
      } else {
        scalarOverview.push([key, value]);
      }
    }

    if (scalarOverview.length > 0) {
      const overviewSection = document.createElement("section");
      const title = document.createElement("h4");
      title.textContent = "Overview";
      overviewSection.appendChild(title);
      overviewSection.appendChild(createKeyValueTable(scalarOverview));
      root.appendChild(overviewSection);
    }

    for (const section of structuredSections.sort((left, right) => String(left.key).localeCompare(String(right.key)))) {
      const group = document.createElement("section");
      const title = document.createElement("h4");
      title.textContent = section.key;
      group.appendChild(title);
      group.appendChild(createKeyValueTable(section.entries));
      root.appendChild(group);
    }

    const rawWrap = document.createElement("section");
    const rawToggle = document.createElement("button");
    rawToggle.type = "button";
    rawToggle.className = "secondary summary-raw-toggle";
    rawToggle.textContent = "Show Raw JSON";
    const rawPre = document.createElement("pre");
    rawPre.className = "result-text hidden";
    rawPre.textContent = rawText;
    rawToggle.addEventListener("click", () => {
      const hidden = rawPre.classList.toggle("hidden");
      rawToggle.textContent = hidden ? "Show Raw JSON" : "Hide Raw JSON";
    });
    rawWrap.appendChild(rawToggle);
    rawWrap.appendChild(rawPre);
    root.appendChild(rawWrap);

    els.resultPreview.appendChild(root);
  }

  function renderResultPreview(payload) {
    const total = Number(payload.total_matches || 0);
    const offset = Number(payload.offset || 0);
    const count = payload.kind === "csv"
      ? (Array.isArray(payload.rows) ? payload.rows.length : 0)
      : (Array.isArray(payload.lines) ? payload.lines.length : 0);
    const from = total === 0 ? 0 : offset + 1;
    const to = total === 0 ? 0 : offset + count;
    els.resultPagingInfo.textContent = `${from}-${to} / ${total}`;

    els.btnPrevPage.disabled = offset <= 0;
    els.btnNextPage.disabled = !payload.has_more;
    els.resultPath.textContent = payload.artifact_path || "";

    els.resultPreview.innerHTML = "";
    if (payload.kind === "csv") {
      const columns = (Array.isArray(payload.columns) ? payload.columns : []).filter((column) => !isSpaceSpecificColumn(column));
      const rows = Array.isArray(payload.rows) ? payload.rows : [];

      if (rows.length === 0) {
        els.resultPreview.textContent = "No matching rows.";
        return;
      }

      const table = document.createElement("table");
      table.className = "result-table";
      const thead = document.createElement("thead");
      const headRow = document.createElement("tr");
      for (const column of columns) {
        const th = document.createElement("th");
        th.textContent = column;
        headRow.appendChild(th);
      }
      thead.appendChild(headRow);
      table.appendChild(thead);

      const tbody = document.createElement("tbody");
      for (const row of rows) {
        const tr = document.createElement("tr");
        for (const column of columns) {
          const td = document.createElement("td");
          td.textContent = formatCsvCellValue(row[column], column);
          tr.appendChild(td);
        }
        tbody.appendChild(tr);
      }
      table.appendChild(tbody);
      els.resultPreview.appendChild(table);
      return;
    }

    const lines = Array.isArray(payload.lines) ? payload.lines : [];
    if (lines.length === 0) {
      els.resultPreview.textContent = "No matching lines.";
      return;
    }
    const pre = document.createElement("pre");
    pre.className = "result-text";
    pre.textContent = lines
      .map((line) => `${line.line_number}: ${line.text}`)
      .join("\n");
    els.resultPreview.appendChild(pre);
  }

  function currentResultParams(resetOffset) {
    if (resetOffset) {
      state.results.offset = 0;
    }
    const searchDisabled = isSearchDisabledArtifactSelected();
    state.results.query = searchDisabled ? "" : els.resultSearchQuery.value.trim();

    return {
      job_id: state.results.latestJob.job_id,
      artifact_key: state.results.artifactKey,
      q: state.results.query,
      offset: String(state.results.offset),
      limit: String(state.results.limit),
    };
  }

  async function previewSelectedArtifact(resetOffset = false) {
    if (!state.results.latestJob || !state.results.artifactKey) {
      return;
    }

    if (isSummaryArtifactSelected()) {
      const params = new URLSearchParams({
        job_id: state.results.latestJob.job_id,
        artifact_key: state.results.artifactKey,
      });
      const response = await fetch(`/api/results/download?${params.toString()}`);
      if (!response.ok) {
        throw new Error(`Could not load ${state.results.artifactKey}.`);
      }
      const rawText = await response.text();
      let parsed;
      try {
        parsed = JSON.parse(rawText);
      } catch (_error) {
        parsed = null;
      }
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        const artifactPath = (state.results.latestJob.artifacts || {})[state.results.artifactKey] || "";
        renderSummaryView(parsed, rawText, artifactPath);
      } else {
        els.resultPath.textContent = (state.results.latestJob.artifacts || {})[state.results.artifactKey] || "";
        els.resultPreview.innerHTML = "";
        const pre = document.createElement("pre");
        pre.className = "result-text";
        pre.textContent = rawText;
        els.resultPreview.appendChild(pre);
      }
      state.results.offset = 0;
      updateArtifactModeControls();
      return;
    }

    const params = new URLSearchParams(currentResultParams(resetOffset));
    const payload = await fetchJSON(`/api/results/preview?${params.toString()}`);
    renderResultPreview(payload);
    state.results.offset = Number(payload.offset || 0);
    updateArtifactModeControls();
  }

  async function refreshLatestResults() {
    const previousJobId = state.results.latestJob ? state.results.latestJob.job_id : "";
    const data = await fetchJSON("/api/results/latest");
    const latest = data.latest_job;
    state.results.latestJob = latest;

    if (!latest) {
      setResultControlsEnabled(false);
      els.resultArtifactSelect.innerHTML = "";
      els.resultsMeta.textContent = "No completed jobs yet.";
      els.resultPath.textContent = "No artifact selected.";
      els.resultPreview.textContent = "No preview loaded.";
      els.resultPagingInfo.textContent = "0-0";
      els.btnPrevPage.disabled = true;
      els.btnNextPage.disabled = true;
      return;
    }

    const artifacts = latest.artifacts || {};
    const artifactKeys = sortArtifactKeys(Object.keys(artifacts));
    els.resultArtifactSelect.innerHTML = "";
    for (const key of artifactKeys) {
      const option = document.createElement("option");
      option.value = key;
      option.textContent = key;
      els.resultArtifactSelect.appendChild(option);
    }

    if (artifactKeys.length === 0) {
      setResultControlsEnabled(false);
      els.resultsMeta.textContent = `Job ${latest.job_id} (${latest.command}) has no artifacts.`;
      return;
    }

    if (latest.job_id !== previousJobId) {
      state.results.artifactKey = pickDefaultArtifactKey(artifactKeys);
      state.results.offset = 0;
    } else if (!artifactKeys.includes(state.results.artifactKey)) {
      state.results.artifactKey = pickDefaultArtifactKey(artifactKeys);
      state.results.offset = 0;
    }
    els.resultArtifactSelect.value = state.results.artifactKey;
    setResultControlsEnabled(true);
    updateArtifactModeControls();

    els.resultsMeta.textContent = `Job ${latest.job_id} (${latest.command}) completed at ${formatTimestamp(latest.updated_at)}.`;
    await previewSelectedArtifact(true);
  }

  async function handleNewConfig() {
    const data = await fetchJSON("/api/config/default");
    setRawConfig(data.raw_config, "new default config", true);
  }

  async function applyExistingResultsOnConfigOpen(existingResults, sourceMessage) {
    const info = existingResults && typeof existingResults === "object" ? existingResults : null;
    const warnings = info && Array.isArray(info.warnings)
      ? info.warnings.map((item) => String(item || "").trim()).filter((item) => item)
      : [];

    if (!info || !info.found) {
      if (warnings.length > 0) {
        setMessage(`${sourceMessage} No existing results attached. ${warnings[0]}`, true);
      } else {
        setMessage(sourceMessage);
      }
      return;
    }

    await refreshLatestResults();
    setView("results");

    const artifacts = info.artifacts && typeof info.artifacts === "object" ? info.artifacts : {};
    const artifactCount = Object.keys(artifacts).length;
    let message = `${sourceMessage} Attached ${artifactCount} existing artifact${artifactCount === 1 ? "" : "s"} for inspection.`;
    if (warnings.length > 0) {
      message = `${message} ${warnings[0]}`;
    }
    setMessage(message, warnings.length > 0);
  }

  async function handleLoadConfig() {
    const path = els.configPath.value.trim();
    if (!path) {
      throw new Error("Provide a config path to load.");
    }
    const data = await fetchJSON("/api/config/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    const openedPath = data.path || path;
    setRawConfig(data.raw_config, openedPath, true);
    els.configPath.value = openedPath;
    await applyExistingResultsOnConfigOpen(data.existing_results, `Loaded ${openedPath}.`);
  }

  async function loadConfigPath(path) {
    const selectedPath = String(path || "").trim();
    if (!selectedPath) {
      throw new Error("Provide a config path to load.");
    }
    els.configPath.value = selectedPath;
    await handleLoadConfig();
  }

  async function handleSaveConfig() {
    const path = els.configPath.value.trim();
    if (!path) {
      throw new Error("Provide a repo JSON path before saving.");
    }
    const config = getLiveConfigForAction();
    const data = await fetchJSON("/api/config/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path, config }),
    });
    setRawConfig(data.raw_config, `saved ${data.path}`, true);
    els.configPath.value = data.path || path;
  }

  async function handleUploadConfig(file) {
    const formData = new FormData();
    formData.append("file", file);
    const data = await fetchJSON("/api/config/upload", {
      method: "POST",
      body: formData,
    });
    setRawConfig(data.raw_config, `uploaded ${data.filename}`, false);
    await applyExistingResultsOnConfigOpen(data.existing_results, `Uploaded ${data.filename}.`);
  }

  function handleDownloadConfig() {
    const config = getLiveConfigForAction();
    const rawPath = els.configPath.value.trim();
    const filename = rawPath ? rawPath.split("/").pop() : "config.json";
    const query = new URLSearchParams({
      filename: filename || "config.json",
      config: JSON.stringify(config),
    });
    window.location.href = `/api/config/download?${query.toString()}`;
  }

  function closeConfigSession() {
    if (!state.editorOpen) {
      return;
    }
    if (state.dirty && !window.confirm("Discard unsaved config changes and close the editor?")) {
      return;
    }
    clearEditorSession();
    setMessage("Closed config session.");
  }

  function showHelpPopover(button, helpKey) {
    const helpText = FIELD_HELP[helpKey] || "No help text available for this field yet.";
    els.helpPopover.textContent = helpText;
    els.helpPopover.classList.remove("hidden");

    const rect = button.getBoundingClientRect();
    const maxLeft = Math.max(8, window.innerWidth - 360);
    const left = Math.min(maxLeft, Math.max(8, rect.left));
    const top = Math.min(window.innerHeight - 140, rect.bottom + 6);

    els.helpPopover.style.left = `${left}px`;
    els.helpPopover.style.top = `${top}px`;
  }

  function hideHelpPopover() {
    els.helpPopover.classList.add("hidden");
  }

  function bindHelpPopovers() {
    document.querySelectorAll(".help-trigger").forEach((button) => {
      button.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        const key = button.getAttribute("data-help-key") || "";
        showHelpPopover(button, key);
      });
    });

    document.addEventListener("click", (event) => {
      if (event.target instanceof Element && event.target.closest(".help-trigger")) {
        return;
      }
      hideHelpPopover();
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        hideHelpPopover();
      }
    });
  }

  function bindDirtyTracking() {
    els.guidedPanel.addEventListener("input", markDirty);
    els.guidedPanel.addEventListener("change", markDirty);
  }

  function bindEvents() {
    els.btnNewConfig.addEventListener("click", async () => {
      try {
        await handleNewConfig();
      } catch (error) {
        setMessage(error.message, true);
      }
    });

    els.btnCloseConfig.addEventListener("click", closeConfigSession);

    els.btnLoadConfig.addEventListener("click", async () => {
      try {
        await handleLoadConfig();
      } catch (error) {
        setMessage(error.message, true);
      }
    });

    els.btnSaveConfig.addEventListener("click", async () => {
      try {
        await handleSaveConfig();
      } catch (error) {
        setMessage(error.message, true);
      }
    });

    els.uploadConfigInput.addEventListener("change", async (event) => {
      const file = event.target.files && event.target.files[0];
      if (!file) {
        return;
      }
      try {
        await handleUploadConfig(file);
      } catch (error) {
        setMessage(error.message, true);
      } finally {
        event.target.value = "";
      }
    });

    els.btnDownloadConfig.addEventListener("click", () => {
      try {
        handleDownloadConfig();
      } catch (error) {
        setMessage(error.message, true);
      }
    });

    els.btnBrowse.addEventListener("click", async () => {
      try {
        await listDirectory();
      } catch (error) {
        setMessage(error.message, true);
      }
    });

    if (els.btnUploadGoldCsv && els.goldCsvUploadInput) {
      els.btnUploadGoldCsv.addEventListener("click", () => {
        els.goldCsvUploadInput.click();
      });
      els.goldCsvUploadInput.addEventListener("change", async (event) => {
        const file = event.target.files && event.target.files[0];
        if (!file) {
          return;
        }
        try {
          const payload = await uploadGuidedFile(file, "gold");
          if (payload && payload.path) {
            setText("dataset_gold", payload.path);
            syncGuidedToRaw();
            markDirty();
            setMessage(`Uploaded ${payload.filename} to ${payload.path}.`);
          }
        } catch (error) {
          setMessage(error.message, true);
        } finally {
          event.target.value = "";
        }
      });
    }

    if (els.btnUploadPredictVocab && els.predictVocabUploadInput) {
      els.btnUploadPredictVocab.addEventListener("click", () => {
        els.predictVocabUploadInput.click();
      });
      els.predictVocabUploadInput.addEventListener("change", async (event) => {
        const file = event.target.files && event.target.files[0];
        if (!file) {
          return;
        }
        try {
          const payload = await uploadGuidedFile(file, "target");
          if (payload && payload.path) {
            setText("prediction_target", payload.path);
            syncGuidedToRaw();
            markDirty();
            setMessage(`Uploaded ${payload.filename} to ${payload.path}.`);
          }
        } catch (error) {
          setMessage(error.message, true);
        } finally {
          event.target.value = "";
        }
      });
    }

    if (els.btnRefreshEmbeddings) {
      els.btnRefreshEmbeddings.addEventListener("click", async () => {
        try {
          await refreshEmbeddingsPicker(getText("emb_single_path"));
          setMessage("Refreshed embedding list.");
        } catch (error) {
          setMessage(error.message, true);
        }
      });
    }

    if (els.embExistingSelect) {
      els.embExistingSelect.addEventListener("change", () => {
        const selected = els.embExistingSelect.value || "";
        if (!selected) {
          return;
        }
        setText("emb_single_path", selected);
        markDirty();
      });
    }

    const embSinglePath = document.getElementById("emb_single_path");
    if (embSinglePath) {
      embSinglePath.addEventListener("input", () => {
        syncEmbeddingSelectFromPath(getText("emb_single_path"));
      });
      embSinglePath.addEventListener("change", () => {
        syncEmbeddingSelectFromPath(getText("emb_single_path"));
      });
    }

    els.btnShowEditor.addEventListener("click", () => setView("editor"));
    els.btnShowResults.addEventListener("click", () => setView("results"));

    els.btnRunPrediction.addEventListener("click", async () => {
      try {
        await submitJob("/api/jobs/prediction-run", "prediction-run");
      } catch (error) {
        setMessage(error.message, true);
      }
    });

    els.btnRefreshResults.addEventListener("click", async () => {
      try {
        await refreshLatestResults();
        setMessage("Refreshed latest results.");
      } catch (error) {
        setMessage(error.message, true);
      }
    });

    els.resultArtifactSelect.addEventListener("change", () => {
      (async () => {
        try {
          state.results.artifactKey = els.resultArtifactSelect.value;
          await previewSelectedArtifact(true);
        } catch (error) {
          setMessage(error.message, true);
        }
      })();
    });

    els.btnSearchArtifact.addEventListener("click", async () => {
      try {
        state.results.artifactKey = els.resultArtifactSelect.value;
        await previewSelectedArtifact(true);
      } catch (error) {
        setMessage(error.message, true);
      }
    });

    els.btnPrevPage.addEventListener("click", async () => {
      try {
        state.results.offset = Math.max(0, state.results.offset - state.results.limit);
        await previewSelectedArtifact(false);
      } catch (error) {
        setMessage(error.message, true);
      }
    });

    els.btnNextPage.addEventListener("click", async () => {
      try {
        state.results.offset += state.results.limit;
        await previewSelectedArtifact(false);
      } catch (error) {
        setMessage(error.message, true);
      }
    });

    els.btnDownloadArtifact.addEventListener("click", () => {
      if (!state.results.latestJob || !state.results.artifactKey) {
        return;
      }
      const query = new URLSearchParams({
        job_id: state.results.latestJob.job_id,
        artifact_key: state.results.artifactKey,
      });
      window.location.href = `/api/results/download?${query.toString()}`;
    });

    els.btnCopyArtifactPath.addEventListener("click", async () => {
      const path = els.resultPath.textContent || "";
      if (!path) {
        return;
      }
      try {
        await navigator.clipboard.writeText(path);
        setMessage("Artifact path copied to clipboard.");
      } catch (_error) {
        setMessage("Could not copy path to clipboard.", true);
      }
    });
  }

  async function boot() {
    bindHelpPopovers();
    bindDirtyTracking();
    bindEvents();
    clearEditorSession();
    updateViewMode();
    updateWorkingIndicator(null);
    setResultControlsEnabled(false);

    try {
      await handleNewConfig();
      await refreshEmbeddingsPicker(getText("emb_single_path"));
    } catch (error) {
      setMessage(error.message, true);
    }

    try {
      await listDirectory();
    } catch (_error) {
      try {
        els.browsePath.value = ".";
        await listDirectory();
        setMessage("configs directory not found; browsing repository root instead.");
      } catch (fallbackError) {
        setMessage(fallbackError.message, true);
      }
    }

    try {
      await refreshLatestResults();
    } catch (error) {
      setMessage(error.message, true);
    }
  }

  boot();
})();
