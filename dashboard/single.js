// Single-video subtitle dashboard — minimal client logic.
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const drop = $("drop");
  const fileInput = $("video");
  const fileMeta = $("fileMeta");
  const startBtn = $("startBtn");
  const modeSeg = $("modeSeg");
  const translateOpts = $("translateOpts");
  const formErr = $("formErr");
  const progCard = $("progCard");
  const setupCard = $("setupCard");
  const resultCard = $("resultCard");
  const stepper = $("stepper");
  const bar = $("bar");
  const stepDetail = $("stepDetail");
  const logEl = $("log");

  let selectedFile = null;
  let mode = "original";
  let pollTimer = null;
  let useLocal = false;

  // ── file picker ──
  drop.addEventListener("click", () => fileInput.click());
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("drag"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("drag"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault(); drop.classList.remove("drag");
    if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) setFile(fileInput.files[0]);
  });
  function setFile(f) {
    selectedFile = f;
    useLocal = false;
    $("useLocal").checked = false;
    $("localBox").classList.add("hidden");
    const mb = (f.size / 1048576).toFixed(1);
    fileMeta.textContent = `Selected: ${f.name} · ${mb} MB`;
    startBtn.disabled = false;
    formErr.textContent = "";
  }

  // ── local path toggle ──
  $("useLocal").addEventListener("change", () => {
    useLocal = $("useLocal").checked;
    $("localBox").classList.toggle("hidden", !useLocal);
    if (useLocal) {
      selectedFile = null;
      fileInput.value = "";
      fileMeta.textContent = "";
      const p = $("local_path").value.trim();
      if (p) fileMeta.textContent = `Local: ${p}`;
      startBtn.disabled = !p;
    } else {
      fileMeta.textContent = "";
      startBtn.disabled = true;
    }
  });
  $("local_path").addEventListener("input", () => {
    if (!useLocal) return;
    const p = $("local_path").value.trim();
    fileMeta.textContent = p ? `Local: ${p}` : "";
    startBtn.disabled = !p;
  });

  // ── mode segmented control ──
  modeSeg.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    mode = b.dataset.v;
    [...modeSeg.children].forEach((x) => x.classList.toggle("on", x === b));
    translateOpts.classList.toggle("hidden", mode !== "translate");
  });

  // ── OpenRouter model preset toggle ──
  const llmPreset = $("llm_model_preset");
  const llmCustom = $("llm_model");
  const llmHint = $("llm_hint");
  llmPreset.addEventListener("change", () => {
    const custom = llmPreset.value === "custom";
    llmCustom.classList.toggle("hidden", !custom);
    llmHint.classList.toggle("hidden", !custom);
  });
  function resolveLLMModel() {
    return llmCustom.value.trim() || llmPreset.value;
  }

  // ── start (real run) ──
  startBtn.addEventListener("click", start);

  // ── mock preview (no upload, no API key) ──
  $("mockBtn").addEventListener("click", () => {
    formErr.textContent = "";
    // Reuse the same progress + result UI as a real run; just hit the mock
    // endpoint. The server burns a sample clip with the REAL subtitle code.
    const fd = new FormData();
    fd.append("subtitle_position", $("subtitle_position").value);
    fd.append("subtitle_mode", mode);  // original / translate
    const fsz = $("subtitle_font_size_pct").value.trim();
    if (fsz) fd.append("subtitle_font_size_pct", fsz);

    startBtn.disabled = true;
    $("mockBtn").disabled = true;
    setupCard.style.opacity = ".6";
    progCard.classList.remove("hidden");
    resultCard.classList.add("hidden");
    buildStepper(["upload", "transcribe", "subtitles", "render", "done"]);
    bar.style.width = "0%";
    logEl.innerHTML = "";
    stepDetail.textContent = "Rendering mock subtitle preview…";

    fetch("/api/single/mock", { method: "POST", body: fd })
      .then((r) => r.json())
      .then((data) => {
        if (data.error) throw new Error(data.error);
        poll(data.job_id);
      })
      .catch((err) => {
        formErr.textContent = err.message || "Mock preview failed.";
        startBtn.disabled = false;
        $("mockBtn").disabled = false;
        setupCard.style.opacity = "1";
        progCard.classList.add("hidden");
      });
  });

  function start() {
    formErr.textContent = "";
    let localPath = "";
    if (useLocal) {
      localPath = $("local_path").value.trim();
      if (!localPath) { formErr.textContent = "Enter a local file path first."; return; }
    } else {
      if (!selectedFile) { formErr.textContent = "Choose a video first."; return; }
    }
    if (mode === "translate" && !$("api_key").value.trim()
        && !hasEnvKey()) {
      // still allow: server may read .env
      formErr.textContent = "Translating needs an API key — paste one, or set OPENROUTER_API_KEY in .env.";
      // do not hard-block; server can fall back to .env
    }
    formErr.textContent = "";

    const fd = new FormData();
    if (useLocal) {
      fd.append("local_path", localPath);
    } else {
      fd.append("video", selectedFile, selectedFile.name);
    }
    fd.append("subtitle_mode", mode);
    fd.append("subtitle_position", $("subtitle_position").value);
    const fsz2 = $("subtitle_font_size_pct").value.trim();
    if (fsz2) fd.append("subtitle_font_size_pct", fsz2);
    fd.append("target_language", $("target_language").value);
    fd.append("model", $("model").value);
    fd.append("lang", $("lang").value);
    fd.append("encoding_crf", $("encoding_crf").value);
    fd.append("remove_silence", $("remove_silence").checked ? "on" : "off");
    if (mode === "translate") fd.append("api_key", $("api_key").value.trim());
    fd.append("llm_model", resolveLLMModel());

    startBtn.disabled = true;
    setupCard.style.opacity = ".6";
    progCard.classList.remove("hidden");
    resultCard.classList.add("hidden");
    buildStepper(["upload", "transcribe", "subtitles", "render", "done"]);
    bar.style.width = "0%";
    logEl.innerHTML = "";
    stepDetail.textContent = useLocal ? "Reading local file…" : "Uploading…";

    fetch("/api/single/process", { method: "POST", body: fd })
      .then((r) => r.json())
      .then((data) => {
        if (data.error) { throw new Error(data.error); }
        poll(data.job_id);
      })
      .catch((err) => {
        formErr.textContent = err.message || "Upload failed.";
        startBtn.disabled = false;
        setupCard.style.opacity = "1";
        progCard.classList.add("hidden");
      });
  }

  function hasEnvKey() {
    // We can't read .env from the browser; assume false and let the server
    // decide. The note above explains this to the user.
    return false;
  }

  // ── stepper ──
  function buildStepper(steps) {
    stepper.innerHTML = "";
    steps.forEach((s) => {
      const d = document.createElement("div");
      d.className = "s"; d.dataset.s = s;
      stepper.appendChild(d);
    });
  }
  function paintStepper(status) {
    [...stepper.children].forEach((el) => {
      const s = el.dataset.s;
      el.className = "s " + (status[s] || "pending");
    });
  }

  // ── polling ──
  function poll(jobId) {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(() => {
      fetch(`/api/single/job/${jobId}`)
        .then((r) => r.json())
        .then((job) => {
          bar.style.width = (job.progress || 0) + "%";
          paintStepper(job.status || {});
          const cur = job.steps ? job.steps[job.step_index] : job.step;
          stepDetail.textContent = (job.detail && job.detail[cur])
            ? job.detail[cur]
            : (cur ? `Step: ${cur}` : "Working…");
          // logs
          if (job.logs) {
            const want = job.logs.slice(-200);
            logEl.innerHTML = want.map((l) => {
              const cls = (l.level === "ERROR") ? "ERR"
                : (l.level === "OK") ? "OK"
                : (l.level === "WARN") ? "WARN" : "";
              return `<span class="${cls}">[${l.t}s] ${escapeHtml(l.msg)}</span>`;
            }).join("\n");
            logEl.scrollTop = logEl.scrollHeight;
          }
          if (job.finished) {
            clearInterval(pollTimer);
            if (job.error) {
              formErr.textContent = "Failed: " + job.error;
              startBtn.disabled = false;
              setupCard.style.opacity = "1";
            } else {
              showResult(job);
            }
          }
        })
        .catch(() => {/* transient; keep polling */});
    }, 1200);
  }

  function showResult(job) {
    const out = job.outputs && job.outputs[0];
    if (!out) {
      formErr.textContent = "Finished but no output file was produced.";
      startBtn.disabled = false;
      setupCard.style.opacity = "1";
      return;
    }
    const url = `/files/single/${job.id}/${encodeURIComponent(out.filename)}`;
    $("player").src = url;
    $("dl").href = url;
    $("dl").setAttribute("download", out.filename);
    const langLabel = out.translated ? "translated" : "original language";
    $("resultMeta").innerHTML =
      `File: <b>${out.filename}</b> · ${out.size_mb} MB · ` +
      `detected language <b>${out.language}</b> · subtitles: <b>${langLabel}</b>`;
    resultCard.classList.remove("hidden");
    // reset for another run
    startBtn.disabled = false;
    setupCard.style.opacity = "1";
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }
})();
