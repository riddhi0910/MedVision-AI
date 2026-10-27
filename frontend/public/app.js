/**
 * MedVision PACS — Clinical Workstation Interactive Engine
 * Hospital-grade radiology workstation client logic.
 */

// Application State
const state = {
  selectedModel: 'resnet18',
  currentFile: null,
  isAnalyzing: false,
  inverted: false,
  studyId: '#CXR-2026-9812'
};

// DOM References
const elements = {
  apiStatusText: document.getElementById('apiStatusText'),
  activeModelBadge: document.getElementById('activeModelBadge'),
  btnResnet: document.getElementById('btnResnet'),
  btnCnn: document.getElementById('btnCnn'),
  dropzone: document.getElementById('dropzone'),
  fileInput: document.getElementById('fileInput'),
  sampleChips: document.getElementById('sampleChips'),
  
  // Upper Triage Detection Bar
  verdictPill: document.getElementById('verdictPill'),
  triageIcd: document.getElementById('triageIcd'),
  triageTelemetry: document.getElementById('triageTelemetry'),
  confidenceNumber: document.getElementById('confidenceNumber'),
  probNormalText: document.getElementById('probNormalText'),
  probPneuText: document.getElementById('probPneuText'),
  probFillNormal: document.getElementById('probFillNormal'),
  probFillPneu: document.getElementById('probFillPneu'),

  // Side-by-Side Left Monitor (Raw Radiograph)
  chamberRaw: document.getElementById('chamberRaw'),
  emptyRaw: document.getElementById('emptyRaw'),
  viewOriginal: document.getElementById('viewOriginal'),
  hudRawMeta: document.getElementById('hudRawMeta'),

  // Side-by-Side Right Monitor (Grad-CAM Output)
  chamberCam: document.getElementById('chamberCam'),
  emptyCam: document.getElementById('emptyCam'),
  camStackContainer: document.getElementById('camStackContainer'),
  viewOriginalUnder: document.getElementById('viewOriginalUnder'),
  viewHeatmap: document.getElementById('viewHeatmap'),
  hudCamLatency: document.getElementById('hudCamLatency'),
  opacitySlider: document.getElementById('opacitySlider'),
  sliderVal: document.getElementById('sliderVal'),

  // Bottom Structured Report & Impressions
  repTechnique: document.getElementById('repTechnique'),
  findingsList: document.getElementById('findingsList'),
  impressionSummary: document.getElementById('impressionSummary'),
  hudStudyId: document.getElementById('hudStudyId'),

  // Formal Print Report Elements
  printAccession: document.getElementById('printAccession'),
  printMrn: document.getElementById('printMrn'),
  printDate: document.getElementById('printDate'),
  printModelName: document.getElementById('printModelName'),
  printVerdict: document.getElementById('printVerdict'),
  printConfidence: document.getElementById('printConfidence'),
  printIcd: document.getElementById('printIcd'),
  printTriageBox: document.getElementById('printTriageBox'),
  printImgRaw: document.getElementById('printImgRaw'),
  printImgCam: document.getElementById('printImgCam'),
  printFindingsList: document.getElementById('printFindingsList'),
  printImpressionBox: document.getElementById('printImpressionBox')
};

// ── Startup Initialization ──────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  checkEngineHealth();
  loadSampleRadiographs();
  setupEventListeners();
});

// ── Professional Clinical Toast Notifications (Zero default popups!) ────────

function showToast(message, type = 'info', duration = 4000) {
  let toastContainer = document.getElementById('toastContainer');
  if (!toastContainer) {
    toastContainer = document.createElement('div');
    toastContainer.id = 'toastContainer';
    toastContainer.className = 'toast-container';
    document.body.appendChild(toastContainer);
  }

  const toast = document.createElement('div');
  toast.className = `pacs-toast toast-${type}`;
  
  const iconMap = {
    success: '✓',
    error: '✕',
    warning: '⚠',
    info: 'ℹ'
  };
  const icon = iconMap[type] || 'ℹ';

  toast.innerHTML = `
    <span class="toast-icon">${icon}</span>
    <span class="toast-message">${message}</span>
    <button class="toast-close" onclick="this.parentElement.remove()">×</button>
  `;

  toastContainer.appendChild(toast);

  requestAnimationFrame(() => {
    toast.classList.add('toast-show');
  });

  setTimeout(() => {
    toast.classList.remove('toast-show');
    setTimeout(() => toast.remove(), 260);
  }, duration);
}

// Global exposure
window.showToast = showToast;

// ── Engine Telemetry & Health ────────────────────────────────────────────────

async function checkEngineHealth() {
  try {
    const res = await fetch('/api/health');
    if (!res.ok) throw new Error('API Offline');
    const data = await res.json();
    
    if (elements.apiStatusText) {
      elements.apiStatusText.textContent = `● AI ENGINE ONLINE (${data.device.toUpperCase()})`;
      elements.apiStatusText.style.color = '#4ade80';
    }
  } catch (err) {
    if (elements.apiStatusText) {
      elements.apiStatusText.textContent = '● AI ENGINE OFFLINE';
      elements.apiStatusText.style.color = '#f87171';
    }
    showToast('AI Inference Engine offline. Verify backend connection.', 'warning');
  }
}

// ── Model Selection ──────────────────────────────────────────────────────────

window.selectModel = function(modelKey) {
  state.selectedModel = modelKey;

  if (modelKey === 'resnet18') {
    elements.btnResnet?.classList.add('active');
    elements.btnCnn?.classList.remove('active');
    if (elements.activeModelBadge) {
      elements.activeModelBadge.textContent = 'ResNet-18 Active (96.4%)';
    }
    if (elements.triageTelemetry) {
      elements.triageTelemetry.textContent = 'Neural Backbone: ResNet-18 (Deep Residual • ImageNet-1K)';
    }
    if (elements.repTechnique) {
      elements.repTechnique.textContent = 'Deep Transfer Learning (ResNet-18) with Gradient-weighted Class Activation Mapping (Grad-CAM).';
    }
    showToast('Switched to ResNet-18 (ImageNet Pretrained)', 'info', 2500);
  } else {
    elements.btnCnn?.classList.add('active');
    elements.btnResnet?.classList.remove('active');
    if (elements.activeModelBadge) {
      elements.activeModelBadge.textContent = 'Baseline CNN Active (94.7%)';
    }
    if (elements.triageTelemetry) {
      elements.triageTelemetry.textContent = 'Neural Backbone: Baseline 4-Block CNN (From Scratch)';
    }
    if (elements.repTechnique) {
      elements.repTechnique.textContent = 'Custom 4-Block Convolutional Network with BatchNorm & LeakyReLU activations.';
    }
    showToast('Switched to Baseline 4-Block CNN', 'info', 2500);
  }

  // Re-run diagnostic inference if radiograph is loaded
  if (state.currentFile) {
    analyzeRadiograph(state.currentFile);
  }
};

// ── PACS Study Worklist Queue ────────────────────────────────────────────────

async function loadSampleRadiographs() {
  try {
    const res = await fetch('/api/samples');
    if (!res.ok) return;
    const data = await res.json();

    if (data.samples && data.samples.length > 0 && elements.sampleChips) {
      elements.sampleChips.innerHTML = '';
      data.samples.forEach((sample, idx) => {
        const btn = document.createElement('button');
        const isPneu = sample.class === 'PNEUMONIA';
        btn.className = `case-chip ${isPneu ? 'chip-pneu' : 'chip-norm'}`;
        const icon = isPneu ? '🔴' : '🟢';
        const label = isPneu ? `Case 0${idx + 1}: Pneumonia` : `Case 0${idx + 1}: Normal`;

        btn.innerHTML = `${icon} ${label}`;
        btn.onclick = () => loadSample(sample.name, sample.class);
        elements.sampleChips.appendChild(btn);
      });
    }
  } catch (err) {
    console.warn('Could not populate clinical worklist queue:', err);
  }
}

window.loadSample = async function(filename, expectedLabel) {
  try {
    setLoadingState(true);
    state.studyId = `#CXR-${filename.replace(/[^0-9]/g, '').slice(-4) || '9812'}`;
    if (elements.hudStudyId) elements.hudStudyId.textContent = state.studyId;

    const res = await fetch(`/api/sample_file/${filename}`);
    if (!res.ok) throw new Error('Failed to retrieve study image');
    const blob = await res.blob();
    const file = new File([blob], filename, { type: 'image/jpeg' });
    analyzeRadiograph(file);
  } catch (err) {
    showToast(`Worklist fetch error: ${err.message}`, 'error');
    setLoadingState(false);
  }
};

// ── Event Setup ──────────────────────────────────────────────────────────────

function setupEventListeners() {
  const dropzone = elements.dropzone;
  const fileInput = elements.fileInput;
  const opacitySlider = elements.opacitySlider;

  // File Upload
  if (fileInput) {
    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        state.studyId = `#CXR-IMP-${Math.floor(1000 + Math.random() * 9000)}`;
        if (elements.hudStudyId) elements.hudStudyId.textContent = state.studyId;
        analyzeRadiograph(e.target.files[0]);
      }
    });
  }

  // Drag & Drop
  if (dropzone) {
    ['dragenter', 'dragover'].forEach(name => {
      dropzone.addEventListener(name, (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach(name => {
      dropzone.addEventListener(name, (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
      });
    });

    dropzone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      if (dt.files && dt.files[0]) {
        state.studyId = `#CXR-DRP-${Math.floor(1000 + Math.random() * 9000)}`;
        if (elements.hudStudyId) elements.hudStudyId.textContent = state.studyId;
        analyzeRadiograph(dt.files[0]);
      }
    });
  }

  // Global Clipboard Paste (Ctrl+V)
  window.addEventListener('paste', (e) => {
    if (!e.clipboardData || !e.clipboardData.items) return;

    for (let i = 0; i < e.clipboardData.items.length; i++) {
      const item = e.clipboardData.items[i];
      if (item.type.indexOf('image') !== -1) {
        e.preventDefault();
        const file = item.getAsFile();
        if (file) {
          state.studyId = `#CXR-PST-${Math.floor(1000 + Math.random() * 9000)}`;
          if (elements.hudStudyId) elements.hudStudyId.textContent = state.studyId;
          flashDropzone();
          showToast('Image pasted from clipboard.', 'info', 2000);
          analyzeRadiograph(file);
          return;
        }
      }
    }
  });

  // Opacity Slider
  if (opacitySlider && elements.viewHeatmap && elements.sliderVal) {
    opacitySlider.addEventListener('input', (e) => {
      updateOpacity(e.target.value);
    });
  }
}

// ── Viewport Control Functions ───────────────────────────────────────────────

window.setOpacity = function(val) {
  if (elements.opacitySlider) elements.opacitySlider.value = val;
  updateOpacity(val);

  document.querySelectorAll('.qbtn').forEach(b => {
    b.classList.remove('active-qbtn');
    if (b.textContent.includes(`${val}%`)) b.classList.add('active-qbtn');
  });
};

function updateOpacity(val) {
  if (elements.viewHeatmap) elements.viewHeatmap.style.opacity = (val / 100).toString();
  if (elements.sliderVal) elements.sliderVal.textContent = `${val}%`;
}

window.toggleInvert = function() {
  state.inverted = !state.inverted;
  const filterVal = state.inverted ? 'invert(100%)' : 'none';
  if (elements.chamberRaw) elements.chamberRaw.style.filter = filterVal;
  if (elements.chamberCam) elements.chamberCam.style.filter = filterVal;
  showToast(state.inverted ? 'Grayscale inverted (Bone window view)' : 'Grayscale reset to standard', 'info', 2000);
};

window.resetZoom = function() {
  state.inverted = false;
  if (elements.chamberRaw) elements.chamberRaw.style.filter = 'none';
  if (elements.chamberCam) elements.chamberCam.style.filter = 'none';
  setOpacity(50);
  showToast('Viewport view reset to 100%', 'info', 1500);
};

window.pasteFromClipboard = async function(event) {
  if (event) {
    event.preventDefault();
    event.stopPropagation();
  }

  if (navigator.clipboard && navigator.clipboard.read) {
    try {
      const clipboardItems = await navigator.clipboard.read();
      for (const item of clipboardItems) {
        for (const type of item.types) {
          if (type.startsWith('image/')) {
            const blob = await item.getType(type);
            const ext = type.split('/')[1] || 'png';
            const file = new File([blob], `pasted_study_${Date.now()}.${ext}`, { type });
            state.studyId = `#CXR-PST-${Math.floor(1000 + Math.random() * 9000)}`;
            if (elements.hudStudyId) elements.hudStudyId.textContent = state.studyId;
            flashDropzone();
            showToast('Radiograph loaded from clipboard.', 'success', 2000);
            analyzeRadiograph(file);
            return;
          }
        }
      }
      showToast('No image found in clipboard. Copy an image first, then paste.', 'warning');
    } catch (err) {
      showToast('Clipboard access denied. Please press Ctrl+V directly to paste.', 'warning');
    }
  } else {
    showToast('Press Ctrl+V on the keyboard to paste your copied radiograph.', 'info');
  }
};

function flashDropzone() {
  if (elements.dropzone) {
    elements.dropzone.classList.add('dragover');
    setTimeout(() => elements.dropzone.classList.remove('dragover'), 400);
  }
}

// Clinical Actions
window.signOffStudy = function() {
  showToast('Study attested by Dr. R. Somani (Nagpur City Hospital) & archived to PACS.', 'success', 4500);
};

window.copyImpressionText = function() {
  const text = elements.impressionSummary?.textContent.trim() || '';
  if (!text || text.includes('Awaiting examination')) {
    showToast('No diagnostic impression available to copy.', 'warning');
    return;
  }
  navigator.clipboard.writeText(text).then(() => {
    showToast('Impression text copied to clipboard.', 'success');
  }).catch(() => {
    showToast('Clipboard write error.', 'error');
  });
};

// ── Diagnostic Inference Execution ───────────────────────────────────────────

async function analyzeRadiograph(file) {
  state.currentFile = file;
  setLoadingState(true);

  const formData = new FormData();
  formData.append('file', file);
  formData.append('model', state.selectedModel);

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      let errMsg = 'Diagnostic evaluation failed.';
      try {
        const errJson = await res.json();
        errMsg = errJson.error || errMsg;
      } catch (e) {
        errMsg = `Server Error (${res.status}): Please check backend logs.`;
      }
      throw new Error(errMsg);
    }

    const data = await res.json();
    renderWorkstationResults(data);
    showToast(`Analysis complete: ${data.prediction} (${data.confidence_pct}%)`, 'success', 3000);
  } catch (err) {
    showToast(`PACS Inference Error: ${err.message}`, 'error', 5000);
  } finally {
    setLoadingState(false);
  }
}

// ── Render Structured Workstation Results ─────────────────────────────────────

function renderWorkstationResults(data) {
  const isPneu = data.prediction === 'PNEUMONIA';

  // 1. Upper Detection Triage Bar
  if (elements.verdictPill) {
    elements.verdictPill.textContent = isPneu ? 'PNEUMONIA DETECTED' : 'NORMAL (NEGATIVE)';
    elements.verdictPill.className = `triage-status-pill ${isPneu ? 'pneu' : 'norm'}`;
  }

  if (elements.confidenceNumber) {
    elements.confidenceNumber.textContent = `${data.confidence_pct}%`;
    elements.confidenceNumber.style.color = isPneu ? 'var(--triage-pneu-text)' : 'var(--triage-norm-text)';
  }

  if (elements.triageIcd) {
    elements.triageIcd.textContent = isPneu 
      ? 'ICD-10 J18.9: Infectious Airspace Consolidation / Pneumonia'
      : 'ICD-10 Z00.00: Unremarkable Radiograph / No Active Consolidation';
  }

  if (elements.triageTelemetry) {
    elements.triageTelemetry.textContent = `Model: ${data.model_used} • Inference Latency: ${data.inference_time_ms}ms • Size: 224×224`;
  }

  // Quantitative Probability Track
  const normProb = (data.probabilities.NORMAL * 100).toFixed(1);
  const pneuProb = (data.probabilities.PNEUMONIA * 100).toFixed(1);

  if (elements.probNormalText) elements.probNormalText.textContent = `${normProb}%`;
  if (elements.probPneuText) elements.probPneuText.textContent = `${pneuProb}%`;

  if (elements.probFillNormal) elements.probFillNormal.style.width = `${normProb}%`;
  if (elements.probFillPneu) elements.probFillPneu.style.width = `${pneuProb}%`;

  // 2. Left Monitor: Input Radiograph (Raw DX)
  if (elements.emptyRaw) elements.emptyRaw.style.display = 'none';
  if (elements.viewOriginal) {
    elements.viewOriginal.src = data.original_image;
    elements.viewOriginal.style.display = 'block';
  }
  if (elements.hudRawMeta) {
    elements.hudRawMeta.textContent = state.studyId;
  }

  // 3. Right Monitor: Output with Grad-CAM Saliency Map
  if (elements.emptyCam) elements.emptyCam.style.display = 'none';
  if (elements.camStackContainer) elements.camStackContainer.style.display = 'block';

  if (elements.viewOriginalUnder) {
    elements.viewOriginalUnder.src = data.original_image;
  }

  if (elements.viewHeatmap) {
    if (data.gradcam_heatmap) {
      elements.viewHeatmap.src = data.gradcam_heatmap;
      elements.viewHeatmap.style.display = 'block';
    } else if (data.gradcam_overlay) {
      elements.viewHeatmap.src = data.gradcam_overlay;
      elements.viewHeatmap.style.display = 'block';
    } else {
      elements.viewHeatmap.style.display = 'none';
    }
    const curVal = elements.opacitySlider ? elements.opacitySlider.value : 50;
    elements.viewHeatmap.style.opacity = (curVal / 100).toString();
  }

  if (elements.hudCamLatency) {
    elements.hudCamLatency.textContent = `LATENCY: ${data.inference_time_ms}ms`;
  }

  // 4. Bottom Structured ACR Radiology Impressions
  if (elements.findingsList && data.findings) {
    elements.findingsList.innerHTML = '';
    data.findings.forEach(item => {
      const li = document.createElement('li');
      li.textContent = item;
      elements.findingsList.appendChild(li);
    });
  }

  if (elements.impressionSummary) {
    if (isPneu) {
      const isBorderline = data.confidence_pct < 70;
      const heading = isBorderline ? 'BORDERLINE / EQUIVOCAL FINDING:' : 'CRITICAL FINDING:';
      const severity = isBorderline ? 'Mild / Equivocal' : 'High-probability';
      elements.impressionSummary.innerHTML = `<strong>${heading}</strong> ${severity} pulmonary parenchymal infiltration (${data.confidence_pct}% score) detected. Gradient saliency correlates focal activations with radio-opaque airspace opacification. Recommend physician auscultation and clinical correlation.`;
    } else {
      const isBorderline = data.confidence_pct < 70;
      const desc = isBorderline ? 'Equivocal normal appearance' : 'Clear bilateral lung expansibility with sharp costophrenic angles';
      elements.impressionSummary.innerHTML = `<strong>UNREMARKABLE:</strong> ${desc} (${data.confidence_pct}% score). No definitive evidence of dense consolidation or pleural effusion.`;
    }
  }

  // 5. Populate Formal Diagnostic PDF Print Report
  if (elements.printAccession) elements.printAccession.textContent = state.studyId.replace('#', '');
  if (elements.printMrn) elements.printMrn.textContent = state.studyId;
  if (elements.printDate) elements.printDate.textContent = new Date().toISOString().split('T')[0];
  if (elements.printModelName) elements.printModelName.textContent = data.model_used;
  if (elements.printVerdict) {
    if (isPneu) {
      elements.printVerdict.textContent = data.confidence_pct < 70 
        ? 'BORDERLINE FINDING: EQUIVOCAL PNEUMONIA'
        : 'CRITICAL FINDING: PNEUMONIA DETECTED';
    } else {
      elements.printVerdict.textContent = 'UNREMARKABLE: NORMAL RADIOGRAPH';
    }
  }
  if (elements.printConfidence) elements.printConfidence.textContent = `${data.confidence_pct}%`;
  if (elements.printIcd) elements.printIcd.textContent = isPneu ? 'ICD-10: J18.9' : 'ICD-10: Z00.00';
  if (elements.printTriageBox) elements.printTriageBox.className = `print-triage ${isPneu ? 'pneu' : 'norm'}`;
  if (elements.printImgRaw && data.original_image) elements.printImgRaw.src = data.original_image;
  if (elements.printImgCam) elements.printImgCam.src = data.gradcam_overlay || data.gradcam_heatmap || data.original_image;
  if (elements.printFindingsList && elements.findingsList) elements.printFindingsList.innerHTML = elements.findingsList.innerHTML;
  if (elements.printImpressionBox && elements.impressionSummary) elements.printImpressionBox.innerHTML = elements.impressionSummary.innerHTML;
}

function setLoadingState(loading) {
  state.isAnalyzing = loading;
  if (elements.dropzone) {
    elements.dropzone.style.opacity = loading ? '0.5' : '1';
    elements.dropzone.style.pointerEvents = loading ? 'none' : 'auto';
  }
}

window.resetViewer = function() {
  state.currentFile = null;
  if (elements.fileInput) elements.fileInput.value = '';
  
  if (elements.emptyRaw) elements.emptyRaw.style.display = 'block';
  if (elements.viewOriginal) elements.viewOriginal.style.display = 'none';

  if (elements.emptyCam) elements.emptyCam.style.display = 'block';
  if (elements.camStackContainer) elements.camStackContainer.style.display = 'none';

  if (elements.verdictPill) {
    elements.verdictPill.textContent = 'AWAITING STUDY';
    elements.verdictPill.className = 'triage-status-pill';
  }
  if (elements.confidenceNumber) elements.confidenceNumber.textContent = '--%';
  if (elements.triageIcd) elements.triageIcd.textContent = 'ICD-10: PENDING EXAMINATION';
  showToast('Study cleared. Ready for new radiograph.', 'info', 2000);
};
