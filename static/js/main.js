// ============================================================
// main.js — MediGAN Web Application JavaScript
// ============================================================

// ── Topbar clock ─────────────────────────────────────────────
function updateClock() {
  const el = document.getElementById('topbarTime');
  if (!el) return;
  const now = new Date();
  el.textContent = now.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}
setInterval(updateClock, 1000);
updateClock();

// ── Animated counters ─────────────────────────────────────────
function animateCounter(el) {
  const target = parseInt(el.getAttribute('data-target') || el.textContent, 10);
  if (isNaN(target)) return;
  const duration = 900;
  const step = target / (duration / 16);
  let current = 0;
  const timer = setInterval(() => {
    current = Math.min(current + step, target);
    el.textContent = Math.floor(current);
    if (current >= target) clearInterval(timer);
  }, 16);
}
document.querySelectorAll('[data-counter]').forEach(el => animateCounter(el));

// ── Drag-and-drop upload zone ─────────────────────────────────
const dropZone = document.querySelector('.drop-zone');
if (dropZone) {
  ['dragenter','dragover'].forEach(evt => {
    dropZone.addEventListener(evt, e => {
      e.preventDefault();
      dropZone.classList.add('drag-over');
    });
  });
  ['dragleave','drop'].forEach(evt => {
    dropZone.addEventListener(evt, e => {
      e.preventDefault();
      dropZone.classList.remove('drag-over');
    });
  });
  dropZone.addEventListener('drop', e => {
    const files = e.dataTransfer.files;
    const input = dropZone.querySelector('input[type="file"]');
    if (input && files.length) {
      // Manually set files and submit
      const dt = new DataTransfer();
      Array.from(files).forEach(f => dt.items.add(f));
      input.files = dt.files;
      input.closest('form').submit();
    }
  });
}

// ── Lightbox ──────────────────────────────────────────────────
function openLightbox(src, label) {
  const lb = document.getElementById('lightbox');
  if (!lb) return;
  lb.querySelector('img').src = src;
  const cap = lb.querySelector('.lightbox-caption');
  if (cap) cap.textContent = label || '';
  lb.classList.add('open');
  document.body.style.overflow = 'hidden';
}
function closeLightbox() {
  const lb = document.getElementById('lightbox');
  if (lb) lb.classList.remove('open');
  document.body.style.overflow = '';
}
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLightbox(); });

// ── Training page ─────────────────────────────────────────────
let trainingChart = null;
let pollInterval = null;

function initTrainingChart() {
  const canvas = document.getElementById('lossChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  trainingChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Generator Loss',
          data: [],
          borderColor: '#42A5F5',
          backgroundColor: 'rgba(66,165,245,0.07)',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.4,
          fill: true,
        },
        {
          label: 'Discriminator Loss',
          data: [],
          borderColor: '#00BCD4',
          backgroundColor: 'rgba(0,188,212,0.07)',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.4,
          fill: true,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 0 },
      plugins: {
        legend: {
          labels: { color: '#90A4AE', font: { family: 'DM Sans' }, boxWidth: 12, padding: 20 }
        },
        tooltip: {
          backgroundColor: '#111D35',
          titleColor: '#E8F0FE',
          bodyColor: '#90A4AE',
          borderColor: '#1A2E50',
          borderWidth: 1,
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(26,46,80,0.6)' },
          ticks: { color: '#546E7A', font: { family: 'DM Mono', size: 10 }, maxTicksLimit: 10 },
        },
        y: {
          grid: { color: 'rgba(26,46,80,0.6)' },
          ticks: { color: '#546E7A', font: { family: 'DM Mono', size: 10 } },
        },
      },
    },
  });
}

function updateTrainingUI(data) {
  // Progress bar
  const bar  = document.getElementById('progressFill');
  const pct  = document.getElementById('progressPct');
  const prog = data.total_epochs > 0 ? (data.epoch / data.total_epochs) * 100 : 0;
  if (bar) bar.style.width = prog.toFixed(1) + '%';
  if (pct) pct.textContent = prog.toFixed(0) + '%';

  // Status message
  const msg = document.getElementById('statusMsg');
  if (msg) msg.textContent = data.message || '—';

  // Epoch counter
  const ep = document.getElementById('epochCounter');
  if (ep) ep.textContent = `${data.epoch} / ${data.total_epochs}`;

  // Chart update
  if (trainingChart && data.g_loss && data.g_loss.length) {
    const labels = data.g_loss.map((_, i) => i + 1);
    trainingChart.data.labels = labels;
    trainingChart.data.datasets[0].data = data.g_loss;
    trainingChart.data.datasets[1].data = data.d_loss;
    trainingChart.update('none');
  }

  // Button states
  const startBtn = document.getElementById('startBtn');
  const stopBtn  = document.getElementById('stopBtn');
  if (startBtn) startBtn.disabled = data.running;
  if (stopBtn)  stopBtn.disabled  = !data.running;

  // Status tag
  const statusTag = document.getElementById('statusTag');
  if (statusTag) {
    statusTag.textContent = data.status.toUpperCase();
    statusTag.className = 'badge ' + (
      data.status === 'training' ? 'badge-cyan' :
      data.status === 'done'     ? 'badge-green' :
      data.status === 'error'    ? 'badge-red' : 'badge-blue'
    );
  }
}

function startPolling() {
  if (pollInterval) return;
  pollInterval = setInterval(async () => {
    try {
      const res = await fetch('/api/train/status');
      const data = await res.json();
      updateTrainingUI(data);
      if (!data.running && data.status !== 'training') {
        clearInterval(pollInterval);
        pollInterval = null;
      }
    } catch (_) {}
  }, 1500);
}

async function startTraining() {
  const form = document.getElementById('trainForm');
  const params = {
    epochs:      parseInt(document.getElementById('epochs')?.value || 100),
    batch_size:  parseInt(document.getElementById('batch_size')?.value || 32),
    latent_dim:  parseInt(document.getElementById('latent_dim')?.value || 100),
  };
  try {
    const res = await fetch('/api/train/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    const data = await res.json();
    if (data.error) { alert(data.error); return; }
    startPolling();
  } catch (e) { alert('Failed to start training: ' + e); }
}

async function stopTraining() {
  await fetch('/api/train/stop', { method: 'POST' });
  if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }
  setTimeout(() => fetch('/api/train/status').then(r => r.json()).then(updateTrainingUI), 300);
}

// Auto-init chart on train page
document.addEventListener('DOMContentLoaded', () => {
  if (document.getElementById('lossChart')) {
    initTrainingChart();
    // Load current status
    fetch('/api/train/status').then(r => r.json()).then(data => {
      updateTrainingUI(data);
      if (data.running) startPolling();
    });
  }

  // Range inputs live display
  document.querySelectorAll('input[type="range"]').forEach(input => {
    const display = document.getElementById(input.id + '_val');
    if (display) {
      display.textContent = input.value;
      input.addEventListener('input', () => { display.textContent = input.value; });
    }
  });
});

// ── Evaluate button ───────────────────────────────────────────
async function runEvaluation() {
  const btn = document.getElementById('evalBtn');
  if (btn) { btn.disabled = true; btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Evaluating…'; }
  try {
    const res  = await fetch('/api/evaluate', { method: 'POST' });
    const data = await res.json();
    if (data.error) { alert('Evaluation error: ' + data.error); return; }
    // Reload the page to show fresh results
    window.location.reload();
  } catch (e) { alert('Error: ' + e); }
  finally { if (btn) { btn.disabled = false; btn.innerHTML = '<i class="fa-solid fa-play"></i> Run Evaluation'; } }
}
