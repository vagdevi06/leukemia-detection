window.onload = function() {
  let selectedFile = null;
  let lastResult = null;
  let pieChartInstance = null;
  let barChartInstance = null;

  const fileInput       = document.getElementById('fileInput');
  const dropZone        = document.getElementById('dropZone');
  const previewZone     = document.getElementById('previewZone');
  const previewImg      = document.getElementById('previewImg');
  const previewName     = document.getElementById('previewName');
  const clearBtn        = document.getElementById('clearBtn');
  const analyzeBtn      = document.getElementById('analyzeBtn');
  const analyzeBtnText  = document.getElementById('analyzeBtnText');
  const analyzeSpinner  = document.getElementById('analyzeSpinner');
  const resultsPanel    = document.getElementById('resultsPanel');

  // Browse button
  document.getElementById('browseBtn').addEventListener('click', function(e) {
    e.stopPropagation();
    fileInput.click();
  });

  // File selected
  fileInput.addEventListener('change', function() {
    if (fileInput.files && fileInput.files[0]) {
      setFile(fileInput.files[0]);
      fileInput.value = '';
    }
  });

  // Drag and drop
  dropZone.addEventListener('dragover', function(e) {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });
  dropZone.addEventListener('dragleave', function() {
    dropZone.classList.remove('drag-over');
  });
  dropZone.addEventListener('drop', function(e) {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
  });

  function setFile(file) {
    const allowed = ['.jpg','.jpeg','.png','.tif','.tiff'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!allowed.includes(ext)) {
      alert('Please upload a JPG, PNG, or TIF image.');
      return;
    }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = function(e) {
      previewImg.src = e.target.result;
      previewName.textContent = file.name;
      dropZone.classList.add('hidden');
      previewZone.classList.remove('hidden');
      resultsPanel.classList.add('hidden');
    };
    reader.readAsDataURL(file);
  }

  clearBtn.addEventListener('click', resetUpload);
  document.getElementById('newAnalysisBtn').addEventListener('click', resetUpload);

  function resetUpload() {
    selectedFile = null;
    lastResult = null;
    fileInput.value = '';
    previewImg.src = '';
    previewZone.classList.add('hidden');
    dropZone.classList.remove('hidden');
    resultsPanel.classList.add('hidden');
    setBusy(false);
    if (pieChartInstance) { pieChartInstance.destroy(); pieChartInstance = null; }
    if (barChartInstance) { barChartInstance.destroy(); barChartInstance = null; }
  }

  analyzeBtn.addEventListener('click', runAnalysis);

  async function runAnalysis() {
    if (!selectedFile) return;
    setBusy(true);
    const form = new FormData();
    form.append('file', selectedFile);
    try {
      const res = await fetch('/api/predict', { method: 'POST', body: form });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || 'Server error');
      }
      const data = await res.json();
      lastResult = data;
      lastResult.imageName = selectedFile.name;
      renderResults(data);
    } catch (err) {
      alert('Analysis failed: ' + err.message);
    } finally {
      setBusy(false);
    }
  }

  function renderResults(d) {
    const isPositive = d.slide_diagnosis.toLowerCase().includes('detected') &&
                       !d.slide_diagnosis.toLowerCase().includes('no ');

    // Verdict banner
    const banner = document.getElementById('verdictBanner');
    banner.className = 'verdict-banner ' + (isPositive ? 'positive' : 'negative');
    document.getElementById('verdictIcon').textContent  = isPositive ? '⚠' : '✓';
    document.getElementById('verdictLabel').textContent = d.slide_diagnosis;
    document.getElementById('verdictSub').textContent   = isPositive
      ? `${pct(d.leukemic_cell_ratio)} of detected cells classified as abnormal.`
      : `No leukemic blasts detected above threshold.`;
    document.getElementById('verdictConfVal').textContent = pct(d.slide_confidence);

    // Quality
    renderQuality(d.quality);

    // Gauge
    drawGauge(d.slide_confidence, isPositive);

    // Stats
    document.getElementById('statTotal').textContent = d.total_cells_detected;
    document.getElementById('statLeuk').textContent  = d.leukemic_cells;
    document.getElementById('statNorm').textContent  = d.normal_cells;
    document.getElementById('statTime').textContent  = d.inference_time_ms.toFixed(0) + ' ms';

    // Charts
    renderCharts(d);

    // Ratio bar
    const ratio = d.leukemic_cell_ratio;
    document.getElementById('ratioValue').textContent = pct(ratio);
    document.getElementById('ratioBarFill').style.width = pct(ratio);

    // Annotated image
    document.getElementById('annotatedImg').src = d.annotated_image_b64;

    // Thumbnails
    renderThumbnails(d);

    // Cell table
    document.getElementById('cellCount').textContent = d.cell_detail.length + ' cells';
    const tbody = document.getElementById('cellTableBody');
    tbody.innerHTML = '';
    d.cell_detail.forEach((cell, i) => {
      const isLeuk = !cell.class_name.toLowerCase().includes('normal');
      const tr = document.createElement('tr');
      tr.className = isLeuk ? 'row-leuk' : 'row-norm';
      const bb = cell.bbox;
      const blastProb = (cell.probabilities[1] ?? 0) * 100;
      tr.innerHTML = `
        <td style="font-family:monospace">${i + 1}</td>
        <td>${cell.class_name}</td>
        <td>${pct(cell.confidence)}</td>
        <td style="font-family:monospace;font-size:.78rem">[${bb.join(', ')}]</td>
        <td>${blastProb.toFixed(1)}%</td>
      `;
      tbody.appendChild(tr);
    });

    resultsPanel.classList.remove('hidden');
    resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // ── Charts ────────────────────────────────────────────────────
  function renderCharts(d) {
    if (pieChartInstance) { pieChartInstance.destroy(); pieChartInstance = null; }
    if (barChartInstance) { barChartInstance.destroy(); barChartInstance = null; }

    // Pie chart
    const pieCtx = document.getElementById('pieChart').getContext('2d');
    pieChartInstance = new Chart(pieCtx, {
      type: 'doughnut',
      data: {
        labels: ['Normal', 'ALL Blast', 'AML Blast', 'CML Blast', 'Suspicious'],
        datasets: [{
          data: [
            d.cell_detail.filter(c => c.class_name === 'Normal Lymphocyte').length,
            d.cell_detail.filter(c => c.class_name === 'ALL Blast').length,
            d.cell_detail.filter(c => c.class_name === 'AML Blast').length,
            d.cell_detail.filter(c => c.class_name === 'CML Blast').length,
            d.cell_detail.filter(c => c.class_name === 'Suspicious Cell').length,
          ],
          backgroundColor: ['#3fb950', '#f85149', '#ff9900', '#d29922', '#a371f7'],
          borderColor: ['#161b22'],
          borderWidth: 3,
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              color: '#7d8590',
              font: { size: 11 },
              padding: 12,
            }
          }
        },
        cutout: '65%',
      }
    });

    // Bar chart
    const barCtx = document.getElementById('barChart').getContext('2d');
    const labels = d.cell_detail.map((_, i) => `Cell ${i + 1}`);
    const confidences = d.cell_detail.map(c => (c.confidence * 100).toFixed(1));
    const colors = d.cell_detail.map(c => {
      const name = c.class_name.toLowerCase();
      if (name.includes('all')) return '#f85149';
      if (name.includes('aml')) return '#ff9900';
      if (name.includes('cml')) return '#d29922';
      if (name.includes('suspicious')) return '#a371f7';
      return '#3fb950';
    });

    barChartInstance = new Chart(barCtx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'CNN Confidence (%)',
          data: confidences,
          backgroundColor: colors,
          borderRadius: 4,
          borderSkipped: false,
        }]
      },
      options: {
        responsive: true,
        scales: {
          y: {
            beginAtZero: true,
            max: 100,
            ticks: { color: '#7d8590', font: { size: 11 } },
            grid: { color: '#30363d' },
          },
          x: {
            ticks: { color: '#7d8590', font: { size: 11 } },
            grid: { color: '#30363d' },
          }
        },
        plugins: {
          legend: {
            labels: {
              color: '#7d8590',
              font: { size: 11 },
            }
          }
        }
      }
    });
  }

  // ── Quality ───────────────────────────────────────────────────
  function renderQuality(quality) {
    if (!quality) return;
    const warningsDiv = document.getElementById('qualityWarnings');
    if (quality.warnings && quality.warnings.length > 0) {
      warningsDiv.innerHTML = quality.warnings.map(w => `
        <div class="quality-warning">⚠ ${w}</div>
      `).join('');
      warningsDiv.classList.remove('hidden');
    } else {
      warningsDiv.innerHTML = '<div class="quality-ok">✓ Image quality is good</div>';
      warningsDiv.classList.remove('hidden');
    }
    const section = document.getElementById('qualitySection');
    section.classList.remove('hidden');
    document.getElementById('qualityScore').textContent = quality.overall_score + ' / 100';
    const blurPct = Math.min((quality.blur_score / 500) * 100, 100).toFixed(1);
    document.getElementById('blurBar').style.width = blurPct + '%';
    const brightPct = (quality.brightness_score / 255 * 100).toFixed(1);
    document.getElementById('brightnessBar').style.width = brightPct + '%';
    const contrastPct = Math.min((quality.contrast_score / 80) * 100, 100).toFixed(1);
    document.getElementById('contrastBar').style.width = contrastPct + '%';
  }

  // ── Confidence Gauge ──────────────────────────────────────────
  function drawGauge(confidence, isPositive) {
    const canvas = document.getElementById('gaugeCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const cx = canvas.width / 2;
    const cy = canvas.height * 0.85;
    const r  = canvas.width * 0.38;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    ctx.arc(cx, cy, r, Math.PI, 2 * Math.PI);
    ctx.lineWidth = 18;
    ctx.strokeStyle = '#1f2937';
    ctx.stroke();
    const angle = Math.PI + (confidence * Math.PI);
    const color = isPositive ? '#f85149' : '#3fb950';
    ctx.beginPath();
    ctx.arc(cx, cy, r, Math.PI, angle);
    ctx.lineWidth = 18;
    ctx.strokeStyle = color;
    ctx.lineCap = 'round';
    ctx.stroke();
    ctx.fillStyle = '#e6edf3';
    ctx.font = `bold ${canvas.width * 0.18}px IBM Plex Mono, monospace`;
    ctx.textAlign = 'center';
    ctx.fillText((confidence * 100).toFixed(1) + '%', cx, cy - 10);
    ctx.fillStyle = '#7d8590';
    ctx.font = `${canvas.width * 0.09}px IBM Plex Sans, sans-serif`;
    ctx.fillText('confidence', cx, cy + 18);
  }

  // ── Cell Thumbnails with Grad-CAM ─────────────────────────────
  function renderThumbnails(d) {
    const container = document.getElementById('thumbnailGrid');
    if (!container) return;
    container.innerHTML = '';

    const img = new Image();
    img.onload = function() {
      d.cell_detail.forEach((cell, i) => {
        const isLeuk = !cell.class_name.toLowerCase().includes('normal');
        const [x1, y1, x2, y2] = cell.bbox;

        const canvas = document.createElement('canvas');
        canvas.width  = 80;
        canvas.height = 80;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, x1, y1, x2 - x1, y2 - y1, 0, 0, 80, 80);

        const wrapper = document.createElement('div');
        wrapper.className = 'thumb-wrap ' + (isLeuk ? 'thumb-leuk' : 'thumb-norm');

        const cellIcon = getCellIcon(cell.class_name);

        if (isLeuk && cell.gradcam_b64) {
          let showingGradcam = false;
          wrapper.innerHTML = `
            <div class="thumb-img-wrap"></div>
            <div class="thumb-label">${cellIcon} ${cell.class_name}</div>
            <div class="thumb-conf">${pct(cell.confidence)}</div>
            <button class="gradcam-toggle">Grad-CAM</button>
          `;
          const imgWrap = wrapper.querySelector('.thumb-img-wrap');
          imgWrap.appendChild(canvas);
          const gcImg = document.createElement('img');
          gcImg.src = cell.gradcam_b64;
          gcImg.style.cssText = 'width:80px;height:80px;border-radius:4px;display:none;';
          imgWrap.appendChild(gcImg);
          wrapper.querySelector('.gradcam-toggle').addEventListener('click', function() {
            showingGradcam = !showingGradcam;
            canvas.style.display = showingGradcam ? 'none' : 'block';
            gcImg.style.display  = showingGradcam ? 'block' : 'none';
            this.textContent = showingGradcam ? 'Original' : 'Grad-CAM';
          });
        } else {
          wrapper.innerHTML = `
            <div class="thumb-img-wrap"></div>
            <div class="thumb-label">${cellIcon} ${cell.class_name}</div>
            <div class="thumb-conf">${pct(cell.confidence)}</div>
          `;
          wrapper.querySelector('.thumb-img-wrap').appendChild(canvas);
        }
        container.appendChild(wrapper);
      });
    };
    img.src = d.annotated_image_b64;
  }

  function getCellIcon(className) {
    const name = className.toLowerCase();
    if (name.includes('normal'))     return '🟢';
    if (name.includes('all'))        return '🔴';
    if (name.includes('aml'))        return '🟠';
    if (name.includes('cml'))        return '🟡';
    if (name.includes('suspicious')) return '🟣';
    return '⚪';
  }

  // ── PDF Report ────────────────────────────────────────────────
  document.getElementById('downloadBtn').addEventListener('click', function() {
    if (!lastResult) return;
    fetch('/api/report', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(lastResult)
    })
    .then(res => res.blob())
    .then(blob => {
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'leukascan_report.pdf';
      a.click();
      URL.revokeObjectURL(url);
    })
    .catch(() => alert('PDF generation failed.'));
  });

  function setBusy(busy) {
    analyzeBtn.disabled = busy;
    analyzeBtnText.textContent = busy ? 'Analysing…' : 'Analyse Image';
    analyzeSpinner.classList.toggle('hidden', !busy);
  }

  function pct(v) {
    return (v * 100).toFixed(1) + '%';
  }
};