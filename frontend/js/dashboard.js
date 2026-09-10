/**
 * SMART AGRI AI - Dashboard Analytics & Chart.js Visualizations
 * Fetches authentic training statistics from /api/dashboard.
 * Strictly presents genuine computed metrics without fabricating values.
 */

document.addEventListener("DOMContentLoaded", () => {
  loadDashboardData();
});

let accuracyChart = null;
let featureChart = null;

async function loadDashboardData() {
  try {
    const response = await fetch("/api/dashboard");
    if (!response.ok) {
      throw new Error(`HTTP error ${response.status}`);
    }

    const data = await response.json();
    updateModelCards(data.models);
    renderAccuracyChart(data.models);
    renderFeatureImportanceChart(data.models.crop);
    renderRecentLogs(data.recent_advisories);

  } catch (err) {
    console.error("Failed to load dashboard metrics:", err);
    showToast("Failed to fetch dashboard data. Check Flask server.", "error");
  }
}

function updateModelCards(models) {
  // 1. Crop Model Card
  const crop = models.crop;
  const cropStatusBadge = document.getElementById("cropStatusBadge");
  cropStatusBadge.className = `status-pill ${crop.ready ? "status-ready" : "status-untrained"}`;
  cropStatusBadge.innerText = crop.status;

  if (crop.metrics) {
    document.getElementById("cropAcc").innerText = `${crop.metrics.accuracy}%`;
    document.getElementById("cropPrec").innerText = `${crop.metrics.precision}%`;
    document.getElementById("cropRec").innerText = `${crop.metrics.recall}%`;
    document.getElementById("cropF1").innerText = `${crop.metrics.f1_score}%`;
  } else {
    document.getElementById("cropAcc").innerText = "N/A";
    document.getElementById("cropPrec").innerText = "N/A";
    document.getElementById("cropRec").innerText = "N/A";
    document.getElementById("cropF1").innerText = "N/A";
  }

  // 2. Fertilizer Model Card
  const fert = models.fertilizer;
  const fertStatusBadge = document.getElementById("fertStatusBadge");
  fertStatusBadge.className = `status-pill ${fert.ready ? "status-ready" : "status-untrained"}`;
  fertStatusBadge.innerText = fert.status;

  if (fert.metrics) {
    document.getElementById("fertAcc").innerText = `${fert.metrics.accuracy}%`;
    document.getElementById("fertPrec").innerText = `${fert.metrics.precision}%`;
    document.getElementById("fertRec").innerText = `${fert.metrics.recall}%`;
    document.getElementById("fertF1").innerText = `${fert.metrics.f1_score}%`;
  } else {
    document.getElementById("fertAcc").innerText = "N/A";
    document.getElementById("fertPrec").innerText = "N/A";
    document.getElementById("fertRec").innerText = "N/A";
    document.getElementById("fertF1").innerText = "N/A";
  }

  // 3. Disease Model Card
  const disease = models.disease;
  const diseaseStatusBadge = document.getElementById("diseaseStatusBadge");
  diseaseStatusBadge.className = `status-pill ${disease.ready ? "status-ready" : "status-untrained"}`;
  diseaseStatusBadge.innerText = disease.status;

  if (disease.metrics) {
    document.getElementById("diseaseAcc").innerText = `${disease.metrics.accuracy}%`;
    document.getElementById("diseaseLoss").innerText = disease.metrics.loss || "N/A";
  } else {
    document.getElementById("diseaseAcc").innerText = "N/A";
    document.getElementById("diseaseLoss").innerText = "Not Trained";
  }
}

function renderAccuracyChart(models) {
  const ctx = document.getElementById("accuracyChartCanvas");
  if (!ctx) return;

  const cropAcc = models.crop.metrics ? models.crop.metrics.accuracy : 0;
  const fertAcc = models.fertilizer.metrics ? models.fertilizer.metrics.accuracy : 0;
  const diseaseAcc = models.disease.metrics ? models.disease.metrics.accuracy : 0;

  if (accuracyChart) {
    accuracyChart.destroy();
  }

  accuracyChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Crop Recommendation (RF)", "Fertilizer Recommender (Pipeline)", "Plant Disease Detection (CNN)"],
      datasets: [
        {
          label: "Test Accuracy (%)",
          data: [cropAcc, fertAcc, diseaseAcc],
          backgroundColor: [
            "rgba(16, 185, 129, 0.85)",
            "rgba(5, 150, 105, 0.85)",
            "rgba(217, 119, 6, 0.75)"
          ],
          borderColor: [
            "#10b981",
            "#059669",
            "#d97706"
          ],
          borderWidth: 1.5,
          borderRadius: 8
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          max: 100,
          title: { display: true, text: "Accuracy Percentage (%)", font: { weight: "bold" } },
          grid: { color: "#f1f5f1" }
        },
        x: {
          grid: { display: false }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (context) => context.raw > 0 ? ` Accuracy: ${context.raw}%` : " Model Not Trained Yet"
          }
        }
      }
    }
  });
}

function renderFeatureImportanceChart(cropModel) {
  const ctx = document.getElementById("featureChartCanvas");
  if (!ctx) return;

  if (!cropModel.metrics || !cropModel.metrics.feature_importances) {
    return;
  }

  const featObj = cropModel.metrics.feature_importances;
  const labels = Object.keys(featObj).map(k => k.toUpperCase());
  const values = Object.values(featObj).map(v => (v * 100).toFixed(1));

  if (featureChart) {
    featureChart.destroy();
  }

  featureChart = new Chart(ctx, {
    type: "polarArea",
    data: {
      labels: labels,
      datasets: [
        {
          data: values,
          backgroundColor: [
            "rgba(16, 185, 129, 0.7)",
            "rgba(52, 211, 153, 0.7)",
            "rgba(5, 150, 105, 0.7)",
            "rgba(132, 204, 22, 0.7)",
            "rgba(180, 83, 9, 0.65)",
            "rgba(245, 158, 11, 0.65)",
            "rgba(59, 130, 246, 0.65)"
          ],
          borderWidth: 1.5,
          borderColor: "#ffffff"
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "right" },
        tooltip: {
          callbacks: {
            label: (context) => ` ${context.label}: ${context.raw}% impact`
          }
        }
      }
    }
  });
}

function renderRecentLogs(logs) {
  const tbody = document.getElementById("historyTableBody");
  if (!tbody) return;

  tbody.innerHTML = "";

  if (!logs || logs.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="5" style="text-align: center; color: var(--text-muted); padding: 2rem;">
          No advisory queries recorded in SQLite database yet. Perform a crop or fertilizer recommendation to view logs.
        </td>
      </tr>
    `;
    return;
  }

  logs.forEach((log) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td style="font-size: 0.82rem; color: var(--text-muted);">${log.timestamp}</td>
      <td><span class="preset-chip" style="cursor: default;">${log.module}</span></td>
      <td style="font-family: monospace; font-size: 0.85rem;">${log.inputs}</td>
      <td style="font-weight: 700; color: var(--primary-900);">${log.prediction}</td>
      <td><span class="service-badge badge-ready">${log.confidence}%</span></td>
    `;
    tbody.appendChild(tr);
  });
}
