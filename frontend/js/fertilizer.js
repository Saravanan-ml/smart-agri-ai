/**
 * SMART AGRI AI - Fertilizer Recommendation Form & Inference Engine
 */

document.addEventListener("DOMContentLoaded", () => {
  const fertForm = document.getElementById("fertilizerForm");
  const recommendBtn = document.getElementById("recommendFertBtn");
  const resultSection = document.getElementById("fertResultSection");
  const alertContainer = document.getElementById("fertAlertContainer");

  // Sample Presets for Quick Testing
  const fertPresets = {
    cotton_clay: { crop: "Cotton", soil: "Clay", N: 61, P: 44, K: 84 },
    maize_silt: { crop: "Maize", soil: "Silt", N: 59, P: 56, K: 18 },
    wheat_sandy: { crop: "Wheat", soil: "Sandy", N: 88, P: 46, K: 34 }
  };

  document.querySelectorAll(".preset-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const key = chip.getAttribute("data-preset");
      const p = fertPresets[key];
      if (p) {
        document.getElementById("cropType").value = p.crop;
        document.getElementById("soilType").value = p.soil;
        document.getElementById("fertNitrogen").value = p.N;
        document.getElementById("fertPhosphorus").value = p.P;
        document.getElementById("fertPotassium").value = p.K;
        showToast(`Loaded ${p.crop} on ${p.soil} Soil preset!`, "info", 2500);
      }
    });
  });

  if (fertForm) {
    fertForm.addEventListener("submit", async (e) => {
      e.preventDefault();

      alertContainer.innerHTML = "";
      resultSection.style.display = "none";

      const crop = document.getElementById("cropType").value;
      const soil = document.getElementById("soilType").value;
      const n = parseFloat(document.getElementById("fertNitrogen").value);
      const p = parseFloat(document.getElementById("fertPhosphorus").value);
      const k = parseFloat(document.getElementById("fertPotassium").value);

      if (!crop || !soil || isNaN(n) || isNaN(p) || isNaN(k)) {
        showAlert("Please select crop, soil type, and enter numeric values for N, P, and K.", "warning");
        return;
      }

      setButtonLoading(recommendBtn, true, "Formulating Fertilizer Advisory...");

      try {
        const response = await fetch("/api/predict-fertilizer", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            crop: crop,
            soil_type: soil,
            N: n,
            P: p,
            K: k
          })
        });

        const data = await response.json();

        if (data.model_status === "not_trained") {
          showAlert("Fertilizer model is not trained yet. Please run: <code>python backend/training/train_fertilizer_model.py</code>", "warning");
          return;
        }

        if (!response.ok || !data.success) {
          showAlert(data.error || "Failed to formulate fertilizer recommendation.", "danger");
          return;
        }

        // Render result
        document.getElementById("resFertilizerName").innerText = data.prediction;
        document.getElementById("resFertConfidence").innerText = `${data.confidence_pct}%`;
        document.getElementById("resFertAdvisory").innerText = data.advisory;
        
        resultSection.style.display = "block";
        resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
        showToast(`Recommended: ${data.prediction}`, "success", 4500);

      } catch (err) {
        console.error("Fertilizer prediction error:", err);
        showAlert("Network error. Verify backend server is running.", "danger");
      } finally {
        setButtonLoading(recommendBtn, false, "🧪 Recommend Fertilizer");
      }
    });
  }

  function showAlert(msg, type = "warning") {
    alertContainer.innerHTML = `
      <div class="alert-box alert-${type}">
        <div class="alert-icon">${type === "danger" ? "✕" : "ℹ️"}</div>
        <div>
          <div class="alert-content-title">${type === "danger" ? "Error" : "Notice"}</div>
          <div class="alert-content-body">${msg}</div>
        </div>
      </div>
    `;
    alertContainer.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  function setButtonLoading(btn, isLoading, text) {
    if (!btn) return;
    if (isLoading) {
      btn.disabled = true;
      btn.innerHTML = `<span class="spinner"></span> <span>${text}</span>`;
    } else {
      btn.disabled = false;
      btn.innerHTML = text;
    }
  }
});
