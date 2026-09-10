/**
 * SMART AGRI AI - Crop Recommendation Form & Prediction Engine
 */

document.addEventListener("DOMContentLoaded", () => {
  const cropForm = document.getElementById("cropForm");
  const predictBtn = document.getElementById("predictCropBtn");
  const resultSection = document.getElementById("cropResultSection");
  const alertContainer = document.getElementById("cropAlertContainer");

  // Sample Agronomic Presets for Viva / Quick Testing
  const presets = {
    rice: { N: 90, P: 42, K: 43, temperature: 20.9, humidity: 82.0, ph: 6.5, rainfall: 202.9 },
    coffee: { N: 104, P: 18, K: 30, temperature: 23.6, humidity: 60.4, ph: 6.7, rainfall: 158.0 },
    cotton: { N: 120, P: 40, K: 20, temperature: 24.5, humidity: 79.8, ph: 7.5, rainfall: 80.5 },
    maize: { N: 80, P: 45, K: 20, temperature: 22.5, humidity: 65.0, ph: 6.2, rainfall: 90.0 },
    apple: { N: 20, P: 130, K: 200, temperature: 22.0, humidity: 92.0, ph: 5.9, rainfall: 110.0 }
  };

  // Preset Click Handlers
  document.querySelectorAll(".preset-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const presetKey = chip.getAttribute("data-preset");
      const values = presets[presetKey];
      if (values) {
        document.getElementById("nitrogen").value = values.N;
        document.getElementById("phosphorus").value = values.P;
        document.getElementById("potassium").value = values.K;
        document.getElementById("temperature").value = values.temperature;
        document.getElementById("humidity").value = values.humidity;
        document.getElementById("ph").value = values.ph;
        document.getElementById("rainfall").value = values.rainfall;
        
        showToast(`Loaded ${presetKey.toUpperCase()} sample agronomic conditions!`, "info", 2500);
      }
    });
  });

  // Form Submission
  if (cropForm) {
    cropForm.addEventListener("submit", async (e) => {
      e.preventDefault();

      // Clear previous alerts and results
      alertContainer.innerHTML = "";
      resultSection.style.display = "none";

      // Collect & Validate Inputs
      const n = parseFloat(document.getElementById("nitrogen").value);
      const p = parseFloat(document.getElementById("phosphorus").value);
      const k = parseFloat(document.getElementById("potassium").value);
      const temp = parseFloat(document.getElementById("temperature").value);
      const humidity = parseFloat(document.getElementById("humidity").value);
      const ph = parseFloat(document.getElementById("ph").value);
      const rainfall = parseFloat(document.getElementById("rainfall").value);

      if (isNaN(n) || isNaN(p) || isNaN(k) || isNaN(temp) || isNaN(humidity) || isNaN(ph) || isNaN(rainfall)) {
        showAlert("Please fill in all 7 soil and environmental input fields with valid numbers.", "warning");
        return;
      }

      if (n < 0 || p < 0 || k < 0 || humidity < 0 || humidity > 100 || ph < 0 || ph > 14 || rainfall < 0) {
        showAlert("One or more values are outside valid agricultural ranges. Please check pH (0-14), humidity (0-100%), and non-negative nutrients.", "warning");
        return;
      }

      // Show Loading State
      setButtonLoading(predictBtn, true, "Analyzing Soil & Climate...");

      try {
        const response = await fetch("/api/predict-crop", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            N: n,
            P: p,
            K: k,
            temperature: temp,
            humidity: humidity,
            ph: ph,
            rainfall: rainfall
          })
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
          if (data.model_status === "not_trained") {
            showAlert("Crop model is not trained yet. Please add the dataset and run the training script: <code>python backend/training/train_crop_model.py</code>", "warning");
          } else {
            showAlert(data.error || "Failed to generate crop recommendation. Please check your inputs.", "danger");
          }
          return;
        }

        // Render Recommendation
        renderCropResult(data);
        showToast(`AI Recommends: ${data.prediction_display} (${data.confidence_pct}% confidence)`, "success", 4500);

      } catch (err) {
        console.error("Prediction error:", err);
        showAlert("Network communication error. Verify that the Flask server is running at <code>http://127.0.0.1:5000</code>.", "danger");
      } finally {
        setButtonLoading(predictBtn, false, "🔍 Predict Best Crop");
      }
    });
  }

  function renderCropResult(data) {
    const meta = data.metadata || {};

    document.getElementById("resCropName").innerText = data.prediction_display || data.prediction;
    document.getElementById("resCropIcon").innerText = meta.icon || "🌾";
    document.getElementById("resScientificName").innerText = meta.scientific_name ? `Scientific Name: ${meta.scientific_name}` : "";
    document.getElementById("resConfidence").innerText = `${data.confidence_pct}%`;
    document.getElementById("resCategory").innerText = meta.category || "Field Crop";
    document.getElementById("resDuration").innerText = meta.growth_duration || "Seasonal";
    document.getElementById("resIdealSoil").innerText = meta.ideal_soil || "Loamy / Clayey";
    document.getElementById("resAdvisory").innerText = meta.advisory || "Follow recommended agronomic package of practices.";

    // Render Top 5 Suitable Crops Probability Bars
    const topCropsContainer = document.getElementById("resTopCrops");
    topCropsContainer.innerHTML = "";

    if (data.top_predictions && data.top_predictions.length > 0) {
      data.top_predictions.forEach((item, index) => {
        const row = document.createElement("div");
        row.className = "probability-row";
        row.innerHTML = `
          <div class="prob-crop-name">
            <span>${item.icon || "🌱"}</span>
            <span>${item.crop_display || item.crop}</span>
          </div>
          <div class="prob-bar-track">
            <div class="prob-bar-fill" style="width: 0%;" data-width="${item.percentage}%"></div>
          </div>
          <div class="prob-percentage">${item.percentage}%</div>
        `;
        topCropsContainer.appendChild(row);
      });

      // Animate progress bars
      setTimeout(() => {
        document.querySelectorAll(".prob-bar-fill").forEach((bar) => {
          bar.style.width = bar.getAttribute("data-width");
        });
      }, 50);
    }

    resultSection.style.display = "block";
    resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function showAlert(message, type = "warning") {
    alertContainer.innerHTML = `
      <div class="alert-box alert-${type}">
        <div class="alert-icon">${type === "danger" ? "⚠️" : "ℹ️"}</div>
        <div>
          <div class="alert-content-title">${type === "danger" ? "Prediction Error" : "Notice"}</div>
          <div class="alert-content-body">${message}</div>
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
