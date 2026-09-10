/**
 * SMART AGRI AI - Plant Disease Detection JavaScript
 * Supports drag-and-drop file upload, instant image preview, dimension extraction,
 * and clear explanation if CNN weights are not yet trained.
 */

document.addEventListener("DOMContentLoaded", () => {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("leafFileInput");
  const previewBox = document.getElementById("previewBox");
  const previewImg = document.getElementById("previewImg");
  const previewName = document.getElementById("previewName");
  const previewMeta = document.getElementById("previewMeta");
  const removeBtn = document.getElementById("removeFileBtn");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const alertContainer = document.getElementById("diseaseAlertContainer");
  const resultSection = document.getElementById("diseaseResultSection");

  let selectedFile = null;

  // File input click trigger
  if (dropzone && fileInput) {
    dropzone.addEventListener("click", () => fileInput.click());

    // Drag & Drop events
    ["dragenter", "dragover"].forEach((eventName) => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove("dragover");
      });
    });

    dropzone.addEventListener("drop", (e) => {
      const dt = e.dataTransfer;
      const files = dt.files;
      if (files.length > 0) {
        handleFileSelection(files[0]);
      }
    });

    fileInput.addEventListener("change", (e) => {
      if (fileInput.files.length > 0) {
        handleFileSelection(fileInput.files[0]);
      }
    });
  }

  if (removeBtn) {
    removeBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      resetFileSelection();
    });
  }

  function handleFileSelection(file) {
    const validTypes = ["image/jpeg", "image/jpg", "image/png"];
    if (!validTypes.includes(file.type.toLowerCase())) {
      showToast("Invalid file format. Please select a JPG, JPEG, or PNG image.", "error");
      return;
    }

    selectedFile = file;
    alertContainer.innerHTML = "";
    resultSection.style.display = "none";

    // Read and preview image
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      previewName.innerText = file.name;
      
      // Calculate image dimensions
      const img = new Image();
      img.onload = () => {
        const sizeKB = (file.size / 1024).toFixed(1);
        previewMeta.innerText = `Dimensions: ${img.width} × ${img.height} px | Size: ${sizeKB} KB`;
      };
      img.src = e.target.result;

      dropzone.style.display = "none";
      previewBox.style.display = "flex";
      analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  function resetFileSelection() {
    selectedFile = null;
    fileInput.value = "";
    dropzone.style.display = "block";
    previewBox.style.display = "none";
    previewImg.src = "";
    analyzeBtn.disabled = true;
    resultSection.style.display = "none";
    alertContainer.innerHTML = "";
  }

  // Analyze Button Submission
  if (analyzeBtn) {
    analyzeBtn.addEventListener("click", async () => {
      if (!selectedFile) {
        showToast("Please upload a plant leaf image first.", "warning");
        return;
      }

      setButtonLoading(analyzeBtn, true, "Running Deep Learning CNN...");
      alertContainer.innerHTML = "";
      resultSection.style.display = "none";

      const formData = new FormData();
      formData.append("image", selectedFile);

      try {
        const response = await fetch("/api/predict-disease", {
          method: "POST",
          body: formData
        });

        const data = await response.json();

        if (data.model_status === "not_trained") {
          // Model is not trained yet - show transparent instruction banner
          alertContainer.innerHTML = `
            <div class="alert-box alert-warning">
              <div class="alert-icon">⚠️</div>
              <div>
                <div class="alert-content-title">Plant Disease CNN Model is Not Trained Yet</div>
                <div class="alert-content-body">
                  <p>Image <strong>${data.file_info.filename}</strong> (${data.file_info.dimensions}) was verified and preprocessed successfully.</p>
                  <p style="margin-top: 0.5rem;">To train the real Deep Learning classifier:</p>
                  <ol style="margin-left: 1.25rem; margin-top: 0.25rem;">
                    <li>Place leaf image folders (e.g. <code>Tomato___healthy</code>, <code>Tomato___Early_blight</code>) into <code>data/disease_dataset/</code>.</li>
                    <li>Execute the CNN training pipeline: <code>python backend/training/train_disease_model.py</code>.</li>
                  </ol>
                  <p style="margin-top: 0.5rem; font-style: italic; color: #78350f;">
                    In accordance with academic integrity guidelines, SMART AGRI AI never presents fabricated AI predictions.
                  </p>
                </div>
              </div>
            </div>
          `;
          alertContainer.scrollIntoView({ behavior: "smooth", block: "center" });
          return;
        }

        if (!response.ok || !data.success) {
          alertContainer.innerHTML = `
            <div class="alert-box alert-danger">
              <div class="alert-icon">✕</div>
              <div>
                <div class="alert-content-title">Analysis Failed</div>
                <div class="alert-content-body">${data.error || "An unexpected error occurred during classification."}</div>
              </div>
            </div>
          `;
          return;
        }

        // Render diagnosis
        document.getElementById("resDiseaseName").innerText = data.prediction;
        document.getElementById("resDiseaseConfidence").innerText = `${data.confidence_pct}%`;
        resultSection.style.display = "block";
        resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
        showToast(`Diagnosis: ${data.prediction} (${data.confidence_pct}%)`, "success", 4500);

      } catch (err) {
        console.error("Leaf analysis error:", err);
        alertContainer.innerHTML = `
          <div class="alert-box alert-danger">
            <div class="alert-icon">⚠️</div>
            <div>
              <div class="alert-content-title">Network Communication Error</div>
              <div class="alert-content-body">Could not connect to Flask API server. Please verify backend is active.</div>
            </div>
          </div>
        `;
      } finally {
        setButtonLoading(analyzeBtn, false, "🔬 Analyze Leaf");
      }
    });
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
