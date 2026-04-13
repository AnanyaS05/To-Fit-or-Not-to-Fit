(function () {
  function parseJsonScript(id) {
    const node = document.getElementById(id);
    if (!node) return null;

    try {
      return JSON.parse(node.textContent || "null");
    } catch (err) {
      return null;
    }
  }

  function formatMetric(value) {
    if (typeof value !== "number" || Number.isNaN(value)) {
      return "N/A";
    }
    return value.toFixed(3);
  }

  function renderVisualizationPage() {
    const cvRows = parseJsonScript("cv-rows-json") || [];
    const mlpCvRows = parseJsonScript("mlp-cv-rows-json") || [];
    const trainDist = parseJsonScript("train-dist-json") || [];
    const bayesSummary = parseJsonScript("bayes-summary-json") || {};
    const bayesPerClass = parseJsonScript("bayes-per-class-json") || [];
    const bayesConfusion = parseJsonScript("bayes-confusion-json") || {};
    const mlpSummary = parseJsonScript("mlp-summary-json") || {};
    const mlpPerClass = parseJsonScript("mlp-per-class-json") || [];
    const mlpConfusion = parseJsonScript("mlp-confusion-json") || {};
    const modelCompareRows = parseJsonScript("model-compare-json") || [];

    const populateFoldTable = (selector, rows, emptyMessage) => {
      const tableBody = document.querySelector(selector);
      if (!tableBody) return;

      tableBody.innerHTML = "";
      rows.forEach((row) => {
        const tr = document.createElement("tr");
        tr.innerHTML = [
          `<td>${row.fold}</td>`,
          `<td>${formatMetric(row.val_accuracy)}</td>`,
          `<td>${formatMetric(row.val_macro_precision)}</td>`,
          `<td>${formatMetric(row.val_macro_recall)}</td>`,
          `<td>${formatMetric(row.val_macro_f1)}</td>`,
        ].join("");
        tableBody.appendChild(tr);
      });

      if (rows.length === 0) {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td colspan="5">${emptyMessage}</td>`;
        tableBody.appendChild(tr);
      }
    };

    populateFoldTable(
      "#bayes-cv-table tbody",
      cvRows,
      "No Bayesian cross-validation file found in Data/brms_cv_fold_metrics.csv."
    );
    populateFoldTable(
      "#mlp-cv-table tbody",
      mlpCvRows,
      "No MLP fold-level scores available."
    );

    const tabButtons = document.querySelectorAll(".fold-tab");
    const tabPanels = document.querySelectorAll(".fold-panel");
    if (tabButtons.length > 0 && tabPanels.length > 0) {
      tabButtons.forEach((button) => {
        button.addEventListener("click", () => {
          const target = button.getAttribute("data-tab-target");

          tabButtons.forEach((btn) => {
            const active = btn === button;
            btn.classList.toggle("is-active", active);
            btn.setAttribute("aria-selected", active ? "true" : "false");
          });

          tabPanels.forEach((panel) => {
            const panelTarget = panel.getAttribute("data-tab-panel");
            const active = panelTarget === target;
            panel.classList.toggle("is-active", active);
            panel.hidden = !active;
          });
        });
      });
    }

    const compareTableBody = document.querySelector("#model-compare-table tbody");
    if (compareTableBody) {
      compareTableBody.innerHTML = "";
      modelCompareRows.forEach((row) => {
        const tr = document.createElement("tr");
        tr.innerHTML = [
          `<td>${row.metric}</td>`,
          `<td>${formatMetric(row.bayesian)}</td>`,
          `<td>${formatMetric(row.mlp)}</td>`,
        ].join("");
        compareTableBody.appendChild(tr);
      });

      if (modelCompareRows.length === 0) {
        const tr = document.createElement("tr");
        tr.innerHTML = '<td colspan="3">No model comparison data available.</td>';
        compareTableBody.appendChild(tr);
      }
    }

    const renderSummaryBar = (targetId, summary, chartTitle, color) => {
      const target = document.getElementById(targetId);
      if (!target) return;

      const metricKeys = ["accuracy", "macro_precision", "macro_recall", "macro_f1"];
      const metricLabels = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"];
      const values = metricKeys.map((k) => {
        const v = summary ? summary[k] : null;
        return typeof v === "number" ? v : null;
      });

      const hasAtLeastOne = values.some((v) => v !== null);
      if (!hasAtLeastOne) {
        target.innerHTML = "<p>Metrics not available.</p>";
        return;
      }

      Plotly.newPlot(
        target,
        [
          {
            type: "bar",
            x: metricLabels,
            y: values,
            marker: { color },
          },
        ],
        {
          margin: { l: 46, r: 12, t: 20, b: 50 },
          title: { text: chartTitle, font: { size: 14 } },
          yaxis: { range: [0, 1], title: "Score" },
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(255,255,255,0.8)",
        },
        { responsive: true, displaylogo: false }
      );
    };

    const renderPerClassF1 = (targetId, rows, color, emptyMessage) => {
      const target = document.getElementById(targetId);
      if (!target) return;

      if (!Array.isArray(rows) || rows.length === 0) {
        target.innerHTML = `<p>${emptyMessage}</p>`;
        return;
      }

      Plotly.newPlot(
        target,
        [
          {
            type: "bar",
            x: rows.map((r) => r.class),
            y: rows.map((r) => r.f1),
            marker: { color },
          },
        ],
        {
          margin: { l: 42, r: 12, t: 18, b: 40 },
          yaxis: { range: [0, 1], title: "F1" },
          xaxis: { title: "Class" },
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(255,255,255,0.8)",
        },
        { responsive: true, displaylogo: false }
      );
    };

    const renderConfusionHeatmap = (targetId, confusion, emptyMessage) => {
      const target = document.getElementById(targetId);
      if (!target) return;

      const matrix = confusion && Array.isArray(confusion.matrix) ? confusion.matrix : [];
      const xLabels = confusion && Array.isArray(confusion.x_labels) ? confusion.x_labels : [];
      const yLabels = confusion && Array.isArray(confusion.y_labels) ? confusion.y_labels : [];

      if (!matrix.length || !xLabels.length || !yLabels.length) {
        target.innerHTML = `<p>${emptyMessage}</p>`;
        return;
      }

      Plotly.newPlot(
        target,
        [
          {
            type: "heatmap",
            z: matrix,
            x: xLabels,
            y: yLabels,
            colorscale: "Blues",
          },
        ],
        {
          margin: { l: 60, r: 20, t: 20, b: 50 },
          xaxis: { title: "Predicted" },
          yaxis: { title: "True" },
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(255,255,255,0.8)",
        },
        { responsive: true, displaylogo: false }
      );
    };

    if (window.Plotly) {
      const cvTarget = document.getElementById("cv-chart");
      if (cvTarget) {
        if (cvRows.length > 0) {
          const folds = cvRows.map((d) => d.fold);
          const acc = cvRows.map((d) => d.val_accuracy);
          const macroF1 = cvRows.map((d) => d.val_macro_f1);

          const traces = [
            {
              x: folds,
              y: acc,
              mode: "lines+markers",
              name: "Accuracy",
              marker: { color: "#0f7c6e" },
              line: { color: "#0f7c6e", width: 3 },
            },
            {
              x: folds,
              y: macroF1,
              mode: "lines+markers",
              name: "Macro F1",
              marker: { color: "#f59f66" },
              line: { color: "#f59f66", width: 3 },
            },
          ];

          const layout = {
            margin: { l: 44, r: 12, t: 18, b: 40 },
            yaxis: { range: [0, 1], title: "Score" },
            xaxis: { title: "Fold" },
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: "rgba(255,255,255,0.8)",
            legend: { orientation: "h" },
          };

          Plotly.newPlot(cvTarget, traces, layout, { responsive: true, displaylogo: false });
        } else {
          cvTarget.innerHTML = "<p>No CV metrics available yet.</p>";
        }
      }

      const distTarget = document.getElementById("class-dist-chart");
      if (distTarget) {
        if (trainDist.length > 0) {
          const labels = trainDist.map((d) => d.class);
          const values = trainDist.map((d) => d.count);

          const traces = [
            {
              type: "pie",
              labels,
              values,
              hole: 0.42,
              marker: {
                colors: ["#117a6d", "#f59f66", "#6ca0dc", "#9ac57d", "#d88bc8"],
              },
              textinfo: "label+percent",
            },
          ];

          const layout = {
            margin: { l: 12, r: 12, t: 16, b: 12 },
            paper_bgcolor: "rgba(0,0,0,0)",
            showlegend: true,
          };

          Plotly.newPlot(distTarget, traces, layout, { responsive: true, displaylogo: false });
        } else {
          distTarget.innerHTML = "<p>No train distribution data available.</p>";
        }
      }

      renderSummaryBar("bayes-summary-chart", bayesSummary, "Bayesian Test Metrics", "#f59f66");
      renderSummaryBar("mlp-summary-chart", mlpSummary, "MLP Test Metrics", "#0f7c6e");

      renderPerClassF1(
        "bayes-per-class-chart",
        bayesPerClass,
        "#f2a66e",
        "No Bayesian per-class metrics available."
      );
      renderPerClassF1(
        "mlp-per-class-chart",
        mlpPerClass,
        "#2c9c8f",
        "No MLP per-class metrics available."
      );

      renderConfusionHeatmap(
        "bayes-cm-chart",
        bayesConfusion,
        "No Bayesian confusion matrix available."
      );
      renderConfusionHeatmap(
        "mlp-cm-chart",
        mlpConfusion,
        "No MLP confusion matrix available."
      );

      const modelCompareTarget = document.getElementById("model-compare-chart");
      if (modelCompareTarget) {
        if (modelCompareRows.length > 0) {
          const metrics = modelCompareRows.map((row) => row.metric);
          const bayesValues = modelCompareRows.map((row) => row.bayesian);
          const mlpValues = modelCompareRows.map((row) => row.mlp);

          Plotly.newPlot(
            modelCompareTarget,
            [
              {
                type: "bar",
                name: "Bayesian",
                x: metrics,
                y: bayesValues,
                marker: { color: "#f59f66" },
              },
              {
                type: "bar",
                name: "MLP",
                x: metrics,
                y: mlpValues,
                marker: { color: "#0f7c6e" },
              },
            ],
            {
              barmode: "group",
              margin: { l: 44, r: 12, t: 20, b: 55 },
              yaxis: { range: [0, 1], title: "Score" },
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(255,255,255,0.8)",
            },
            { responsive: true, displaylogo: false }
          );
        } else {
          modelCompareTarget.innerHTML = "<p>No model comparison metrics available.</p>";
        }
      }
    }
  }

  function wirePredictionForm() {
    const form = document.getElementById("predict-form");
    if (!form) return;

    const summaryNode = document.getElementById("prediction-summary");
    const barsNode = document.getElementById("probability-bars");
    const inputsNode = document.getElementById("normalized-inputs");

    form.addEventListener("submit", async (event) => {
      event.preventDefault();

      const formData = new FormData(form);
      const payload = Object.fromEntries(formData.entries());

      summaryNode.textContent = "Running prediction...";
      barsNode.innerHTML = "";
      inputsNode.textContent = "";

      try {
        const response = await fetch("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        const body = await response.json();
        if (!response.ok || !body.ok) {
          throw new Error(body.error || "Prediction failed.");
        }

        const confidencePct = (body.confidence * 100).toFixed(1);
        summaryNode.innerHTML =
          `Prediction: <strong>${body.prediction.toUpperCase()}</strong> ` +
          `with ${confidencePct}% confidence.`;

        const probs = body.probabilities || {};
        const entries = Object.entries(probs).sort((a, b) => b[1] - a[1]);
        entries.forEach(([label, value]) => {
          const row = document.createElement("div");
          row.className = "prob-row";
          row.innerHTML = `
            <span>${label}</span>
            <div class="prob-track"><div class="prob-fill" style="width:${(value * 100).toFixed(1)}%"></div></div>
            <span>${(value * 100).toFixed(1)}%</span>
          `;
          barsNode.appendChild(row);
        });

        inputsNode.textContent = `Model Input Snapshot:\n${JSON.stringify(body.normalized_inputs, null, 2)}`;
      } catch (err) {
        summaryNode.textContent = err.message || "Prediction failed.";
      }
    });
  }

  function renderPredictionReferenceCharts() {
    const sizeRows = parseJsonScript("size-chart-json") || [];
    const cupRows = parseJsonScript("cup-map-json") || [];

    const sizeTableBody = document.querySelector("#size-reference-table tbody");
    if (sizeTableBody) {
      sizeTableBody.innerHTML = "";
      sizeRows.forEach((row) => {
        const tr = document.createElement("tr");
        tr.innerHTML = [
          `<td>${row.garment_type}</td>`,
          `<td>${row.size_label}</td>`,
          `<td>${row.bust_min}-${row.bust_max}</td>`,
          `<td>${row.waist_min}-${row.waist_max}</td>`,
          `<td>${row.hips_min}-${row.hips_max}</td>`,
          `<td>${row.height_min_inches}-${row.height_max_inches}</td>`,
        ].join("");
        sizeTableBody.appendChild(tr);
      });

      if (sizeRows.length === 0) {
        const tr = document.createElement("tr");
        tr.innerHTML = '<td colspan="6">No demo size chart file found.</td>';
        sizeTableBody.appendChild(tr);
      }
    }

    const cupTableBody = document.querySelector("#cup-map-table tbody");
    if (cupTableBody) {
      cupTableBody.innerHTML = "";
      cupRows.forEach((row) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${row.letter}</td><td>${row.code}</td>`;
        cupTableBody.appendChild(tr);
      });
    }

    if (!window.Plotly) return;

    const chartTarget = document.getElementById("size-reference-chart");
    if (!chartTarget) return;

    if (sizeRows.length === 0) {
      chartTarget.innerHTML = "<p>No reference sizing chart available.</p>";
      return;
    }

    const byType = {};
    sizeRows.forEach((row) => {
      const key = row.garment_type;
      if (!byType[key]) byType[key] = [];
      byType[key].push(row);
    });

    const traces = Object.entries(byType).map(([garmentType, rows]) => {
      const ordered = [...rows].sort((a, b) => a.size_order - b.size_order);
      const x = ordered.map((r) => r.size_label);
      const yCenter = ordered.map((r) => (r.hips_min + r.hips_max) / 2);
      const yHalfRange = ordered.map((r) => (r.hips_max - r.hips_min) / 2);

      return {
        x,
        y: yCenter,
        mode: "lines+markers",
        name: garmentType,
        error_y: {
          type: "data",
          array: yHalfRange,
          visible: true,
        },
      };
    });

    const layout = {
      margin: { l: 48, r: 12, t: 20, b: 45 },
      xaxis: { title: "Size Label" },
      yaxis: { title: "Hips (inches), center +/- range" },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(255,255,255,0.8)",
      legend: { orientation: "h" },
    };

    Plotly.newPlot(chartTarget, traces, layout, { responsive: true, displaylogo: false });
  }

  window.ToFitApp = {
    renderVisualizationPage,
    wirePredictionForm,
    renderPredictionReferenceCharts,
  };
})();
