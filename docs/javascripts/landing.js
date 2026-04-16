document.addEventListener("DOMContentLoaded", () => {
  if (document.querySelector(".landing-page")) {
    document.body.classList.add("is-landing");
  }

  const workflowExplorer = document.querySelector('[data-module="workflow-explorer"]');
  if (!workflowExplorer) {
    return;
  }

  const workflows = {
    regression: {
      badge: "Default starting point",
      title: "<code>SelfCalibratingConformalPredictor</code>",
      summary:
        "Start here when you have a point predictor for <code>y</code> and want calibrated predictions together with intervals induced by that calibrated scale.",
      visualNote:
        "Best for the shortest path from a fitted regression model to calibrated predictions, Venn-Abers style multi-predictions, and interval outputs.",
      inputTitle: "Core inputs",
      inputCopy:
        "A predictor for <code>y</code>, a calibration split, and the target miscoverage level <code>alpha</code>.",
      outputTitle: "What you get",
      outputCopy:
        "Calibrated point predictions, Venn-Abers style prediction sets, intervals, and empirical coverage summaries.",
      code:
        "from selfcalibratingconformal import SelfCalibratingConformalPredictor\n\nmodel = SelfCalibratingConformalPredictor(predictor)\nmodel.fit(X_cal, y_cal, alpha=0.1)\nintervals = model.predict_interval(X_test)",
      schematic:
        '<div class="workflow-chip">predictor</div><div class="workflow-arrow" aria-hidden="true"></div><div class="workflow-chip workflow-chip-accent">calibration map</div><div class="workflow-arrow" aria-hidden="true"></div><div class="workflow-chip workflow-chip-warm">intervals</div>',
    },
    quantile: {
      badge: "Conformal threshold path",
      title: "<code>VennAbersQuantileConformalPredictor</code>",
      summary:
        "Use this path when your workflow estimates the <code>(1 - alpha)</code> quantile of a conformity score and you want calibrated thresholds that translate into prediction intervals.",
      visualNote:
        "Best when your upstream model already produces score thresholds instead of direct outcome predictions.",
      inputTitle: "Core inputs",
      inputCopy:
        "A score-quantile predictor, an optional center predictor, a calibration split, and the target level <code>alpha</code>.",
      outputTitle: "What you get",
      outputCopy:
        "Calibrated score thresholds, conformal intervals, and both coverage and threshold-calibration diagnostics.",
      code:
        "from selfcalibratingconformal import VennAbersQuantileConformalPredictor\n\ncp = VennAbersQuantileConformalPredictor(\n    score_quantile_predictor=score_model,\n    center_predictor=center_model,\n    alpha=0.1,\n)\ncp.fit(X_cal, y_cal)",
      schematic:
        '<div class="workflow-chip">score quantile</div><div class="workflow-arrow" aria-hidden="true"></div><div class="workflow-chip workflow-chip-accent">threshold calibration</div><div class="workflow-arrow" aria-hidden="true"></div><div class="workflow-chip workflow-chip-warm">interval solver</div>',
    },
  };

  const badge = workflowExplorer.querySelector("[data-workflow-badge]");
  const title = workflowExplorer.querySelector("[data-workflow-title]");
  const summary = workflowExplorer.querySelector("[data-workflow-summary]");
  const visualNote = workflowExplorer.querySelector("[data-workflow-visual-note]");
  const inputTitle = workflowExplorer.querySelector("[data-workflow-input-title]");
  const inputCopy = workflowExplorer.querySelector("[data-workflow-input-copy]");
  const outputTitle = workflowExplorer.querySelector("[data-workflow-output-title]");
  const outputCopy = workflowExplorer.querySelector("[data-workflow-output-copy]");
  const code = workflowExplorer.querySelector("[data-workflow-code] code");
  const schematic = workflowExplorer.querySelector("[data-workflow-schematic]");
  const buttons = workflowExplorer.querySelectorAll("[data-workflow]");

  const renderWorkflow = (key) => {
    const workflow = workflows[key];
    if (!workflow) {
      return;
    }
    badge.textContent = workflow.badge;
    title.innerHTML = workflow.title;
    summary.innerHTML = workflow.summary;
    visualNote.innerHTML = workflow.visualNote;
    inputTitle.textContent = workflow.inputTitle;
    inputCopy.innerHTML = workflow.inputCopy;
    outputTitle.textContent = workflow.outputTitle;
    outputCopy.innerHTML = workflow.outputCopy;
    code.textContent = workflow.code;
    schematic.innerHTML = workflow.schematic;

    buttons.forEach((button) => {
      button.classList.toggle("is-active", button.dataset.workflow === key);
    });
  };

  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      renderWorkflow(button.dataset.workflow);
    });
  });

  renderWorkflow("regression");
});
