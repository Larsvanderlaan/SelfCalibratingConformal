---
hide:
  - navigation
  - toc
---

<div class="landing-page" id="top">
  <header class="landing-hero">
    <nav class="landing-topbar" aria-label="Section navigation">
      <div class="landing-brand"><a href="#top">selfcalibratingconformal</a></div>
      <div class="landing-links">
        <a href="#overview">Overview</a>
        <a href="#workflows">Workflows</a>
        <a href="#why-calibration">Why Calibration</a>
        <a href="#install">Install</a>
        <a href="#quickstart">Quickstart</a>
        <a href="#resources">Resources</a>
        <a href="#references">References</a>
        <a href="https://github.com/Larsvanderlaan/SelfCalibratingConformal">GitHub</a>
      </div>
    </nav>

    <div class="hero-grid">
      <section class="hero-copy">
        <p class="landing-eyebrow">selfcalibratingconformal</p>
        <div class="hero-head">
          <h1>self-calibrating conformal prediction</h1>
          <div class="hero-art">
            <img
              src="./assets/selfcalibratingconformal-badge-mark.svg"
              alt="Circular Self-Calibrating Conformal badge with a calibration curve and interval band"
              width="1024"
              height="1024"
            />
          </div>
        </div>
        <p class="landing-lede">
          <code>selfcalibratingconformal</code> brings post-hoc calibration and conformal
          prediction to black-box regression models. Choose one of two routes: calibrate a point
          prediction, or calibrate a conformity-score threshold. In both, interval width adapts
          across data-adaptive isotonic bins.
        </p>
        <div class="landing-actions">
          <a class="landing-button landing-button-primary" href="#quickstart">Quickstart</a>
          <a class="landing-button landing-button-secondary" href="#workflows">Choose a workflow</a>
          <a
            class="landing-button landing-button-secondary"
            href="https://github.com/Larsvanderlaan/SelfCalibratingConformal"
          >
            GitHub
          </a>
        </div>
        <ul class="hero-pills">
          <li>Point-prediction and score-threshold workflows</li>
          <li>Adaptive intervals across isotonic bins</li>
          <li>Backward-compatible API with typed configs</li>
        </ul>
      </section>

      <aside class="hero-panel">
        <div class="landing-card landing-card-gradient">
          <p class="landing-card-label">At a glance</p>
          <div class="stat-grid">
            <div class="stat-item">
              <strong>Core use case</strong>
              <span>Wrap an existing regression model to improve calibration and intervals.</span>
            </div>
            <div class="stat-item">
              <strong>Two entrypoints</strong>
              <span>Calibrate predicted responses or score thresholds.</span>
            </div>
            <div class="stat-item">
              <strong>Outputs</strong>
              <span>Return calibrated predictions, intervals, and diagnostics.</span>
            </div>
          </div>
        </div>

        <div class="landing-card">
          <p class="landing-card-label">Core classes</p>
          <pre class="landing-code"><code>from selfcalibratingconformal import (
    SelfCalibratingConformalPredictor,
    VennAbersQuantileConformalPredictor,
)</code></pre>
        </div>
      </aside>
    </div>
  </header>

  <section class="landing-section" id="overview">
    <div class="section-heading">
      <p class="landing-eyebrow">Overview</p>
      <h2>What the package is for</h2>
      <p>
        Use the package when you already have a regression model and need better calibration and
        intervals. It adds post-hoc calibration and conformal inference without changing the
        underlying model.
      </p>
    </div>

    <div class="overview-grid">
      <article class="landing-card">
        <p class="mini-label">Self-calibrating conformal</p>
        <h3>Calibrate the predicted response, then build an interval around it</h3>
        <p>
          This workflow starts with a point prediction for <code>y</code>. It learns a monotone
          correction on a calibration set, then builds adaptive intervals from the calibrated
          prediction. Width varies across data-adaptive isotonic bins.
        </p>
      </article>
      <article class="landing-card">
        <p class="mini-label">Quantile workflow</p>
        <h3>Predict how large the error can be, then calibrate that threshold</h3>
        <p>
          This workflow starts with a model for the <code>(1 - alpha)</code> quantile of a
          conformity score. The package calibrates that threshold, then includes outcomes whose
          scores fall below the calibrated cutoff. Width varies across data-adaptive isotonic bins
          of the calibrated quantile.
        </p>
      </article>
      <article class="landing-card">
        <p class="mini-label">Conformity score</p>
        <h3>Measure how compatible a candidate outcome is with the input</h3>
        <p>
          A conformity score is small when a candidate outcome looks plausible and large when it
          does not. The default score is <code>|y - mu(x)|</code>, the absolute distance from the
          center prediction.
        </p>
      </article>
    </div>
  </section>

  <section class="landing-section" id="workflows">
    <div class="section-heading">
      <p class="landing-eyebrow">Workflows</p>
      <h2>Choose the workflow that matches your upstream model</h2>
      <p>
        Use the regression path for point predictors and the quantile path for score-threshold
        models.
      </p>
    </div>

    <div class="method-explorer" data-module="workflow-explorer">
      <div class="method-tabs" role="tablist" aria-label="Package workflows">
        <button class="tab-button is-active" type="button" data-workflow="regression">Regression workflow</button>
        <button class="tab-button" type="button" data-workflow="quantile">Quantile-score workflow</button>
      </div>

      <div class="method-panel">
        <div class="panel-banner">
          <span class="badge" data-workflow-badge>Default starting point</span>
          <h3 data-workflow-title><code>SelfCalibratingConformalPredictor</code></h3>
        </div>
        <p class="method-summary" data-workflow-summary>
          Start here if you have a point predictor for <code>y</code> and want calibrated
          predictions and intervals.
        </p>

        <div class="method-visual-card">
          <div class="method-visual-copy">
            <span class="curve-kicker">When to use it</span>
            <p data-workflow-visual-note>
              Best when you want to wrap a fitted regression model.
            </p>
          </div>
          <div class="workflow-schematic" data-workflow-schematic="regression">
            <div class="workflow-chip">predictor</div>
            <div class="workflow-arrow" aria-hidden="true"></div>
            <div class="workflow-chip workflow-chip-accent">calibration map</div>
            <div class="workflow-arrow" aria-hidden="true"></div>
            <div class="workflow-chip workflow-chip-warm">intervals</div>
          </div>
        </div>

        <div class="argument-grid argument-grid-two">
          <article class="argument-card">
            <h3 data-workflow-input-title>Core inputs</h3>
            <p data-workflow-input-copy>
              A predictor for <code>y</code>, a calibration split, and the target miscoverage
              level <code>alpha</code>.
            </p>
          </article>
          <article class="argument-card">
            <h3 data-workflow-output-title>What you get</h3>
            <p data-workflow-output-copy>
              Calibrated point predictions, prediction sets, intervals, and coverage summaries.
            </p>
          </article>
        </div>

        <pre class="landing-code" data-workflow-code><code>from selfcalibratingconformal import SelfCalibratingConformalPredictor

model = SelfCalibratingConformalPredictor(predictor)
model.fit(X_cal, y_cal, alpha=0.1)
intervals = model.predict_interval(X_test)</code></pre>
      </div>
    </div>
  </section>

  <section class="landing-section" id="why-calibration">
    <div class="section-heading">
      <p class="landing-eyebrow">Why Calibration</p>
      <h2>Predict, calibrate, then form the interval</h2>
      <p>
        Predict a value or threshold, calibrate it on held-out data, then build the interval.
        Width adapts across data-adaptive isotonic bins.
      </p>
    </div>

    <div class="process-strip" role="img" aria-label="Workflow showing predict, calibrate, and form intervals.">
      <article class="process-step">
        <p class="process-label">Step 1</p>
        <h3>Predict</h3>
        <p>Fit or supply a model that outputs a prediction or score threshold.</p>
      </article>
      <div class="process-connector" aria-hidden="true"></div>
      <article class="process-step process-step-accent">
        <p class="process-label">Step 2</p>
        <h3>Calibrate</h3>
        <p>Use a calibration set to learn a monotone correction.</p>
      </article>
      <div class="process-connector" aria-hidden="true"></div>
      <article class="process-step process-step-warm">
        <p class="process-label">Step 3</p>
        <h3>Form intervals</h3>
        <p>Build the interval from the calibrated output and check coverage.</p>
      </article>
    </div>

    <div class="overview-grid">
      <article class="landing-card">
        <p class="mini-label">Direct regression path</p>
        <h3>How the self-calibrating path makes an interval</h3>
        <p>
          It calibrates the point prediction and then uses residual-based conformity scores to set
          interval width around the calibrated center. Width varies across data-adaptive isotonic
          bins.
        </p>
      </article>
      <article class="landing-card">
        <p class="mini-label">Score-threshold path</p>
        <h3>How the quantile path makes an interval</h3>
        <p>
          It predicts a threshold for the conformity score, calibrates that threshold, and includes
          outcomes whose scores stay below the calibrated cutoff. With the default absolute-residual
          score, this gives a symmetric adaptive interval around the center prediction. Width varies
          across data-adaptive isotonic bins of the calibrated quantile.
        </p>
      </article>
    </div>
  </section>

  <section class="landing-section" id="install">
    <div class="section-heading">
      <p class="landing-eyebrow">Install</p>
      <h2>Install the package and start from your own predictor</h2>
      <p>
        Most users only need the base package.
      </p>
    </div>
    <div class="single-code-card">
      <pre class="landing-code"><code>python -m pip install selfcalibratingconformal</code></pre>
    </div>
  </section>

  <section class="landing-section" id="quickstart">
    <div class="section-heading">
      <p class="landing-eyebrow">Quickstart</p>
      <h2>A compact regression workflow</h2>
      <p>
        The example below fits the regression workflow and returns calibrated intervals.
      </p>
    </div>
    <div class="single-code-card">
      <pre class="landing-code"><code>import numpy as np
from selfcalibratingconformal import SelfCalibratingConformalPredictor

class MeanPredictor:
    def predict(self, x):
        x = np.asarray(x)
        return 1.5 * x[:, 0]

model = SelfCalibratingConformalPredictor(MeanPredictor())
model.fit(X_cal, y_cal, alpha=0.1)
point_predictions = model.predict_point(X_test)
intervals = model.predict_interval(X_test)
coverage, width = model.check_coverage(X_test, y_test)</code></pre>
    </div>
  </section>

  <section class="landing-section" id="resources">
    <div class="section-heading">
      <p class="landing-eyebrow">Resources</p>
      <h2>Documentation and papers</h2>
      <p>
        Use these links for notebooks, docs, and the main papers.
      </p>
    </div>

    <div class="resource-grid">
      <article class="resource-card">
        <p class="mini-label">Paper PDF</p>
        <h3>Self-Calibrating Conformal Prediction</h3>
        <p>Main paper for the self-calibrating conformal prediction workflow.</p>
        <a href="https://proceedings.neurips.cc/paper_files/paper/2024/file/c1c49aba08e6c90f2b1f85751f497a2f-Paper-Conference.pdf">Open PDF</a>
      </article>
      <article class="resource-card">
        <p class="mini-label">Paper PDF</p>
        <h3>Generalized Venn and Venn-Abers Calibration</h3>
        <p>Companion paper covering generalized Venn and Venn-Abers calibration.</p>
        <a href="https://openreview.net/pdf?id=kl2SA1N03E">Open PDF</a>
      </article>
      <article class="resource-card">
        <p class="mini-label">Notebook</p>
        <h3>Regression quickstart</h3>
        <p>Example of the regression workflow.</p>
        <a href="https://github.com/Larsvanderlaan/SelfCalibratingConformal/blob/master/quickstart_regression.ipynb">Open notebook</a>
      </article>
      <article class="resource-card">
        <p class="mini-label">Notebook</p>
        <h3>Quantile conformal quickstart</h3>
        <p>Example of the score-threshold workflow.</p>
        <a href="https://github.com/Larsvanderlaan/SelfCalibratingConformal/blob/master/quickstart_quantile_cp.ipynb">Open notebook</a>
      </article>
      <article class="resource-card">
        <p class="mini-label">Notebook</p>
        <h3>Advanced customization</h3>
        <p>Custom scores, calibrators, and solver settings.</p>
        <a href="https://github.com/Larsvanderlaan/SelfCalibratingConformal/blob/master/advanced_customization.ipynb">Open notebook</a>
      </article>
      <article class="resource-card">
        <p class="mini-label">Docs</p>
        <h3>API reference</h3>
        <p>Public classes and config options.</p>
        <a href="./api/">Open API page</a>
      </article>
      <article class="resource-card">
        <p class="mini-label">Docs</p>
        <h3>Guides</h3>
        <p>Short descriptions of the regression and quantile workflows.</p>
        <a href="./guides/">Open guides</a>
      </article>
      <article class="resource-card">
        <p class="mini-label">Repository</p>
        <h3>README and source</h3>
        <p>Installation details and source code.</p>
        <a href="https://github.com/Larsvanderlaan/SelfCalibratingConformal">Open repository</a>
      </article>
    </div>
  </section>

  <section class="landing-section landing-section-last" id="references">
    <div class="section-heading">
      <p class="landing-eyebrow">References</p>
      <h2>Core background</h2>
      <p>
        The package draws on Venn-Abers prediction and conformalized quantile regression.
      </p>
    </div>

    <ul class="reference-list">
      <li>van der Laan, L. and Alaa, A. M. (2024). <em>Self-Calibrating Conformal Prediction</em>.</li>
      <li>van der Laan, L. and Alaa, A. M. (2025). <em>Generalized Venn and Venn-Abers Calibration with Applications in Conformal Prediction</em>.</li>
      <li>Vovk, V., Petej, I., and Fedorova, V. (2015). <em>Large-scale probabilistic predictors with and without guarantees of validity</em>.</li>
      <li>Angelopoulos, A. N. and Bates, S. (2023). <em>Conformal prediction: A gentle introduction</em>.</li>
      <li>Romano, Y., Patterson, E., and Candes, E. (2019). <em>Conformalized quantile regression</em>.</li>
    </ul>
  </section>
</div>
