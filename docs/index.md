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
          <h1>Calibrated regression and conformal intervals for black-box models</h1>
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
          prediction to black-box regression models. It supports two routes: calibrate a point
          prediction directly, or calibrate a conformity-score threshold and use that threshold to
          form an interval.
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
          <li>Intervals built from calibrated outputs</li>
          <li>Backward-compatible API with typed configs</li>
        </ul>
      </section>

      <aside class="hero-panel">
        <div class="landing-card landing-card-gradient">
          <p class="landing-card-label">At a glance</p>
          <div class="stat-grid">
            <div class="stat-item">
              <strong>Core use case</strong>
              <span>Start from an existing regression model and improve the scale used for reported predictions and intervals.</span>
            </div>
            <div class="stat-item">
              <strong>Two entrypoints</strong>
              <span>Choose between direct calibration of predicted responses and calibration of score thresholds.</span>
            </div>
            <div class="stat-item">
              <strong>Outputs</strong>
              <span>Return calibrated predictions, prediction intervals, and diagnostics for interval quality.</span>
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
        Use the package when you already have a useful regression model but need better calibration
        and interval construction. It wraps the fitted learner with post-hoc calibration and
        conformal logic, so the original model can remain a black box.
      </p>
    </div>

    <div class="overview-grid">
      <article class="landing-card">
        <p class="mini-label">Self-calibrating conformal</p>
        <h3>Calibrate the predicted response, then build an interval around it</h3>
        <p>
          This workflow starts with a point prediction for <code>y</code>. It learns a monotone
          correction from predicted values to observed outcomes on a calibration set, then builds
          intervals from the calibrated prediction.
        </p>
      </article>
      <article class="landing-card">
        <p class="mini-label">Quantile workflow</p>
        <h3>Predict how large the error can be, then calibrate that threshold</h3>
        <p>
          This workflow starts with a model for the <code>(1 - alpha)</code> quantile of a
          conformity score. The package calibrates that threshold and then includes candidate
          outcomes whose scores fall below the calibrated cutoff.
        </p>
      </article>
      <article class="landing-card">
        <p class="mini-label">Conformity score</p>
        <h3>Measure how compatible a candidate outcome is with the input</h3>
        <p>
          A conformity score is a number that is small when a candidate outcome looks plausible and
          large when it looks implausible. The default score is <code>|y - mu(x)|</code>, the
          absolute distance between the candidate outcome and a center prediction.
        </p>
      </article>
    </div>
  </section>

  <section class="landing-section" id="workflows">
    <div class="section-heading">
      <p class="landing-eyebrow">Workflows</p>
      <h2>Choose the workflow that matches your upstream model</h2>
      <p>
        Choose the regression path when you start from a point predictor, and the quantile path
        when you start from a score-threshold model.
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
          Start here when you have a point predictor for <code>y</code> and want calibrated
          predictions together with intervals derived from that calibrated scale.
        </p>

        <div class="method-visual-card">
          <div class="method-visual-copy">
            <span class="curve-kicker">When to use it</span>
            <p data-workflow-visual-note>
              Best for the shortest path from a fitted regression model to calibrated predictions
              and interval outputs.
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
              Calibrated point predictions, Venn-Abers style prediction sets, intervals, and
              empirical coverage summaries.
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
        The model provides a raw prediction or score threshold, calibration adjusts that quantity
        on held-out data, and the interval is built from the calibrated result.
      </p>
    </div>

    <div class="process-strip" role="img" aria-label="Workflow showing predict, calibrate, and form intervals.">
      <article class="process-step">
        <p class="process-label">Step 1</p>
        <h3>Predict</h3>
        <p>Fit or supply a model that outputs either a response prediction or a score threshold.</p>
      </article>
      <div class="process-connector" aria-hidden="true"></div>
      <article class="process-step process-step-accent">
        <p class="process-label">Step 2</p>
        <h3>Calibrate</h3>
        <p>Use a calibration set to learn a monotone correction so the reported scale better matches the data.</p>
      </article>
      <div class="process-connector" aria-hidden="true"></div>
      <article class="process-step process-step-warm">
        <p class="process-label">Step 3</p>
        <h3>Form intervals</h3>
        <p>Turn the calibrated prediction or threshold into an interval and check coverage empirically.</p>
      </article>
    </div>

    <div class="overview-grid">
      <article class="landing-card">
        <p class="mini-label">Direct regression path</p>
        <h3>How the self-calibrating path makes an interval</h3>
        <p>
          It calibrates the point prediction and then uses residual-based conformity scores to set
          the interval width around that calibrated center.
        </p>
      </article>
      <article class="landing-card">
        <p class="mini-label">Score-threshold path</p>
        <h3>How the quantile path makes an interval</h3>
        <p>
          It predicts a threshold for the conformity score, calibrates that threshold, and includes
          outcomes whose scores stay below the calibrated cutoff. With the default absolute-residual
          score, this gives a symmetric interval around the center prediction.
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
        The example below fits the regression workflow on a calibration split and then produces
        calibrated intervals on a test set.
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
        Use these links for notebooks, API documentation, and the main references behind the two
        workflows.
      </p>
    </div>

    <div class="resource-grid">
      <article class="resource-card">
        <p class="mini-label">Paper PDF</p>
        <h3>Inductive Venn-Abers predictive distribution</h3>
        <p>Primary reference for the regression-style Venn-Abers workflow.</p>
        <a href="http://proceedings.mlr.press/v91/nouretdinov18a/nouretdinov18a.pdf">Open PDF</a>
      </article>
      <article class="resource-card">
        <p class="mini-label">Paper PDF</p>
        <h3>Conformalized quantile regression</h3>
        <p>Primary reference for the quantile-based interval construction.</p>
        <a href="https://papers.nips.cc/paper_files/paper/2019/file/5103c3584b063c431bd12689b5e76fb-Conference.pdf">Open PDF</a>
      </article>
      <article class="resource-card">
        <p class="mini-label">Notebook</p>
        <h3>Regression quickstart</h3>
        <p>Example of the direct regression workflow.</p>
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
        <p>Review custom scores, custom calibrators, and solver configuration.</p>
        <a href="https://github.com/Larsvanderlaan/SelfCalibratingConformal/blob/master/advanced_customization.ipynb">Open notebook</a>
      </article>
      <article class="resource-card">
        <p class="mini-label">Docs</p>
        <h3>API reference</h3>
        <p>See the public class and config surface.</p>
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
        <p>Find installation details and repository context.</p>
        <a href="https://github.com/Larsvanderlaan/SelfCalibratingConformal">Open repository</a>
      </article>
    </div>
  </section>

  <section class="landing-section landing-section-last" id="references">
    <div class="section-heading">
      <p class="landing-eyebrow">References</p>
      <h2>Core background</h2>
      <p>
        The package draws on Venn-Abers predictive distributions and conformalized quantile
        regression.
      </p>
    </div>

    <ul class="reference-list">
      <li>Vovk, V., Petej, I., and Fedorova, V. (2015). <em>Large-scale probabilistic predictors with and without guarantees of validity</em>.</li>
      <li>Angelopoulos, A. N. and Bates, S. (2023). <em>Conformal prediction: A gentle introduction</em>.</li>
      <li>Romano, Y., Patterson, E., and Candes, E. (2019). <em>Conformalized quantile regression</em>.</li>
    </ul>
  </section>
</div>
