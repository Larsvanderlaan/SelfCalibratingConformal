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
        <p class="landing-eyebrow">Calibrated Regression and Conformal Prediction</p>
        <div class="hero-head">
          <h1>Self-Calibrating Conformal for calibrated regression and conformal intervals</h1>
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
          prediction to black-box regression models. Start from any predictor, calibrate it with
          Venn-Abers style machinery, and return calibrated predictions or intervals through a
          compact Python interface.
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
          <li>Post-hoc calibration around black-box predictors</li>
          <li>Regression and quantile-score conformal workflows</li>
          <li>Backward-compatible API with typed configs</li>
        </ul>
      </section>

      <aside class="hero-panel">
        <div class="landing-card landing-card-gradient">
          <p class="landing-card-label">At a glance</p>
          <div class="stat-grid">
            <div class="stat-item">
              <strong>Core use case</strong>
              <span>Calibrate predictions from an existing regression model without retraining the learner.</span>
            </div>
            <div class="stat-item">
              <strong>Two entrypoints</strong>
              <span>Choose between direct regression calibration and a conformity-score quantile workflow.</span>
            </div>
            <div class="stat-item">
              <strong>Outputs</strong>
              <span>Return calibrated predictions, Venn-Abers style sets, intervals, and coverage diagnostics.</span>
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
        and uncertainty quantification. It wraps the fitted learner with post-hoc calibration and
        conformal logic, so the original model can remain a black box.
      </p>
    </div>

    <div class="overview-grid">
      <article class="landing-card">
        <p class="mini-label">Regression calibration</p>
        <h3>Calibrate point predictions after the model is trained</h3>
        <p>
          Start here when you have a point predictor for <code>y</code> and want calibrated
          predictions together with intervals induced by that calibrated scale.
        </p>
      </article>
      <article class="landing-card">
        <p class="mini-label">Quantile-score calibration</p>
        <h3>Calibrate a learned score threshold for conformal prediction</h3>
        <p>
          Use this path when you can predict the <code>(1 - alpha)</code> quantile of a conformity
          score and want intervals built from a calibrated threshold.
        </p>
      </article>
      <article class="landing-card">
        <p class="mini-label">Extensibility</p>
        <h3>Keep the learner fixed and swap the calibration pieces</h3>
        <p>
          Predictors can be callables or objects with <code>.predict(...)</code>, and both
          workflows expose hooks for custom calibrators, conformity scores, and typed algorithm
          configs.
        </p>
      </article>
    </div>
  </section>

  <section class="landing-section" id="workflows">
    <div class="section-heading">
      <p class="landing-eyebrow">Workflows</p>
      <h2>Two tight paths, depending on what your model gives you</h2>
      <p>
        The decision is simple: start from a point predictor if your model estimates the response
        directly, or from a score-quantile predictor if your workflow is built around conformity
        scores.
      </p>
    </div>

    <div class="workflow-grid">
      <article class="workflow-card">
        <p class="mini-label">When to use it</p>
        <h3><code>SelfCalibratingConformalPredictor</code></h3>
        <p>
          Choose this when you want the shortest route from a regression model for <code>y</code>
          to calibrated predictions and intervals.
        </p>
        <pre class="landing-code"><code>from selfcalibratingconformal import SelfCalibratingConformalPredictor

model = SelfCalibratingConformalPredictor(predictor)
model.fit(X_cal, y_cal, alpha=0.1)
intervals = model.predict_interval(X_test)</code></pre>
      </article>

      <article class="workflow-card">
        <p class="mini-label">When to use it</p>
        <h3><code>VennAbersQuantileConformalPredictor</code></h3>
        <p>
          Choose this when your workflow estimates a conformity-score quantile and you want
          calibrated thresholds that translate into prediction intervals.
        </p>
        <pre class="landing-code"><code>from selfcalibratingconformal import VennAbersQuantileConformalPredictor

cp = VennAbersQuantileConformalPredictor(
    score_quantile_predictor=score_model,
    center_predictor=center_model,
    alpha=0.1,
)
cp.fit(X_cal, y_cal)</code></pre>
      </article>
    </div>
  </section>

  <section class="landing-section" id="why-calibration">
    <div class="section-heading">
      <p class="landing-eyebrow">Why Calibration</p>
      <h2>Predict, calibrate, then form intervals</h2>
      <p>
        Useful predictive structure is not the same as reliable scale. This package learns a
        one-dimensional correction on a calibration sample and uses that corrected scale to form
        predictions or interval thresholds.
      </p>
    </div>

    <div class="process-strip" role="img" aria-label="Workflow showing predict, calibrate, and form intervals.">
      <article class="process-step">
        <p class="process-label">Step 1</p>
        <h3>Predict</h3>
        <p>Start from a black-box model that outputs either a response prediction or a score quantile.</p>
      </article>
      <div class="process-connector" aria-hidden="true"></div>
      <article class="process-step process-step-accent">
        <p class="process-label">Step 2</p>
        <h3>Calibrate</h3>
        <p>Apply Venn-Abers style isotonic calibration to put predictions or score thresholds on the right empirical scale.</p>
      </article>
      <div class="process-connector" aria-hidden="true"></div>
      <article class="process-step process-step-warm">
        <p class="process-label">Step 3</p>
        <h3>Form intervals</h3>
        <p>Return calibrated outputs and check coverage on held-out data.</p>
      </article>
    </div>

    <div class="overview-grid">
      <article class="landing-card">
        <p class="mini-label">Direct regression path</p>
        <h3>Calibrate the response scale itself</h3>
        <p>
          This workflow calibrates point predictions directly and derives outputs from the corrected
          response scale.
        </p>
      </article>
      <article class="landing-card">
        <p class="mini-label">Score-threshold path</p>
        <h3>Calibrate the quantile of a conformity score</h3>
        <p>
          This workflow calibrates a learned score threshold and solves the score level set to
          obtain intervals.
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
        The direct regression path is the shortest route into the package. The example below fits
        on a calibration split and then produces calibrated intervals on a test set.
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
      <h2>Go deeper only when you need to</h2>
      <p>
        Use these links for notebooks, repository context, and the secondary reference material
        that remains in the docs.
      </p>
    </div>

    <div class="resource-grid">
      <article class="resource-card">
        <p class="mini-label">Notebook</p>
        <h3>Regression quickstart</h3>
        <p>Start with the direct regression workflow.</p>
        <a href="https://github.com/Larsvanderlaan/SelfCalibratingConformal/blob/master/quickstart_regression.ipynb">Open notebook</a>
      </article>
      <article class="resource-card">
        <p class="mini-label">Notebook</p>
        <h3>Quantile conformal quickstart</h3>
        <p>See the score-quantile conformal workflow.</p>
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
        <p>Review the short workflow guide.</p>
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
        These references cover the main ingredients behind the package.
      </p>
    </div>

    <ul class="reference-list">
      <li>Vovk, V., Petej, I., and Fedorova, V. (2015). <em>Large-scale probabilistic predictors with and without guarantees of validity</em>.</li>
      <li>Angelopoulos, A. N. and Bates, S. (2023). <em>Conformal prediction: A gentle introduction</em>.</li>
      <li>Romano, Y., Patterson, E., and Candes, E. (2019). <em>Conformalized quantile regression</em>.</li>
    </ul>
  </section>
</div>
