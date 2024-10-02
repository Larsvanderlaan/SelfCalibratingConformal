# Reviewer #1



## Response to Weaknesses

1. Thank you for this suggestions. In the revised paper, we will include additional real data experiments, specifically the "bike," "bio,", "star", "concrete", and "community" datasets used in CQR reference. 

3. In the revised real experiments, we will include CQR as a feature-conditional baseline to investigate for which datasets prediction-conditional validity provides an adequate (or poor) approximation of feature-conditional validity.

## Response to Questions

1. Thank you for this question. In the revised version of the paper, rather than approximating the algorithm via discretization, we instead propose running the algorithm exactly using a discretized outcome and a discretized model. The outcome can be discretized by binning the outcome space into, say, 200 bins, and the model can be discretized similarly. In this case, the algorithm can be computed exactly, and the coverage and calibration guarantees are with respect to the discretized outcome. Discretizing the model output into 100-200 bins is generally sufficient to preserve predictive performance, especially isotonic regression/calibration already bins the model output. Discretizing the outcome allows the user to directly control how much the discretized outcome can deviate from the true outcome. We also discuss how to increase the width of the prediction interval by the outcome approximation error to guarantee coverage for the true outcome.▍

2. We have added a discussion on model choice to the problem setup and experiment section. Our method, like most conformal prediction methods, provides tighter prediction intervals as the model's predictiveness (e.g., MSE) improves. In our algorithm, isotonic regression learns an optimal monotone transformation of the original predictor (Theorem 4.3) and, therefore, can asymptotically only improve the MSE of the original predictor. However, if the original predictor is poorly predictive, the calibrated predictor, albeit calibrated, will typically also not be predictive. In addition, prediction-conditional validity is more useful the more predictive the model is. For example, if the predictor is constant, then prediction-conditional validity reduces to marginal validity.▍

## Response to Limitations.

1. We have moved our discussion of the limitations to its own subsection in Section 4. This paragraph includes all the limitations discussed in the checklist, as well as other limitations mentioned throughout the submitted version of the paper.

# Reviewer #2 

## Weaknesses

1. **Weakness:** One of the main disadvantages of the proposed approach is the computational complexity... it may not offer significant advantage over other existing methods.

   **Response:** While our method has greater computational complexity than the split-CP approach for marginal validity, it is faster compared to methods like full CP and the conditional split CP approach by Gibbs et al. (2023). It is scalable to large datasets, as isotonic regression can be computed with XGBoost. In our experiments with calibration datasets (n=10000-50000), the computational time ranged from 1 to 5 minutes. Given that training the initial/uncalibrated model can take much longer, we believe the computational complexity of our method is not a significant weakness.

    Advantages of our method over existing methods are discussed in Section 4.1 and shown empirically in Section 5. Mondrian-CP approaches can only provide prediction-conditional validity and do not achieve our objective of self-calibration, which offers both calibrated point predictions and prediction-conditional validity. Another limitation of Mondrian-CP is the need for pre-specification of a binning scheme for the predictor f(·), introducing a trade-off between model performance and prediction interval width. In contrast, SC-CP data-adaptively discretizes the predictor f(·) using isotonic calibration, providing calibrated predictions, improved conformity scores, and self-calibrated intervals.

## Questions

1. What is the meaning of the subscript of $f_{n+1}$ in the desiderata (line 90)?
   **Response:** The subscript indicates that $f_{n+1}$ is obtained by calibrating the original model $f$ using the calibration data $\{(X_i,Y_i)\}_{i=1}^n$ and the new context $X_{n+1}$. We have clarified in our objective how $f_{n+1}$ is obtained and explained this notation in the revised paper.

2. ... Would it be better to highlight which methods perform best?

    **Response:** We have changed the coloring in the figure to highlight which method performs best.

3. "Is there a reason you are not comparing to the method in [42]?"

    **Response:** We did not include the method of [42] because their aim is to construct cumulative distribution function (CDF) estimates for a continuous outcome with marginal calibration guarantees. Our objective is to construct prediction intervals centered around a calibrated point prediction with prediction-conditional coverage guarantees. Inverting the CDF estimates from [42] yields quantile estimates and prediction intervals. However, these intervals are not centered around a point prediction, don't use conformity scores, and do not ensure prediction-conditional validity. While both approaches use Venn-Abers calibration, we use it to calibrate the regression model and employ conformal prediction techniques to construct prediction intervals. In contrast, they use Venn-Abers calibration with the indicator outcome 1(Y ≤ t) to construct calibrated CDF estimates.

4. "The example presented in the experiment section is interesting ... "

    **Response:** In the revised paper, we plan to include additional real data experiments, specifically the "bike," "bio,", "star", "concrete", and "community" datasets used in [1]. Our method generally will not fail to achieve self-calibration, barring failure of standard CP assumptions like exchangeability. However, there are scenarios where the prediction-conditional validity of our approach may lead to poorly adaptive prediction intervals relative to CP methods targeting feature-conditional validity.

In Appendix B.3, we use synthetic datasets to illustrate how prediction-conditional validity can approximate feature-conditional validity when the heteroscedasticity/variance of the outcomes is related to its conditional mean. In such cases, our approach offers narrow interval widths regardless of feature dimensions, leveraging model predictions as a scalar dimension reduction. We also show that in scenarios without a mean-variance relationship, our approach may provide poor feature-conditional coverage.

In the revised paper, we discuss these scenarios and point to the synthetic data experiments. To strengthen the real-data experiments, we will include the CQR method of [1] as a baseline for feature-conditional validity to investigate if prediction-conditional validity approximates feature-conditional validity in real data.

   [1] Romano, Yaniv, Evan Patterson, and Emmanuel Candes. "Conformalized quantile regression." Advances in neural information processing systems 32 (2019).

   [2] Gibbs, Isaac, John J. Cherian, and Emmanuel J. Candès. "Conformal prediction with conditional guarantees." arXiv preprint arXiv:2305.12616 (2023).

5. "Did your theoretical analysis offer any general insights..."

    **Response:** Our main contribution is integrating two areas of trustworthy machine learning: (1) calibration of model outputs and (2) uncertainty quantification via prediction intervals, proposing self-calibration and self-calibrating conformal prediction. Our theoretical analysis offers further insights and potential extensions. Our techniques can analyze conformal prediction methods that involve calibrating model predictions followed by constructing conditionally valid prediction intervals. One could apply a feature-conditional CP method with conformity scores and/or the conditioning variables depending on calibrated model predictions and derive feature-conditional validity guarantees using a straightforward modification of our arguments. We plan to explore generalizations of our procedure in future work and have added a concluding paragraph discussing these insights and extensions.

## Limitations

**Response:** We have moved our discussion of the limitations to its own subsection in Section 4. This paragraph includes all the limitations discussed in the checklist, as well as other limitations mentioned throughout the submitted version of the paper.
 






# Review 3

## Weaknesses and Responses

1.  We have added clarification on when our proposed self-calibration objective can approximate feature-conditional validity. Our self-calibration objective aims to provide a relaxation of feature-conditional validity that is feasible in finite samples (Section 2.3). A key benefit is that it involves conditioning on a one-dimensional variable, thus avoiding the curse of dimensionality (Section 2.2). Self-calibration is not a replacement for feature-conditional validity. Prediction-conditional validity can approximate it when the outcome's heteroscedasticity/variance is a function of its conditional mean. Appendix B.3 experimentally confirms this using synthetic data, and Section 5 provides additional evidence for this using real data that appears to have a mean-variance relationship.▍

2. We have redrafted the description for clarity and provided the algorithm before describing its qualitative properties. The revised sentences include:
   
    1. "Isotonic calibration can overfit, leading to poorly calibrated predictions. When this occurs, the Venn-Abers set prediction widens, reflecting greater (epistemic) uncertainty in the perfectly calibrated point prediction within the set."
    2. "The Venn-Abers calibration set is guaranteed to contain at least one perfectly calibrated point prediction in finite samples, and each prediction, being obtained via isotonic calibration, still enjoys the same large-sample calibration guarantees as isotonic calibration."

3. We clarified the role of calibration in the self-calibration objective, which aims to obtain prediction intervals that are centered around calibrated point predictions and provide valid coverage conditional on the calibrated point prediction. Point calibration ensures that the prediction interval is centered around a conditionally unbiased point prediction. Prediction-conditional validity is only attainable in finite samples for predictors with discrete outputs. Venn-Abers calibration discretizes the predictor, enabling prediction-conditional validity while mitigating the loss in predictive power due to discretization.

4.   Along with Mondrian-CP, we also included the kernel smoothing approach of Gibbs et al. (2023). Our baselines are sufficient to illustrate the benefit of our approach combining point calibration and prediction-conditional validity. The baselines for prediction-conditional involve estimating a one-dimensional quantile function, where Mondrian-CP uses histogram regression and Gibbs et al. use kernel smoothing. Kernel smoothing and histogram regression are minimax optimal for 1D functions under weak assumptions. Also see our response to Limitation #1.

## Response to Questions

1. See also our response to Weakness 1. On the difference between conditioning on \( Y \) and \( f(X) \):  \( Y \) is typically a noisy signal \( f_{true}(X) + \varepsilon \). Thus, two contexts \( X_1 \) and \( X_2 \) with \(Y_1 = Y_2\) can have very different signals \( f_{true}(X_1) \) and \( f_{true}(X_2) \). Outcome conditional validity is primarily useful in classification settings where the outcome \( Y \) is a ground-truth label and thus not subject to noise. We have added this discussion to related work.

2.  Yes, it is the set indicator. The covariate shift (CS) terminology for the multicalibration objective in (3) is from Gibbs et al. (2023). In equation (3), \( f \) is not the predictor but an arbitrary element of \(\mathcal{F}\). Thank you for pointing out this confusion in notation. We now denote the CS \( f \) in (3) by \( h \). The CS terminology arises in (3) because if \( h \) is a density ratio between a source and target distribution then multicalibration with respect to \( h \) in the source population implies marginal coverage with respect to the target distribution as well.

3. Prediction-conditional validity as a formal objective has not been proposed before. Some works have used bins based on predictions as an application of Mondrian CP to provide a form of prediction-conditional validity. Our contributions include extending Venn-Abers calibration to regression and proposing the self-calibration objective and our solution, combining prediction-conditional validity with the calibration of model outputs.

4.   \( f_n^{(x,y)} \) is defined in Algorithm 1. The subscript \( n \) indicates that the model depends on the data \(\mathcal{C}_n\), and the superscript \((x,y)\) indicates that the model also depends on the context \( x \) and the imputed outcome \( y \). In our notation for \( f_{n+1} \), we suppressed the dependence on the context \( x \). To avoid confusion and make the dependence on \( x \) explicit, we have now changed the notation from \( f_{n+1} \) to \( f_{n,x} \). We fixed a typo on line 153, where \( f_{n+1}(x) \) is now \( f_{n+1}(X_{n+1}) \).

5.  It is not possible to compute the marginal bands obtained using split CP around the calibrated predictor without sacrificing finite-sample validity, as Venn-Abers calibration is performed using the same data used to construct CP bands.  

## Response to limitations

1. We have clarified our choice of baselines, focusing on those targeting prediction-conditional validity as that is part of our self-calibration objective. We will add Conformal Quantile Regression as a feature-conditional baseline in the revised experiments. For direct comparison with our method, our baselines employed the commonly-used absolute residual error conformity score. While different conformity scores can improve feature-conditional validity, our primary goal is to show how calibrated point predictions and prediction-conditional validity enhance interval efficiency and adaptivity. Our method can be adapted to any conformity score, including the error reweighting score, to provide variance-adaptive prediction bands. We now discuss this in the conclusion and provide an explicit algorithm for the Error Reweighting modification in the Appendix.












 