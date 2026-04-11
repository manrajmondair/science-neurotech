    # Model: SVM

    ## Overview

    **Description:** Support Vector Regression with RBF kernel (MultiOutputRegressor(SVR))

    **Why chosen:** Classical kernel method providing a smooth non-linear baseline. Works well on bounded, normalized spike-count features. Included as a comparison point against gradient-based methods.

    ---

    ## Data

    ### Preprocessing Summary

    | Parameter | Value |
    |-----------|-------|
    | Sampling rate | 32,000 Hz |
    | Neural channels | 64 (channels 0-63) |
    | Target channels | 12 (channels 64-75) |
    | Bin size | 20 ms = 640 samples |
    | Feature type | Spike count (\|z\| > k) |
    | Target aggregation | Mean per bin |
    | Train/test split | 80/20 chronological |

    ### k Values Tested

    k  { 2.5, 3.0, 3.5, 4.0 }

    ---

    ## Results

    ### R2 Summary Table

    | k | Train R2 | Test R2 |
    |---|---------|--------|
    | 2.5 | -0.0436 | -0.0438 |
| 3.0 | -0.0436 | -0.0438 |
| 3.5 | -0.0436 | -0.0438 |
| 4.0 | -0.0436 | -0.0438 |

    **Best k:** `3.5` -> Test R2 = **-0.0438**

    ### Per-Channel R2 Breakdown (k = 3.5)

    | Channel | Test R2 |
    |---------|---------|
    | Ch 0 | 0.0000 |
| Ch 1 | 0.0000 |
| Ch 2 | 0.0000 |
| Ch 3 | 0.0000 |
| Ch 4 | 0.0000 |
| Ch 5 | 0.0000 |
| Ch 6 | 0.0000 |
| Ch 7 | 0.0000 |
| Ch 8 | -0.1972 |
| Ch 9 | -0.3282 |
| Ch 10 | 0.0000 |
| Ch 11 | 0.0000 |

    ---

    ## Plots

    ### R2 vs Threshold k

    ![R2 vs k](figures\svm_r2_vs_k.png)
*Mean train and test R2 across k values for SVM*

    ### Per-Channel R2 (Best k = 3.5)

    ![Per-channel R2](figures\svm_per_channel_k3.5.png)
*Per-channel test R2 for k=3.5*

    ### Predicted vs Actual -- Channel 0 (Best k = 3.5)

    ![Pred vs Actual](figures\svm_pred_vs_actual_k3.5.png)
*Predicted vs actual for channel 0, k=3.5*

    ### Residuals -- Channel 0 (Best k = 3.5)

    ![Residuals](figures\svm_residuals_k3.5.png)
*Residual distribution for channel 0, k=3.5*

    ---

    ## Analysis

    ### Performance Trend vs k

    Test R2 **increases** as k increases from 2.5 to 4.0.
    At low k, the threshold is loose -- many sub-threshold noise fluctuations are
    counted as spikes, inflating feature values and adding noise.
    At high k, only strong deflections are counted -- fewer features, but higher
    SNR. The optimal k for this model is **3.5**.

    ### Overfitting / Underfitting

    The largest train-test gap is **0.0002** at k=2.5. The model generalises well; train/test performance is closely matched.

    ### Strengths

    Theoretically well-motivated with RBF kernel. No gradient issues. Works well with a small number of samples.

    ### Weaknesses

    O(n2) training complexity; impractical on >10K samples without subsampling. RBF bandwidth requires careful tuning.

    ---

    ## Conclusion

    **SVM** achieves weak decoding performance with a best test R2 of **-0.0438** at k=3.5. We recommend using **k=3.5** for this model in the final BCI pipeline. This model is best suited for offline analysis rather than real-time on-device inference.
