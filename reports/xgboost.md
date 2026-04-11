    # Model: XGBOOST

    ## Overview

    **Description:** XGBoost Gradient-Boosted Trees (MultiOutputRegressor, 100 estimators)

    **Why chosen:** Strong classical baseline for tabular spike-count data. Handles non-linear feature interactions without explicit engineering. One independent ensemble is trained per output channel.

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
    | 2.5 | 0.9778 | -0.0136 |
| 3.0 | 0.9697 | -0.0136 |
| 3.5 | 0.9653 | -0.0113 |
| 4.0 | 0.9678 | -0.0096 |

    **Best k:** `4.0` -> Test R2 = **-0.0096**

    ### Per-Channel R2 Breakdown (k = 4.0)

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
| Ch 8 | 0.0364 |
| Ch 9 | -0.1514 |
| Ch 10 | 0.0000 |
| Ch 11 | 0.0000 |

    ---

    ## Plots

    ### R2 vs Threshold k

    ![R2 vs k](figures\xgboost_r2_vs_k.png)
*Mean train and test R2 across k values for XGBOOST*

    ### Per-Channel R2 (Best k = 4.0)

    ![Per-channel R2](figures\xgboost_per_channel_k4.0.png)
*Per-channel test R2 for k=4.0*

    ### Predicted vs Actual -- Channel 0 (Best k = 4.0)

    ![Pred vs Actual](figures\xgboost_pred_vs_actual_k4.0.png)
*Predicted vs actual for channel 0, k=4.0*

    ### Residuals -- Channel 0 (Best k = 4.0)

    ![Residuals](figures\xgboost_residuals_k4.0.png)
*Residual distribution for channel 0, k=4.0*

    ---

    ## Analysis

    ### Performance Trend vs k

    Test R2 **increases** as k increases from 2.5 to 4.0.
    At low k, the threshold is loose -- many sub-threshold noise fluctuations are
    counted as spikes, inflating feature values and adding noise.
    At high k, only strong deflections are counted -- fewer features, but higher
    SNR. The optimal k for this model is **4.0**.

    ### Overfitting / Underfitting

    The largest train-test gap is **0.9914** at k=2.5. This suggests meaningful overfitting -- the model has memorised training patterns that don't generalise. Consider adding dropout, reducing model capacity, or collecting more data.

    ### Strengths

    Excellent on tabular spike-count data. Robust to outlier counts. No normalisation of inputs required. Interpretable via feature importance.

    ### Weaknesses

    Trains independent regressors per channel -- misses cross-channel correlations. Slow with large datasets.

    ---

    ## Conclusion

    **XGBOOST** achieves weak decoding performance with a best test R2 of **-0.0096** at k=4.0. We recommend using **k=4.0** for this model in the final BCI pipeline. This model is best suited for offline analysis rather than real-time on-device inference.
