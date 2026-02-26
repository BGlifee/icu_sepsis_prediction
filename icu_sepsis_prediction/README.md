✍️ Polished README Version (Professional Tone)
Why XGBoost for Time-Series Sepsis Prediction?

XGBoost was selected as the primary modeling approach for the following reasons:

Robust to Noise
ICU vital sign data are inherently noisy and irregular. Tree-based models such as XGBoost are resilient to outliers and nonlinear patterns compared to linear models.

Stable and Efficient Training
XGBoost provides strong regularization, shrinkage (learning rate), and early stopping, making it stable even with high-dimensional feature spaces derived from sliding windows.

Strong Performance on Imbalanced Data
Sepsis prediction is a highly imbalanced classification problem (approximately 26:1 ratio of negative to positive cases).
XGBoost handles imbalance effectively through:

scale_pos_weight

Gradient boosting framework

Robust split finding

Interpretability
Feature importance and SHAP values can be extracted to explain which physiological signals (e.g., HR variability, MAP trends) contribute most to risk predictions.