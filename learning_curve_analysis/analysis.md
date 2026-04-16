# Learning Curve Diagnostic Analysis

**Metric Selection Justification**  
Due to the dataset's class imbalance (where churners are a minority), accuracy is an unhelpful metric because a naive baseline could achieve high accuracy simply by predicting the majority class. Instead, F1-score (which balances precision and recall) was chosen as the scoring metric since it provides a more truthful representation of the model's ability to actually detect churning customers. 

**Diagnostic Interpretations**

Based on the shape of the generated learning curves, the **Logistic Regression model is primarily suffering from high bias (underfitting)**. Early on, with less than 200 training examples, there is a visible gap between training and validation performance. However, as the number of data points increases towards 1,200, the training score rapidly drops and both the training and validation F1-scores tightly converge around ~0.35–0.37. This narrow gap, combined with a relatively low absolute performance ceiling, demonstrates that the linear model lacks the capacity to capture the true underlying complexity of the telecom churn data.

Because the model has reached a high bias plateau, **collecting more data would not yield meaningful improvements in validation performance.** The validation score essentially flattening out after ~660 training examples provides strong evidence that the model has already learned all the generalized patterns it can from the features provided; adding rows will only extend the flat plateau. 

Conversely, **increasing model complexity is highly likely to help.** Since a high bias issue suggests the model's restrictions are too rigid (a straight linear decision boundary), injecting flexibility will allow the model to learn closer to the training data. The **recommended next steps** involve replacing Logistic Regression with more sophisticated, nonlinear models capable of higher capacity, such as Random Forest or Gradient Boosting (e.g., XGBoost). If retaining Logistic Regression is a requirement, one should engineer polynomial and interaction features to grant the linear model a more flexible, pseudo-nonlinear decision space.
