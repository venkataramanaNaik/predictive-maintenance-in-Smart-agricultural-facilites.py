import shap

import lime

import lime.lime_tabular

# SHAP explainability
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)

# LIME explainability

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train, 
    feature_names=['temperature', 'humidity', 'vibration', 'soilmoisture', 'timesincelastmaintenance'],
    class_names=['No Failure', 'Failure'], 
    discretize_continuous=True
)
lime_explanation = lime_explainer.explain_instance(X_test[0], xgb_model.predict, num_features=5)
lime_explanation.show_in_notebook()



# implementation SHAP $ LIME Explainability Integration models in XgBoost And LSTM model training on it

#   SHAP: global + local insights; shows how all features globally contribute to outcomes across the dataset. 
#   LIME: local insights; focuses on explaining an individual prediction.




  
