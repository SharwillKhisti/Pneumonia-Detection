def explain_naturally(prediction, confidence, model_name, attention_focus="central lung fields"):
    """
    Generate a human-readable explanation based on model prediction and attention.

    Args:
        prediction (str): "Normal" or "Pneumonia"
        confidence (float): Prediction confidence score (0.0 to 1.0)
        model_name (str): Identifier of the model used
        attention_focus (str): Region where Grad-CAM/IG showed highest activation

    Returns:
        str: Natural language explanation
    """
    confidence_pct = f"{confidence * 100:.2f}%"

    if prediction == "Pneumonia":
        return f"""
        🫁 **Result:** The model **{model_name}** predicts **Pneumonia**  
        🔒 **Confidence:** {confidence_pct}

        🔍 **Model Attention:**  
        The explanation maps highlight strong focus in the **{attention_focus}**,  
        which often correspond to areas where pulmonary infections appear  
        (e.g., opacities, patchy densities, or asymmetrical textures).

        🧠 **Interpretation:**  
        The model likely detected abnormal density patterns or disrupted lung symmetry  
        that are consistent with pneumonia indicators.

        📌 **Note:** This is an AI-assisted interpretation. A radiologist’s review  
        is recommended for a final clinical decision.
        """
    else:
        return f"""
        🫁 **Result:** The model **{model_name}** predicts **Normal**  
        🔒 **Confidence:** {confidence_pct}

        🔍 **Model Attention:**  
        The explanation maps show low or diffuse activation across the lungs,  
        suggesting no significant anomalies in the **{attention_focus}**.

        🧠 **Interpretation:**  
        The lung fields appear balanced, with clear costophrenic angles and  
        no signs of consolidation or abnormal texture.

        📌 **Note:** While the model suggests normal findings, this does not  
        replace expert medical review. Clinical correlation is advised.
        """
