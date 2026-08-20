"""
Hypertension-specific safety filter
Replace the SafetyFilter in generation_layer.py with this
"""

from typing import Optional

class HypertensionSafetyFilter:
    """Pre-retrieval safety filter for hypertension queries."""

    OUT_OF_SCOPE_KEYWORDS = {
        # Conditions NOT related to hypertension management
        "covid", "coronavirus", "diabetes treatment", "cancer", "tumor",
        "oncology", "asthma", "copd", "respiratory", "pneumonia",
        "hip fracture", "osteoporosis", "bone", "arthritis",
        "pregnancy", "obstetric", "psychiatric", "depression",
        "heart attack", "myocardial infarction",  # Acute, not chronic HTN
        "stroke acute", "tia",  # Acute events, not HTN management
    }

    IN_SCOPE_KEYWORDS = {
        # Hypertension & blood pressure management
        "hypertension", "high blood pressure", "blood pressure",
        "systolic", "diastolic", "elevated bp", "stage 1 hypertension",
        "stage 2 hypertension", "hypertensive crisis",
        
        # Drug classes used in HTN
        "ace inhibitor", "acei", "lisinopril", "enalapril",
        "arb", "angiotensin receptor", "losartan", "valsartan",
        "calcium channel blocker", "ccb", "amlodipine", "diltiazem",
        "beta blocker", "metoprolol", "atenolol", "carvedilol",
        "thiazide", "hydrochlorothiazide", "hctz", "chlorthalidone",
        "diuretic", "spironolactone", "potassium sparing",
        "vasodilator", "minoxidil", "hydralazine",
        
        # Related conditions that affect HTN management
        "diabetes", "chronic kidney disease", "ckd", "egfr",
        "cardiovascular risk", "cvd", "coronary artery disease",
        "left ventricular hypertrophy", "lvh", "heart failure",
        "atrial fibrillation", "afib",
        
        # Management terms
        "blood pressure target", "bp target", "treatment goal",
        "antihypertensive", "lifestyle modification", "dash diet",
        "sodium restriction", "weight loss",
        "screening", "diagnosis", "monitoring", "follow-up",
    }

    @staticmethod
    def check(query: str) -> tuple[bool, Optional[str]]:
        """
        Returns (is_safe, refusal_reason_if_unsafe).
        True = safe to proceed; False = refuse.
        """
        query_lower = query.lower()

        # Check for explicit out-of-scope terms
        for keyword in HypertensionSafetyFilter.OUT_OF_SCOPE_KEYWORDS:
            if keyword in query_lower:
                return False, (
                    f"Your query mentions '{keyword}', which is outside the scope "
                    f"of our hypertension management guidelines. Our system covers "
                    f"blood pressure screening, diagnosis, and treatment based on "
                    f"ESC 2021 guidelines. Please consult appropriate clinical "
                    f"resources for other conditions."
                )

        # Require at least one in-scope keyword
        has_in_scope = any(kw in query_lower for kw in HypertensionSafetyFilter.IN_SCOPE_KEYWORDS)
        if not has_in_scope:
            return False, (
                "Your query does not appear to relate to hypertension management. "
                "Our system provides evidence-based recommendations for blood pressure "
                "screening, diagnosis, and treatment based on ESC 2021 guidelines. "
                "Please rephrase your question to focus on hypertension."
            )

        return True, None
