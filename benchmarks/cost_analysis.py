def estimate_cost(model_name):
    if model_name == "Gemini":
        return {
            "cost_per_1M_tokens": 0.0,
            "note": "Free tier Gemini API (rate-limited)"
        }

    return {
        "cost_per_1M_tokens": 0.0,
        "note": "Open-source, self-hosted (CPU/GPU cost excluded)"
    }
