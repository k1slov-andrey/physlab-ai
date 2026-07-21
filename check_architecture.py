from __future__ import annotations

from core.lab_registry import list_labs
from core.recommendation_engine import build_feedback
from core.schemas import ModelPrediction


EXPECTED_LAB_IDS = {
    "cooling",
    "boyle_mariotte",
    "isochoric",
    "heat_balance",
}


def main() -> None:
    labs = list_labs()
    lab_ids = {lab.lab_id for lab in labs}
    if lab_ids != EXPECTED_LAB_IDS:
        missing = sorted(EXPECTED_LAB_IDS.difference(lab_ids))
        extra = sorted(lab_ids.difference(EXPECTED_LAB_IDS))
        raise SystemExit(f"Lab registry mismatch. Missing={missing}; extra={extra}")

    for lab in labs:
        if lab.implementation_status != "ml_ready":
            raise SystemExit(
                f"Lab '{lab.lab_id}' is not ready: {lab.implementation_status}"
            )
        print(f"{lab.lab_id}: {lab.short_title} [{lab.implementation_status}]")

    prediction = ModelPrediction(
        lab_id="boyle_mariotte",
        predicted_class="air_leak",
        confidence=0.91,
        probabilities={"air_leak": 0.91},
        features={"pv_slope": -0.012},
    )
    feedback = build_feedback(prediction)
    if not feedback.title or not feedback.recommended_action:
        raise SystemExit("Recommendation engine returned incomplete feedback")
    print(f"Recommendation engine: {feedback.title}")


if __name__ == "__main__":
    main()
