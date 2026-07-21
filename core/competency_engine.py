from __future__ import annotations

from core.schemas import CompetencyScores, ModelPrediction


VALID_DECISIONS = {
    "Повторить измерение",
    "Проверить оборудование",
    "Пересчитать и проверить единицы",
    "Принять данные и обосновать вывод",
}


def _text_score(text: str, medium_length: int, high_length: int) -> float:
    length = len(text.strip())
    if length >= high_length:
        return 3.0
    if length >= medium_length:
        return 2.0
    if length > 0:
        return 1.0
    return 0.0


def score_competencies(
    prediction: ModelPrediction,
    hypothesis_text: str,
    student_decision: str,
    conclusion_text: str,
) -> CompetencyScores:
    hypothesis_score = _text_score(hypothesis_text, 15, 40)
    conclusion_score = _text_score(conclusion_text, 30, 70)
    decision_score = 3.0 if student_decision in VALID_DECISIONS else 1.0

    if not prediction.accepted:
        data_score = 1.0
    elif prediction.confidence >= 0.85:
        data_score = 3.0
    elif prediction.confidence >= 0.65:
        data_score = 2.0
    else:
        data_score = 1.0

    return CompetencyScores(
        research_question=2.0,
        hypothesis=hypothesis_score,
        experiment_planning=decision_score,
        equipment_usage=2.0,
        data_analysis=data_score,
        physics_interpretation=conclusion_score,
        conclusion_argumentation=conclusion_score,
        critical_ai_usage=decision_score,
    )
