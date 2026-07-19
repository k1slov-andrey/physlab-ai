from core.schemas import CompetencyScores, ModelPrediction

def score_competencies(prediction: ModelPrediction, hypothesis_text: str, student_decision: str, conclusion_text: str) -> CompetencyScores:
    hyp=3.0 if len(hypothesis_text.strip())>=40 else 2.0 if len(hypothesis_text.strip())>=15 else 1.0 if hypothesis_text.strip() else 0.0
    conclusion=3.0 if len(conclusion_text.strip())>=70 else 2.0 if len(conclusion_text.strip())>=30 else 1.0 if conclusion_text.strip() else 0.0
    decision=3.0 if student_decision in {"Повторить измерение","Проверить оборудование","Пересчитать и проверить единицы","Принять данные и обосновать вывод"} else 1.0
    data_score=3.0 if prediction.confidence>=0.85 else 2.0 if prediction.confidence>=0.65 else 1.0
    return CompetencyScores(
        research_question=2.0,
        hypothesis=hyp,
        experiment_planning=decision,
        equipment_usage=2.0,
        data_analysis=data_score,
        physics_interpretation=conclusion,
        conclusion_argumentation=conclusion,
        critical_ai_usage=decision,
    )
