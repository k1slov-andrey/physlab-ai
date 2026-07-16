from core.lab_registry import list_labs
from core.recommendation_engine import build_feedback
from core.schemas import ModelPrediction


def main() -> None:
    print("Доступные лабораторные работы:\n")

    for lab in list_labs():
        print(
            f"- {lab.lab_id}: {lab.short_title} "
            f"[{lab.implementation_status}]"
        )
        print(f"  Классы: {', '.join(lab.classes)}")
        print(f"  Цель: {lab.educational_goal}\n")

    demo_prediction = ModelPrediction(
        lab_id="boyle_mariotte",
        predicted_class="air_leak",
        confidence=0.91,
        probabilities={
            "normal": 0.03,
            "air_leak": 0.91,
            "temperature_change": 0.04,
            "volume_measurement_error": 0.02,
        },
        features={
            "pv_relative_change": -0.18,
            "pv_slope": -0.012,
            "temperature_range": 0.7,
        },
    )

    feedback = build_feedback(demo_prediction)

    print("Тест педагогической рекомендации:\n")
    print(f"Заголовок: {feedback.title}")
    print(f"Объяснение: {feedback.explanation}")
    print(f"Вопрос ученику: {feedback.student_question}")
    print(f"Действие: {feedback.recommended_action}")
    print(f"Повтор опыта: {feedback.requires_repeat}")


if __name__ == "__main__":
    main()