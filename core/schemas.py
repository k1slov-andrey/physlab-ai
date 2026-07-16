from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LabConfig:
    """Описание одной лабораторной работы."""

    lab_id: str
    title: str
    short_title: str
    description: str
    physics_model: str
    input_columns: tuple[str, ...]
    classes: tuple[str, ...]
    educational_goal: str
    implementation_status: str


@dataclass
class ModelPrediction:
    """Унифицированный результат работы ML-модели."""

    lab_id: str
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    features: dict[str, float] = field(default_factory=dict)


@dataclass
class ResearchFeedback:
    """Педагогическая интерпретация результата модели."""

    title: str
    explanation: str
    evidence: list[str]
    student_question: str
    recommended_action: str
    teacher_note: str
    requires_repeat: bool


@dataclass
class CompetencyScores:
    """Результаты исследовательских действий ученика по шкале 0–3."""

    research_question: float = 0.0
    hypothesis: float = 0.0
    experiment_planning: float = 0.0
    equipment_usage: float = 0.0
    data_analysis: float = 0.0
    physics_interpretation: float = 0.0
    conclusion_argumentation: float = 0.0
    critical_ai_usage: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "Исследовательский вопрос": self.research_question,
            "Формулирование гипотезы": self.hypothesis,
            "Планирование эксперимента": self.experiment_planning,
            "Работа с оборудованием": self.equipment_usage,
            "Анализ данных": self.data_analysis,
            "Физическая интерпретация": self.physics_interpretation,
            "Аргументация вывода": self.conclusion_argumentation,
            "Критическая работа с ИИ": self.critical_ai_usage,
        }


@dataclass
class LabAnalysis:
    """Полный результат прохождения лабораторного модуля."""

    prediction: ModelPrediction
    feedback: ResearchFeedback
    competencies: CompetencyScores
    metadata: dict[str, Any] = field(default_factory=dict)