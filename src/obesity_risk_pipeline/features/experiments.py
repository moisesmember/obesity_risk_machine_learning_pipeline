"""Declarative ablation catalog used consistently by CV and inference."""

from __future__ import annotations

from dataclasses import dataclass

from obesity_risk_pipeline.data.modeling import BEHAVIORAL_FEATURES, MODEL_FEATURES


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """One governed feature/representation hypothesis."""

    name: str
    description: str
    source_features: tuple[str, ...]
    age_mode: str = "continuous"
    include_bmi: bool = False
    categorical_encoding: str = "nominal"
    proxy_note: str = ""

    @property
    def output_features(self) -> tuple[str, ...]:
        names = list(self.source_features)
        if self.include_bmi:
            names.append("BMI")
        if "Age" in names and self.age_mode != "continuous":
            names.remove("Age")
            names.append(
                "Age_completed" if self.age_mode == "completed" else "Age_group"
            )
        return tuple(names)


def build_experiment_catalog() -> dict[str, ExperimentSpec]:
    """Return required A-F ablations plus explicit comparison variants."""

    without_gender = tuple(name for name in MODEL_FEATURES if name != "Gender")
    without_weight = tuple(name for name in MODEL_FEATURES if name != "Weight")
    without_anthropometrics = tuple(
        name for name in MODEL_FEATURES if name not in {"Height", "Weight"}
    )
    specs = (
        ExperimentSpec(
            "A_full",
            "Todas as variáveis preditoras com Age contínua.",
            MODEL_FEATURES,
            proxy_note="Weight e Height podem reconstruir o estado corporal.",
        ),
        ExperimentSpec(
            "A_full_ordinal",
            "Modelo completo com CAEC e CALC ordinais.",
            MODEL_FEATURES,
            categorical_encoding="ordinal",
            proxy_note="Weight e Height permanecem disponíveis.",
        ),
        ExperimentSpec(
            "B_without_gender",
            "Ablação para medir dependência do atalho Gender.",
            without_gender,
            proxy_note="Weight e Height permanecem disponíveis.",
        ),
        ExperimentSpec(
            "C_without_weight",
            "Remove Weight e mantém Height para comparação controlada.",
            without_weight,
            proxy_note="Height isolada permanece como medida corporal.",
        ),
        ExperimentSpec(
            "C_without_weight_height",
            "Remove Weight e Height para auditar proxies corporais.",
            without_anthropometrics,
        ),
        ExperimentSpec(
            "D_behavioral",
            "Hábitos, atividade, água, tecnologia, transporte e histórico familiar.",
            BEHAVIORAL_FEATURES,
        ),
        ExperimentSpec(
            "E_body_bmi",
            "Baseline corporal com Weight, Height e BMI derivado.",
            ("Height", "Weight"),
            include_bmi=True,
            proxy_note="Experimento deliberadamente dominado por proxies do alvo.",
        ),
        ExperimentSpec(
            "F_age_continuous",
            "Modelo completo com Age contínua.",
            MODEL_FEATURES,
        ),
        ExperimentSpec(
            "F_age_completed",
            "Modelo completo com floor(Age), sem Age contínua.",
            MODEL_FEATURES,
            age_mode="completed",
        ),
        ExperimentSpec(
            "F_age_grouped",
            "Modelo completo com faixas etárias, sem Age contínua.",
            MODEL_FEATURES,
            age_mode="grouped",
        ),
    )
    return {spec.name: spec for spec in specs}
