from labs.boyle_mariotte.module import predict as predict_boyle
from labs.boyle_mariotte.module import simulate as simulate_boyle
from labs.cooling.module import predict as predict_cooling
from labs.cooling.module import simulate as simulate_cooling
from labs.heat_balance.module import predict as predict_heat_balance
from labs.heat_balance.module import simulate as simulate_heat_balance
from labs.isochoric.module import predict as predict_isochoric
from labs.isochoric.module import simulate as simulate_isochoric


CASES = [
    (
        "cooling",
        simulate_cooling,
        predict_cooling,
        {
            "normal": 1,
            "single_outlier": 1,
            "sensor_drift": 7,
            "high_noise": 1,
        },
    ),
    (
        "boyle_mariotte",
        simulate_boyle,
        predict_boyle,
        {
            "normal": 2,
            "air_leak": 1,
            "temperature_change": 1,
            "volume_measurement_error": 1,
        },
    ),
    (
        "isochoric",
        simulate_isochoric,
        predict_isochoric,
        {
            "normal": 1,
            "air_leak": 1,
            "volume_instability": 1,
            "temperature_sensor_lag": 1,
        },
    ),
    (
        "heat_balance",
        simulate_heat_balance,
        predict_heat_balance,
        {
            "normal": 3,
            "heat_loss": 1,
            "mass_measurement_error": 1,
            "insufficient_mixing": 1,
        },
    ),
]


def main() -> None:
    all_ok = True
    for lab_id, simulator, predictor, scenarios in CASES:
        print(f"\n{lab_id}")
        for class_name, seed in scenarios.items():
            result = predictor(simulator(class_name, seed=seed))
            is_correct = result.predicted_class == class_name
            all_ok = all_ok and is_correct
            status = "OK" if is_correct else "CHECK"
            print(
                f"[{status}] {class_name:28s} -> "
                f"{result.predicted_class:28s} "
                f"{result.confidence:.3f}"
            )

    if not all_ok:
        raise SystemExit("At least one control scenario was classified incorrectly")
    print("\nВсе контрольные сценарии распознаны корректно.")


if __name__ == "__main__":
    main()
