import numpy as np

from src.calibration import BinaryCalibrator, evaluate_oof_calibration


def test_identity_calibrator_preserves_probabilities() -> None:
    probabilities = np.array([0.1, 0.5, 0.9])
    calibrated = BinaryCalibrator("uncalibrated").predict(probabilities)
    assert np.allclose(calibrated, probabilities)


def test_oof_calibration_is_deterministic_and_finite() -> None:
    targets = np.tile([0, 1], 20)
    probabilities = np.where(targets == 1, 0.65, 0.35)
    first, first_report, first_predictions = evaluate_oof_calibration(
        targets, probabilities, folds=5, seed=17
    )
    second, second_report, second_predictions = evaluate_oof_calibration(
        targets, probabilities, folds=5, seed=17
    )
    assert first == second
    assert first_report == second_report
    assert first_report["selected_method"] in {"uncalibrated", "temperature", "platt"}
    for method in first_predictions:
        assert np.allclose(first_predictions[method], second_predictions[method])
        assert np.isfinite(first_predictions[method]).all()


def test_oof_requires_enough_samples_per_class() -> None:
    try:
        evaluate_oof_calibration(np.array([0, 0, 1]), np.array([0.1, 0.2, 0.8]), folds=2)
    except ValueError as error:
        assert "each class" in str(error)
    else:
        raise AssertionError("expected calibration split validation to fail")
