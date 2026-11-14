import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.supplylens.training import train_currency_model


def main():
    parser = argparse.ArgumentParser(description="Train a MYR SupplyLens V2.1 CNN-LSTM and benchmark-selection bundle.")
    parser.add_argument("--currency", default="USD")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-arima", action="store_true",
                        help="Skip ARIMA during selection, calibration and final testing.")
    parser.add_argument("--calibration-stride", type=int, default=5,
                        help="Use every Nth calibration origin to limit benchmark fitting time.")
    args = parser.parse_args()

    result = train_currency_model(
        currency=args.currency,
        epochs=args.epochs,
        batch_size=args.batch_size,
        force_download=args.force_download,
        include_arima=not args.skip_arima,
        calibration_stride=args.calibration_stride,
    )
    print("\nTraining complete:", result["model_path"])
    print("\nValidation-selected method policy:")
    print(result["selection_policy"].to_string(index=False))
    print("\nUntouched final-test benchmark metrics:")
    print(result["metrics"].to_string(index=False))
    print("\nMethod-specific calibration sample counts:")
    print(result["calibration"].groupby("model")["calibration_samples"].max().to_string())


if __name__ == "__main__":
    main()
