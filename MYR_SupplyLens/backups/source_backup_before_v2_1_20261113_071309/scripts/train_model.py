import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.supplylens.training import train_currency_model


def main():
    parser = argparse.ArgumentParser(description="Train a MYR SupplyLens V2 CNN-LSTM model.")
    parser.add_argument("--currency", default="USD")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-arima", action="store_true",
                        help="Skip the ARIMA benchmark if training time is constrained.")
    args = parser.parse_args()

    result = train_currency_model(
        currency=args.currency,
        epochs=args.epochs,
        batch_size=args.batch_size,
        force_download=args.force_download,
        include_arima=not args.skip_arima,
    )
    print("\nTraining complete:", result["model_path"])
    print("\nWalk-forward benchmark metrics:")
    print(result["metrics"].to_string(index=False))
    print("\nCalibration sample counts:")
    print(result["calibration"].tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
