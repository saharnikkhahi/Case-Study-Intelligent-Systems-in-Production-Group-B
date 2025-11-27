import sys
from pathlib import Path

from train import ModelTrainer


def main():
    print("\n" + "=" * 80)
    print("AI-DRIVEN ROUTE OPTIMIZATION AND DELAY PREDICTION SYSTEM")
    print("=" * 80)
    
    data_path = Path("data/cleaned_delivery_data.csv")
    
    if not data_path.exists():
        print(f"\nError: Data file not found at {data_path}")
        print("Please ensure the cleaned data file exists.")
        sys.exit(1)
    
    output_dir = Path("outputs")
    
    trainer = ModelTrainer(str(data_path), output_dir=str(output_dir))
    
    trainer.preprocess_data()
    
    trainer.get_summary_statistics()
    
    trainer.train_all_models(
        lstm_epochs=50,
        lstm_batch_size=64,
        lstm_sequence_length=10
    )
    
    print("\n" + "=" * 80)
    print("MODEL TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nModel outputs available in: {output_dir.absolute()}")
    print("\nNext steps:")
    print("  1. Review evaluation results in outputs/results/")
    print("  2. Check feature importance plots")
    print("  3. Compare model performance metrics")
    print("  4. Run route optimization: python run_optimization.py")
    print("  5. Start API server: cd api && fastapi dev main.py")


if __name__ == "__main__":
    main()
