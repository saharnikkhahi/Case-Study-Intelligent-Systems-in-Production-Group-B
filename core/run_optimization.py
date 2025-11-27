import sys
from pathlib import Path
import pandas as pd

from integrated_optimizer import IntegratedRouteOptimizer


def run_full_optimization():
    print("\n" + "=" * 80)
    print("FULL ROUTE OPTIMIZATION PIPELINE")
    print("=" * 80)
    
    data_path = Path("data/cleaned_delivery_data.csv")
    
    if not data_path.exists():
        print(f"\nError: Data file not found at {data_path}")
        print("Please run training first: python main.py")
        sys.exit(1)
    
    models_exist = (
        Path("outputs/models/random_forest_classifier.pkl").exists() and
        Path("outputs/preprocessor/scaler.pkl").exists()
    )
    
    if not models_exist:
        print("\nError: Trained models not found")
        print("Please run training first: python main.py")
        sys.exit(1)
    
    print("\nLoading data...")
    df = pd.read_csv(data_path)
    
    print(f"Total stops in dataset: {len(df)}")
    print(f"Total routes: {df['route_id'].nunique()}")
    
    sample_size = min(500, len(df))
    sample_df = df.head(sample_size)
    
    print(f"\nUsing sample of {sample_size} stops for optimization")
    print("(For production, process full dataset in batches)")
    
    print("\nInitializing integrated optimizer...")
    integrated = IntegratedRouteOptimizer(
        models_dir="outputs/models",
        preprocessor_dir="outputs/preprocessor"
    )
    
    print("\nRunning optimization pipeline...")
    result = integrated.predict_and_optimize(sample_df, time_limit=60)
    
    if result:
        print("\n" + "=" * 80)
        print("SUCCESS: Optimization Complete!")
        print("=" * 80)
        print("\nResults saved to:")
        print("  - outputs/optimization/optimization_report.txt")
        print("  - outputs/optimization/optimized_routes.json")
        print("  - outputs/optimization/delay_predictions.csv")
        print("  - outputs/optimization/reassignments.csv")
        
        print("\nKey Metrics:")
        print(f"  Vehicles used: {result['solution']['num_vehicles_used']}")
        print(f"  Total distance: {result['solution']['total_distance']:.2f}")
        print(f"  Reassignments: {len(result['reassignments'])}")
    else:
        print("\n" + "=" * 80)
        print("Optimization failed")
        print("=" * 80)
        sys.exit(1)


def run_scenario_comparison():
    print("\n" + "=" * 80)
    print("SCENARIO COMPARISON")
    print("=" * 80)
    
    data_path = Path("data/cleaned_delivery_data.csv")
    df = pd.read_csv(data_path)
    sample_df = df.head(300)
    
    integrated = IntegratedRouteOptimizer(
        models_dir="outputs/models",
        preprocessor_dir="outputs/preprocessor"
    )
    
    scenarios = integrated.compare_scenarios(sample_df)
    
    print("\n" + "=" * 80)
    print("SCENARIO COMPARISON COMPLETE")
    print("=" * 80)


def run_batch_optimization():
    print("\n" + "=" * 80)
    print("BATCH OPTIMIZATION")
    print("=" * 80)
    
    data_path = Path("data/cleaned_delivery_data.csv")
    df = pd.read_csv(data_path)
    
    integrated = IntegratedRouteOptimizer(
        models_dir="outputs/models",
        preprocessor_dir="outputs/preprocessor"
    )
    
    result = integrated.batch_optimization(df.head(1000), batch_size=200)
    
    print("\n" + "=" * 80)
    print("BATCH OPTIMIZATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Route Optimization Pipeline")
    parser.add_argument(
        '--mode',
        choices=['full', 'scenario', 'batch'],
        default='full',
        help='Optimization mode'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        run_full_optimization()
    elif args.mode == 'scenario':
        run_scenario_comparison()
    elif args.mode == 'batch':
        run_batch_optimization()

