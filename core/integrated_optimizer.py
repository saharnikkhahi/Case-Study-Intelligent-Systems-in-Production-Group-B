import pandas as pd
import numpy as np
from pathlib import Path

from predict import ModelPredictor
from route_optimizer import RouteOptimizer, RouteReassignment, RouteOptimizationEvaluator


class IntegratedRouteOptimizer:
    def __init__(self, models_dir="outputs/models", preprocessor_dir="outputs/preprocessor"):
        self.predictor = ModelPredictor(models_dir, preprocessor_dir)
        self.predictor.load_all_models()
        
        self.optimizer = RouteOptimizer()
        self.reassignment = RouteReassignment(self.optimizer)
        self.evaluator = RouteOptimizationEvaluator()
        
        print("Integrated optimizer initialized with trained models")
    
    def predict_and_optimize(self, routes_df, time_limit=30):
        print("\n" + "=" * 80)
        print("STEP 1: PREDICTING DELAYS")
        print("=" * 80)
        
        predictions = self.predictor.predict_route_delays(routes_df)
        
        delay_rate = predictions['delayed_flag_pred'].mean()
        avg_delay = predictions['delay_minutes_pred'].mean()
        
        print(f"Predicted delay rate: {delay_rate:.2%}")
        print(f"Average predicted delay: {avg_delay:.2f} minutes")
        
        print("\n" + "=" * 80)
        print("STEP 2: OPTIMIZING ROUTES WITH DELAY PREDICTIONS")
        print("=" * 80)
        
        routes_with_pred = routes_df.copy()
        routes_with_pred = routes_with_pred.merge(
            predictions[['stop_id', 'delay_probability', 'delay_minutes_pred']],
            on='stop_id',
            how='left'
        )
        
        solution = self.optimizer.optimize_from_dataframe(
            routes_with_pred,
            predictor=self.predictor
        )
        
        if solution:
            print(f"Optimized routes using {solution['num_vehicles_used']} vehicles")
            print(f"Total distance: {solution['total_distance']:.2f}")
            print(f"Total time: {solution['total_time']:.2f} minutes")
        else:
            print("Optimization failed to find solution")
            return None
        
        print("\n" + "=" * 80)
        print("STEP 3: IDENTIFYING REASSIGNMENT OPPORTUNITIES")
        print("=" * 80)
        
        available_drivers = routes_df['driver_id'].unique().tolist()
        
        routes_reassigned, reassignments = self.reassignment.reassign_stops(
            routes_df,
            predictions,
            available_drivers
        )
        
        print(f"Identified {len(reassignments)} reassignment opportunities")
        
        if reassignments:
            top_reassignments = sorted(
                reassignments, 
                key=lambda x: x['expected_improvement'], 
                reverse=True
            )[:5]
            
            print("\nTop reassignments:")
            for r in top_reassignments:
                print(f"  Stop {r['stop_id']}: Expected improvement: {r['expected_improvement']:.2%}")
        
        print("\n" + "=" * 80)
        print("STEP 4: GENERATING REPORTS")
        print("=" * 80)
        
        self.evaluator.generate_report(solution, reassignments, routes_df)
        self.evaluator.save_optimized_routes(solution)
        
        self.save_predictions_and_reassignments(predictions, reassignments)
        
        return {
            'solution': solution,
            'predictions': predictions,
            'reassignments': reassignments,
            'routes_reassigned': routes_reassigned
        }
    
    def save_predictions_and_reassignments(self, predictions, reassignments):
        pred_path = self.evaluator.results_dir / "delay_predictions.csv"
        predictions.to_csv(pred_path, index=False)
        print(f"Predictions saved to {pred_path}")
        
        if reassignments:
            reassign_df = pd.DataFrame(reassignments)
            reassign_path = self.evaluator.results_dir / "reassignments.csv"
            reassign_df.to_csv(reassign_path, index=False)
            print(f"Reassignments saved to {reassign_path}")
    
    def real_time_adjustment(self, current_routes_df, live_delays_df):
        print("\n" + "=" * 80)
        print("REAL-TIME ROUTE ADJUSTMENT")
        print("=" * 80)
        
        print("Detecting actual delays in progress...")
        delayed_stops = live_delays_df[live_delays_df['actual_delay'] > 5]
        
        if len(delayed_stops) == 0:
            print("No significant delays detected")
            return current_routes_df
        
        print(f"Found {len(delayed_stops)} stops with delays > 5 minutes")
        
        remaining_stops = current_routes_df[
            ~current_routes_df['stop_id'].isin(delayed_stops['stop_id'])
        ]
        
        print(f"Re-optimizing {len(remaining_stops)} remaining stops...")
        
        solution = self.optimizer.optimize_from_dataframe(
            remaining_stops,
            predictor=self.predictor
        )
        
        if solution:
            print("Successfully re-optimized remaining routes")
            return solution
        else:
            print("Re-optimization failed, keeping current routes")
            return current_routes_df
    
    def batch_optimization(self, routes_df, batch_size=100):
        print("\n" + "=" * 80)
        print(f"BATCH OPTIMIZATION (batch size: {batch_size})")
        print("=" * 80)
        
        unique_routes = routes_df['route_id'].unique()
        num_batches = len(unique_routes) // batch_size + 1
        
        all_solutions = []
        all_reassignments = []
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(unique_routes))
            batch_routes = unique_routes[start_idx:end_idx]
            
            print(f"\nProcessing batch {i+1}/{num_batches} ({len(batch_routes)} routes)...")
            
            batch_df = routes_df[routes_df['route_id'].isin(batch_routes)]
            
            result = self.predict_and_optimize(batch_df, time_limit=15)
            
            if result:
                all_solutions.append(result['solution'])
                all_reassignments.extend(result['reassignments'])
        
        print("\n" + "=" * 80)
        print("BATCH OPTIMIZATION COMPLETE")
        print("=" * 80)
        print(f"Processed {num_batches} batches")
        print(f"Total reassignments: {len(all_reassignments)}")
        
        return {
            'solutions': all_solutions,
            'reassignments': all_reassignments
        }
    
    def compare_scenarios(self, routes_df):
        print("\n" + "=" * 80)
        print("SCENARIO COMPARISON")
        print("=" * 80)
        
        print("\nScenario 1: Current routes (no optimization)")
        original_distance = routes_df['distancep'].sum()
        original_routes = routes_df['route_id'].nunique()
        print(f"  Total distance: {original_distance:.2f}")
        print(f"  Number of routes: {original_routes}")
        
        print("\nScenario 2: Basic optimization (no delay predictions)")
        solution_basic = self.optimizer.optimize_from_dataframe(routes_df)
        if solution_basic:
            print(f"  Total distance: {solution_basic['total_distance']:.2f}")
            print(f"  Vehicles used: {solution_basic['num_vehicles_used']}")
            print(f"  Improvement: {(original_distance - solution_basic['total_distance']) / original_distance * 100:.2f}%")
        
        print("\nScenario 3: Delay-aware optimization")
        solution_smart = self.optimizer.optimize_from_dataframe(
            routes_df, 
            predictor=self.predictor
        )
        if solution_smart:
            print(f"  Total distance: {solution_smart['total_distance']:.2f}")
            print(f"  Vehicles used: {solution_smart['num_vehicles_used']}")
            print(f"  Improvement: {(original_distance - solution_smart['total_distance']) / original_distance * 100:.2f}%")
        
        return {
            'original': {'distance': original_distance, 'routes': original_routes},
            'basic_optimization': solution_basic,
            'delay_aware_optimization': solution_smart
        }


def main():
    print("\n" + "=" * 80)
    print("INTEGRATED ROUTE OPTIMIZATION SYSTEM")
    print("=" * 80)
    
    data_path = Path("data/cleaned_delivery_data.csv")
    
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return
    
    df = pd.read_csv(data_path)
    sample_df = df.head(200)
    
    print(f"\nLoaded {len(sample_df)} stops for optimization demo")
    
    integrated = IntegratedRouteOptimizer(
        models_dir="outputs/models",
        preprocessor_dir="outputs/preprocessor"
    )
    
    result = integrated.predict_and_optimize(sample_df, time_limit=30)
    
    if result:
        print("\n" + "=" * 80)
        print("OPTIMIZATION COMPLETE")
        print("=" * 80)
        print(f"\nCheck outputs/optimization/ for detailed results")


if __name__ == "__main__":
    main()

