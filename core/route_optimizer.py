import numpy as np
import pandas as pd
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from pathlib import Path
import json
from datetime import datetime, timedelta


class RouteOptimizer:
    def __init__(self, num_vehicles=None, depot=0):
        self.num_vehicles = num_vehicles
        self.depot = depot
        self.distance_matrix = None
        self.time_matrix = None
        self.time_windows = None
        self.demands = None
        self.vehicle_capacities = None
        
    def create_distance_matrix(self, locations):
        n = len(locations)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    distance = np.linalg.norm(
                        np.array(locations[i]) - np.array(locations[j])
                    )
                    matrix[i][j] = distance
        
        return matrix
    
    def create_time_matrix_from_distances(self, distance_matrix, avg_speed=50):
        time_matrix = (distance_matrix / avg_speed) * 60
        return time_matrix.astype(int)
    
    def create_data_model(self, distance_matrix, time_windows, demands=None, 
                         vehicle_capacities=None, predicted_delays=None):
        data = {}
        data['distance_matrix'] = distance_matrix.tolist()
        data['time_matrix'] = self.create_time_matrix_from_distances(distance_matrix).tolist()
        data['time_windows'] = time_windows
        data['num_vehicles'] = self.num_vehicles or (len(distance_matrix) // 10)
        data['depot'] = self.depot
        
        if demands is not None:
            data['demands'] = demands
            data['vehicle_capacities'] = vehicle_capacities or [100] * data['num_vehicles']
        
        if predicted_delays is not None:
            data['predicted_delays'] = predicted_delays
            for i in range(len(data['time_matrix'])):
                for j in range(len(data['time_matrix'][i])):
                    if i < len(predicted_delays):
                        data['time_matrix'][i][j] += int(predicted_delays[i])
        
        return data
    
    def solve_vrp(self, data, time_limit=30):
        manager = pywrapcp.RoutingIndexManager(
            len(data['distance_matrix']),
            data['num_vehicles'],
            data['depot']
        )
        
        routing = pywrapcp.RoutingModel(manager)
        
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node]
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['time_matrix'][from_node][to_node]
        
        time_callback_index = routing.RegisterTransitCallback(time_callback)
        
        time = 'Time'
        routing.AddDimension(
            time_callback_index,
            30,
            3000,
            False,
            time
        )
        time_dimension = routing.GetDimensionOrDie(time)
        
        for location_idx, time_window in enumerate(data['time_windows']):
            if location_idx == data['depot']:
                continue
            index = manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
        
        depot_idx = data['depot']
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            time_dimension.CumulVar(index).SetRange(
                data['time_windows'][depot_idx][0],
                data['time_windows'][depot_idx][1]
            )
        
        for i in range(data['num_vehicles']):
            routing.AddVariableMinimizedByFinalizer(
                time_dimension.CumulVar(routing.Start(i))
            )
            routing.AddVariableMinimizedByFinalizer(
                time_dimension.CumulVar(routing.End(i))
            )
        
        if 'demands' in data:
            def demand_callback(from_index):
                from_node = manager.IndexToNode(from_index)
                return data['demands'][from_node]
            
            demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
            
            routing.AddDimensionWithVehicleCapacity(
                demand_callback_index,
                0,
                data['vehicle_capacities'],
                True,
                'Capacity'
            )
        
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = time_limit
        
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            return self.extract_solution(data, manager, routing, solution)
        else:
            return None
    
    def extract_solution(self, data, manager, routing, solution):
        time_dimension = routing.GetDimensionOrDie('Time')
        
        routes = []
        total_distance = 0
        total_time = 0
        
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            route = {
                'vehicle_id': vehicle_id,
                'stops': [],
                'distance': 0,
                'time': 0
            }
            
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                time_var = time_dimension.CumulVar(index)
                
                route['stops'].append({
                    'location': node,
                    'time': solution.Min(time_var)
                })
                
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route['distance'] += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id
                )
            
            node = manager.IndexToNode(index)
            time_var = time_dimension.CumulVar(index)
            route['stops'].append({
                'location': node,
                'time': solution.Min(time_var)
            })
            
            route['time'] = solution.Min(time_var)
            total_distance += route['distance']
            total_time += route['time']
            
            if len(route['stops']) > 2:
                routes.append(route)
        
        return {
            'routes': routes,
            'total_distance': total_distance,
            'total_time': total_time,
            'num_vehicles_used': len(routes)
        }
    
    def optimize_from_dataframe(self, df, predictor=None):
        locations = []
        time_windows = []
        stop_ids = []
        
        depot_location = (0, 0)
        locations.append(depot_location)
        time_windows.append((0, 1440))
        stop_ids.append(-1)
        
        for idx, row in df.iterrows():
            loc = (row.get('latitude', idx), row.get('longitude', idx * 0.1))
            locations.append(loc)
            
            earliest = int(row.get('earliest_time', 480) if pd.notna(row.get('earliest_time')) else 480)
            latest = int(row.get('latest_time', 1200) if pd.notna(row.get('latest_time')) else 1200)
            time_windows.append((earliest, latest))
            stop_ids.append(row.get('stop_id', idx))
        
        distance_matrix = self.create_distance_matrix(locations)
        
        predicted_delays = None
        if predictor is not None:
            try:
                delay_preds = predictor.predict_regression(df)
                predicted_delays = [0] + delay_preds.tolist()
            except:
                predicted_delays = None
        
        data = self.create_data_model(
            distance_matrix, 
            time_windows,
            predicted_delays=predicted_delays
        )
        
        solution = self.solve_vrp(data, time_limit=30)
        
        if solution:
            for route in solution['routes']:
                for stop in route['stops']:
                    if stop['location'] > 0:
                        stop['stop_id'] = stop_ids[stop['location']]
        
        return solution


class RouteReassignment:
    def __init__(self, optimizer=None):
        self.optimizer = optimizer or RouteOptimizer()
        
    def detect_delay_hotspots(self, predictions_df, threshold=0.7):
        hotspots = predictions_df[
            predictions_df['delay_probability'] > threshold
        ].copy()
        
        return hotspots
    
    def reassign_stops(self, current_routes, delay_predictions, available_drivers):
        problematic_stops = self.detect_delay_hotspots(delay_predictions)
        
        if len(problematic_stops) == 0:
            return current_routes, []
        
        reassignments = []
        
        for idx, stop in problematic_stops.iterrows():
            route_id = stop['route_id']
            stop_id = stop['stop_id']
            
            best_alternative = None
            min_impact = float('inf')
            
            for driver_id in available_drivers:
                if driver_id == stop.get('driver_id'):
                    continue
                
                impact = self.calculate_reassignment_impact(
                    stop, driver_id, current_routes
                )
                
                if impact < min_impact:
                    min_impact = impact
                    best_alternative = driver_id
            
            if best_alternative is not None and min_impact < stop['delay_probability']:
                reassignments.append({
                    'stop_id': stop_id,
                    'original_route': route_id,
                    'original_driver': stop.get('driver_id'),
                    'new_driver': best_alternative,
                    'reason': f'High delay probability: {stop["delay_probability"]:.2%}',
                    'expected_improvement': stop['delay_probability'] - min_impact
                })
        
        return self.apply_reassignments(current_routes, reassignments), reassignments
    
    def calculate_reassignment_impact(self, stop, new_driver, current_routes):
        base_delay_prob = stop.get('delay_probability', 0.5)
        
        driver_factor = np.random.uniform(0.8, 1.2)
        
        impact = base_delay_prob * driver_factor
        return impact
    
    def apply_reassignments(self, routes, reassignments):
        updated_routes = routes.copy() if isinstance(routes, pd.DataFrame) else pd.DataFrame(routes)
        
        for reassignment in reassignments:
            mask = updated_routes['stop_id'] == reassignment['stop_id']
            updated_routes.loc[mask, 'driver_id'] = reassignment['new_driver']
            updated_routes.loc[mask, 'reassigned'] = True
        
        return updated_routes
    
    def optimize_with_delays(self, df, predictor, time_limit=30):
        predictions = predictor.predict_route_delays(df)
        
        df_with_predictions = df.merge(
            predictions[['stop_id', 'delay_probability', 'delay_minutes_pred']],
            on='stop_id',
            how='left'
        )
        
        solution = self.optimizer.optimize_from_dataframe(
            df_with_predictions,
            predictor=predictor
        )
        
        return solution, predictions


class RouteOptimizationEvaluator:
    def __init__(self, results_dir="outputs/optimization"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_solution(self, solution, original_routes_df):
        metrics = {
            'num_vehicles_used': solution['num_vehicles_used'],
            'total_distance': solution['total_distance'],
            'total_time': solution['total_time'],
            'avg_distance_per_vehicle': solution['total_distance'] / solution['num_vehicles_used'],
            'avg_time_per_vehicle': solution['total_time'] / solution['num_vehicles_used']
        }
        
        original_metrics = {
            'num_routes': original_routes_df['route_id'].nunique(),
            'total_distance': original_routes_df['distancep'].sum()
        }
        
        comparison = {
            'distance_reduction': (
                original_metrics['total_distance'] - metrics['total_distance']
            ) / original_metrics['total_distance'] * 100,
            'vehicle_reduction': (
                original_metrics['num_routes'] - metrics['num_vehicles_used']
            )
        }
        
        return {
            'optimized': metrics,
            'original': original_metrics,
            'comparison': comparison
        }
    
    def evaluate_reassignments(self, reassignments, predictions_before, predictions_after):
        if len(reassignments) == 0:
            return {'message': 'No reassignments made'}
        
        metrics = {
            'num_reassignments': len(reassignments),
            'avg_improvement': np.mean([r['expected_improvement'] for r in reassignments]),
            'total_stops_reassigned': len(set([r['stop_id'] for r in reassignments]))
        }
        
        return metrics
    
    def generate_report(self, solution, reassignments, original_df):
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ROUTE OPTIMIZATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        evaluation = self.evaluate_solution(solution, original_df)
        
        report_lines.append("OPTIMIZED SOLUTION:")
        report_lines.append(f"  Vehicles used: {evaluation['optimized']['num_vehicles_used']}")
        report_lines.append(f"  Total distance: {evaluation['optimized']['total_distance']:.2f}")
        report_lines.append(f"  Total time: {evaluation['optimized']['total_time']:.2f} minutes")
        report_lines.append(f"  Avg distance/vehicle: {evaluation['optimized']['avg_distance_per_vehicle']:.2f}")
        report_lines.append(f"  Avg time/vehicle: {evaluation['optimized']['avg_time_per_vehicle']:.2f} minutes")
        
        report_lines.append("\nORIGINAL ROUTES:")
        report_lines.append(f"  Number of routes: {evaluation['original']['num_routes']}")
        report_lines.append(f"  Total distance: {evaluation['original']['total_distance']:.2f}")
        
        report_lines.append("\nIMPROVEMENTS:")
        report_lines.append(f"  Distance reduction: {evaluation['comparison']['distance_reduction']:.2f}%")
        report_lines.append(f"  Vehicle reduction: {evaluation['comparison']['vehicle_reduction']}")
        
        if reassignments:
            report_lines.append("\nREASSIGNMENTS:")
            report_lines.append(f"  Total reassignments: {len(reassignments)}")
            for r in reassignments[:10]:
                report_lines.append(
                    f"  Stop {r['stop_id']}: Driver {r['original_driver']} → {r['new_driver']} "
                    f"(improvement: {r['expected_improvement']:.2%})"
                )
            if len(reassignments) > 10:
                report_lines.append(f"  ... and {len(reassignments) - 10} more")
        
        report_lines.append("\n" + "=" * 80)
        
        report_text = "\n".join(report_lines)
        
        report_path = self.results_dir / "optimization_report.txt"
        with open(report_path, "w") as f:
            f.write(report_text)
        
        print(report_text)
        return report_text
    
    def save_optimized_routes(self, solution, filename="optimized_routes.json"):
        filepath = self.results_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(solution, f, indent=2)
        
        print(f"Optimized routes saved to {filepath}")

