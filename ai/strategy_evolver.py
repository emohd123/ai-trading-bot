"""
Strategy Evolver - Genetic Algorithm for Strategy Optimization
Evolves trading strategies over time to find optimal parameters
"""

import random
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import copy

from ai.backtester import Backtester, BacktestResult, save_backtest_result
import config


class Gene:
    """Represents a single parameter that can evolve"""
    
    def __init__(self, name: str, min_val: float, max_val: float, 
                 current: float = None, step: float = None, is_int: bool = False):
        self.name = name
        self.min_val = min_val
        self.max_val = max_val
        self.is_int = is_int
        self.step = step or (max_val - min_val) / 20
        
        if current is not None:
            self.value = current
        else:
            self.value = random.uniform(min_val, max_val)
            if is_int:
                self.value = int(self.value)
    
    def mutate(self, mutation_rate: float = 0.1):
        """Randomly mutate this gene"""
        if random.random() < mutation_rate:
            # Small mutation
            delta = random.gauss(0, self.step)
            self.value += delta
            self.value = max(self.min_val, min(self.max_val, self.value))
            if self.is_int:
                self.value = int(self.value)
    
    def crossover(self, other: 'Gene') -> 'Gene':
        """Create offspring gene from two parents"""
        if random.random() < 0.5:
            new_value = self.value
        else:
            new_value = other.value
        
        # Sometimes blend values
        if random.random() < 0.3:
            new_value = (self.value + other.value) / 2
        
        return Gene(self.name, self.min_val, self.max_val, 
                   new_value, self.step, self.is_int)


class Strategy:
    """Represents a complete trading strategy with evolvable genes"""
    
    def __init__(self, genes: Dict[str, Gene] = None):
        self.genes = genes or self._create_default_genes()
        self.fitness = 0
        self.backtest_result = None
        self.generation = 0
        self.id = f"strat_{datetime.now().strftime('%H%M%S')}_{random.randint(1000, 9999)}"
    
    def _create_default_genes(self) -> Dict[str, Gene]:
        """Create default gene set for trading strategy"""
        return {
            # Entry thresholds
            "BUY_THRESHOLD": Gene("BUY_THRESHOLD", 0.15, 0.60, 0.35),
            "SELL_THRESHOLD": Gene("SELL_THRESHOLD", -0.50, -0.10, -0.25),
            
            # Profit/Loss targets
            "PROFIT_TARGET": Gene("PROFIT_TARGET", 0.005, 0.03, 0.015),
            "STOP_LOSS": Gene("STOP_LOSS", 0.005, 0.025, 0.01),
            "MIN_PROFIT": Gene("MIN_PROFIT", 0.002, 0.015, 0.005),
            
            # Confluence requirements
            "MIN_CONFLUENCE_BUY": Gene("MIN_CONFLUENCE_BUY", 3, 8, 5, is_int=True),
            "MIN_CONFIDENCE_BUY": Gene("MIN_CONFIDENCE_BUY", 0.30, 0.70, 0.45),
            
            # Trailing stop
            "TRAILING_ACTIVATION": Gene("TRAILING_ACTIVATION", 0.005, 0.025, 0.01),
            
            # Indicator weights (will be normalized)
            "W_MOMENTUM": Gene("W_MOMENTUM", 0.05, 0.30, 0.18),
            "W_MACD": Gene("W_MACD", 0.05, 0.20, 0.12),
            "W_BOLLINGER": Gene("W_BOLLINGER", 0.05, 0.25, 0.15),
            "W_EMA": Gene("W_EMA", 0.02, 0.15, 0.05),
            "W_SR": Gene("W_SR", 0.05, 0.25, 0.15),
            "W_ML": Gene("W_ML", 0.02, 0.15, 0.08),
        }
    
    def to_params(self) -> Dict:
        """Convert genes to config parameters"""
        params = {}
        weights = {}
        
        for name, gene in self.genes.items():
            if name.startswith("W_"):
                # Collect indicator weights
                indicator_name = name[2:].lower()
                weight_map = {
                    "momentum": "momentum",
                    "macd": "macd", 
                    "bollinger": "bollinger",
                    "ema": "ema",
                    "sr": "support_resistance",
                    "ml": "ml_prediction"
                }
                if indicator_name in weight_map:
                    weights[weight_map[indicator_name]] = gene.value
            else:
                params[name] = gene.value
        
        # Normalize weights to sum to 1
        if weights:
            total = sum(weights.values())
            # Add remaining indicators with small weights
            remaining_weight = 0.27  # For ichimoku, mfi, williams_r, cci
            for ind in ["ichimoku", "mfi", "williams_r", "cci"]:
                weights[ind] = remaining_weight / 4
            
            total = sum(weights.values())
            weights = {k: v/total for k, v in weights.items()}
            params["INDICATOR_WEIGHTS"] = weights
        
        return params
    
    def mutate(self, mutation_rate: float = 0.15):
        """Mutate all genes"""
        for gene in self.genes.values():
            gene.mutate(mutation_rate)
    
    def crossover(self, other: 'Strategy') -> 'Strategy':
        """Create offspring strategy from two parents"""
        new_genes = {}
        for name in self.genes:
            new_genes[name] = self.genes[name].crossover(other.genes[name])
        
        child = Strategy(new_genes)
        child.generation = max(self.generation, other.generation) + 1
        return child
    
    def copy(self) -> 'Strategy':
        """Create a copy of this strategy"""
        new_genes = {name: Gene(g.name, g.min_val, g.max_val, g.value, g.step, g.is_int) 
                    for name, g in self.genes.items()}
        strategy = Strategy(new_genes)
        strategy.fitness = self.fitness
        strategy.generation = self.generation
        return strategy


class StrategyEvolver:
    """
    Genetic algorithm for evolving trading strategies
    """
    
    def __init__(self, population_size: int = 10):
        self.population_size = population_size
        self.population: List[Strategy] = []
        self.best_strategy: Optional[Strategy] = None
        self.generation = 0
        self.backtester = Backtester()
        
        # Evolution parameters
        self.mutation_rate = 0.15
        self.elite_count = 2  # Best strategies to keep unchanged
        self.tournament_size = 3
        
        # History
        self.evolution_history = []
        
    def initialize_population(self):
        """Create initial random population"""
        self.population = []
        
        # Add one strategy based on current config
        current_strategy = self._create_from_current_config()
        self.population.append(current_strategy)
        
        # Add random variations
        for _ in range(self.population_size - 1):
            strategy = Strategy()
            self.population.append(strategy)
        
        print(f"[EVOLVER] Initialized population with {len(self.population)} strategies")
    
    def _create_from_current_config(self) -> Strategy:
        """Create strategy from current config values"""
        import config
        
        genes = {
            "BUY_THRESHOLD": Gene("BUY_THRESHOLD", 0.15, 0.60, 
                                 getattr(config, 'BUY_THRESHOLD', 0.35)),
            "SELL_THRESHOLD": Gene("SELL_THRESHOLD", -0.50, -0.10,
                                  getattr(config, 'SELL_THRESHOLD', -0.25)),
            "PROFIT_TARGET": Gene("PROFIT_TARGET", 0.005, 0.03,
                                 getattr(config, 'PROFIT_TARGET', 0.015)),
            "STOP_LOSS": Gene("STOP_LOSS", 0.005, 0.025,
                             getattr(config, 'STOP_LOSS', 0.01)),
            "MIN_PROFIT": Gene("MIN_PROFIT", 0.002, 0.015,
                              getattr(config, 'MIN_PROFIT', 0.005)),
            "MIN_CONFLUENCE_BUY": Gene("MIN_CONFLUENCE_BUY", 3, 8,
                                      getattr(config, 'MIN_CONFLUENCE_BUY', 5), is_int=True),
            "MIN_CONFIDENCE_BUY": Gene("MIN_CONFIDENCE_BUY", 0.30, 0.70,
                                      getattr(config, 'MIN_CONFIDENCE_BUY', 0.45)),
            "TRAILING_ACTIVATION": Gene("TRAILING_ACTIVATION", 0.005, 0.025,
                                       getattr(config, 'TRAILING_ACTIVATION', 0.01)),
            "W_MOMENTUM": Gene("W_MOMENTUM", 0.05, 0.30, 0.18),
            "W_MACD": Gene("W_MACD", 0.05, 0.20, 0.12),
            "W_BOLLINGER": Gene("W_BOLLINGER", 0.05, 0.25, 0.15),
            "W_EMA": Gene("W_EMA", 0.02, 0.15, 0.05),
            "W_SR": Gene("W_SR", 0.05, 0.25, 0.15),
            "W_ML": Gene("W_ML", 0.02, 0.15, 0.08),
        }
        
        strategy = Strategy(genes)
        strategy.id = "current_config"
        return strategy
    
    def evaluate_population(self, days: int = 7):
        """Evaluate all strategies in population via backtesting"""
        print(f"[EVOLVER] Evaluating {len(self.population)} strategies...")
        
        for i, strategy in enumerate(self.population):
            if strategy.backtest_result is None:
                print(f"  Testing strategy {i+1}/{len(self.population)} ({strategy.id})...")
                params = strategy.to_params()
                result = self.backtester.run_backtest(strategy_params=params, days=days)
                strategy.backtest_result = result
                strategy.fitness = result.score()
                
                # Save to history
                save_backtest_result(params, result)
        
        # Sort by fitness
        self.population.sort(key=lambda s: s.fitness, reverse=True)
        
        # Update best
        if self.population and (self.best_strategy is None or 
                                self.population[0].fitness > self.best_strategy.fitness):
            self.best_strategy = self.population[0].copy()
            print(f"[EVOLVER] New best strategy! Score: {self.best_strategy.fitness:.1f}")
    
    def select_parent(self) -> Strategy:
        """Select parent using tournament selection"""
        tournament = random.sample(self.population, min(self.tournament_size, len(self.population)))
        return max(tournament, key=lambda s: s.fitness)
    
    def evolve_generation(self):
        """Create next generation through selection, crossover, and mutation"""
        self.generation += 1
        print(f"\n[EVOLVER] === Generation {self.generation} ===")
        
        new_population = []
        
        # Keep elite (best strategies unchanged)
        elites = sorted(self.population, key=lambda s: s.fitness, reverse=True)[:self.elite_count]
        for elite in elites:
            elite_copy = elite.copy()
            elite_copy.backtest_result = None  # Re-evaluate in new generation
            new_population.append(elite_copy)
        
        # Create rest through crossover and mutation
        while len(new_population) < self.population_size:
            parent1 = self.select_parent()
            parent2 = self.select_parent()
            
            child = parent1.crossover(parent2)
            child.mutate(self.mutation_rate)
            child.backtest_result = None
            new_population.append(child)
        
        self.population = new_population
        
        # Record history
        self.evolution_history.append({
            "generation": self.generation,
            "best_fitness": self.best_strategy.fitness if self.best_strategy else 0,
            "avg_fitness": sum(s.fitness for s in self.population) / len(self.population),
            "timestamp": datetime.now().isoformat()
        })
    
    def run_evolution(self, generations: int = 5, days: int = 7) -> Strategy:
        """
        Run complete evolution cycle
        
        Args:
            generations: Number of generations to evolve
            days: Days of historical data for backtesting
            
        Returns:
            Best strategy found
        """
        print(f"\n{'='*50}")
        print(f"[EVOLVER] Starting Evolution: {generations} generations")
        print(f"{'='*50}\n")
        
        # Initialize if needed
        if not self.population:
            self.initialize_population()
        
        # Initial evaluation
        self.evaluate_population(days=days)
        
        # Evolution loop
        for gen in range(generations):
            self.evolve_generation()
            self.evaluate_population(days=days)
            
            # Adaptive mutation rate
            if gen > 0 and self.evolution_history[-1]["avg_fitness"] == self.evolution_history[-2].get("avg_fitness", 0):
                # No improvement - increase mutation
                self.mutation_rate = min(0.3, self.mutation_rate * 1.2)
            else:
                # Improvement - decrease mutation
                self.mutation_rate = max(0.05, self.mutation_rate * 0.9)
            
            print(f"  Best: {self.best_strategy.fitness:.1f} | Avg: {self.evolution_history[-1]['avg_fitness']:.1f} | Mutation: {self.mutation_rate:.2f}")
        
        print(f"\n{'='*50}")
        print(f"[EVOLVER] Evolution Complete!")
        print(f"  Best Score: {self.best_strategy.fitness:.1f}")
        if self.best_strategy.backtest_result:
            result = self.best_strategy.backtest_result.to_dict()
            print(f"  Win Rate: {result['win_rate']}%")
            print(f"  Profit Factor: {result['profit_factor']}")
            print(f"  Return: {result['return_pct']}%")
        print(f"{'='*50}\n")
        
        return self.best_strategy
    
    def get_evolved_params(self) -> Optional[Dict]:
        """Get parameters from best evolved strategy"""
        if self.best_strategy:
            return self.best_strategy.to_params()
        return None
    
    def save_state(self, filepath: str = None):
        """Save evolution state to file"""
        if filepath is None:
            filepath = os.path.join(config.DATA_DIR, "evolution_state.json")
        try:
            state = {
                "generation": self.generation,
                "best_fitness": self.best_strategy.fitness if self.best_strategy else 0,
                "best_params": self.best_strategy.to_params() if self.best_strategy else None,
                "history": self.evolution_history,
                "mutation_rate": self.mutation_rate,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
                
            print(f"[EVOLVER] State saved to {filepath}")
            
        except Exception as e:
            print(f"[EVOLVER] Error saving state: {e}")
    
    def load_state(self, filepath: str = None):
        """Load evolution state from file"""
        if filepath is None:
            filepath = os.path.join(config.DATA_DIR, "evolution_state.json")
        try:
            if not os.path.exists(filepath):
                return False
                
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.generation = state.get("generation", 0)
            self.mutation_rate = state.get("mutation_rate", 0.15)
            self.evolution_history = state.get("history", [])
            
            # Recreate best strategy from params
            best_params = state.get("best_params")
            if best_params:
                self.best_strategy = self._params_to_strategy(best_params)
                self.best_strategy.fitness = state.get("best_fitness", 0)
            
            print(f"[EVOLVER] State loaded: Generation {self.generation}")
            return True
            
        except Exception as e:
            print(f"[EVOLVER] Error loading state: {e}")
            return False
    
    def _params_to_strategy(self, params: Dict) -> Strategy:
        """Convert parameters back to a Strategy object"""
        strategy = Strategy()
        
        for name, gene in strategy.genes.items():
            if name in params:
                gene.value = params[name]
            elif name.startswith("W_"):
                # Extract from INDICATOR_WEIGHTS
                weights = params.get("INDICATOR_WEIGHTS", {})
                indicator_map = {
                    "W_MOMENTUM": "momentum",
                    "W_MACD": "macd",
                    "W_BOLLINGER": "bollinger", 
                    "W_EMA": "ema",
                    "W_SR": "support_resistance",
                    "W_ML": "ml_prediction"
                }
                if name in indicator_map and indicator_map[name] in weights:
                    gene.value = weights[indicator_map[name]]
        
        return strategy


# Quick evolution function
def quick_evolve(generations: int = 3, days: int = 7) -> Dict:
    """Run a quick evolution and return best parameters"""
    evolver = StrategyEvolver(population_size=8)
    best = evolver.run_evolution(generations=generations, days=days)
    evolver.save_state()
    return best.to_params() if best else None
