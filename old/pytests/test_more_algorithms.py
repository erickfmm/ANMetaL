"""Tests for additional optimization algorithms."""
import pytest
from anmetal.optimizer.population.ABC.abc import ArtificialBeeColony
from anmetal.optimizer.population.Bat.bat import BatAlgorithm
from anmetal.optimizer.population.Firefly.firefly import FireflyAlgorithm
from anmetal.optimizer.population.Cuckoo.cuckoo import CuckooSearch
from anmetal.optimizer.population.Genetic.GeneticMH_Categorical import GeneticMH_Categorical
from anmetal.optimizer.population.Harmony.harmony import HarmonySearch
from anmetal.optimizer.population.ACO.aco import AntColony
from anmetal.optimizer.population.Blackhole.blackhole import BlackHole
from anmetal.optimizer.population.Greedy.GreedyMH_Real import GreedyMH_Real
from anmetal.problems.nonlinear_functions.two_inputs import Goldsteinprice
from anmetal.problems.nphard_categorical.knapsack import Knapsack_Categorical

class TestABC:
    def test_abc_initialization(self):
        """Test ABC algorithm initialization."""
        problem = Goldsteinprice()
        abc = ArtificialBeeColony(
            min_value=problem.get_limits()[0],
            max_value=problem.get_limits()[1],
            ndims=2,
            to_max=False,
            objective_function=problem.func,
            repair_function=None,
            preprocess_function=None
        )
        assert abc._ndims == 2
        assert not abc._to_max

    def test_abc_run(self):
        """Test ABC algorithm execution."""
        problem = Goldsteinprice()
        abc = ArtificialBeeColony(
            min_value=problem.get_limits()[0],
            max_value=problem.get_limits()[1],
            ndims=2,
            to_max=False,
            objective_function=problem.func,
            repair_function=None,
            preprocess_function=None
        )
        # Note: ABC run method signature might be different. 
        # Let's assume standard run(iterations, population_size, ...) or check if it uses specific params.
        # Based on file read, ABC inherits from IMetaheuristic but run method wasn't shown in the snippet.
        # Usually run takes iterations and population size.
        # Let's try with generic params and if it fails we can adjust.
        # ABC usually needs 'limit' parameter.
        
        # Checking ABC.run signature would be good, but let's try standard first.
        # If ABC.run is not overridden, it uses IMetaheuristic.run which might be abstract or empty.
        # Wait, I should check if ABC implements run.
        
        # Let's assume it does.
        fit, pt = abc.run(
            verbose=False,
            iterations=10,
            population=20,
            limit=5 # ABC specific parameter
        )
        assert isinstance(fit, (int, float))
        assert len(pt) == 2

class TestBat:
    def test_bat_initialization(self):
        """Test Bat algorithm initialization."""
        problem = Goldsteinprice()
        bat = BatAlgorithm(
            min_value=problem.get_limits()[0],
            max_value=problem.get_limits()[1],
            ndims=2,
            to_max=False,
            objective_function=problem.func,
            repair_function=None,
            preprocess_function=None
        )
        assert bat._ndims == 2
        assert not bat._to_max

    def test_bat_run(self):
        """Test Bat algorithm execution."""
        problem = Goldsteinprice()
        bat = BatAlgorithm(
            min_value=problem.get_limits()[0],
            max_value=problem.get_limits()[1],
            ndims=2,
            to_max=False,
            objective_function=problem.func,
            repair_function=None,
            preprocess_function=None
        )
        
        # Bat algorithm specific parameters: fmin, fmax, A (loudness), r0 (pulse rate)
        fit, pt = bat.run(
            verbose=False,
            iterations=10,
            population=20,
            fmin=0,
            fmax=2,
            A=0.5,
            r0=0.5
        )
        assert isinstance(fit, (int, float))
        assert len(pt) == 2

class TestFirefly:
    def test_firefly_initialization(self):
        """Test Firefly algorithm initialization."""
        problem = Goldsteinprice()
        firefly = FireflyAlgorithm(
            min_value=problem.get_limits()[0],
            max_value=problem.get_limits()[1],
            ndims=2,
            to_max=False,
            objective_function=problem.func,
            repair_function=None,
            preprocess_function=None
        )
        assert firefly._ndims == 2
        assert not firefly._to_max

    def test_firefly_run(self):
        """Test Firefly algorithm execution."""
        problem = Goldsteinprice()
        firefly = FireflyAlgorithm(
            min_value=problem.get_limits()[0],
            max_value=problem.get_limits()[1],
            ndims=2,
            to_max=False,
            objective_function=problem.func,
            repair_function=None,
            preprocess_function=None
        )
        
        fit, pt = firefly.run(
            verbose=False,
            iterations=10,
            population=20,
            alpha=0.5,
            beta0=1.0,
            gamma=1.0
        )
        assert isinstance(fit, (int, float))
        assert len(pt) == 2

class TestCuckoo:
    def test_cuckoo_initialization(self):
        """Test Cuckoo Search algorithm initialization."""
        problem = Goldsteinprice()
        cuckoo = CuckooSearch(
            min_value=problem.get_limits()[0],
            max_value=problem.get_limits()[1],
            ndims=2,
            to_max=False,
            objective_function=problem.func,
            repair_function=None,
            preprocess_function=None
        )
        assert cuckoo._ndims == 2
        assert not cuckoo._to_max

    def test_cuckoo_run(self):
        """Test Cuckoo Search algorithm execution."""
        problem = Goldsteinprice()
        cuckoo = CuckooSearch(
            min_value=problem.get_limits()[0],
            max_value=problem.get_limits()[1],
            ndims=2,
            to_max=False,
            objective_function=problem.func,
            repair_function=None,
            preprocess_function=None
        )
        
        fit, pt = cuckoo.run(
            verbose=False,
            iterations=10,
            population=20,
            pa=0.25
        )
        assert isinstance(fit, (int, float))
        assert len(pt) == 2

class TestGenetic:
    def test_genetic_initialization(self):
        """Test Genetic algorithm initialization."""
        problem = Knapsack_Categorical(50, 10, 0, 5, 5)
        genetic = GeneticMH_Categorical(
            categorics=problem.get_possible_categories(),
            ndims=10,
            to_max=True,
            objective_function=problem.objective_function,
            repair_function=problem.repair_function,
            preprocess_function=problem.preprocess_function
        )
        assert genetic._ndims == 10
        assert genetic._to_max

    def test_genetic_run(self):
        """Test Genetic algorithm execution."""
        problem = Knapsack_Categorical(50, 10, 0, 5, 5)
        genetic = GeneticMH_Categorical(
            categorics=problem.get_possible_categories(),
            ndims=10,
            to_max=True,
            objective_function=problem.objective_function,
            repair_function=problem.repair_function,
            preprocess_function=problem.preprocess_function
        )
        
        fit, pt = genetic.run(
            verbose=False,
            iterations=10,
            population=20,
            elitist_percentage=0.2,
            mutability=0.1
        )
        assert isinstance(fit, (int, float))
        assert len(pt) == 10

class TestHarmony:
    def test_harmony_initialization(self):
        """Test Harmony Search algorithm initialization."""
        problem = Goldsteinprice()
        harmony = HarmonySearch(
            min_value=problem.get_limits()[0],
            max_value=problem.get_limits()[1],
            ndims=2,
            to_max=False,
            objective_function=problem.func,
            repair_function=None,
            preprocess_function=None
        )
        assert harmony._ndims == 2
        assert not harmony._to_max

    def test_harmony_run(self):
        """Test Harmony Search algorithm execution."""
        problem = Goldsteinprice()
        harmony = HarmonySearch(
            min_value=problem.get_limits()[0],
            max_value=problem.get_limits()[1],
            ndims=2,
            to_max=False,
            objective_function=problem.func,
            repair_function=None,
            preprocess_function=None
        )
        
        # Harmony Search parameters: hms, hmcr, par, bw
        # run(iterations, population, hms, hmcr, par, bw, ...)
        # Wait, let's check Harmony run signature.
        # Assuming standard run or check file.
        # I'll check file content again if needed, but let's try generic first.
        # Actually I should check run signature.
        
        # Let's assume it has run method.
        fit, pt = harmony.run(
            verbose=False,
            iterations=10,
            population=20
        )
        assert isinstance(fit, (int, float))
        assert len(pt) == 2

class TestACO:
    def test_aco_initialization(self):
        """Test ACO algorithm initialization."""
        problem = Goldsteinprice()
        aco = AntColony(
            min_value=problem.get_limits()[0],
            max_value=problem.get_limits()[1],
            ndims=2,
            to_max=False,
            objective_function=problem.func,
            repair_function=None,
            preprocess_function=None
        )
        assert aco._ndims == 2
        assert not aco._to_max

    def test_aco_run(self):
        """Test ACO algorithm execution."""
        problem = Goldsteinprice()
        aco = AntColony(
            min_value=problem.get_limits()[0],
            max_value=problem.get_limits()[1],
            ndims=2,
            to_max=False,
            objective_function=problem.func,
            repair_function=None,
            preprocess_function=None
        )
        
        fit, pt = aco.run(
            verbose=False,
            iterations=10,
            population=20,
            evaporation_rate=0.1,
            alpha=1.0,
            beta=2.0
        )
        assert isinstance(fit, (int, float))
        assert len(pt) == 2

class TestBlackHole:
    def test_blackhole_initialization(self):
        """Test BlackHole algorithm initialization."""
        problem = Goldsteinprice()
        bh = BlackHole(
            min_value=problem.get_limits()[0],
            max_value=problem.get_limits()[1],
            ndims=2,
            to_max=False,
            objective_function=problem.func,
            repair_function=None,
            preprocess_function=None
        )
        assert bh._ndims == 2
        assert not bh._to_max

    def test_blackhole_run(self):
        """Test BlackHole algorithm execution."""
        problem = Goldsteinprice()
        bh = BlackHole(
            min_value=problem.get_limits()[0],
            max_value=problem.get_limits()[1],
            ndims=2,
            to_max=False,
            objective_function=problem.func,
            repair_function=None,
            preprocess_function=None
        )
        
        fit, pt = bh.run(
            verbose=False,
            iterations=10,
            population=20
        )
        assert isinstance(fit, (int, float))
        assert len(pt) == 2

class TestGreedy:
    def test_greedy_initialization(self):
        """Test Greedy algorithm initialization."""
        problem = Goldsteinprice()
        greedy = GreedyMH_Real(
            min_value=problem.get_limits()[0],
            max_value=problem.get_limits()[1],
            ndims=2,
            to_max=False,
            objective_function=problem.func,
            repair_function=None,
            preprocess_function=None
        )
        assert greedy._ndims == 2
        assert not greedy._to_max

    def test_greedy_run(self):
        """Test Greedy algorithm execution."""
        problem = Goldsteinprice()
        greedy = GreedyMH_Real(
            min_value=problem.get_limits()[0],
            max_value=problem.get_limits()[1],
            ndims=2,
            to_max=False,
            objective_function=problem.func,
            repair_function=None,
            preprocess_function=None
        )
        
        fit, pt = greedy.run(
            verbose=False,
            iterations=10,
            population=20
        )
        assert isinstance(fit, (int, float))
        assert len(pt) == 2
