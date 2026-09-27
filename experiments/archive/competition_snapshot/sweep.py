import concurrent.futures
import itertools
from engine import play_match
from game_config import GameConfig
from strategies.adaptive_bidder import Bot as Adaptive
from strategies.rational import Bot as Rational
import test3 
from adversaries import QuoteLiar, TEHoarder, Turn6Forcer

# 1. Define the search space
PARAM_GRID = {
    'EDGE_MULT': [0.15, 0.20, 0.25, 0.30],
    'BASE_SHADE': [0.50, 0.55, 0.60],
    'TE_WEIGHT': [0.010, 0.015, 0.020]
}

# 2. Define the Gauntlet
OPPONENTS = [Adaptive, Rational, QuoteLiar, TEHoarder, Turn6Forcer]

# 3. Generate Fixed CRN Seeds
N_DEALS_PER_MATCH = 20
SEEDS = [1000, 2000, 3000, 4000, 5000] # 5 seeds * 20 deals * 2 (mirror) = 200 deals per opponent

def make_bot_factory(edge_mult, base_shade, te_weight):
    """Creates a custom bot factory to inject parameters for this thread."""
    def factory():
        bot = test3.Bot()
        bot.EDGE_MULT = edge_mult
        bot.BASE_SHADE = base_shade
        bot.TE_WEIGHT = te_weight
        return bot
    return factory

def evaluate_params(params):
    edge_mult, base_shade, te_weight = params
    bot_factory = make_bot_factory(edge_mult, base_shade, te_weight)
    config = GameConfig()
    
    total_pnl = 0
    
    for opp_class in OPPONENTS:
        for seed in SEEDS:
            # Run mirrored match
            result = play_match(
                bot_a_factory=bot_factory,
                bot_b_factory=opp_class,
                config=config,
                seed=seed,
                mirror=True,
                n_deals=N_DEALS_PER_MATCH,
                verbose=False
            )
            total_pnl += result.pnl[0]
            
    return params, total_pnl

if __name__ == '__main__':
    # Generate all combinations of parameters
    keys, values = zip(*PARAM_GRID.items())
    param_combinations = [v for v in itertools.product(*values)]
    
    print(f"Starting sweep of {len(param_combinations)} configurations...")
    
    best_pnl = -float('inf')
    best_params = None
    
    # Run in parallel across all CPU cores
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(evaluate_params, param_combinations)
        
        for params, pnl in results:
            print(f"Params (Edge: {params[0]}, Shade: {params[1]}, TE: {params[2]}) -> PnL: {pnl:.2f}")
            if pnl > best_pnl:
                best_pnl = pnl
                best_params = params
                
    print("\n" + "="*50)
    print(f"OPTIMIZATION COMPLETE.")
    print(f"Best PnL: {best_pnl:.2f}")
    print(f"Best Params: EDGE_MULT={best_params[0]}, BASE_SHADE={best_params[1]}, TE_WEIGHT={best_params[2]}")
    print("="*50)