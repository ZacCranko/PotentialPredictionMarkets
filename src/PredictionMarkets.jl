module PredictionMarkets

using NLopt, Devectorize, Distributions

# agents
export Agent, ExpectedUtility, HARA, CobbDouglas, Logarithmic, Exponential
export utility, marginal_utility, compute_demand, num_states, beliefs

# market makers
export MarketMaker, LMSR, LinearCost
export cost, price, price!, num_securities, execute_trade!

# simulations
export execute_trade!, simulate_market!, simulate_market, update_beliefs!

# includes

include("market_makers.jl")
include("agents.jl")
include("market_simulation.jl")
include("random.jl")

end # module