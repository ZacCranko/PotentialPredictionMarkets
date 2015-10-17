using Base.Test

using PredictionMarkets

prices = [0.3, 0.7]
wealth = [1.0, 1.0]
belief = [0.1, 0.9]

pref  = ExpectedUtility(Logarithmic{Float64}(), belief)
agent = Agent(pref, wealth)

lmsr   = MarketMaker(LMSR(100.0), 2)
linear = MarketMaker(LinearCost([0.3,0.7]), 2)

numeric_demand  = compute_demand(linear, agent)
analytic_demand = dot(prices, wealth) * belief ./ prices

# test if numeric demand is feasible
@test dot(numeric_demand, prices) <= dot(wealth, prices)
@test_approx_eq_eps numeric_demand analytic_demand 1e-6

numeric_demand_convex  = compute_demand(lmsr, agent)