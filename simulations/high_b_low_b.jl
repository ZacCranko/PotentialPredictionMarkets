using PredictionMarkets
using Distributions: Pareto, Dirichlet, LogNormal, Uniform

function unzip(input)
    n = length(input)
    types  = [ typeof(tup) for tup in first(input) ]
    output = [ Vector{typ}(n) for typ in types ]

    for i = 1:n
        @inbounds for (j, x) in enumerate(input[i])
            (output[j])[i] = x
        end
    end
    return (output...)
end

function running_average(v::Vector, lags = 400)
    avg = deepcopy(v)
    for i = 1:length(v)
        k = max(i-lags,1)
        @inbounds avg[i] = mean(v[k:i])
    end
    return avg
end


using PyPlot

function price_plot(prices)
    figure(figsize=(10, 6))

    for p in unzip(prices)
        ind = collect(1:(length(p)-1))
        plot(ind, p[1:end-1],                  color="black", linewidth=1.0, linestyle="-", alpha=0.5)
        plot(ind, running_average(p[1:end-1]), color="black", linewidth=1.0, linestyle="-", alpha=1.0)
    end
end

# Two state simulation, low b
# -----------------------------

mm     = MarketMaker(LMSR(10), 2)
agents = rand(Agent,
              500,
              wealth_dist = Pareto(2),
              belief_dist = Dirichlet([3,6]),
              risk_aversion_dist = LogNormal())

(mm, agents, trace) = simulate_market(mm, agents, 5, xtol=1e-4, bc_tol=1e-4)
market_positions, prices, agent_utility = unzip(trace)
price_plot(prices)
ind = collect(1:(length(agent_utility)-1))
plot(ind, running_average(agent_utility[1:end-1]), color="black", linewidth=1.0, linestyle="-", alpha=1.0)

# Two state simulation, high b
# -----------------------------

mm     = MarketMaker(LMSR(100), 2)
agents = rand(Agent,
              500,
              wealth_dist = Pareto(2),
              belief_dist = Dirichlet([3,6]),
              risk_aversion_dist = LogNormal())

(mm, agents, trace) = simulate_market(mm, agents, 5, xtol=1e-4, bc_tol=1e-4)
market_positions, prices, agent_utility = unzip(trace)
price_plot(prices)
ind = collect(1:(length(agent_utility)-1))
plot(ind, running_average(agent_utility[1:end-1]), color="black", linewidth=1.0, linestyle="-", alpha=1.0)