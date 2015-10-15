# Bernoulli utility Functions
# ---------------------------

immutable HARA{T} <: Utility{T}
    a::T
    b::T
    risk_aversion::T
end

function utility(u::HARA, x)
    if u.risk_aversion == 1
        return (x * u.a + u.b)/u.risk_aversion
    end
    (1-u.risk_aversion) * (x * u.a/(1-u.risk_aversion) + u.b).^u.risk_aversion / u.risk_aversion
end

function marginal_utility(u::HARA, x)
    u.a*(x * u.a/(1-u.risk_aversion) + u.b).^(u.risk_aversion-1)
end

type LogarithmicUtility{T} <: Utility{T}
end

utility(u::LogarithmicUtility, x) = log(x)
marginal_utility(u::LogarithmicUtility, x) = 1 ./ x


# Expected utility
# ----------------

immutable ExpectedUtility{T} <: AbstractUtility{T}
    u::Utility{T}
    beliefs::Vector{T}
end

function utility(vnm::ExpectedUtility, x)
    return dot(utility(vnm.u,x), vnm.beliefs)
end

function marginal_utility(vnm::ExpectedUtility, x)
    marginal_utility(vnm.u,x) .* vnm.beliefs
end

beliefs(u::ExpectedUtility) =  u.beliefs
num_states(u::ExpectedUtility) = length(u.beliefs)

# Agents
# ------

type Agent{U<:AbstractUtility, T}
    vnm::U
    w::Vector{T}
end

typealias AgentVector{T} Vector{Agent{T}}

Agent{T}(u::ExpectedUtility{T}, w::Vector{T}) = Agent{T}(u, w)
Agent{T}(u::ExpectedUtility{T}, w::Real) = Agent{T}(u, w * ones(num_states(u)))

beliefs(a::Agent) = beliefs(a.vnm)

utility(a::Agent) = utility(a.vnm, a.w)
utility(a::Agent, x) = utility(a.vnm, x)

marginal_utility(a::Agent)    = marginal_utility(a.vnm, a.w)
marginal_utility(a::Agent, x) = marginal_utility(a.vnm, x)


# Consumer Behaviour
# ------------------

function compute_demand(mm::MarketMaker,
                        agent::Agent;
                        xtol = 1e-8,
                        bc_tol = 1e-8,
                        method = :LN_AUGLAG)

    @assert length(mm.market_position) == length(beliefs(agent))
        "the market position vector length must match the agent's belief vector length"

    function myobjective(agent::Agent, x::Vector, grad::Vector)
        grad = marginal_utility(agent, x)
        return utility(agent, x)
    end

    function myconstraint(mm::MarketMaker, agent::Agent, x::Vector, grad::Vector)
        price!(mm, x - agent.w, grad)
        return cost(mm, x - agent.w)
    end

    opt = Opt(method, length(agent.w))
    xtol_rel!(opt, xtol)
    max_objective!(opt, (x,g) -> myobjective(agent, x, g))
    inequality_constraint!(opt, (x,g) -> myconstraint(mm, agent, x, g), bc_tol)

    (max_f, max_x, ret) = optimize(opt,  agent.w ./ 10)
    return max_x
end

function execute_trade!{C,U,T}(mm::MarketMaker{C,T}, agent::Agent{U,T}, demand::Vector{T})
    trade = demand - agent.w
    mm.market_position += trade     # update market position with net difference
    agent.w = demand                # update traded wealth
    return mm.market_position
end

function simulate_market!{C,T}(mm::MarketMaker{C,T},
                               agents::AgentVector{T},
                               num_rounds::Int;
                               xtol   = 1e-8,
                               bc_tol = 1e-8,
                               method = :LN_AUGLAG) # augmented lagrangian solver for demand computation

    num_agents = length(agents)

    demand     = Vector{T}(num_securities(mm))
    market_pos = Vector{T}(num_securities(mm))

    for t = 1:num_rounds
        for (tradeno, agent) in enumerate(agents)
            demand     = compute_demand(mm, agent, xtol = xtol, bc_tol = bc_tol, method = method)
            market_pos = execute_trade!(mm, agent, demand)

            @printf "%f percent complete\r" 100*tradeno/(num_rounds*num_agents)
        end
    end
end

function simulate_market{C,T}(mm::MarketMaker{C,T}, agents::AgentVector{T}, num_rounds::Int)
    simulate_market!{C,T}(deepcopy(mm), deepcopy(agents), num_rounds)
end

