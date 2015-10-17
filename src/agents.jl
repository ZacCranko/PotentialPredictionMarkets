abstract AbstractUtility{T<:Real}

abstract Utility{T} <: AbstractUtility{T}

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
    log_u = log(inv(u.risk_aversion) - 1) + (u.risk_aversion) * log(x * u.a/(1-u.risk_aversion) + u.b)
    return exp(log_u)
end
function marginal_utility(u::HARA, x)
    thing = x * u.a/(1-u.risk_aversion) + u.b

    fixed = any(thing .<= 0) ? ones(thing)*eps() : thing
    log_marg = log(u.a) + (u.risk_aversion-1)*log(fixed)
    exp(log_marg)
end

type LogarithmicUtility{T} <: Utility{T}
end

utility(u::LogarithmicUtility, x) = log(x)
marginal_utility(u::LogarithmicUtility, x) = 1 ./ x

type CobbDouglas{T} <: Utility{T}
    exponent::T
end

utility(u::CobbDouglas, x) = x .^ u.exponent
function marginal_utility{T}(u::CobbDouglas{T}, x)
    u.exponent == 1 && return one(T)
    u.exponent .* x .^ (u.exponent-1)
end


type ExponentialUtility{T} <: Utility{T}
    a::T
end

function utility(u::ExponentialUtility, x)
    if u.a == 1
        return x
    else
        return (1-exp(-u.a*x))/u.a
    end
end

function marginal_utility{T}(u::ExponentialUtility{T}, x)
    if u.a == 1
        return one(T)
    else
        return exp(-u.a*x)/u.a
    end
end

# Expected utility
# ----------------

type ExpectedUtility{T} <: AbstractUtility{T}
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

type Agent{T}
    vnm::AbstractUtility
    wealth::Vector{T}
end

typealias AgentVector{T} Vector{Agent{T}}

Agent{T}(u::ExpectedUtility{T}, wealth::T) = Agent(u, wealth * ones(T,num_states(u)))

beliefs(a::Agent) = beliefs(a.vnm)

utility(a::Agent) = utility(a.vnm, a.wealth)

function utility{T}(agents::Vector{Agent{T}})
    u = zero(T)
    for a in agents
        u += utility(a)
    end
    return u
end

utility(a::Agent, x) = utility(a.vnm, x)

marginal_utility(a::Agent)    = marginal_utility(a.vnm, a.wealth)
marginal_utility(a::Agent, x) = marginal_utility(a.vnm, x)

function update_beliefs!{T}(agents::AgentVector{T}, new_beliefs::Distribution)
    for agent in agents
        agent.vnm.beliefs = rand(new_beliefs)
    end
    return agents
end

function Base.mean{T}(agents::AgentVector{T})
    n = length(agents)
    b = zero(beliefs(first(agents)))
    for a in agents
        b += beliefs(a)
    end

    return b/n
end


# Consumer Behaviour
# ------------------

function agent_objective(agent::Agent, x::Vector, grad::Vector)
    grad = marginal_utility(agent, x)
    return utility(agent, x)
end

function agent_constraint(mm::MarketMaker, agent::Agent, x::Vector, grad::Vector)
    price!(mm, x - agent.wealth, grad)
    return cost(mm, x - agent.wealth)
end

function compute_demand(mm::MarketMaker,
                        agent::Agent;
                        xtol = 1e-4,
                        bc_tol = 1e-4,
                        method = :LN_AUGLAG)

    @assert length(mm.market_position) == length(beliefs(agent))
        "the market position vector length must match the agent's belief vector length"

    opt = Opt(method, length(agent.wealth))
    xtol_rel!(opt, xtol)
    max_objective!(opt, (x,g) -> agent_objective(agent, x, g))
    inequality_constraint!(opt, (x,g) -> agent_constraint(mm, agent, x, g), bc_tol)
    (max_f, max_x, ret) = optimize(opt,  agent.wealth/2)
    return max_x
end