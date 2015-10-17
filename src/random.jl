function Base.rand(T::Type{HARA}; a=0.0, b=0.0, risk_aversion_dist=LogNormal())
    risk_aversion = rand(risk_aversion_dist)
    HARA(a, b, risk_aversion)
end

function Base.rand(T::Type{ExponentialUtility}; risk_aversion_dist=LogNormal())
    a = rand(risk_aversion_dist)
    ExponentialUtility(a)
end

function Base.rand(T::Type{HARA},
                   n::Int;
                   a=00.5,
                   b=10.0,
                   risk_aversion_dist = LogNormal())
    random_haras = Vector{T}(n)
    for i = 1:n
        @inbounds random_haras[i] = rand(T, a=a, b=b, risk_aversion_dist=risk_aversion_dist)
    end
    return random_haras
end


function Base.rand(T::Type{ExponentialUtility},
                   n::Int;
                   risk_aversion_dist = LogNormal())
    random_exponentials = Vector{T}(n)
    for i = 1:n
        @inbounds random_exponentials[i] = rand(T, risk_aversion_dist=risk_aversion_dist)
    end
    return random_exponentials
end


function Base.rand(T::Type{Agent},
                   n::Int;
                   wealth_dist = Pareto(),
                   belief_dist = Dirichlet([3,6]),
                   risk_aversion_dist = LogNormal())

    random_agents  = AgentVector{Float64}(n)
    random_wealth  = rand(wealth_dist, n)
    random_beliefs = rand(belief_dist, n)
    random_pref    = rand(ExponentialUtility, n, risk_aversion_dist=risk_aversion_dist)

    for i = 1:n
        @inbounds random_agents[i] =
            Agent(ExpectedUtility(random_pref[i], random_beliefs[:,i]), random_wealth[i])
    end
    return random_agents
end


# function Base.rand(T::Type{Agent},
#                    n::Int;
#                    wealth_dist = Pareto(),
#                    belief_dist = Dirichlet([3,6]),
#                    a=00.5,
#                    b=10.0,
#                    risk_aversion_dist = LogNormal())

#     random_agents  = AgentVector{Float64}(n)
#     random_wealth  = rand(wealth_dist, n)
#     random_beliefs = rand(belief_dist, n)
#     random_pref    = rand(HARA, n, a=a, b=b, risk_aversion_dist=risk_aversion_dist)

#     for i = 1:n
#         @inbounds random_agents[i] =
#             Agent(ExpectedUtility(random_pref[i], random_beliefs[:,i]), random_wealth[i])
#     end
#     return random_agents
# end