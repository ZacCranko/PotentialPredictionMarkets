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