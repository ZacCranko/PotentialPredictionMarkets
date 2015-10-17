function execute_trade!{C,T}(mm::MarketMaker{C,T}, agent::Agent{T}, demand::Vector{T})
    trade = demand - agent.wealth
    mm.market_position += trade     # update market position with net difference
    agent.wealth = demand                # update traded wealth
    return mm.market_position
end

function simulate_market!{C,T}(mm::MarketMaker{C,T},
                               agents::AgentVector{T},
                               num_rounds::Int;
                               trace  = Vector(),
                               xtol   = 1e-4,
                               bc_tol = 1e-4,
                               method = :LN_AUGLAG) # augmented lagrangian solver for demand computation

    num_agents = length(agents)

    demand     = Vector{T}(num_securities(mm))
    market_pos = copy(mm.market_position)

    push!(trace, (market_pos, price(mm), utility(agents)))
    i = 0
    for t = 1:num_rounds
        for (tradeno, agent) in enumerate(agents)
            i += 1
            demand     = compute_demand(mm, agent, xtol = xtol, bc_tol = bc_tol, method = method)
            market_pos = execute_trade!(mm, agent, demand)

            push!(trace, (market_pos, price(mm), utility(agents)))
            @printf "%f percent complete...\r" 100*i/(num_rounds*num_agents)
        end
    end
    info("market simulation complete")
    return mm, agents, trace
end

function simulate_market{C,T}(mm::MarketMaker{C,T},
                              agents::AgentVector{T},
                              num_rounds::Int;
                              xtol = 1e-4,
                              bc_tol = 1e-4,
                              method = :LN_AUGLAG,
                              trace = Vector())

    simulate_market!(deepcopy(mm), deepcopy(agents), num_rounds, xtol=xtol, bc_tol=bc_tol, method=method, trace=trace)
end