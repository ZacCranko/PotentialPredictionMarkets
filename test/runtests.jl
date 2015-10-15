pref  = ExpectedUtility(LogarithmicUtility{Float64}(), [.1, .9])
agent = Agent(pref, [0.5,0.5])

lmsr = MarketMaker(LMSR(100.0),2)

d = compute_demand(lmsr,agent)

mpos = execute_trade!(lmsr, agent, d)
