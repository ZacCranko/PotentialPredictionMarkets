module PotentialPredictionMarkets

using Distributions, NLopt, Devectorize

includes = ["agents.jl", "files.txt", "market_makers.jl", "market_simulation.jl", "random.jl"]

for inc in includes
  include(inc)
end

end # module