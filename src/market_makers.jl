# Cost functions
# --------------

abstract CostFunction

typealias MarketPosition Vector

type LMSR <: CostFunction
    b::Real
end

LMSR() = LMSR(1.0)

cost(c::LMSR, q::Vector)  = c.b*log(sum(exp(q/c.b)))
function price!(c::LMSR, q::Vector, prices::Vector)
    @devec prices = exp(q ./ c.b)
    prices /= sum(prices)
end


type LinearCost <: CostFunction
    p::Vector
end

cost(c::LinearCost, q::Vector)  = dot(c.p, q)
price!(c::LinearCost, q::Vector, prices::Vector) = (prices = c.p)



price(c::CostFunction, q::Vector) = price!(c,q,copy(q))

cost(c::CostFunction, x::Vector, q::Vector) = cost(c, q + x) - cost(c, q)
price!(c::CostFunction, x::Vector, q::Vector, prices::Vector) = price!(c, q + x, prices)


# Potential-based Market Maker
# ------------

type MarketMaker{C<:CostFunction, T<:Real}
    cost_function::C
    market_position::Vector{T}
end

MarketMaker(c::CostFunction, n::Integer) = MarketMaker(c, zeros(n))

cost(mm::MarketMaker, x::Vector) = cost(mm.cost_function, mm.market_position)
cost(mm::MarketMaker, x::Vector) = cost(mm.cost_function, x, mm.market_position) # cost of a transaction

# instantaneous price at market position
price!(mm::MarketMaker, prices::Vector) = price!(mm.c, mm.market_position, prices)
# instantaneous price after trade x
price!(mm::MarketMaker, x::Vector, prices::Vector) = price!(mm.cost_function, x, mm.market_position, prices)


num_securities(mm:MarketMaker) = length(mm.market_position)