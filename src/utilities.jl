#####
##### miscellaneous
#####

function increment_at!(array, index_lists)
    for index_list in index_lists
        array[index_list...] += 1
    end
    return array
end

"""
    area_under_curve(x, y)

Calculates the area under the curve specified by the `x` vector and `y` vector
using the trapezoidal rule.
"""
function area_under_curve(x, y)
    length(x) == length(y) || throw(ArgumentError("Length of inputs must match."))
    length(x) == 0 && throw(ArgumentError("Inputs must be nonempty."))
    auc = zero(middle(one(eltype(x)), one(eltype(y))))
    perms = sortperm(x)
    sorted_x = view(x, perms)
    sorted_y = view(y, perms)
    # calculate the trapazoidal method https://en.wikipedia.org/wiki/Trapezoidal_rule
    for i in 2:length(x)
        auc += middle(sorted_y[i], sorted_y[i - 1]) * (sorted_x[i] - sorted_x[i - 1])
    end
    return auc
end

"""
    area_under_curve_unit_square(x, y)

Calculates the area under the curve specified by the `x` vector and `y` vector
for a unit square, using the trapezoidal rule.
"""
function area_under_curve_unit_square(x, y)
    length(x) == length(y) || throw(ArgumentError("Length of inputs must match."))
    length(x) == 0 && throw(ArgumentError("Inputs must be nonempty."))
    kept = [(i, j)
            for (i, j) in zip(x, y)
            if !(ismissing(i) || ismissing(j)) && (0 <= i <= 1 && 0 <= j <= 1)]
    return area_under_curve(map(k -> k[1], kept), map(k -> k[2], kept))
end

"""
    majority([rng::AbstractRNG=Random.GLOBAL_RNG], hard_labels, among::UnitRange)

Return the majority label within `among` out of `hard_labels`:

```
julia> majority([1, 2, 1, 3, 2, 2, 3], 1:3)
2

julia> majority([1, 2, 1, 3, 2, 2, 3, 4], 3:4)
3
```

In the event of a tie, a winner is randomly selected from the tied labels via `rng`.
"""
function majority(rng::AbstractRNG, labels, among::UnitRange)
    return rand(rng, StatsBase.modes(labels, among))
end
majority(labels, among::UnitRange) = majority(Random.GLOBAL_RNG, labels, among)

#####
##### `ResourceInfo`
#####

struct ResourceInfo
    time_in_seconds::Float64
    gc_time_in_seconds::Float64
    allocations::Int64
    memory_in_mb::Float64
end

# NOTE: This is suitable for "coarse" (i.e. long-running) `f`, not "fine" `f`;
# the latter requires the kind of infrastructure provided by BenchmarkTools.
function call_with_resource_info(f)
    gc_start = Base.gc_num()
    time_start = time_ns()
    result = f()
    time_in_seconds = ((time_ns() - time_start) / 1_000_000_000)
    gc_diff = Base.GC_Diff(Base.gc_num(), gc_start)
    gc_time_in_seconds = gc_diff.total_time / 1_000_000_000
    allocations = gc_diff.malloc + gc_diff.realloc + gc_diff.poolalloc + gc_diff.bigalloc
    memory_in_mb = gc_diff.allocd / 1024 / 1024
    info = ResourceInfo(time_in_seconds, gc_time_in_seconds, allocations, memory_in_mb)
    return result, info
end
