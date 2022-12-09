#####
##### `LearnLogger` implementation of logging interface
#####

"""
    LearnLogger

A struct that wraps a `TensorBoardLogger.TBLogger` in order to enforce the following:

- all values logged to Tensorboard should be accessible to the `post_epoch_callback`
  argument to [`learn!`](@ref)
- all values that are cached during [`learn!`](@ref) should be logged to Tensorboard

To access values logged to a `LearnLogger` instance, inspect the instance's `logged` field.
"""
struct LearnLogger
    path::String
    tensorboard_logger::TensorBoardLogger.TBLogger
    logged::Dict{String,Vector{Any}}
end

function LearnLogger(path, run_name; kwargs...)
    tensorboard_logger = TBLogger(joinpath(path, run_name); kwargs...)
    return LearnLogger(path, tensorboard_logger, Dict{String,Any}())
end

function log_value!(logger::LearnLogger, field::AbstractString, value)
    values = get!(() -> Any[], logger.logged, field)
    push!(values, value)
    TensorBoardLogger.log_value(logger.tensorboard_logger, field, value;
                                step=length(values))
    return value
end

function log_event!(logger::LearnLogger, value::AbstractString)
    logged = string(now(), " | ", value)
    TensorBoardLogger.log_text(logger.tensorboard_logger, "events", logged)
    return logged
end

function log_plot!(logger::LearnLogger, field::AbstractString, plot, plot_data)
    values = get!(() -> Any[], logger.logged, field)
    push!(values, plot_data)
    TensorBoardLogger.log_image(logger.tensorboard_logger, field, plot; step=length(values))
    return plot
end

function log_line_series!(logger::LearnLogger, field::AbstractString, curves,
                          labels=1:length(curves))
    @warn "`log_line_series!` not implemented for `LearnLogger`" maxlog = 1
    return nothing
end

"""
    Base.flush(logger::LearnLogger)

Persist possibly transient logger state.
"""
Base.flush(logger::LearnLogger) = nothing

"""
    forwarding_task = forward_logs(channel, logger::LearnLogger)

Forwards logs with values supported by `TensorBoardLogger` to `logger::LearnLogger`:
- string events of type `AbstractString`
- scalars of type `Union{Real,Complex}`
- plots that `TensorBoardLogger` can convert to raster images

returns the `forwarding_task:::Task` that does the forwarding.
To cleanly stop forwarding, `close(channel)` and `wait(forwarding_task)`.

outbox is a Channel or RemoteChannel of Pair{String, Any}
field names starting with "__plot__" forward to TensorBoardLogger.log_image
"""
function forward_logs(outbox, logger::LearnLogger)
    @async try
        while true
            (field, value) = take!(outbox)
            if typeof(value) <: AbstractString
                log_event!(logger, value)
            elseif startswith(field, "__plot__")
                original_field = field[9:end]
                values = get!(() -> Any[], logger.logged, original_field)
                TensorBoardLogger.log_image(logger.tensorboard_logger, original_field,
                                            value; step=length(values))
            elseif typeof(value) <: Union{Real,Complex}
                log_value!(logger, field, value)
            end
        end
    catch e
        if !(isa(e, InvalidStateException) && e.state == :closed)
            @error "error forwarding logs, STOPPING FORWARDING!" exception = (e,
                                                                              catch_backtrace())
        end
    end
end
