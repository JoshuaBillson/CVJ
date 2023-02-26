import CVJ.AbstractMask, CVJ.AbstractImage

import Flux

using CategoricalArrays

abstract type AbstractModel end

abstract type AbstractClassifier <: AbstractModel end

abstract type AbstractSegmentor <: AbstractModel end

"Specify the scitypes of the input and label expected by the given AbstractClassifier, respectively."
function scitypes(::Type{<:AbstractClassifier})
    return AbstractImage, CategoricalValue
end

"Specify the scitypes of the input and label expected by the given AbstractSegmentor, respectively."
function scitypes(::Type{<:AbstractSegmentor})
    return AbstractImage, AbstractMask
end

"""
    validate_model(model_type, images, labels)

Validate that the scitypes of the provided images and labels matches those expected by the specified model. 
"""
function validate_model(::M, ::AbstractVector{I}, ::AbstractVector{L}) where {M<:AbstractModel, I, L}
    f_type, l_type = scitypes(M)
    scitypes_match = (I <: f_type) && (L <: l_type)
    m="Model $M expects types ($f_type, $l_type), received types ($I, $L)!"
    @assert scitypes_match m
end

"""
    construct_predictor(model)

Must be implemented by all AbstractModel instances.

Construct a predictor (Flux neural network) matching the hyperparameters specified by model.
"""
function construct_predictor(::T) where {T <: AbstractModel}
    error("You forgot to implement construct_predictor(model) for model type $(T)!")
end

"""
    get_params(model, predictor)

Can be optionally implemented for an AbstractModel instance.

Return the trainable parameters for the predictor specified by the given model. By default, returns all parameters in the predictor.
Useful for pre-trained networks, where you want to train only the output layer before fine-tuning the encoder.
"""
function get_params(::Any, predictor)
    return Flux.params(predictor)
end