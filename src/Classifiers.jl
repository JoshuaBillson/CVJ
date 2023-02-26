import CVJ.AbstractClassifier, CVJ.construct_predictor
using Metalhead
using Flux

struct ResNetClassifier <: AbstractClassifier
    n_layers::Int
    n_classes::Int
    pretrain::Bool
    η::Float64
    λ::Float64
    epochs::Int
    fine_tune::Bool
end

function ResNetClassifier(
    ;n_layers=34, 
    n_classes=2, 
    pretrain=true, 
    η=1e-3, 
    λ=0.0, 
    epochs=10, 
    fine_tune=false)
    
    return ResNetClassifier(n_layers, n_classes, pretrain, η, λ, epochs, fine_tune)
    
end

function CVJ.construct_predictor(model::ResNetClassifier)
    Flux.Chain(
        Metalhead.ResNet(model.n_layers, pretrain=model.pretrain), 
        Flux.Dense(1000, model.n_classes), 
        Flux.softmax
    )
end

function CVJ.get_params(model::ResNetClassifier, predictor)
    return model.fine_tune ? Flux.params(predictor[1:end-1]) : Flux.params(predictor[end-1])
end