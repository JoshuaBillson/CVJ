using Images, FileIO, CategoricalArrays, Augmentor
import Flux
using Pipe: @pipe

abstract type AbstractImage end

abstract type AbstractMask end

"Denotes the features in the pipeline as an RGB image saved to disk."
struct RGBImage <: AbstractImage
    src::String
end

"Denotes the features in the pipeline as a grayscale image saved to disk."
struct GrayImage <: AbstractImage
    src::String
end

"Denotes the label in the pipeline as a mask where pixels are already encoded as the appropriate class."
struct Mask <: AbstractMask
    src::String
    classes::Int
end

"Represents a data pipeline for training a model. Stores the feature images, labels, and augmentation pipeline."
struct DataPipeline{I<:AbstractImage, L}
    imgs::AbstractVector{I}
    labels::AbstractVector{L}
    sz::Tuple{Int, Int}
    split::Symbol
end

"""
    n_channels(img_type)

Return the number of channels for the given AbstractImage type. 
"""
function n_channels(t::Type{<:AbstractImage})
    return 3
end

"Read a label into a Float32 Array."
function n_classes(l::Type{T}) where {T <: AbstractMask}
    error("Error: You forgot to implement n_classes(mask_type) for type $(T)!")
end

function load_data(x::AbstractImage, y::CategoricalValue, sz::Tuple{Int, Int}, split::Symbol)
    x = @pipe load(x.src) |> imresize(_, sz) |> image_to_tensor(_)
    y = Flux.onehot(y, levels(y)) .|> Float32
    return x, y
end

function load_data(x::AbstractImage, y::AbstractMask, sz::Tuple{Int, Int}, split::Symbol)
    x = @pipe load(x.src) |> imresize(_, sz) |> image_to_tensor(_)
    y = Flux.onehot(y, levels(y)) .|> Float32
    return x, y
end

function Base.length(X::DataPipeline)
	return length(X.imgs)
end

function Base.getindex(X::DataPipeline{I, <:CategoricalValue}, i::AbstractArray{Int}) where {I}
    imgs = X.imgs[i]
    labels = X.labels[i]
    x_dst = zeros(Float32, (X.sz..., n_channels(I), length(i)))
    y_dst = zeros(Float32, (length(levels(labels)), length(i)))
    @Threads.threads for idx in 1:length(imgs)
        x, y = load_data(imgs[idx], labels[idx], X.sz, X.split)
        x_dst[:,:,:,idx] .= x 
        y_dst[:,idx] .= y
    end
    return x_dst |> Flux.gpu, y_dst |> Flux.gpu
end

function Base.getindex(X::DataPipeline, i::Int)
    return X[[i]]
end

function augment_data(x::Array{Float32, 3}, aug)
    return @pipe make_slices(x) |> augment(_, aug) |> join_slices(_...)
end

function augment_data(x::Array{Float32, 3}, y::Array{Float32, 3}, aug)
    x_y = @pipe (make_slices(x)..., make_slices(y)...) |> augment(_, aug)
    return join_slices(x_y[1:3]...), join_slices(x_y[4:6]...)
end

"Turn a 3-dimensional array image into a tuple of channel slices."
function make_slices(x::Array{Float32, 3})
	return Tuple(x[:,:,i] for i in eachindex(x[1,1,:]))
end

"Turn a collection of channel slices into a 3-dimensional array"
function join_slices(x::Vararg{Matrix{Float32}})
	cat(x..., dims=3)
end

"Encode a mask as a one-hot array where the number of channels is equal to the number of classes."
function one_hot_mask(mask::Array{UInt8, 2}, classes::Int)
    m = zeros(Float32, size(mask)..., classes)
    for row in 1:size(mask, 1)
        for col in 1:size(mask, 2)
            i = mask[row,col] + 1
            m[row,col,i] = 1.0f0
        end
    end
    return m
end

function image_to_tensor(img::Array{<:Colorant, 2})::Array{Float32, 3}
    @pipe float32.(img) |> channelview(_) |> permutedims(_, (3, 2, 1))
end

function image_to_tensor(img::Array{<:Gray, 2})::Array{Float32, 3}
    x = float32.(img) |> channelview
    return @pipe reshape(x, size(x)..., 1) |> permutedims(_, (3, 2, 1))
end

function get_images(src::String, ext::String, ::Type{T}) where {T <: AbstractImage}
    imgs = []
    for (root, dirs, files) in walkdir(src)
        for file in files
            if contains(file, ".$ext")
                push!(imgs, "$root/$file")
            end
        end
    end
    return T.(imgs)
end

function get_labels(f, imgs::Vector{T}) where {T <: AbstractImage}
    @assert (:src in fieldnames(T)) "Error: Image Does Not Contain 'src' Field!"
    map(f, [img.src for img in imgs]) |> categorical
end

function get_labels(imgs::Vector{T}) where {T <: AbstractImage}
    @assert (:src in fieldnames(T)) "Error: Image Does Not Contain 'src' Field!"
    map(x->splitpath(x)[end-1], [img.src for img in imgs]) |> categorical
end