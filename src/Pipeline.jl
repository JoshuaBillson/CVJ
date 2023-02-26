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
    aug::Augmentor.Pipeline
end

"""
    DataPipeline(imgs, labels, sz, p_flipx, p_flipy, p_rotate)

Construct a data pipeline from the given images and labels. 

Images/masks will be resized to the resolution specified by sz. 

p_flipx, p_flipy, and p_ratate denote the probability that the given operations will be applied during augmentation.
"""
function DataPipeline(i::AbstractVector{I}, l::AbstractVector{L}, sz::Tuple{Int, Int}, p_flipx, p_flipy, p_rotate) where {I <:AbstractImage, L}
    aug = FlipX(p_flipx) |> FlipY(p_flipy) |> Rotate90(p_rotate) |> Resize(sz)
    DataPipeline(i, l, aug)
end

"""
    n_channels(img_type)

Return the number of channels for the given AbstractImage type. 
"""
function n_channels(t::Type{<:AbstractImage})
    return 3
end

"""
    load_image(img)

Read an image into a 3-dimensional Float32 array. 

Must be implemented for all AbstractImage types.
"""
function load_image(img::AbstractImage)::Array{Float32, 3}
    return load(img.src) |> image_to_tensor
end

"Read a label into a Float32 Array."
function load_mask(l::AbstractMask)
    @pipe load(l.src) |>
    channelview |>
    permutedims(_, (2, 1)) |>
    rawview |>
    Matrix |>
    one_hot_mask(_, l.classes)
end

"Read a label into a Float32 Array."
function n_classes(l::Type{T}) where {T <: AbstractMask}
    error("Error: You forgot to implement n_classes(mask_type) for type $(T)!")
end

function load_data(xs::Vector{I}, ys::CategoricalVector, aug, sz) where {I <: AbstractImage}
    x_dst = zeros(Float32, (sz..., n_channels(I), length(xs)))
    y_dst = Flux.onehotbatch(ys, levels(ys)) .|> Float32
    @Threads.threads for idx in 1:length(xs)
        x_dst[:,:,:,idx] .= @pipe load_image(xs[idx]) |> augment_data(_, aug)
    end
    return x_dst, y_dst
end

function load_data(xs::Vector{I}, ys::Vector{M}, aug, sz) where {I <: AbstractImage, M <: AbstractMask}
    x_dst = zeros(Float32, (sz..., n_channels(I), length(i)))
    y_dst = zeros(Float32, (sz..., n_classes(M), length(i)))
    @Threads.threads for idx in 1:length(xs)
        x, y = augment_data(load_image(xs[idx]), load_mask(ys[idx]), aug)
        x_dst[:,:,:,idx] .= x 
        y_dst .= y
    end
    return x_dst, y_dst
end

function Base.length(X::DataPipeline)
	return length(X.imgs)
end

function Base.getindex(X::DataPipeline{I, <:CategoricalValue}, i::AbstractArray{Int}) where {I}
    xs, ys = load_data(X.imgs[i], X.labels[i], X.aug, X.aug.operations[end].size)
    return xs |> Flux.gpu, ys |> Flux.gpu
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

function get_images(src::String, ::Type{T}) where {T <: AbstractImage}
    imgs = []
    for (root, dirs, files) in walkdir(src)
        for file in files
            if contains(file, ".jpg")
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