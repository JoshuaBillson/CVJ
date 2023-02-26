import CVJ.AbstractImage, CVJ.AbstractModel, CVJ.construct_predictor, CVJ.DataPipeline, CVJ.validate_model, CVJ.get_params
import Flux
using Random, ProgressBars, Statistics
using Pipe: @pipe

mutable struct FitResult
	predictor
	train::Vector{Int}
	test::Vector{Int}
	val::Vector{Int}
	learning_curve::Vector{Float32}
	validation_curve::Vector{Float32}
end

function FitResult()
    FitResult(nothing, [], [], [], [], [])
end

struct Machine{M<:AbstractModel, I<:AbstractImage, L}
	model::M
	X::Vector{I}
	Y::AbstractVector{L}
	imsize::Tuple{Int, Int}
    p_flipx::Float64
    p_flipy::Float64
    p_rotate::Float64
	fitresult::FitResult
end

function Machine(model::AbstractModel, X, Y, imsize::Tuple{Int, Int}, p_flipx=0.5, p_flipy=0.5, p_rotate=0.25)
	validate_model(model, X, Y)
	Machine(model, X, Y, imsize, p_flipx, p_flipy, p_rotate, FitResult())
end

function compile!(mach::Machine, train=0.65, test=0.20, val=0.15, rng=MersenneTwister(123))
	@assert (train + test + val) ≈ 1.0 "Error: Train/Test/Val Split Must Sum To 1 (Got $(train + test + val))!"
	n = length(mach.X)
	indices = randperm(rng, n)
	train_index = floor(Int, train * n)
	test_index = floor(Int, test * n) + train_index
	compile!(mach, indices[1:train_index], indices[train_index+1:test_index], indices[test_index+1:end])
end

function compile!(mach::Machine, train::AbstractVector{Int}, test::AbstractVector{Int}, val::AbstractVector{Int})
	mach.fitresult.predictor = construct_predictor(mach.model)
	mach.fitresult.train = Vector(train)
	mach.fitresult.test = Vector(test)
	mach.fitresult.val = Vector(val)
end

function fit!(machine::Machine, epochs::Int; opt=Flux.Adam(), loss=Flux.crossentropy, callbacks=nothing)
	# Get Train Data
	trainX = machine.X[machine.fitresult.train]
	trainY = machine.Y[machine.fitresult.train]
	trainPipeline = DataPipeline(trainX, trainY, machine.imsize, machine.p_flipx, machine.p_flipy, machine.p_rotate)
	trainData = Flux.DataLoader(trainPipeline, batchsize=16, shuffle=true)

	# Get Validation Data
	valX = machine.X[machine.fitresult.val]
	valY = machine.Y[machine.fitresult.val]
	valPipeline = DataPipeline(valX, valY, machine.imsize, 0.0, 0.0, 0.0)
	valData = Flux.DataLoader(valPipeline, batchsize=16, shuffle=false)

	# Get Params
	params = get_params(machine.model, machine.fitresult.predictor)

	# Iterate Over Epochs
	for epoch in 1:epochs

		# Iterate Over Training Data
		avg_loss = 0.0
		iter = ProgressBar(trainData)
		for (i, (x, y)) in enumerate(iter)

			# Compute Gradient
			l, grads = Flux.withgradient(() -> loss(machine.fitresult.predictor(x), y), params)

			# Update Params
			Flux.update!(opt, params, grads)

			# Update Running Average
			avg_loss += (l - avg_loss) / i
			set_description(iter, "Epoch: $epoch, Loss: $avg_loss")

		end
	end
end

function evaluate(machine::Machine, metric, split=:test)
	# Get Evaluation Data
	@assert split in [:test, :train, :val] "Error: Split must be one of (:test, :train, :val)!"
	splits = Dict(:test => machine.fitresult.test, :train => machine.fitresult.train, :val => machine.fitresult.val)
	X = machine.X[splits[split]]
	Y = machine.Y[splits[split]]
	pipeline = DataPipeline(X, Y, machine.imsize, 0.0, 0.0, 0.0)
	data = Flux.DataLoader(pipeline, batchsize=16, shuffle=false)

	# Get Predictions
	predictions = [(machine.fitresult.predictor(x), y) for (x, y) in data]
	y = cat([y for (ŷ, y) in predictions]..., dims=2)
	ŷ = cat([ŷ for (ŷ, y) in predictions]..., dims=2)
	
	# Return Evaluation
	return metric(ŷ, y)

end

function flatten_mask(mask)
	nclasses = size(mask, 3)
	obs = size(mask, 1) * size(mask, 2) * size(mask, 4)
	return @pipe permutedims(mask, (3, 1, 2, 4)) |> reshape(_, (nclasses, obs))
end