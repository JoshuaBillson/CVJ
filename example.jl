import CVJ, Random, Flux, Metrics
using Pipe: @pipe

function main()
    X = @pipe CVJ.get_images("data", "jpg", CVJ.RGBImage) |> Random.shuffle(_)[1:5000]
    Y = CVJ.get_labels(X)

    model = CVJ.ResNetClassifier()
    machine = CVJ.Machine(model, X, Y, (256, 256))
    CVJ.compile!(machine)

    CVJ.fit!(machine, 1)
    CVJ.evaluate(machine, [Metrics.categorical_accuracy])
end

main()