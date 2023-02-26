module CVJ

include("Pipeline.jl")
include("Models.jl")
include("Machines.jl")
include("Classifiers.jl")

export AbstractImage, AbstractMask, RGBImage, GrayImage, Mask, load_image, load_mask, n_channels, n_classes, load_data, augment_data, one_hot_mask, image_to_tensor, get_images, get_labels
export AbstractModel, AbstractClassifier, AbstractSegmentor, construct_predictor, get_params
export FitResult, Machine, compile!, fit!, evaluate
export ResNetClassifier

end