** Using python saved model in c++
	Must run the model through torchscript. e.g.,

	traced_script_module = torch.jit.trace(model, input1)
	traced_script_module.save("traced.pt")
	
	But torchscript can only process model with tensor or tuple of tensor
	as output.
	So may need to make a wrapper module:
	
class wrapper(torch.nn.Module):
    def __init__(self, model):
        super(wrapper, self).__init__()
        self.model = model
    
    def forward(self, input):
        results = []
        output = self.model(input)
        for k, v in output.items():
            results.append(v)
        return tuple(results)

model = wrapper(deeplap_model)

To access the tuple content in c++:
		auto a = module.forward(input);
		auto tup = a.toTuple();

// access element 0
	auto tensor1 = tup->elements()[0].toTensor();


c++ tensor permutation:
tensor = tensor.permute({ 0, 3, 1, 2 }); // e..g, 5x6x7x8 becomes 5x8x6x7


Helper to get tensor shape:
static std::vector<int> get_tensor_shape(torch::Tensor& tensor) {
	std::vector<int> shape;
	int num_dimensions = tensor.dim();
	for (int ii_dim = 0; ii_dim < num_dimensions; ii_dim++) {
		shape.push_back(tensor.size(ii_dim));
	}
	return shape;
}
