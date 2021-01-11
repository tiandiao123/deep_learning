import onnx
import argparse



parser = argparse.ArgumentParser(description='Convert TensorFlow model into TVM')
parser.add_argument("--model_path", help="model to Convert", type=str)
args = parser.parse_args()


model_path = args.model_path

model = onnx.load(model_path)

# The model is represented as a protobuf structure and it can be accessed
# using the standard python-for-protobuf methods

# iterate through inputs of the graph
for input in model.graph.input:
    print (input.name, end=": ")
    # get type of input tensor
    tensor_type = input.type.tensor_type
    # check if it has a shape:
    if (tensor_type.HasField("shape")):
        # iterate through dimensions of the shape:
        for d in tensor_type.shape.dim:
            # the dimension may have a definite (integer) value or a symbolic identifier or neither:
            if (d.HasField("dim_value")):
                print (d.dim_value, end=", ")  # known dimension
            elif (d.HasField("dim_param")):
                print (d.dim_param, end=", ")  # unknown dimension with symbolic name
            else:
                print ("?", end=", ")  # unknown dimension with no name
    else:
        print ("unknown rank", end="")
    print()
