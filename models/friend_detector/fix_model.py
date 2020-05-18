import onnx

model = onnx.load('arcfaceresnet100-8.onnx')

for node in model.graph.node:
    if(node.op_type == "BatchNormalization"):
        for attr in node.attribute:
            if (attr.name == "spatial"):
                attr.i = 1
onnx.save(model, 'updated_arcface.onnx')
