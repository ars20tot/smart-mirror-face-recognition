import pathlib, sys, torch

rootPath = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(rootPath))

from models import get_model

modelName = "edgeface_xs_gamma_06"
model = get_model(modelName)

stateDictPath = rootPath / "checkpoints" / f"{modelName}.pt"
stateDict = torch.load(stateDictPath, map_location="cpu")
model.load_state_dict(stateDict.get("state_dict", stateDict), strict=False)
model.eval()

dummyInput = torch.zeros(1, 3, 112, 112)

onnxDir = rootPath / "onnx"
onnxDir.mkdir(exist_ok=True)
onnxPath = onnxDir / f"{modelName}.onnx"

torch.onnx.export(
    model,
    dummyInput,
    onnxPath.as_posix(),
    input_names=["input"],
    output_names=["embedding"],
    dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
    opset_version=17,
)

print(f"ONNX exported to {onnxPath}")


