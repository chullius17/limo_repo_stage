import os
import tensorrt as trt
from ament_index_python.packages import get_package_share_directory
import onnx

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
package_share_dir = get_package_share_directory('cv_package')
onnx_path = os.path.join(package_share_dir, 'best.onnx')
engine_path = os.path.join(package_share_dir, 'best.engine')

model = onnx.load(onnx_path)
print([i.name for i in model.graph.input])

def build_engine(onnx_path, engine_path):
    builder = trt.Builder(TRT_LOGGER)

    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )

    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("❌ ONNX parse failed")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return None

    config = builder.create_builder_config()

    # 🔥 CRITICO
    config.set_flag(trt.BuilderFlag.FP16)
    config.max_workspace_size = 2 << 30

    # 🔥 MOLTO IMPORTANTE
    config.builder_optimization_level = 5

    # 🔥 profiling (fondamentale su Jetson)
    profile = builder.create_optimization_profile()

    profile.set_shape(
        "images",
        (1, 3, 448, 448),
        (1, 3, 448, 448),
        (1, 3, 448, 448)
    )

    config.add_optimization_profile(profile)

    engine = builder.build_engine(network, config)

    with open(engine_path, "wb") as f:
        f.write(engine.serialize())

def main(args=None):
    import rclpy  # non serve davvero, ma ok per standard ROS
    build_engine(onnx_path, engine_path)
    return

if __name__ == "__main__":
    main()