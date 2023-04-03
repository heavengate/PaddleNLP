import os
import paddle
import numpy as np
from modeling import BloomBlock
from configuration import BloomConfig

def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./100", help="The directory of model.")
    parser.add_argument("--fuse_mt", type=int, default=2, help="The batch size of data.")
    return parser.parse_args()

# def main():
#     args = parse_arguments()
#
#     config = BloomConfig(args.model_path)
#     print("hidden_size", config.hidden_size)
#     block = BloomBlock(config, 0)
#     block.set_state_dict(paddle.load(os.path.join(args.model_path, "model_state.pdparams")))
#
#     data = np.random.random([1, 20, 64]).astype("float16")
#     p_data = paddle.to_tensor(data)
#     ret = block(p_data)
#     print("ret", ret.shape, ret.numpy())

def main():
    args = parse_arguments()

    state_dict = paddle.load(os.path.join(args.model_path, "model_state.pdparams"))
    keys = [k for k in state_dict.keys() if k.startswith('bloom.h.0')]
    for k in keys:
        print(k, state_dict[k].shape)
    print("qkv_weight0", state_dict["bloom.h.0.self_attention.query_key_value.weight"].transpose([1, 0]))
    print("qkv_bias0", state_dict["bloom.h.0.self_attention.query_key_value.bias"])
    print("linear_weight0", state_dict["bloom.h.0.self_attention.dense.weight"])
    print("linear_bias0", state_dict["bloom.h.0.self_attention.dense.bias"])
    print("ffn1_weight0", state_dict["bloom.h.0.mlp.dense_h_to_4h.weight"])
    print("ffn1_bias0", state_dict["bloom.h.0.mlp.dense_h_to_4h.bias"])


if __name__ == "__main__":
    # main()
    a = paddle.load('./20000/model_state.pdparams')
    for k, v in a.items():
        print(k, v.dtype)
