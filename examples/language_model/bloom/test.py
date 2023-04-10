import paddle

# state_dict = paddle.load("./bloom-7b1/model_state.pdparams")
state_dict = paddle.load("./checkpoint-50/model_state.pdparams")
print("keys", state_dict.keys())
print("qkv_weight0", state_dict['bloom.h.0.self_attention.query_key_value.weight'])
print("qkv_bias0", state_dict['bloom.h.0.self_attention.query_key_value.bias'])
print("linear_weight0", state_dict['bloom.h.0.self_attention.dense.weight'])
print("linear_bias0", state_dict['bloom.h.0.self_attention.dense.bias'])
