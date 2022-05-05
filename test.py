import json

a = {"LR": 3e-5,
     "GLOBAL_SEED": 6,
     "SEQ_MAX_LEN": 512,
     "warm_up_ratio": 0.1,
     "tensorboard": "D:\\python_code\\paper\\summary\\",
     "use_model": "albert"
     }

c = json.dumps(a, indent=2)
print(c)
