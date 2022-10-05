import json
import uuid

import os
import numpy as np

def write_data(root, data_type, data_dict):
    scene_id = uuid.uuid4().hex
    if not os.path.isdir(root / "scenes" / data_type):
        os.mkdir(root / "scenes" / data_type)
    path = root / "scenes" / data_type/ (scene_id + ".npz")
    assert not path.exists()
    np.savez_compressed(path, **data_dict)
    return scene_id