"""CPU-only ONNX inference with explicit threading and a validator adapter."""
import onnx_bootstrap
import cv2
import numpy as np
import onnxruntime as ort
import torch


def prepare(image):
    rgb=cv2.cvtColor(cv2.resize(image,(320,320)),cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb.transpose(2,0,1)[None],dtype=np.float32)/255


def session(path):
    options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    result=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    assert result.get_providers()==['CPUExecutionProvider']
    return result


class OrtAdapter(torch.nn.Module):
    """Share the existing box metric calculation; all network inference is ORT CPU."""
    def __init__(self,ort_session,names):
        super().__init__();self.session=ort_session;self.names=names;self.stride=torch.tensor([8.,16.,32.])

    def fuse(self,verbose=False):return self  # Graph was fused before export.

    def forward(self,x,*args,**kwargs):
        output=self.session.run(['predictions'],{'images':x.detach().cpu().numpy()})[0]
        return torch.from_numpy(output).to(x.device),None
