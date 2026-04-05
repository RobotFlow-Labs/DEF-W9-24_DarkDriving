# PRD-06: Export Pipeline

## Objective
Export trained enhancement models to production formats.

## Deliverables
1. Export script supporting:
   - PyTorch -> SafeTensors
   - PyTorch -> ONNX (opset 17, dynamic batch)
   - ONNX -> TensorRT FP16 + FP32 (using shared TRT toolkit)
2. Validation: compare outputs between PyTorch and exported models (max diff < 1e-3)
3. Benchmarking: measure latency per format on L4

## Export Formats
| Format | Use Case | Path |
|--------|----------|------|
| SafeTensors | HF upload, portable | /mnt/artifacts-datai/exports/project_darkdriving/*.safetensors |
| ONNX | Cross-platform | /mnt/artifacts-datai/exports/project_darkdriving/*.onnx |
| TensorRT FP16 | Production inference | /mnt/artifacts-datai/exports/project_darkdriving/*.trt |
| TensorRT FP32 | High-precision inference | /mnt/artifacts-datai/exports/project_darkdriving/*.trt |

## Export Flow
```
best.pth -> safetensors -> ONNX -> TRT FP16 + TRT FP32
                             |
                        validate outputs match
```

## Acceptance Criteria
- ONNX model loads in onnxruntime
- TRT models load and produce valid output
- Output diff < 1e-3 between PyTorch and ONNX
