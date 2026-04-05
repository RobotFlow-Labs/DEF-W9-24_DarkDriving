# PRD-07: Integration

## Objective
Docker serving, ROS2 node, and HuggingFace push.

## Deliverables
1. `Dockerfile.serve` -- 3-layer Docker image (FROM anima-serve:jazzy)
2. `docker-compose.serve.yml` -- profiles: serve, ros2, api, test
3. `.env.serve` -- module identity + runtime config
4. `src/dark_driving/serve.py` -- AnimaNode subclass
5. `anima_module.yaml` -- updated with docker/ros2 fields
6. HF push script for checkpoints + exports

## Docker Architecture
```
Layer 1: anima-base:jazzy (ROS2 + CycloneDDS)
Layer 2: anima-serve:jazzy (FastAPI + AnimaNode)
Layer 3: anima-darkdriving:latest (this module)
```

## API Endpoints
- `GET /health` -- service health
- `GET /ready` -- model loaded check
- `POST /predict` -- enhance a low-light image
  - Input: base64 encoded image or file upload
  - Output: enhanced image (base64) + quality metrics

## ROS2 Topics
- Subscribe: `/camera/night/image_raw` (sensor_msgs/Image)
- Publish: `/darkdriving/enhanced` (sensor_msgs/Image)
- Publish: `/darkdriving/metrics` (std_msgs/String, JSON)

## Acceptance Criteria
- Docker builds without error
- `docker compose --profile api up` serves /health
- HF push script uploads to ilessio-aiflowlab/project_darkdriving-checkpoint
