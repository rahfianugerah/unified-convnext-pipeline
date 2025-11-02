# Production-Ready ConvNeXt Multi-Task Pipeline

A comprehensive production deployment system for the ConvNeXt computer vision pipeline supporting classification, object detection, and OCR tasks.

### Features

### High Priority
- **Inference Script**: Standalone inference for production predictions
- **Docker Containerization**: Complete Docker setup with docker-compose
- **Input Validation**: Robust validation and error handling for data preprocessing
- **Model Serialization**: Export models to ONNX and TorchScript formats

### Medium Priority
- **FastAPI REST API**: High-performance API for model serving
- **Prediction Logging**: Comprehensive logging and monitoring system
- **Data Validation Pipeline**: Multi-stage validation for inputs
- **Visualization Tools**: Tools for visualizing predictions

### Nice to Have
- **Model Explainability**: Grad-CAM for understanding predictions
- **Automated Retraining**: Pipeline for automatic model retraining
- **A/B Testing Framework**: Compare model versions in production
- **Model Versioning System**: Track and manage model versions

### Additional
- **Comprehensive Documentation**: Detailed guides and API documentation
- **Test Suite**: Unit tests for all components
- **CI/CD Pipeline**: Automated testing and deployment

### Project Structure

```
production/
├── api/
│   └── server.py              # FastAPI server
├── inference/
│   ├── inference.py           # Inference engine
│   └── model_export.py        # Model export utilities
├── validation/
│   └── input_validator.py     # Input validation
├── monitoring/
│   └── logger.py              # Prediction logging
├── utils/
│   └── visualization.py       # Visualization tools
├── explainability/
│   └── gradcam.py            # Grad-CAM explainability
├── retraining/
│   └── pipeline.py           # Automated retraining
├── ab_testing/
│   └── manager.py            # A/B testing
└── versioning/
    └── registry.py           # Model versioning

tests/
└── test_production.py        # Unit tests

docs/
├── DEPLOYMENT.md             # Deployment guide
└── API.md                    # API reference

.github/
└── workflows/
    └── ci-cd.yml            # CI/CD pipeline

Dockerfile                    # Docker image definition
docker-compose.yml           # Multi-container setup
requirements.txt             # Python dependencies
```

### Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA (optional, for GPU support)
- Docker (optional, for containerized deployment)

### Install from Source

```bash
# Clone repository
git clone https://github.com/rahfianugerah/gemastik.git
cd gemastik

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

### 1. Train a Model

```bash
python main.py \
  --data_root ./data/cls \
  --theme classification \
  --epochs 10 \
  --n_trials 3
```

### 2. Run Inference

```bash
python production/inference/inference.py \
  --checkpoint ./artifacts/classification/best_model.pth \
  --theme classification \
  --image test.jpg \
  --output predictions.json
```

### 3. Start API Server

```bash
# Set environment variables
export MODEL_PATH=./artifacts/classification/best_model.pth
export THEME=classification

# Start server
python -m production.api.server
```

### 4. Make Predictions via API

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@test.jpg"
```

### Docker Deployment

### Build and Run

```bash
# Build image
docker build -t convnext-api:latest .

# Run container
docker run -d \
  --name convnext-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -e MODEL_PATH=/app/models/best_model.pth \
  -e THEME=classification \
  --gpus all \
  convnext-api:latest
```

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

### Usage Examples

### Batch Inference

```python
from production.inference.inference import InferenceEngine

# Initialize engine
engine = InferenceEngine(
    checkpoint_path="./artifacts/classification/best_model.pth",
    theme="classification",
    device="cuda"
)

# Batch prediction
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = engine.predict_batch(image_paths)
```

### Model Export

```bash
# Export to ONNX and TorchScript
python production/inference/model_export.py \
  --checkpoint ./artifacts/classification/best_model.pth \
  --theme classification \
  --output_dir ./exported_models \
  --format all
```

### Input Validation

```python
from production.validation.input_validator import InputValidator

validator = InputValidator(theme="classification", strict=True)
is_valid, report = validator.validate_and_preprocess("image.jpg")

if is_valid:
    print("Image is valid!")
else:
    print(f"Validation errors: {report['errors']}")
```

### Visualization

```bash
# Visualize predictions
python production/utils/visualization.py \
  --predictions predictions.json \
  --images_dir ./test_images \
  --output_dir ./visualizations
```

### Grad-CAM Explainability

```bash
# Generate Grad-CAM heatmap
python production/explainability/gradcam.py \
  --checkpoint ./artifacts/classification/best_model.pth \
  --image test.jpg \
  --output gradcam.png
```

### Model Versioning

```bash
# Register new version
python production/versioning/registry.py \
  --action register \
  --model_path ./artifacts/classification/best_model.pth \
  --theme classification

# Promote to production
python production/versioning/registry.py \
  --action promote \
  --theme classification \
  --version v1 \
  --stage production
```

### A/B Testing

```bash
# Create experiment
python production/ab_testing/manager.py \
  --action create \
  --name model_comparison \
  --model_a ./models/v1.pth \
  --model_b ./models/v2.pth \
  --split 0.5

# Analyze results
python production/ab_testing/manager.py \
  --action analyze \
  --name model_comparison
```

### Automated Retraining

```bash
# Run retraining pipeline
python production/retraining/pipeline.py \
  --config retraining_config.json
```

### Documentation

- [Deployment Guide](docs/DEPLOYMENT.md) - Complete deployment instructions
- [API Reference](docs/API.md) - REST API documentation
- [Main README](README.md) - Training pipeline documentation

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=production --cov-report=html

# Run specific test
pytest tests/test_production.py::test_input_validator -v
```

### CI/CD

The project includes a GitHub Actions workflow for:
- Code linting (flake8, black, isort)
- Unit testing with coverage
- Docker image building
- Automated deployment

See `.github/workflows/ci-cd.yml` for details.

### Monitoring

### Metrics

Access metrics at:
- API: `http://localhost:8000/metrics`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

### Logs

- Predictions: `logs/predictions.jsonl`
- Errors: `logs/errors.jsonl`
- Metrics: `logs/metrics.json`

### Configuration

### Environment Variables

- `MODEL_PATH`: Path to model checkpoint
- `THEME`: Task type (classification/object/ocr)
- `DEVICE`: Device to use (cuda/cpu/auto)
- `LOG_LEVEL`: Logging level (INFO/DEBUG/ERROR)
- `MAX_WORKERS`: Number of API workers

### Configuration Files

- `retraining_config.json`: Retraining pipeline settings
- `ab_test_config.json`: A/B testing configuration
- `docker-compose.yml`: Multi-container orchestration

### Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### License

This project is licensed under the MIT License.

### Acknowledgments

- PyTorch team for the deep learning framework
- timm library for model architectures
- FastAPI for the web framework
- The open-source community

### Support

For issues and questions:
- Open an issue on GitHub
- Check the documentation
- Contact the development team