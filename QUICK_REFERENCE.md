# Quick Reference Guide

##  Common Commands

### Development

Small overview of important shell commands that are might be needed for this project.


```bash
# Start development server
make dev
# or
uvicorn app.main:app --reload

# Run tests
make test

# Format code
make format

# Check code quality
make lint
```

### Docker

```bash
# Build and run
make docker-build
make docker-run

# View logs
make docker-logs

# Stop services
make docker-stop
```

### Testing

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_api.py -v

# With coverage
pytest --cov=app --cov-report=html

# Only unit tests
pytest tests/test_api.py -v

# Only integration tests
pytest tests/integration/ -v
```

### Model Validation

```bash
# Validate model before deployment
python scripts/validate_model.py

# Run smoke tests
python scripts/smoke_tests.py
```

##  Monitoring URLs (Default)

| Service | URL | Credentials |
|---------|-----|-------------|
| API Docs | http://localhost:8000/docs | - |
| API Health | http://localhost:8000/health | - |
| Prometheus | http://localhost:9090 | - |
| Grafana | http://localhost:3000 | admin/admin |
| MLflow | http://localhost:5000 | - |

## 🔍 Debugging

### Check API Logs
```bash
docker-compose logs -f api
```

### Check All Logs
```bash
docker-compose logs -f
```

### Enter Container
```bash
docker exec -it brain-tumor-api bash
```

### Check Model Loading
```bash
curl http://localhost:8000/health | jq '.model_loaded'
```

### Test Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@test_image.jpg" | jq '.'
```

## 🛠️ Troubleshooting

### Port Already in Use
```bash
# Find process using port 8000
lsof -i :8000
# Kill the process
kill -9 <PID>
```

### Docker Issues
```bash
# Clean up Docker
docker system prune -a
docker volume prune

# Rebuild without cache
docker-compose build --no-cache
```

### Python Environment
```bash
# Recreate virtual environment
deactivate
rm -rf venv/
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 📈 Performance Monitoring

### Check API Metrics
```bash
curl http://localhost:8000/metrics
```

### View Prediction History
```bash
# Recent predictions
tail -f logs/predictions_$(date +%Y%m%d).jsonl
```

### Monitor Resource Usage
```bash
# Docker stats
docker stats brain-tumor-api
```

## 🔄 Update & Deploy

### Update Model
```bash
# 1. Copy new model
cp path/to/new_model.pth models/best_model.pth

# 2. Restart API
docker-compose restart api

# 3. Verify
curl http://localhost:8000/health
```

### Deploy Updates
```bash
# 1. Commit changes
git add .
git commit -m "Update model/code"

# 2. Push to trigger CI/CD
git push origin main

# 3. Monitor deployment
# Check GitHub Actions tab
```

## 📝 File Locations

```
Important Files:
├── app/main.py              # Main API code
├── models/best_model.pth    # Model weights
├── requirements.txt         # Dependencies
├── Dockerfile              # Container config
├── docker-compose.yml      # Multi-container setup
└── .github/workflows/ci-cd.yml  # CI/CD pipeline
```

## 🎯 Quick Wins

1. **Add New Endpoint**: Edit `app/main.py`
2. **Update Model**: Replace `models/best_model.pth`
3. **Change Dependencies**: Edit `requirements.txt`
4. **Modify Tests**: Edit `tests/test_api.py`
5. **Update CI/CD**: Edit `.github/workflows/ci-cd.yml`

## 💡 Pro Tips

- Use `make help` to see all available commands
- Run `pytest -v -s` for detailed test output
- Use `docker-compose up` without `-d` to see live logs
- Monitor Grafana dashboards for performance insights

---

**Need Help?** Check README.md
