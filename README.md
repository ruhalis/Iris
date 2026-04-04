# Iris
### Prepare model
```
python3.10 -m venv .venv
```
```
source .venv/bin/activate
```
```
pip install -r requirements.txt
```

```
python train.py
```

Test it
```
python predict.py
```

### Build and deploy
```
docker compose up -d --build
```

### Check

```
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```


You will get 
```
{"predicted_class":0,"predicted_label":"setosa","probabilities":{"setosa":0.9999,"versicolor":0.0001,"virginica":0.0}}% 
```