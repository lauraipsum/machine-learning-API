import uvicorn
from fastapi import FastAPI
from Model import IrisModel, IrisSpecies

#   Instância das classes
app = FastAPI() 
model = IrisModel()

#   Declaração da primeira rota
#   http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'olá, humano!'}

#   Declaração da segunda rota
#    http://127.0.0.1:8000/nome
@app.get('/{name}')
def get_name(name: str):
    return {'message': f'Olá, {name}'}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
# 
@app.post('/predict')
def predict_species(iris: IrisSpecies):
    data = iris.dict()
    prediction, probability = model.predict_species(
        data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']
    )
    return {
        'prediction': prediction,
        'probability': probability
    }
