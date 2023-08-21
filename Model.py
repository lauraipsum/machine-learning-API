"""
Ensemble: 
    >>Método que constroi  vários modelos de machine learning, 
    utilizando o resultado de cada modelo na definição de um único resultado, 
    obtendo-se assim um valor final único. 

    Random Forest: 
        >>Algoritmo de ensemble que combina várias árvores de decisão individuais 
        para criar um modelo mais robusto e preciso.

        RandomForestClassifier: 
            >>Classe da biblioteca sklearn que implementa o algoritmo 
            Random Forest para tarefas de classificação. 
            
            Parametros do RandomForestClassifier:


"""

import pandas as pd 
from sklearn.ensemble import RandomForestClassifier # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
from pydantic import BaseModel
import joblib # Salva e carrega modelos

#   Medidas de uma
class IrisSpecies(BaseModel):
    sepal_length: float 
    sepal_width: float 
    petal_length: float 
    petal_width: float
    
#   Construtor
class IrisModel:

   #Carrega o dataset e o modelo se existirem
   #Do contrário, chama o método _train_model e salva o modelo
    def __init__(self):
        self.df = pd.read_csv('iris.csv')
        self.model_fname_ = 'iris_model.pkl'
        try:
            self.model = joblib.load(self.model_fname_)
        except Exception as _:
            self.model = self._train_model()
            joblib.dump(self.model, self.model_fname_)
            
#   Treina o modelo usando o classificador RandomForest
    def _train_model(self):
        X = self.df.drop('species') #conjunto self.df, removendo apenas a coluna species
        y = self.df['species'] #armazena os rótulos das classes
        model = RandomForestClassifier.fit(X, y)
        
        return model
    
#   Predição baseada nos dados providos pelo usuário

    def predict_species(self, sepal_length, sepal_width, petal_length, petal_width):
        data_in = [[sepal_length, sepal_width, petal_length, petal_length]]
        prediction = self.model.predict(data_in)
        probability = self.model.predict_proba(data_in).max()
        
        #   Retorna a espécie predita e sua probabilidade
        return prediction[0], probability