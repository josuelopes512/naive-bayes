import pandas as pd
import csv

class Bayes_Classifier:
    def __init__(self, genero, idade, escolaridade, profissao, target):
        self.genero = genero
        self.idade = idade
        self.escolaridade = escolaridade
        self.profissao = profissao
        self.target = target

    def carrega_tabela(self, tabela):
        self.csv = pd.read_csv(tabela, sep=',')
        print(self.csv)
    
    def load_csv(filename):
        """Load CSV data from a file and convert the attributes to numbers.
        :param filename: The name of the CSV file to load.
        :return: A list of the dataset.
        """
        f = open(filename)
        lines = csv.reader(f)
        dataset = list(lines)
        for i in range(len(dataset)):
            dataset[i] = [float(x) for x in dataset[i]]
        f.close()
        return dataset

a = Bayes_Classifier("F","a - Ate 25 anos ","Fundamental","b",0)
a.carrega_tabela("naive-bayes-classificador-2.csv")

print(a.csv['genero'])
