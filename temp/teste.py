import numpy as np
import pandas as pd



class Bayes_Classifier:
    def __init__(self, outlook, temp, humidity, windy, play_golf):
        self.outlook = outlook
        self.temp = temp
        self.humidity = humidity
        self.windy = windy
        self.play_golf = play_golf
        self.carrega_tabela("exemplo.csv")

    def carrega_tabela(self, tabela):
        self.csv = pd.read_csv(tabela, sep=',')
    
    def calculate_prior(self, Y):
        tabela = self.csv
        classes = sorted(list(tabela[Y].unique()))
        print(classes)
        prior = []
        for i in classes:
            prior.append(len(tabela[tabela[Y]==i])/len(tabela))
        return prior
    
    def likelihood(self, target, value_target):
        tabela = self.csv
        total_target_total = len(tabela[target])
        tabela = tabela[tabela[target]==value_target]
        total_target = len(tabela[target])
        print(total_target/total_target_total)
        
        
        # tabela = tabela[tabela[Y]==label]
        # total_Y = len(tabela[Y])
        # result = total_Y/total_target
        # print(result)

        # return tabela
    def calculate_likelihood_gaussian(self, feat_name, X, Y, label):
        """
            feat_name: target
            X: valor a testar
            Y: tabela selecionada
            label: preditor
        """
        tabela = self.csv
        feat = list(tabela.columns)

        tabela = tabela[tabela[Y]==label]
        media, desv_pad = tabela[feat_name].mean(), tabela[feat_name].std()

        # função de densidade
        p_x_given_y = (1 / (np.sqrt(2 * np.pi) * (desv_pad**2))) *  np.exp(-((X-media)**2 / (2 * desv_pad**2 )))
        return p_x_given_y

    def naive_bayes_gaussian(self, X, Y):
        """
            feat_name: target
            X: valor a testar
            Y: tabela selecionada
            label: preditor
        """
        tabela = self.csv
        # get feature names
        features = list(tabela.columns)[:-1]

        # calculate prior
        prior = self.calculate_prior(Y)

        Y_pred = []
        # loop over every data sample
        for x in X:
            # calculate likelihood
            labels = sorted(list(tabela[Y].unique()))
            likelihood = [1]*len(labels)
            for j in range(len(labels)):
                for i in range(len(features)):
                    likelihood[j] *= self.calculate_likelihood_gaussian(features[i], x[i], Y, labels[j])

            # calculate posterior probability (numerator only)
            post_prob = [1]*len(labels)
            for j in range(len(labels)):
                post_prob[j] = likelihood[j] * prior[j]

            Y_pred.append(np.argmax(post_prob))

        return np.array(Y_pred)

a = Bayes_Classifier("Rainy","Hot","High","False","No")
a.likelihood("Outlook","Rainy")
print(a.calculate_prior("Outlook"))