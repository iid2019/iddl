'''
Classes associated with ensemble learning. Currently implemented:
- AdaBoostClassifier: https://github.com/aimacode/aima-pseudocode/blob/master/md/AdaBoost.md
'''
import numpy as np


class AdaBoostClassifier():
    __base_classifier = None
    __sample_weight_array = None
    __base_classifier_list = []
    __weight_list = []


    def __init__(self, base_classifier):
        self.__base_classifier = base_classifier

        return


    def train(self, dataloader, classifier_num=5):
        # Initialize the sample weight array
        n = len(dataloader)
        self.__sample_weight_array = np.repeat(1/n, n)

        for i in range(classifier_num):
            trained_classifier = self.__base_classifier(dataloader, self.__sample_weight_array)
            self.__base_classifier_list.append(trained_classifier)
            error = 0
            for index, (x, y) in enumerate(dataloader):
                output = trained_classifier(x)
                predicted = output.data.max(1)[1]
                if predicted.eq(y.data).cpu().sum():
                    error += self.__sample_weight[index]
            for index, (x, y) in enumerate(dataloader):
                # TODO: The logic here should stay outside of this class
                x = x.to('cuda')
                y = y.to('cuda')

                output = self.__base_classifier_list[-1](x)
                predicted = output.data.max(1)[1]
                if predicted.eq(y.data).cpu().sum():
                    self.__sample_weight[index] *= (1 - error) / error

            print('error = {}'.format(error))

            # Normalize the sample weight array
            self.__sample_weight_array / self.__sample_weight_array.sum(0)

            # Calculate the weight for the current hypothesis
            self.__weight_list.append(np.log((1 - error)/error))
            

        return


    def predict(self, input):
        return
