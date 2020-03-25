'''
Classes associated with ensemble learning. Currently implemented:
- AdaBoostClassifier: https://github.com/aimacode/aima-pseudocode/blob/master/md/AdaBoost.md
'''
import numpy as np
import operator


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
            print('Training hypothesis {}...'.format(i + 1))
            trained_classifier = self.__base_classifier(dataloader, self.__sample_weight_array)
            self.__base_classifier_list.append(trained_classifier)
            error = 0
            for index, (x, y) in enumerate(dataloader):
                # TODO: The logic here should stay outside of this class
                x = x.to('cuda')
                y = y.to('cuda')

                output = trained_classifier(x)
                predicted = output.data.max(1)[1]
                if not predicted.eq(y.data).cpu().sum():
                    error += self.__sample_weight_array[index]
            for index, (x, y) in enumerate(dataloader):
                # TODO: The logic here should stay outside of this class
                x = x.to('cuda')
                y = y.to('cuda')

                output = self.__base_classifier_list[-1](x)
                predicted = output.data.max(1)[1]
                if predicted.eq(y.data).cpu().sum():
                    self.__sample_weight_array[index] *= (1 - error) / error

            print('    error: {}'.format(error))

            # Normalize the sample weight array
            self.__sample_weight_array / self.__sample_weight_array.sum(0)

            # Calculate the weight for the current hypothesis
            weight = np.log((1 - error)/error)
            print('    weight: {}'.format(weight))
            self.__weight_list.append(weight)
            

        return


    def predict(self, input):
        vote_dict = {}

        for index, classifier in enumerate(self.__classifier_list):
            output = classifier(input)
            predicted = output.data.max(1)[1]
            if vote_dict[predicted] is None:
                vote_dict[predicted] = self.__weight_list[index]
            else:
                vote_dict[predicted] += self.__weight_list[index]
        
        # Calculate the predicted category with the maximum vote
        return max(vote_dict.items(), key=operator.itemgetter(1))[0]
