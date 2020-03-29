'''
Classes associated with ensemble learning.

Currently implemented:
- AdaBoostClassifier: http://ww.web.stanford.edu/~hastie/Papers/SII-2-3-A8-Zhu.pdf
'''
import numpy as np
import operator
import math


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
            print('Training classifier {}...'.format(i + 1))

            # Train the classifier
            trained_classifier = self.__base_classifier(dataloader, self.__sample_weight_array)
            self.__base_classifier_list.append(trained_classifier)

            # Calculate the error
            error = 0
            for index, (x, y) in enumerate(dataloader):
                # TODO: The logic here should stay outside of this class
                x = x.to('cuda')
                y = y.to('cuda')

                output = trained_classifier(x)
                predicted = output.max(1)[1]
                if not predicted.eq(y).cpu().sum():
                    error += self.__sample_weight_array[index]
            print('    Error: {}'.format(error))

            # Calculate the weight for the current hypothesis
            # TODO: Make 10 as a parameter
            weight = math.log((1 - error)/error) + math.log(10 - 1)
            print('    Weight: {}'.format(weight))
            self.__weight_list.append(weight)
            
            # Update the weights
            for index, (x, y) in enumerate(dataloader):
                # print(y)
                # TODO: The logic here should stay outside of this class
                x = x.to('cuda')
                y = y.to('cuda')

                output = self.__base_classifier_list[-1](x)
                predicted = output.data.max(1)[1]
                if predicted.eq(y.data).cpu().sum():
                    # TODO: Make 10 as a parameter
                    self.__sample_weight_array[index] *= (1 - error) * (10 - 1) / error

            # Normalize the sample weight array
            self.__sample_weight_array /= self.__sample_weight_array.sum()

        return


    def predict(self, input):
        vote_dict = {}

        for index, classifier in enumerate(self.__base_classifier_list):
            output = classifier(input)
            predicted = output.max(1)[1]
            category = predicted[0].cpu().numpy().item()
            if category in vote_dict:
                vote_dict[category] += self.__weight_list[index]
            else:
                vote_dict[category] = self.__weight_list[index]
        
        # Calculate the predicted category with the maximum vote
        return max(vote_dict.items(), key=operator.itemgetter(1))[0]

    
    def predict_using_base_classifier(self, base_classifier_index, input):
        output = self.__base_classifier_list[base_classifier_index](input)
        predicted = output.max(1)[1]
        category = predicted[0].cpu().numpy().item()
        return category
