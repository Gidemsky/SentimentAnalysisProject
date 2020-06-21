from Models import TestModel
from Models.modelHelperTest import *

if __name__ == '__main__':
    model_helper = modelHelperTest()
    fet_ids, train_X, train_Y, test_ids, test_X, test_Y = TestModel.set_data(model_helper)
    while True:
        print("Witch model do you want to run? svm / random forest / naive bayes")
        print("for Exit - Press 1")
        user_input = input()
        if user_input == "1":  
            break
        elif user_input == "svm" or user_input == "random forest" or user_input == "naive bayes":
            TestModel.run_model(model_helper, user_input, fet_ids, train_X, train_Y, test_ids, test_X, test_Y)
        else:
            raise Exception('unknown model')


