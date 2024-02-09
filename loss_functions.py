import numpy as np

class loss_functions():
    def __init__(self, y_preds, y_target):
        self.y_preds = y_preds #softmax layer: list of probabilities
        self.y_target = y_target

    '''def CrossEntropyLoss(self):
        n_classes = len(self.y_preds)
        encoded_target = np.zeros_like(self.y_preds)
        encoded_target[self.y_target-1] = 1
        print(f'target_in_loss: {encoded_target}')

        temp = 0
        for i in range(n_classes):
            print(temp)
            temp+=(encoded_target[i] * np.log(self.y_preds[i]))
        return temp'''
    
    def cross_entropy_loss(self, epsilon=1e-15):
        self.y_preds = np.clip(self.y_preds, epsilon, 1 - epsilon)
        ce_loss = - (self.y_target * np.log(self.y_preds) + (1 - self.y_target) * np.log(1 - self.y_preds))
        return np.sum(ce_loss)

    def l2loss(self):
        loss = np.sum(np.square(self.y_preds - self.y_target))
        return loss


