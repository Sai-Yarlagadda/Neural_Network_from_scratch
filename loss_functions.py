import numpy as np

class loss_functions():
    def __init__(self, y_preds, y_target):
        self.y_preds = y_preds #softmax layer: list of probabilities
        self.y_target = y_target

    def CrossEntropyLoss(self):
        n_classes = len(self.y_preds)
        encoded_target = np.zeros_like(self.y_preds)
        encoded_target[self.y_target -1] = 1
        print(encoded_target)

        temp = 0
        for i in range(n_classes):
            print(temp)
            temp+=(encoded_target[i] * np.log(self.y_preds[i]))
        return temp

    def l2loss(self):
        loss = np.sum(np.square(self.y_preds - self.y_target))
        return loss


#cross checking class defined above
y1 = np.array([0.1,0.2,0.8])
y = 3
loss = loss_functions(y_preds=y1, y_target=y)
CE = loss.CrossEntropyLoss()
print(CE)

y1 = np.array([1,2,3])
y2 =np.array([0,1,2])
loss = loss_functions(y_preds=y1, y_target=y2)
l2 = loss.l2loss()
print(l2)