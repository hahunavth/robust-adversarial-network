from torch import optim

def create_optimizer(name, model, lr):
    if name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, 
                    momentum=0.9, weight_decay=0.0005)
    if name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, 
                    weight_decay=0.0005, amsgrad=False)
    raise Exception('invalid optimizer')