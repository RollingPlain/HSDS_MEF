import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])  


class Architect(object):

    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model  
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(), lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, float(args.epochs), eta_min=args.learning_rate_min)


    def _compute_unrolled_model(self, input, target, eta,
                                network_optimizer):

        x =self.model(input)
        loss = self.model.calculate_loss(x, target)[0]

        theta = _concat(self.model.net_parameters()).data 
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.net_parameters()).mul_(self.network_momentum)  
        except:
            moment = torch.zeros_like(theta) 
        dtheta = _concat(torch.autograd.grad(loss, self.model.net_parameters())).data + self.network_weight_decay * theta 

        unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment + dtheta))  
        return unrolled_model

    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled): 
        self.optimizer.zero_grad()  
        if unrolled:  
            self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer) 
        else:  
            self._backward_step(input_valid, target_valid)
        self.optimizer.step()  
        self.scheduler.step()


    def _backward_step(self, input_valid, target_valid):
        x = self.model(input_valid)
        loss = self.model.calculate_loss(x, target_valid)[0]
        loss.backward()

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):

        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer) 
        
        x = unrolled_model(input_valid)
        unrolled_loss = unrolled_model.calculate_loss(x, target_valid)[0]

        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]

        vector = [v.grad.data if v.grad!=None else torch.zeros_like(v) for v in unrolled_model.net_parameters()] 
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)
        for v, g in zip(self.model.arch_parameters(), dalpha): 
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)
            

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        params, offset = {}, 0
        for v1,v2 in zip(model_new.net_parameters(), self.model.net_parameters()):
            v_length = np.prod(v2.size())
            v1.data.copy_(theta[offset: offset + v_length].view(v2.size()))
            offset += v_length
        assert offset == len(theta)
        return model_new.cuda()


    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        R = r / _concat(vector).norm()  
        for p, v in zip(self.model.net_parameters(), vector):
            p.data.add_(R, v)

        x = self.model(input)
        loss = self.model.calculate_loss(x, target)[0]

        grads_p = torch.autograd.grad(loss, self.model.arch_parameters()) 

        for p, v in zip(self.model.net_parameters(), vector):
            p.data.sub_(2 * R, v)

        
        x = self.model(input)
        loss = self.model.calculate_loss(x, target)[0]

        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())  

        for p, v in zip(self.model.net_parameters(), vector):  
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]  
