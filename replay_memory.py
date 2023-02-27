# Copyright 2019 The TensorTrade Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License


import random

from collections import namedtuple
from typing import List
import numpy as np


Transition = namedtuple('Transition', ['state', 'action', 'reward', 'done'])


class ReplayMemory(object):

    def __init__(self, capacity: int, transition_type: namedtuple = Transition):
        self.capacity = capacity
        self.Transition = transition_type

        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(self.Transition(*args))
            #print(self.Transition(*args))
            #print(len(self.memory))
        else:
            self.memory.pop(0)
            self.memory.append(self.Transition(*args))
            #self.memory[self.position] = self.Transition(*args)

        self.position = (self.position + 1) % self.capacity
    
    def pop(self)-> List[namedtuple]:
        #print("HAKUBA")
        #print("antes",self.memory[-1])
        self.memory.pop()
        #print("depois", self.memory[-1])

    def sample(self, batch_size) -> List[namedtuple]:
        return random.sample(self.memory, batch_size)

    def head(self, batch_size) -> List[namedtuple]:
        return self.memory[:batch_size]

    def tail(self, batch_size) -> List[namedtuple]:
        return self.memory[-batch_size:]

    def __len__(self):
        return len(self.memory)

    def sample_DRQN_errado(self, batch_size, trace_length = 6) -> List[namedtuple]:
        #print(self.memory)
        print(self.position)
        indices = []
        sample = []
        #print(sample)
        while len(indices) < batch_size:
            index = random.randint(trace_length, len(self.memory)-1) 
            #ids = np.random.randint(low=0, high = len(self.memory)-1, size=batch_size)#pode repetir indice
            print("OLHA AQUI", ids           )
            history = self.memory[index - trace_length:index]
            #print(history)
            #print(len(indices))
            for item in history:
                #print(item)
                sample.append(item) 
            #print("final sample", sample)
            indices.append(index)
              
              
        return sample
        
    def sample_DRQN(self, batch_size, trace_length = 4) -> List[namedtuple]:
        #print(self.memory)
        print(self.position)
        indices = []
        sample = []
        #print(sample)
        while len(indices) < batch_size:
            index = random.randint(trace_length, len(self.memory)-1) 
            #ids = np.random.randint(low=0, high = len(self.memory)-1, size=batch_size)
            #print("OLHA AQUI", ids           )
            history = self.memory[index - trace_length:index]
            #print(history)
            #print(len(indices))
            for item in history:
                #print(item)
                sample.append(item) 
            #print("final sample", sample)
            indices.append(index)
              
        #print(sample)     
        return sample

    def sample_DRQN_pos_DODO_gambia(self,  trace_length ) -> List[namedtuple]:
        #print(self.memory)
        #print(self.position)
        indices = []
        sample = []
        #print(sample)
        
        index = random.randint(trace_length, len(self.memory)-1) 
        #ids = np.random.randint(low=0, high = len(self.memory)-1, size=batch_size)
        #print("OLHA AQUI", ids           )
        history = self.memory[index - trace_length:index]
        #print(history)
        #print(len(indices))
        for item in history:
            #print(item)
            sample.append(item) 
            #print("final sample", sample)
        indices.append(index)
              
        #print(sample)     
        return sample


class ReplayMemoryDRQN():
    def __init__(self, capacity = int):
        self.buffer = []
        self.buffer_size = capacity
    
    def add(self,experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1+len(self.buffer))-self.buffer_size] = []
        self.buffer.append(experience)
            
    def sample(self,batch_size,trace_length):
        sampled_episodes = random.sample(self.buffer,batch_size)
        sampledTraces = []
        for episode in sampled_episodes:
            point = np.random.randint(0,len(episode)+1-trace_length)
            sampledTraces.append(episode[point:point+trace_length])
        sampledTraces = np.array(sampledTraces)
        return np.reshape(sampledTraces,[batch_size*trace_length,5])
