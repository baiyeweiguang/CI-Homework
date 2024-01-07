# coding: utf-8
# Copyright(C) Chengfu Zou 

import numpy as np
import yaml
import math

class Config:
  '''
  @brief: 参数
  '''
  def __init__(self, config_path):
    with open(config_path, 'r') as f:
      config = yaml.load(f, Loader=yaml.FullLoader)
    # 个体数
    self.pop_size = config['pop_size']
    # 精度
    self.precision = config['precision']
    # 交叉概率
    self.Cp = config['crossover_rate']
    # 变异概率
    self.Mp = config['mutation_rate']
    # 迭代次数
    self.iter_num = config['iter_num']
    
class ObjectiveFunc:
  '''
  @brief: 目标函数与适应度函数
  '''
  def calc_objective(self, x):
    # numpy 中 (n,) 和 (n, 1) 有区别，统一转换为 (n, 1)
    if x.ndim == 1:
       x = x.reshape(1, -1)
    val =  np.sum(-x**2 + 10 * np.cos(2 * math.pi * x) + 30, axis=1)
    return -val
    
  def calc_fitness(self, x):
    return -self.calc_objective(x)    
  
class GA:
  '''
  @brief: 遗传算法
  '''
  def __init__(self, objective_func, dim: int, bounds: tuple, cfg: Config):
    self.obj_func = objective_func.calc_objective
    self.fitness_func = objective_func.calc_fitness
    self.dim = dim
    self.bounds = bounds
    self.cfg = Config(cfg)
    
    self.encoding_length = math.ceil(math.log2((bounds[1] - bounds[0]) / self.cfg.precision)) * self.dim
    self.population = self.init_population(self.cfg.pop_size, self.encoding_length)
    
  def evolution(self):
    '''
    @brief: 迭代求解
    '''
    objective_list = []
    for i in range(self.cfg.iter_num):
      self.selection()
      self.crossover()
      self.mutation()
      decoded = self.decode(self.population, self.bounds, self.dim)
      fitness = self.fitness_func(decoded)
      max_fitness_index = np.argmax(fitness)
      
      objective = self.obj_func(decoded)
      objective_list.append(objective[max_fitness_index])
      
      print('iter: {}, best solution x: {}, f(x): {}'.format(i, decoded[max_fitness_index], objective[max_fitness_index]))
      
    decoded = self.decode(self.population, self.bounds, self.dim)
    fitness = self.fitness_func(decoded)
    objective = self.obj_func(decoded)
    max_fitness_index = np.argmax(fitness)
    
    return decoded[max_fitness_index], objective[max_fitness_index], objective_list
    
  @staticmethod 
  def decode(population: np.ndarray, bounds, dim) -> np.ndarray:
    '''
    @brief: 将二进制编码解码为实数
    '''
    pop_size, length = population.shape
    chorosome_len = length // dim
    lower, upper = bounds
    precision = (upper - lower) / (2 ** chorosome_len - 1)
    
    # weight = [..., 8, 4, 2, 1]
    weight = 2 ** np.arange(chorosome_len)
    weight = weight[::-1]
    weight = np.stack([weight] * dim, axis=0)
    
    decoded = lower + np.sum(population.reshape(pop_size, dim, chorosome_len) * weight, axis=2) * precision 
    
    return decoded
     
  @staticmethod
  def init_population(pop_size: int, length: int) -> np.ndarray:
    '''
    @brief: 初始化种群
    '''
    population = np.random.randint(2, size=(pop_size, length))
    return population 
  
  def selection(self):
    '''
    @brief: 轮盘赌选择
    '''
    fitness : np.ndarray = self.fitness_func(self.decode(self.population, self.bounds, self.dim))
    print(fitness)
    
    # max_fitness = np.max(fitness)
    max_fitness_index = np.argmax(fitness)
    min_fitness = np.min(fitness)
    parents = np.empty((self.cfg.pop_size, self.encoding_length))
    
    # 保留最优个体，且最优个体不参与选择
    parents[0] = self.population[max_fitness_index]
    fitness[max_fitness_index] = min_fitness
    
    # 随机选择剩余个体
    # fitness最小的个体被选中的概率为0
    min_fitness = abs(min_fitness)
    keep_prob = (fitness + min_fitness) / np.sum(fitness + min_fitness)
    for i in range(1, self.cfg.pop_size):
      parents[i] = self.population[np.random.choice(self.cfg.pop_size, p=keep_prob)] 
    
    self.population = parents
    
  def crossover(self):
    '''
    @brief: 交叉
    '''
    offspring = np.empty((self.cfg.pop_size, self.encoding_length))
    
    # 0是最优个体，不会在交叉中发生变化
    for i in range(1, self.cfg.pop_size):
      if np.random.rand() < self.cfg.Cp:
        # 随机选择交叉点
        cross_point = np.random.randint(0, self.encoding_length)
        father = self.population[i]
        mother = self.population[np.random.randint(0, self.cfg.pop_size)]
        offspring[i] = np.concatenate((father[:cross_point], mother[cross_point:]))
      else:
        offspring[i] = self.population[i]   
  
  def mutation(self):
    '''
    @brief: 变异
    '''
    # 0是最优个体，不参与变异
    for i in range(1, self.cfg.pop_size):
      mutation_points = np.random.rand(self.encoding_length) < self.cfg.Mp
      self.population[i][mutation_points] = 1 - self.population[i][mutation_points] 
        
if __name__ == '__main__':
  
  ga = GA(ObjectiveFunc(), dim=2, bounds=(-5, 5), cfg='GA.yaml')
  
  best_solution, objective, objective_list = ga.evolution()
  print('best solution: {} with objective value {}'.format(best_solution, objective))
  import matplotlib.pyplot as plt
  plt.plot(objective_list)
  plt.show()