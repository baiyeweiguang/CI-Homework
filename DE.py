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
    # crossover control paramter
    self.Cr = config['crossover_rate']
    # 缩放因子
    self.F = config['F']
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
    
class DE:
  '''
  @brief: 差分进化算法 用的rand/1/bin
  '''
  def __init__(self, objective_func, dim: int, bounds: tuple, cfg: Config):
    self.obj_func = objective_func.calc_objective
    self.fitness_func = objective_func.calc_fitness
    self.dim = dim
    self.bounds = bounds
    self.cfg = Config(cfg)
    
    self.population = self.init_population(self.cfg.pop_size, self.dim, self.bounds)
    
  def evolution(self):
    '''
    @brief: 迭代求解
    '''
    objective_list = []
    for iter in range(self.cfg.iter_num):
      mutant_vectors = self.mutation()
      trial_vectors = self.crossover(mutant_vectors)
      
      population_fitness = self.fitness_func(self.population)
      trial_fitness = self.fitness_func(trial_vectors)
      
      # 如果trial_fitness > fitness 则替换原个体
      better = trial_fitness > population_fitness
      self.population[better] = trial_vectors[better]
       
      objective = self.obj_func(self.population)
      fitness = self.fitness_func(self.population)
      max_fitness_index = np.argmax(fitness)
      objective_list.append(objective[max_fitness_index])
      print('iter: {}, best solution x: {}, f(x): {}'.format(iter, self.population[max_fitness_index], objective[max_fitness_index]))
      
    objective = self.obj_func(self.population)  
    fitness = self.fitness_func(self.population)
    max_fitness_index = np.argmax(fitness)
    
    return self.population[max_fitness_index], objective[max_fitness_index], objective_list
  
  def mutation(self) -> np.ndarray:
    '''
    @brief: rand/1 变异操作
    '''
    mutant_vectors = np.zeros((self.cfg.pop_size, self.dim))
    for i in range(self.cfg.pop_size):
      # 随机选择三个个体
      r1, r2, r3 = np.random.choice(self.cfg.pop_size, 3, replace=False)
      mutant_vectors[i] = self.population[r1] + self.cfg.F * (self.population[r2] - self.population[r3])
    
    return mutant_vectors 
          
  def crossover(self, mutant_vectors: np.ndarray) -> np.ndarray:
    '''
    @brief: binary 交叉操作
    '''
    lower, upper = self.bounds
    trial_vectors = np.zeros((self.cfg.pop_size, self.dim))
    for i in range(self.cfg.pop_size):
      for j in range(self.dim):
        j_rand = np.random.randint(self.dim)
        # j == j_rand 保证至少有一个维度发生变化
        if np.random.rand() < self.cfg.Cr or j == j_rand:
          trial_vectors[i][j] = mutant_vectors[i][j]
        else:
          trial_vectors[i][j] = self.population[i][j]
          
        # 变异向量的每个维度都在边界内
        if trial_vectors[i][j] < lower:
          trial_vectors[i][j] = min(upper, 2 * lower - trial_vectors[i][j])
        if trial_vectors[i][j] > upper:
          trial_vectors[i][j] = max(lower, 2 * upper - trial_vectors[i][j])

    return trial_vectors
          
    
  @staticmethod
  def init_population(pop_size, dim, bounds):
    '''
    @brief: 初始化种群
    '''
    lower, upper = bounds
    # 随机初始化种群（0，1）
    population = np.random.rand(pop_size, dim)
    # 缩放到指定范围
    population = lower + population * (upper - lower)
        
    return population
  
if __name__ == '__main__':
  
  de = DE(ObjectiveFunc(), dim=2, bounds=(-5, 5), cfg='DE.yaml')
  best_solution, best_value, objective_list = de.evolution()
  print('best solution: {}, with objective value: {}'.format(best_solution, best_value))
  import matplotlib.pyplot as plt
  plt.plot(objective_list)
  plt.show()