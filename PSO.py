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
    # 迭代次数
    self.iter_num = config['iter_num']
    # c1
    self.c1 = config['c1']
    # c2
    self.c2 = config['c2']
    # vmax
    self.vmax = config['vmax']
    
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
    
   
   
class PSO:
  '''
  @brief: 粒子群优化算法
  '''
  def __init__(self, objective_func, dim, bounds, cfg) :
    self.obj_func = objective_func.calc_objective
    self.fitness_func = objective_func.calc_fitness
    self.dim = dim
    self.bounds = bounds
    self.cfg = Config(cfg)
    
    self.population = self.velocity = self.personal_best = np.zeros((self.cfg.pop_size, self.dim))
    self.global_best = np.zeros(self.dim)
  
    self.init_population()
    
  def evolution(self) -> (np.ndarray, float, list):
    '''
    @brief: 迭代求解
    @return 最优解，最优解对应的目标函数值，迭代过程中目标函数值列表
    '''
    objective_list = []
    best = self.global_best
    best_value = np.max(self.obj_func(best))
    
    for i in range(self.cfg.iter_num):
      best, best_value = self.move()
      print('iter: {}, best solution x: {}, f(x): {}'.format(i, best, best_value))
      objective_list.append(best_value)
      
    return best, best_value, objective_list
    
  def move(self) -> (np.ndarray, float):
    '''
    @brief: 粒子移动
    @return: 最优解，最优解对应的目标函数值
    '''
    self.update_velocity()
    self.update_position()
    
    # 更新gbest和pbest
    fitness = self.fitness_func(self.population)
    
    for i in range(self.cfg.pop_size):
      print(self.fitness_func(self.population[i]), self.fitness_func(self.personal_best[i]))
      if fitness[i] > self.fitness_func(self.personal_best[i]):
        self.personal_best[i] = self.population[i]
        
      if fitness[i] > self.fitness_func(self.global_best):
        self.global_best = self.population[i]
    
    return self.global_best, self.obj_func(self.global_best)
    
    
  def init_population(self) -> None:
    '''
    @brief: 初始化种群
    '''
    lower, upper = self.bounds
    self.population = np.random.rand(self.cfg.pop_size, self.dim) * (upper - lower) + lower
    self.velocity = np.random.rand(self.cfg.pop_size, self.dim) * (upper - lower)     
    self.personal_best = self.population
    
    fitness = self.fitness_func(self.population)
    gbest_fitness = fitness[0] 
    for i in range(self.cfg.pop_size):
      if fitness[i] > gbest_fitness:
        gbest_fitness = fitness[i]
        self.global_best = self.population[i]
    

  def update_velocity(self) -> None:
    '''
    @brief: 更新粒子速度
    '''
    for i in range(self.cfg.pop_size):
      r1, r2 = np.random.rand(2)
      c1, c2 = self.cfg.c1, self.cfg.c2
      pbest = self.personal_best[i]
      gbest = self.global_best
      xi = self.population[i]
      
      # 速度更新公式
      self.velocity[i] = self.velocity[i] + c1 * r1 * (pbest - xi) + c2 * r2 * (gbest - xi)
      # 避免越界
      self.velocity[i] = np.clip(self.velocity[i], -self.cfg.vmax, self.cfg.vmax)
        
      
  def update_position(self) -> None:
    '''
    @brief: 更新粒子位置
    '''
    self.population = self.population + self.velocity 
    
    # 避免越界
    lower, upper = self.bounds
    self.population = np.clip(self.population, lower, upper)
    
if __name__ == '__main__':
    
  pso = PSO(ObjectiveFunc(), 2, (-10, 10), 'PSO.yaml')
  best_solution, objective, objective_list = pso.evolution()
  print('best solution: {}, with objective value: {}'.format(best_solution, objective))
  
  import matplotlib.pyplot as plt
  plt.plot(objective_list)
  plt.show()
    
    