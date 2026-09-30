from cobaya.likelihoods.roman_real._cosmolike_prototype_base import _cosmolike_prototype_base
import cosmolike_roman_real_interface as ci
import numpy as np

class cluster_lensing(_cosmolike_prototype_base):
  def initialize(self):
    super(cluster_lensing,self).initialize(probe="cluster_lensing")
