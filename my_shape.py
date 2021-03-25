#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 10:38:04 2021

@author: matt
"""

import numpy as np

#following description in Thob et al. 2019
#(see e.g. Dubinski & Carlberg 1991; Bett 2012; Schneider et al. 2012)

def reduced_quadrupole_moments_of_mass_tensor(r_p, m_p, e2_p):
  '''Calculates the reduced inertia tensor
  M_i,j = sum_p m_p/r_~p^2 . r_p,i r_p,j / sum_p m_p/r_p^2
  Itterative selection is done in the other function.
  '''
  norm = m_p / e2_p

  m = np.zeros((3,3))

  for i in range(3):
    for j in range(3):
      m[i,j] = np.sum(norm * r_p[:, i] * r_p[:, j])

  m /= np.sum(norm)

  return(m)

def process_tensor(m):
  '''
  '''
  #DO NOT use np.linalg.eigh(m)
  #looks like there is a bug
  if np.any(np.isnan(m)):
    print('Warning: Nans in tensor. Returning identity instead.')
    return(np.ones(3, dtype=np.float64), np.identity(3, dtype=np.float64))

  (eigan_values, eigan_vectors) = np.linalg.eig(m)

  order = np.flip(np.argsort(eigan_values))

  eigan_values  = eigan_values[order]
  eigan_vectors = eigan_vectors[order]

  return(eigan_values, eigan_vectors)

def defined_particles(pos, mass, eigan_values, eigan_vectors):
  '''Assumes eigan values are sorted
  '''
  #projection along each axis
  projected_a = (pos[:,0] * eigan_vectors[0,0] + pos[:,1] * eigan_vectors[0,1] +
                 pos[:,2] * eigan_vectors[0,2])
  projected_b = (pos[:,0] * eigan_vectors[1,0] + pos[:,1] * eigan_vectors[1,1] +
                 pos[:,2] * eigan_vectors[1,2])
  projected_c = (pos[:,0] * eigan_vectors[2,0] + pos[:,1] * eigan_vectors[2,1] +
                 pos[:,2] * eigan_vectors[2,2])

  #ellipse distance #Thob et al. 2019 eqn 4.
  ellipse_distance = (np.square(projected_a) + np.square(projected_b) / (eigan_values[1]/eigan_values[0]) +
                      np.square(projected_c) / (eigan_values[2]/eigan_values[0]) )

  #ellipse radius #Thob et al. 2019 eqn 4.
  ellipse_radius = np.power(np.square(eigan_values[0]) / (eigan_values[1] * eigan_values[2]), 1/3
                            ) * np.square(30)

  #Thob et al. 2019 eqn 4.
  inside_mask = ellipse_distance <= ellipse_radius

  return(pos[inside_mask], mass[inside_mask], ellipse_distance[inside_mask])

def find_abc(pos, mass):
  '''Finds the major, intermediate and minor axes.
  Follows Thob et al. 2019 using quadrupole moment of mass to bias towards
  particles closer to the centre
  '''
  #start off speherical
  r2 = np.square(np.linalg.norm(pos, axis=1))

  #stop problems with r=0
  pos  = pos[r2 != 0]
  mass = mass[r2 != 0]
  r2   = r2[r2 != 0]

  #mass tensor of particles
  m = reduced_quadrupole_moments_of_mass_tensor(pos, mass, r2)

  #linear algebra stuff
  (eigan_values, eigan_vectors) = process_tensor(m)

  #to see convergeance
  cona = np.sqrt(eigan_values[2] / eigan_values[0])
  bona = np.sqrt(eigan_values[1] / eigan_values[0])

  done = False
  for i in range(100):

    #redefine particles, calculate ellipse distance
    (pos, mass, ellipse_r2) = defined_particles(pos, mass, eigan_values, eigan_vectors)

    #mass tensor of new particles
    m = reduced_quadrupole_moments_of_mass_tensor(pos, mass, ellipse_r2)

    #linear algebra stuff
    (eigan_values, eigan_vectors) = process_tensor(m)

    if (1 - np.sqrt(eigan_values[2] / eigan_values[0]) / cona < 0.01) and (
        1 - np.sqrt(eigan_values[1] / eigan_values[0]) / bona < 0.01):
      #converged
      done = True
      break

    else:
      cona = np.sqrt(eigan_values[2] / eigan_values[0])
      bona = np.sqrt(eigan_values[1] / eigan_values[0])

  if not done:
    print('Warning: Shape did not converge.')

  if len(mass) < 100:
    print('Warning: Defining shape with <100 particles.')

  return(np.sqrt(eigan_values))

##############################################################################
#everything else is for testing
def nfw_m(r, rs=10):
  '''
  '''
  x = r/rs

  # m = np.log(1 + x) - x / (1 + x)
  #hernquist
  m = np.square(x) / np.square(1 + x)

  return(m)

def nfw_r(m, rs=10):
  '''
  '''
  sqm = np.sqrt(m)

  #hernquist
  x = - sqm / (sqm - 1)

  r = x * rs

  return(r)

def spherical_r(r):
  '''
  '''
  n = len(r)
  u = 2*np.random.rand(n, 3) -1
  u = u / np.linalg.norm(u, axis=1)[:, np.newaxis]

  out = u * r[:, np.newaxis]

  return(out)

def mock_nfw(n=10000, cona=0.2, bona=0.8, rs=10, rmax=1000):
  '''Is actually Hernquist. Couldn't work out inverse cumulate mass profile.
  '''
  mmax = 1.2

  mrand = np.random.rand(n)

  rrand = nfw_r(mrand / mmax, rs)

  pos = spherical_r(rrand)

  pos[:, 1] *= bona
  pos[:, 2] *= cona

  return(pos)

if __name__ == '__main__':

  pos = mock_nfw()
  mass = np.ones(len(pos))

  #random rotation
  from scipy.spatial.transform import Rotation
  random = Rotation.from_euler('yx', [360*np.random.rand(),360*np.random.rand()], degrees=True)
  pos = random.apply(pos)

  out = find_abc(pos, mass)

  # print(out)
  print(out[2] / out[0], out[1] / out[0])

  pass
