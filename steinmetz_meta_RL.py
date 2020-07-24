# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 19:58:21 2020

@author: 44796
"""
#TODO: parallelise? maybe not worth it
#TODO: save params/accs and plots

import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import differential_evolution as de
f0 = np.load('C:\\Users\\44796\\Downloads\\steinmetz_part0.npz',allow_pickle=True)['dat']
f1 = np.load('C:\\Users\\44796\\Downloads\\steinmetz_part1.npz',allow_pickle=True)['dat']
f2 = np.load('C:\\Users\\44796\\Downloads\\steinmetz_part2.npz',allow_pickle=True)['dat']

alldat = np.hstack([f0,f1,f2])

class RL:   
    def __init__(self,dat,alpha=0.1,gamma=0.9):#pass mice data
        self.dat = dat
        self.states = [0,1,2,3,4,5] #0=no stimuli, #1=equal but not zero #2=left only #3=right only #4=left higher #5=right higher
        self.action = [0,1,2] #0=left 1=no-go 2=right
        #self.reward =[-1,1]
        self.q_values = np.zeros((len(self.states),len(self.action)))
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # how greedy mouse was 1 = full greed
        self.mouse_action = dat['response'] +1 # to make it 0,1,2 rather than -1,0,1
        self.mouse_reward = np.array(dat['feedback_type'])
        self.logLik = []
        self.model_action = []
        self.elen = len(self.mouse_action)
        self.condition = np.zeros(self.elen)
    
    #discretise states
    def setStates(self):    
        for trial in range(self.elen):
          right = self.dat['contrast_right'][trial]
          left = self.dat['contrast_left'][trial]
          if right > left and left > 0: #both present but right higher contrast
              self.condition[trial] = 5
          elif left > right and right >0: # both present but left is higher
              self.condition[trial] = 4
          elif left == 0 and right ==0: # no stimuli are present
              self.condition[trial] = 0
          elif right >0 and left==0: # right is present and left is not
              self.condition[trial] = 3
          elif left > 0 and right ==0: # left is present and right is not
              self.condition[trial] = 2
          elif left == right:
              self.condition[trial] =1
              

    def q_learn_update(self,state,action,reward):
      #print(state, action)
      old_q = self.q_values[state][action]
      prediction_error = reward - old_q
      new_q = old_q + (self.alpha * prediction_error)
      self.q_values[state][action] = new_q
    
    # model predict action
    def choose_action(self,state):
      state = int(state)
      # high gamma = exploit ----- low gamma = explore
      actionProbs = scipy.special.softmax(self.gamma*self.q_values[state])
      return actionProbs
    
    def compare_mouse_model(self):
      same = np.sum(self.mouse_action == self.model_action)  # comparison
      return same / self.elen # return proportion correct
    
    def run_model(self):
      ts = self.elen
      for i in range(ts):
        mouse_state = self.condition[i]
        mod_action = self.choose_action(mouse_state) 
        mouse_act = int(self.mouse_action[i])
        #we are basically maximising lik, the model's probability for choosing the action that the mouse chose
        lik = mod_action[mouse_act]
        # minimum of 1e-7 because log(0) is bad
        logLik = np.log(max(lik,1e-7))
        #print(logLik)
        self.logLik.append(logLik)    
        self.model_action.append(np.random.choice(3,p=mod_action))
        self.q_learn_update(int(mouse_state),int(mouse_act), int(self.mouse_reward[i]))


  
def testParamsVan(params,dat):
    global accuracy
    alpha,gamma = params
    #print(alpha,gamma)
    rl = RL(dat,alpha,gamma)
    rl.setStates()
    rl.run_model()
    return -np.sum(rl.logLik)

#important that gamma doesn't go too low or optimization breaks
accs = []
params = []
negLL = []
accs2 = []
params2 = []
negLL2 = []
for i in alldat:
  accuracy = 0
  # initial values for alpha and gamma
  test2 = [0.1,10]
  dat = i
  # bounds = bounds for params during optimisation
  #res2 = minimize(testParamsVan, bounds=[(0,1),(1e-1,1e+2)], x0=test2, args=dat,method='L-BFGS-B',options={'eps':1e-2})
  res2 = de(testParamsVan, bounds=[(0,1),(1e-1,1e+2)],args=(dat,)) # , needed in args for tuple
  rl2 = RL(dat,res2.x[0],res2.x[1])
  rl2.setStates()
  rl2.run_model()
  #hacky way of getting accuracy for param settings - will lead to variance in accuracy, but the mean should be correct and unbiased
  acclist = []
  print(res2)
  for j in range(100):
    acclist.append(rl2.compare_mouse_model())
  accuracy = np.mean(acclist)
  accs2.append(accuracy)
  params2.append(res2.x)
  negLL2.append(res2.fun)

print(accs2)
print(params2)
print(negLL2)

# break into sessions properly?
import seaborn as sns
alphas = [i[0] for i in params2]
gammas = [i[1] for i in params2]
ps2 = [alphas,gammas,accs2]
labs = ['alpha','gamma','accuracy']
ax = sns.swarmplot(data=ps2)
ax.set(xticklabels=labs)
plt.ylabel('Learning rate (alpha)')
print(params2)

#WSLS model - the switch chooses randomly between the other two actions - this could be more sophisticated
class WSLS:
  def __init__(self,dat,pWS=0.5,pLS=0.5,pWSFin=1,pLSFin=0,thetaW=0.01,thetaL=0.01):
    self.pWS = pWS#probability of staying aftr win
    self.pLS = pLS # probability of switching after loss 
    self.pWSFin = pWSFin #asymptotic value
    self.pLSFin = pLSFin #asymptotic value
    self.thetaW = thetaW #incremental value
    self.thetaL = thetaL #incremental value
    self.mouse_action = dat['response'] +1 # to make it 0,1,2 rather than -1,0,1
    self.mouse_reward = np.array(dat['feedback_type'])
    self.actions = [0,1,2]
    self.mod_acts = []
    self.elen = len(self.mouse_action)
    self.liks = []
    self.a_probs = []

  def choose_action(self,prev_act,prev_r,curr_act):
    lik = 0
    prev_act = int(prev_act)
    probs = np.array([1.,1.,1.])
    if prev_r == 1:
      switch_prob = (1-self.pWS) / 2
      probs *= switch_prob
      probs[prev_act] = self.pWS
      if prev_act == curr_act:
        lik = self.pWS
      else:
        lik = (1 - self.pWS)/2
      if np.random.uniform() < self.pWS: # repeat winning action
        act = prev_act
      else: # choose between other two actions randomly
        possible = [i for i in self.actions if i != prev_act]
        act = possible[np.random.randint(2)]
    else: # prev punished
      switch_prob = self.pLS / 2
      probs *= switch_prob
      probs[prev_act] = (1-self.pLS)
      if prev_act == curr_act:
        lik = 1 - self.pLS
      else:
        lik = self.pLS / 2
      if np.random.uniform() > self.pLS: # repeat losing action
        act = prev_act
      else: # choose between other two actions randomly
        possible = [i for i in self.actions if i != prev_act]
        act = possible[np.random.randint(2)]
    self.a_probs = probs
    return act,lik
 
  def update_probs(self):
    if self.pWS < self.pWSFin:
      self.pWS = min(self.pWSFin,self.pWS + self.thetaW) # ensure doesn't cross maximum
    if self.pLS < self.pLSFin:
      self.pLS = min(self.pLSFin,self.pLS + self.thetaL)

  def run_model(self):
    prev_r = 0
    prev_act = 0
    curr_act = 0
    for i in range(self.elen-1):
      if i > 0:
        prev_r = self.mouse_reward[i-1]
        prev_act = self.mouse_action[i-1]
        curr_act = self.mouse_action[i]
      mod_act,lik = self.choose_action(prev_act,prev_r,curr_act)
      self.mod_acts.append(float(mod_act))
      self.liks.append(max(lik,1e-7))
      self.update_probs()
  
  def compare_mouse_model(self):
    same = np.sum(self.mouse_action == self.mod_acts)  # comparison
    return same / self.elen # return proportion correct



def testParamsWS(params,dat):
    pWS,pLS,pWSFin,pLSFin,thetaW,thetaL = params
    #print(alpha,gamma)
    errs = []
    for i in range(10):
      wsls = WSLS(dat,pWS,pLS,pWSFin,pLSFin,thetaW,thetaL)
      #print(params)
      wsls.run_model()
      errs.append(sum([i!=j for i,j in zip(wsls.mod_acts,wsls.mouse_action)])/wsls.elen)
    return np.mean(errs)
    #return -np.sum(np.log(wsls.liks)) # neg lok lik

#TODO need to update to get continues values to minimise (eg, repeat prob)
accsWS = []
paramsWS = []
nLLWS = []
for i in alldat:
  accuracy = 0
  # initial values for params
  test = [0.5,0.5,1,1,1e-2,1e-2]
  dat = i
  # bounds = bounds for params during optimisation
  pmax = 0
  #res = minimize(testParamsWS, bounds=[(0,1),(0,1),(1,1),(1,1),(0,0),(0,0)], x0=test, args=dat,method='L-BFGS-B',options={'eps':1e-2})
  res = de(testParamsWS, bounds=[(0,1),(0,1),(0,1),(0,1),(0,1e-1),(0,1e-1)],args=(dat,)) # , needed in args for tuple
  print(res)
  acclist = []
  for j in range(100):
    wsls = WSLS(dat,res.x[0],res.x[1],res.x[2],res.x[3],res.x[4],res.x[5])
    wsls.run_model()
    #hacky way of getting accuracy for param settings - will lead to variance in accuracy, but the mean should be correct and unbiased
    acclist.append(sum([i==j for i,j in zip(wsls.mod_acts,wsls.mouse_action)])/wsls.elen)
  accuracy = np.mean(acclist)
  accsWS.append(accuracy)
  paramsWS.append(res.x)
  nLLWS.append(res.fun)
 

print(accsWS)
print(paramsWS)


# WSLS-RL model from Worthy & Maddox 2014

class META:
  def __init__(self,dat,pWS,pLS,pWSFin,pLSFin,thetaW,thetaL,alpha,gamma,kws):
    self.kws = kws # weight for wsls 
    self.wsls=WSLS(dat,pWS,pLS,pWSFin,pLSFin,thetaW,thetaL)
    self.rl=RL(dat,alpha,gamma)
    self.mouse_action = dat['response'] +1 # to make it 0,1,2 rather than -1,0,1
    self.mouse_reward = np.array(dat['feedback_type'])
    self.elen = len(self.mouse_action)
    self.mod_acts = []
    self.liks = []

  def run_model(self):
    prev_r = 0
    prev_act = 0
    curr_act = 0
    self.rl.setStates()
    for i in range(self.elen-1):
      if i > 0:
        prev_r = self.mouse_reward[i]
        prev_act = self.mouse_action[i]
        curr_act = self.mouse_action[i+1]
      ws_act,ws_lik = self.wsls.choose_action(prev_act,prev_r,curr_act)
      ws_probs = wsls.a_probs
      mouse_state = self.rl.condition[i]
      rl_probs = self.rl.choose_action(mouse_state) 
      mouse_act = int(self.mouse_action[i])
      meta_probs = (self.kws*ws_probs) + ((1-self.kws)*rl_probs)
      meta_act = np.random.choice(3,p=meta_probs)
      self.mod_acts.append(meta_act)
      meta_lik = meta_probs[mouse_act]
      self.liks.append(max(meta_lik,1e-5))
      self.wsls.update_probs()
      self.rl.q_learn_update(int(mouse_state),int(mouse_act), int(self.mouse_reward[i]))
      
def testParamsMeta(params,dat):
    pWS,pLS,pWSFin,pLSFin,thetaW,thetaL,alpha,gamma,kws = params
    #print(alpha,gamma)
    errs = []
    for i in range(10): #reduce variance
      meta = META(dat,pWS,pLS,pWSFin,pLSFin,thetaW,thetaL,alpha,gamma,kws)
      #print(params)
      meta.run_model()
      err = sum([i!=j for i,j in zip(meta.mod_acts,meta.mouse_action)])/meta.elen
      errs.append(err)
    meanErr = np.mean(errs) #sum or mean
    return meanErr # 
    #return -np.sum(np.log(meta.liks)) # neg lok lik

#TODO time varying param for kws? or just try HMM?
# what should determine HMM belief - previous mouse behaviour?

accsMeta = []
paramsMeta = []
nLLMeta = []
for i in alldat:
  accuracy = 0
  # initial values for params
  test = [0.9,0.9,1,1,0,0,0.5,10,0.5]
  dat = i
  # bounds = bounds for params during optimisation
  #res = minimize(testParamsMeta, bounds=[(0,1),(0,1),(1,1),(1,1),(0,0),(0,0),(0,1),(1e-1,1e+2),(0.01,0.99)], x0=test, args=dat,method='L-BFGS-B',options={'eps':1e-2})
  res = de(testParamsMeta, bounds=[(0,1),(0,1),(0,1),(0,1),(0,1e-1),(0,1e-1),(0,1),(1e-1,1e+2),(0.01,0.99)],args=(dat,)) # , needed in args for tuple
  print(res)
  acclist = []
  for j in range(100):
    meta = META(dat,res.x[0],res.x[1],res.x[2],res.x[3],res.x[4],res.x[5],res.x[6],res.x[7],res.x[8])
    meta.run_model()
    #hacky way of getting accuracy for param settings - will lead to variance in accuracy, but the mean should be correct and unbiased
    acclist.append(sum([i==j for i,j in zip(meta.mod_acts,meta.mouse_action)])/meta.elen)
  accuracy = np.mean(acclist)
  accsMeta.append(accuracy)
  paramsMeta.append(res.x)
  nLLMeta.append(res.fun)
 

print(accsWS)
print(paramsWS)

plt.bar(x = np.arange(len(accsMeta)),height=accsMeta,alpha=0.5,color = 'magenta')
plt.bar(x = np.arange(len(accs2)),height=accs2,alpha=0.5,color = 'cyan')
plt.bar(x = np.arange(len(accsWS)),height=accsWS,alpha=0.5,color = 'yellow')
plt.xlabel('session')
plt.ylabel('proportion correct model choices')

dat = alldat[0]
response = dat['response'] #+1 # to make it 0,1,2 rather than -1,0,1
feedb = np.array(dat['feedback_type'])
con_right = dat['contrast_right']
con_left = dat['contrast_left']
#need to calculate what correct action actually was
correct_acts = []
for i in range(len(feedb)):
    if con_right[i] == 0 and con_left[i] == 0:
        correct_acts.append(0)
    elif con_right[i] > con_left[i]:
        correct_acts.append(-1)
    elif con_right[i] < con_left[i]:
        correct_acts.append(1)
    else: # equal contrast - maybe remove
        correct_acts.append(np.random.choice([-1,1]))
correct_acts = np.array(correct_acts) + 1 #to make it 0,1,2 rather than -1,0,1

testPs = paramsMeta[0]
testAccs = []
correct = []
meanSame = []
for i in range(100):
    tmeta = META(dat,testPs[0],testPs[1],testPs[2],testPs[3],testPs[4],testPs[5],testPs[6],testPs[7],testPs[8])
    tmeta.run_model()
    correct.append([p==j for p,j in zip(tmeta.mod_acts,correct_acts)])
    testAccs.append(([p==j for p,j in zip(tmeta.mod_acts,tmeta.mouse_action)]))
meanAcc = np.mean(correct,axis=0)
meanSame = np.mean(testAccs,axis=0)

#plotting mouse accuracy over time

accuracy = np.zeros(len(feedb))
accuracy_rm = np.zeros(len(feedb))
mod_acc_rm = np.zeros(len(feedb))
mean_same_rm = np.zeros(len(feedb))
window = 50
for i in range(len(feedb)):
    #get rid of same level contrast trials with 50% reward chance regardless of left or right
    if feedb[i] == 1:# and (con_right[i] != con_left[i] or con_right[i] == 0): #allow a no-go trial
        accuracy[i] = 1.
    else: #elif (con_right[i] != con_left[i] or con_right[i] == 0):
        accuracy[i] = 0.
    # Accuracy running mean
    accuracy_rm[i] = np.mean(accuracy[max(0,i-window):i])
    mod_acc_rm[i] = np.mean(meanAcc[max(0,i-window):i])
    mean_same_rm[i] = np.mean(meanSame[max(0,i-window):i])
#print(accuracy_rm)
plt.plot(accuracy_rm)
plt.plot(mod_acc_rm)
plt.plot(mean_same_rm)
plt.xlabel('trials')
plt.ylabel('running average accuracy - window size = ' + str(window))
plt.show()
print(np.mean(testAccs))
