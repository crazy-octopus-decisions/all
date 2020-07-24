def behaviour(data):

  '''
  INPUTS: session level data

  OUTPUTS:
  0. [array] Trial Type: Switch = -1 / Repeat = 1
  1. [list] as above but as a list (S,R)
  2. [array] Accuracy for each trial
  3. [array] Accuracy cumulat at each trial
  4. [array] Target Location: Left -1 / Right = 1 / No diff or no target = 0
  5. [array] Target contrast difference left - right
  6. [array] Maximum wheel value (how much to the right)
  7. [array] Minimum wheel value (how much to the left)
  8. [array] Maximum - minimum wheel value
  9. [list] Condition; cL/cR = higher contrast left/right, cE = equal contrast, nG = No go (no stim) 
  10. [array] Trial Type, Previous Trial
  11. [array] Reward, Previous Trial
  '''
  
  wheel_data = data['wheel']
  response = data['response']
  feedb = data['feedback_type']
  stim_left = data['contrast_left']
  stim_right = data['contrast_right']
  
  # for Outputs
  # Which one was the target based on the contrast difference
  target_location = np.zeros_like(response)
  
  # Contrast difference LEFT - RIGHT
  target_difference = np.zeros_like(response)
  
  # Coded as 1 = correct and -1 = incorrect
  # If response was not 0 and feedback = 1 --> Correct (1)
  # If response was not 0 and feedback = -1 --> Incorrect
  accuracy = np.zeros_like(response)

  # Accuracy running mean
  accuracy_rm = np.zeros_like(response)
  
  # Maximum value of wheel (to the right)
  wheel_max = np.zeros_like(response)
  
  # Minimum value of wheel (to the left)
  wheel_min = np.zeros_like(response)
  
  # Codes each trial as:
  # Switch = 1
  # Repeat = -1
  trial_sr = np.zeros_like(response)
  trial_sr[0] = None # no n-1 on first trial
  trial_sr_list = list()
  trial_sr_list.append(None)

  # Code conditions as:
  # cL - Contrast left higher
  # cR - Contrast right higher
  # cE - Contrast Equal
  # nG - No go
  conds = list()

  # For previous trial, first trial has None
  feedb_prev = np.append(None, feedb[0:len(feedb)-1])
  
  for i in range(0, len(target_location)):
    
    # specify target location as contrast difference
    if stim_left[i] > stim_right[i]:
      target_location[i] = -1.0
    elif stim_left[i] < stim_right[i]:
      target_location[i] = 1.0
    
    # calculate contrast diff left v right
    target_difference[i] = stim_left[i] - stim_right[i]

    # Accurate (1) if decission was left/right and rewarded
    # Inaccurate (-1) if decission was left/right and punished (noise)
    if response[i] != 0 and feedb[i] == 1:
      accuracy[i] = 1.
    elif response[i] != 0 and feedb[i] == -1:
      accuracy[i] = 0.

    # Accuracy running mean
    if i != 0:
      accuracy_rm[i] = np.sum(accuracy[0:i]) / i

    # Condition
    if stim_left[i] == stim_right[i] == 0:
      conds.append('nG')
    elif stim_left[i] > stim_right[i]:
      conds.append('cL')
    elif stim_left[i] < stim_right[i]:
      conds.append('cR')
    elif (stim_left[i] == stim_right[i]) and stim_left[i] != 0:
      conds.append('cE')
    else:
      conds.append(None)

    # wheel
    wheel_max[i] = np.max(wheel_data[0,i,:])
    wheel_min[i] = np.min(wheel_data[0,i,:])

    # code switch repeat
    if i != 0:
      if response[i] == response[i-1]:
        trial_sr[i] = -1.0
        trial_sr_list.append('Repeat')
      elif response[i] != response[i-1]:
        trial_sr[i] = 1.0
        trial_sr_list.append('Switch')
      else:
        trial_sr_list.append(None)

    # For previous trial, first trial has None
    trial_sr_prev = np.append(None, trial_sr[0:len(trial_sr)-1])
    
    
  return trial_sr, trial_sr_list, accuracy, accuracy_rm, target_location, \
   target_difference, wheel_max, wheel_min, wheel_max - wheel_min, conds, \
   trial_sr_prev, feedb_prev