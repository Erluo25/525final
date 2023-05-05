#import carla
import time

#from automatic_control_GRAIC import RACE_ENV
from utils import *
from starformer import *
from starformer_critic import *
from automatic_control_GRAIC import RACE_ENV

class StarAgent():
  def __init__(self, episode_num, gamma, a_lr, c_lr, batch_size,\
               batch_round, update_round, step_limit,\
               action_dim, action_bound, rb_max, input_dim,\
              collision_weight, distance_weight, center_line_weight, 
              render, round_precision, stuck_counter_limit, maxT, patch_length):
    
    # Basic parameters for the A2C methods
    self.episode_num = episode_num
    self.gamma = gamma
    self.batch_size = batch_size
    self.batch_round = batch_round
    self.update_round = update_round # Every update_round update the critic copy net
    self.step_limit = step_limit
    self.action_dim = action_dim
    self.action_bound = action_bound
    self.input_dim = input_dim
    self.training_reward_x = []
    self.training_reward_y = []
    self.maxT = maxT
    self.patch_length = patch_length

    # Basic starformer configurations
    self.actor_config = StarformerConfig(action_dim, action_dim + action_dim * action_dim, vector_length = self.input_dim, patch_length = self.patch_length,
                         context_length=30, pos_drop=0.1, resid_drop=0.1,
                          N_head=8, D=192, local_N_head=4, local_D=64, model_type='star', max_timestep=100, n_layer=6, maxT=10, 
                          action_type='continuous')
    self.critic_config = StarformerConfig_C(1, vector_length = input_dim, patch_length = self.patch_length, 
                        context_length=30, pos_drop=0.1, resid_drop=0.1,
                          N_head=8, D=192, local_N_head=4, local_D=64, model_type='star', max_timestep=100, n_layer=6, maxT=10, 
                          action_type='continuous')
    self.actor_config_train = TrainerConfig(learning_rate=a_lr, lr_decay=True, maxT=self.maxT)

    self.critic_config_train = TrainerConfig(learning_rate=c_lr, lr_decay=True, maxT=self.maxT)
    
    # Actor and Critic networks setup and loss function for ciritc network
    self.act_net = Starformer(self.actor_config).to(device)
    self.critic_net = Starformer_C(self.critic_config).to(device)

    self.actor_optimizer = self.act_net.configure_optimizers(self.actor_config_train)
    self.critic_optimizer = self.critic_net.configure_optimizers(self.critic_config_train)

    self.critic_net_copy = None
    self.critic_loss_func = nn.MSELoss()

    # The following is the replay buffers
    self.current_state_rb = None
    self.next_state_rb = None
    self.action_rb = None
    self.done_list_rb = None
    self.reward_rb = None
    self.index_rb = None
    self.rb_size = 0
    self.rb_max = rb_max

    self.collision_weight = collision_weight
    self.distance_weight = distance_weight
    self.center_line_weight = center_line_weight
    self.render = render
    self.round_precision = round_precision
    self.stuck_counter_limit = stuck_counter_limit
  
  def run_step(self, state):
    result_state = convert_state_to_tensor(state)
    action = self.sample_action_from_state_gaussian(result_state)
    control = get_control_from_action(action)
    return control
  
  def forward_state(self, visiting_states, visiting_actions):
    act_net_result, _, _ = self.act_net(visiting_states, visiting_actions, None) 
    mean = act_net_result[..., :self.action_dim]
    cov_mat = act_net_result[..., self.action_dim:].view(-1, self.action_dim, self.action_dim)
    transpose_cov_mat = cov_mat.transpose(1, 2)
    cov_mat = torch.bmm(cov_mat, transpose_cov_mat)
    id_mat = torch.eye(self.action_dim).repeat(cov_mat.size(0), 1, 1).to(device)
    cov_mat = cov_mat + id_mat
    return mean, cov_mat
  
  def sample_action_from_state_gaussian(self, visiting_states, visiting_actions):
    visiting_states = visiting_states.to(torch.float32)
    mean, cov_mat = self.forward_state(visiting_states, visiting_actions)
    mean = mean.detach()
    cov_mat = cov_mat.detach()

    # Create the Gaussian distribution
    gaussian_distribution = MultivariateNormal(mean, cov_mat)

    # Get the action from the sample
    action = gaussian_distribution.sample().detach() 
    return action

  def train(self):
    update_counter = 0
    total_train_time = 0
    self.critic_net_copy = deepcopy(self.critic_net)
    try:
      env = RACE_ENV(args, collision_weight=self.collision_weight, \
                    distance_weight=self.distance_weight, \
                    center_line_weight=self.center_line_weight, \
                    render=self.render, round_precision=self.round_precision,\
                    stuck_counter_limit=self.stuck_counter_limit)
      
      for i in range (0, self.episode_num):
        # Main entry of the episode.
        episode_start_time = time.time()
        episode_reward = 0
        
        if update_counter == self.update_round:
          self.critic_net_copy = deepcopy(self.critic_net)
          update_counter = 0
        update_counter += 1
        
        # Reset the environment
        current_state, info = env.reset()
        current_state = convert_state_to_tensor(current_state)

        visiting_states = None
        visiting_actions = None
        input_counter = 0

        step_count = 0
        loop_end = False
        while loop_end is False:
          
          step_count += 1
          if step_count >= self.step_limit:
            loop_end = True
          
          assert (visiting_states is None and visiting_actions is None) or \
            (((visiting_states.size(0) == visiting_actions.size(0) + 1) or \
              (visiting_states.size(0)==1 and visiting_actions is None))\
             and visiting_states.size(0) <=self.maxT), "Error stacking visiting states"

          if visiting_states is None:
            visiting_states = deepcopy(current_state)
          else:
            if visiting_states.size(0) == self.maxT:
              assert visiting_actions.size(0) == self.maxT -1, "Max truncation error"
              visiting_states = visiting_states[1:, ...]
              visiting_actions = visiting_actions[1:, ...]
              visiting_states = torch.vstack((visiting_states, current_state))
          # Step the environment based on the selected action
          # Note: this action is on GPU and is a tensor
          # Add a dummpy action padding
          if visiting_actions is None:
            visiting_actions = torch.zeros(1, self.action_dim).to(device)
          else:
            visiting_actions = torch.vstack((visiting_actions, torch.zeros(1, visiting_actions.size(1)).to(device)))
          
          # Convert the form of input
          visiting_states = visiting_states.view(1, visiting_states.size(0), visiting_states.size(1))
          visiting_actions = visiting_actions.view(1, visiting_actions.size(0), visiting_actions.size(1))
          action = self.sample_action_from_state_gaussian(visiting_states, visiting_actions)
          
          # Convert the forms back
          visiting_states = visiting_states.view(visiting_states.size(1), -1)
          visiting_actions = visiting_actions.view(visiting_actions.size(1), -1)
          visiting_actions = visiting_actions[:-1, ...] # Throuw away the dummy action
          visiting_actions = torch.vstack((visiting_actions, action))
          assert visiting_states.size(0) == visiting_actions.size(0), "Visting states and actions are not eqaul"
        
          # Convert the action to control object before stepping.
          control = get_control_from_action(action)
          
          next_state, reward, terminated, truncated = env.step(control)
          #print("Reward is: ", reward)
          
          next_state = convert_state_to_tensor(next_state)
          episode_reward += reward
        
          # Process the done tensor and reward tensor and index tensor 
          done_tensor = torch.tensor([[1]]).to(device) if terminated or truncated\
                                    else torch.tensor([[0]]).to(device)
          reward_tensor = torch.tensor([[reward]]).to(device).to(torch.float32)
          
          # Need a tensor to store the current index in the trajectory
          index_tensor = torch.tensor([[step_count - 1]]).to(device)

          # Stacking the replay buffer
          if self.rb_size == 0:
            # Initializing the replay buffer
            assert self.current_state_rb is None and self.next_state_rb is None \
              and self.action_rb is None and self.done_list_rb is None and \
              self.reward_rb is None and self.index_rb is None, "Invalid initial replay buffer size"
            self.current_state_rb = deepcopy(current_state)
            self.next_state_rb = deepcopy(next_state)
            self.action_rb = deepcopy(action)
            self.done_list_rb = deepcopy(done_tensor)
            self.reward_rb = deepcopy(reward_tensor)
            self.index_rb = deepcopy(index_tensor)
            self.rb_size += 1
          else:
            # Checking the replay buffer is stacking in the right size
            assert self.current_state_rb.size(0) == self.next_state_rb.size(0)\
              == self.action_rb.size(0) == self.done_list_rb.size(0) ==\
              self.reward_rb.size(0) == self.rb_size, "Invalid stack size during stacking"
            assert self.current_state_rb.size() == self.next_state_rb.size(),\
                "Current state stack size not equal to next state stack size "
            assert self.action_rb.size(1) == self.action_dim, \
                "Action stack width is incorrect"
            assert self.done_list_rb.size(1) == self.reward_rb.size(1) == self.index_rb.size(1) == 1,\
                "Reward stack width and done list stack width is not eqaul to 1"
            
            # Stack to the existing replay buffer
            self.current_state_rb = torch.vstack((self.current_state_rb, current_state))
            self.next_state_rb = torch.vstack((self.next_state_rb, next_state))
            self.action_rb = torch.vstack((self.action_rb, action))
            self.done_list_rb = torch.vstack((self.done_list_rb, done_tensor))
            self.reward_rb = torch.vstack((self.reward_rb, reward_tensor))
            self.index_rb = torch.vstack((self.index_rb, index_tensor))
            self.rb_size += 1

            # Under the case where rb exceed max, throw out prevous elements
            if self.rb_size > self.rb_max:
              self.current_state_rb = self.current_state_rb[-self.rb_max:]
              self.next_state_rb = self.next_state_rb[-self.rb_max:]
              self.action_rb = self.action_rb[-self.rb_max:]
              self.done_list_rb = self.done_list_rb[-self.rb_max:]
              self.reward_rb = self.reward_rb[-self.rb_max:]
              self.index_rb = self.index_rb[-self.index_rb]
              self.rb_size = self.rb_max
              assert self.current_state_rb.size(0) == self.rb_size,\
                  "Resizing replay buffer error."

          # Don't forget to update current state
          current_state = next_state

          # Perform the batch update for both actor and critic network
          for _ in range(0, self.update_round):
            self.batch_update()

          # If this state action reaches a final state, end the episode
          if terminated or truncated:
            loop_end = True
            
        # Here reach the end of each episode
        episode_end_time = time.time()
        episode_duration = episode_end_time - episode_start_time
        total_train_time += episode_duration
        self.training_reward_x.append(i)
        self.training_reward_y.append(episode_reward)
        print("Episode ", i, " finish takes time: ", episode_duration,\
              " with reward: ", episode_reward)
        if (i % 100 == 0):
          torch.save(self.act_net.state_dict(), "./actor_str.pth")
          torch.save(self.critic_net.state_dict(), "./critic_str.pth")
          x = torch.tensor(self.training_reward_x)
          y = torch.tensor(self.training_reward_y)
          torch.save(x, 'tx_str.pt')
          torch.save(y, 'ty_str.pt')

      print("Total training time is: ", total_train_time)
    finally:
      env.close()
    return


  def batch_update(self):
    # Under the StARformer case the training process will be 
    # 1. pick a batch size
    # 2. randomly select the corresponding state action reward pairs
    # 3. Find back T position of the sequence and pad if any element missing

    batch_size = min(self.batch_size, self.rb_size)
    if self.rb_size != 0:
      assert self.current_state_rb.size(0) == self.next_state_rb.size(0)\
            == self.action_rb.size(0) == self.done_list_rb.size(0) ==\
            self.reward_rb.size(0) == self.rb_size, "Invalid stack size (batch)"
    else:
      assert True, "No replay buffer exist!"

    assert self.current_state_rb.size() == self.next_state_rb.size(),\
            "Current state stack size not equal to next state stack size (batch)"
    assert self.action_rb.size(1) == self.action_dim, \
            "Action stack width is incorrect (batch)"
    assert self.done_list_rb.size(1) == self.reward_rb.size(1) == self.index_rb.size(1) == 1,\
            "Reward stack width and done list stack width is not eqaul to 1 (batch)"
    
    # Compute the batch indices
    batch_idx_range = list(range(0, self.rb_size))
    select_batch_idxs = torch.tensor(random.sample(batch_idx_range, batch_size)).to(device)

    # Expand to trajectories
    position_in_traj = self.index_rb[select_batch_idxs, ...].view(-1, 1)

    # Final stack list
    state_stack_list = []
    next_state_stack_list = []
    action_stack_list = []

    for i in range(0, batch_size):
        select_idx_global = select_batch_idxs[i].item()
        corr_pos_in_traj = position_in_traj[i].item()
        min_go_back_dist = min(corr_pos_in_traj, self.maxT-1)

        if select_idx_global < min_go_back_dist:
            expanded_idx_range = list(range(0, select_idx_global + 1))
            length = select_idx_global + 1
        else:
            expanded_idx_range = list(range(select_idx_global - min_go_back_dist, select_idx_global + 1))
            length = min_go_back_dist + 1


        expanded_idx_range = torch.tensor(expanded_idx_range)

        state_stack_list.append(self.current_state_rb[expanded_idx_range, ...].to(torch.float32))
        next_state_stack_list.append(self.next_state_rb[expanded_idx_range, ...].to(torch.float32))
        action_stack_list.append(self.action_rb[expanded_idx_range, ...])

        # Add the padding to the lists
        pad_num = self.maxT - length
        assert pad_num >=0, "Invalid padding length"
        if pad_num > 0:
          state_stack_list.append(torch.zeros(pad_num, self.current_state_rb.size(1)).to(device))
          next_state_stack_list.append(torch.zeros(pad_num, self.next_state_rb.size(1)).to(device))
          action_stack_list.append(torch.zeros(pad_num, self.action_rb.size(1)).to(device))

    
    # Combine the stack lists
    current_state_stack = torch.vstack(state_stack_list).view(batch_size, self.maxT, -1)
    next_state_stack = torch.vstack(next_state_stack_list).view(batch_size, self.maxT, -1)
    action_stack = torch.vstack(action_stack_list).view(batch_size, self.maxT, -1)
    done_list = self.done_list_rb[select_batch_idxs, ...].view(-1, 1)
    trans_reward = self.reward_rb[select_batch_idxs, ...].view(-1, 1)

    # Get the indices indicate the transaction terminates the trajectory
    done_indices = torch.where(done_list == 1)[0].long()

    # 1. Regress the critic network
    
    # 1-1. Use the critic network to compute the current output
    critic_output, _, _ = self.critic_net(current_state_stack) # Contain gradients
    critic_output = critic_output.view(-1, 1)

    # 1-2. Use the copy critic network to compute the target
    critic_target, _, _ = self.critic_net_copy(next_state_stack) # No gradients
    critic_target = critic_target.detach().view(-1, 1)

    # Zero out the case where the end of trajectory reach and no need to compute
    # the future expected reward.
    critic_target[done_indices] = 0

    # 1-3 Compute the traget of the critic network
    critic_target = self.gamma * critic_target + trans_reward

    # 1-4 Regress the critic network
    critic_loss = self.critic_loss_func(critic_output, critic_target)
    for cp in self.critic_net.parameters():
      cp.grad = None
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.critic_config_train.grad_norm_clip)
    self.critic_optimizer.step()

    # 2. Compute the new critic weight A(s, a) value
    
    # 2-1 To construct Q(s_{t}, a) = r(s_{t}, a) + \gamma V(s_{t+1}), we need 
    # to compute V(s_{t+1})
    expt_next_state_val, _, _ = self.critic_net(next_state_stack)
    expt_next_state_val = expt_next_state_val.detach().view(-1, 1)
    expt_next_state_val[done_indices] = 0

    # 2-2 Compose Q(s_{t}, a)
    q_value = self.gamma * expt_next_state_val + trans_reward

    # 2-3 Compute V(s_{t})
    v_value, _, _ = self.critic_net(current_state_stack)
    v_value = v_value.detach().view(-1, 1)

    # 2-4 Compose A(s_t, a_t) = Q(s_t, a_t) - V(s_t)
    a_value = q_value - v_value

    # 3. Use actor network to compute the log prob of the current_state, action pair
    
    # 3-1 Feed the group of trajs into the actor to get the mean and covariance
    mean, cov_mat = self.forward_state(current_state_stack)

    # 3-2 Construct the Gaussian distribution
    distribution = MultivariateNormal(mean, cov_mat) 

    # 3-3 Compute the log prob
    log_prob = distribution.log_prob(action_stack)
    log_prob = log_prob.view(1, -1)#.to(torch.float64)

    # 3-4 Compute the multiplication with a_value
    dot_prod = -1 * log_prob @ a_value

    # 3-5 Backprop and update gradients
    #self.act_net.zero_grad() 
    for ap in self.act_net.parameters():
      ap.grad = None
    dot_prod.backward()
    torch.nn.utils.clip_grad_norm_(self.act_net.parameters(), self.actor_config_train.grad_norm_clip)
    self.actor_optimizer.step()
    #with torch.no_grad():
    #  for param in self.act_net.parameters():
    #    param += self.act_net.lr * torch.clip(param.grad, -1, 1)
    # Done with the single batch update
    return
