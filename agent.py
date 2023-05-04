#import carla
import time

#from automatic_control_GRAIC import RACE_ENV
from utils import *
from automatic_control_GRAIC import RACE_ENV

class Agent1():
    def __init__(self, vehicle=None):
        self.vehicle = vehicle

    def run_step(self, filtered_obstacles, waypoints, vel, transform, boundary, distance):
        """
        Execute one step of navigation.
        Args:
        filtered_obstacles
            - Type:        List[carla.Actor(), ...]
            - Description: All actors except for EGO within sensoring distance
        waypoints 
            - Type:         List[[x,y,z], ...] 
            - Description:  List All future waypoints to reach in (x,y,z) format
        vel
            - Type:         carla.Vector3D 
            - Description:  Ego's current velocity in (x, y, z) in m/s
        transform
            - Type:         carla.Transform 
            - Description:  Ego's current transform
        boundary 
            - Type:         List[List[left_boundry], List[right_boundry]]
            - Description:  left/right boundary each consists of 20 waypoints,
                            they defines the track boundary of the next 20 meters.
        Return: carla.VehicleControl()
        """
        # Actions to take during each simulation step
        # Feel Free to use carla API; however, since we already provide info to you, using API will only add to your delay time
        # Currently the timeout is set to 10s
        # Print the state information 
        #print("Obstacles are: ", filtered_obstacles)
        #wps = np.array(waypoints)
        #print("Waypoints shape: ", wps.shape)
        #print("Velocity are: ", vel.x, vel.y, vel.z)
        #print(f"Current transformation location x { transform.location.x}, y {transform.location.y} ")
        #print(f"Carla transform rotation yaw {transform.rotation.yaw}, orientation: {np.deg2rad(transform.rotation.yaw)}")
        #left, right = extract_road_boundary(boundary)
        #print(f"Boundary information Left shape: {left.shape}, Right shape: {right.shape}")
        #print("The distance is: ", distance)



        #state = (filtered_obstacles, waypoints, vel, transform, boundary,distance)
        #state = convert_state_to_tensor(state)
        # 
        print("Reach Customized Agent")
        control = carla.VehicleControl()
        control.throttle = 0.5
        return control


# Actor Critic Implementation

class Actor_Net_Num(nn.Module):
  def __init__(self, action_dim, action_bound, lr, input_dim):
     super(Actor_Net_Num, self).__init__()
     self.action_dim = action_dim
     self.action_bound = action_bound
     self.lr = lr
     self.input_dim = input_dim
     
     self.fc0 = nn.Linear(input_dim, 2048)
     self.fc2 = nn.Linear(2048, 1024)
     self.fc3 = nn.Linear(1024, 512)
     self.fc4 = nn.Linear(512, self.action_dim + self.action_dim * self.action_dim)
     
     # Need to have tanh to bound the output action in range [-3, 3]
     self.tanh = nn.Tanh()
     self.sigmoid = nn.Sigmoid()

     self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

  def forward(self, x):
    x = F.leaky_relu(self.fc0(x))
    #x = F.leaky_relu(self.fc1(x))
    x = F.leaky_relu(self.fc2(x))
    x = F.leaky_relu(self.fc3(x))
    x = self.fc4(x)

    # Obtain the mean and bound the mean
    m = x[..., :self.action_dim]
    #print("Dimension of mean is: ", m.size())
    #print("Dimension of action_bound is: ", self.action_bound.size())
    mean = self.action_bound * self.tanh(m)

    # Obtain the covariance matrix
    cov_mat = x[..., self.action_dim:].view(-1, self.action_dim, self.action_dim)
    cov_mat = self.sigmoid(cov_mat)
    transpose_cov_mat = cov_mat.transpose(1, 2)
    cov_mat = torch.bmm(cov_mat, transpose_cov_mat)
    id_mat = torch.eye(self.action_dim).repeat(cov_mat.size(0), 1, 1).to(device)
    cov_mat = cov_mat + id_mat
    # Return the bounded mean and covariance matrix as output
    return mean, cov_mat

  def sample_action_from_state_gaussian(self, state):
    state = state.to(torch.float32)
    mean, cov_mat = self.forward(state)
    mean = mean.detach()
    cov_mat = cov_mat.detach()

    # Create the Gaussian distribution
    gaussian_distribution = MultivariateNormal(mean, cov_mat)

    # Get the action from the sample
    action = gaussian_distribution.sample().detach() 
    return action
  
class Critic_Net_Num(nn.Module):
  def __init__(self, lr, input_dim):
     super(Critic_Net_Num, self).__init__()
     self.lr = lr
     self.input_dim = input_dim
     
     self.fc0 = nn.Linear(input_dim, 2048)
     self.fc2 = nn.Linear(2048, 1024)
     self.fc3 = nn.Linear(1024, 512)
     self.fc4 = nn.Linear(512, 1)
     self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
  
  def forward(self, x):
      x = F.leaky_relu(self.fc0(x))
      #x = F.leaky_relu(self.fc1(x))
      x = F.leaky_relu(self.fc2(x))
      x = F.leaky_relu(self.fc3(x))
      x = F.leaky_relu(self.fc4(x))
      return x

class Agent():
  def __init__(self, episode_num, gamma, a_lr, c_lr, batch_size,\
               batch_round, update_round, step_limit,\
               action_dim, action_bound, rb_max, input_dim,\
              collision_weight, distance_weight, center_line_weight, 
              render, round_precision, stuck_counter_limit):
    
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
    
    # Actor and Critic networks setup and loss function for ciritc network
    self.act_net = Actor_Net_Num(action_dim=action_dim, action_bound=action_bound, lr=a_lr, \
                                 input_dim = self.input_dim).to(device)
    self.critic_net = Critic_Net_Num(lr = c_lr, input_dim=self.input_dim).to(device)
    self.critic_net_copy = None
    self.critic_loss_func = nn.MSELoss()

    # The following is the replay buffers
    self.current_state_rb = None
    self.next_state_rb = None
    self.action_rb = None
    self.done_list_rb = None
    self.reward_rb = None
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
    action = self.act_net.sample_action_from_state_gaussian(result_state)
    control = get_control_from_action(action)
    return control
  	
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

        step_count = 0
        loop_end = False
        while loop_end is False:
          
          step_count += 1
          if step_count >= self.step_limit:
            loop_end = True

          # Step the environment based on the selected action
          # Note: this action is on GPU and is a tensor
          action = self.act_net.sample_action_from_state_gaussian(current_state)
          
          # Convert the action to control object before stepping.
          control = get_control_from_action(action)
          
          next_state, reward, terminated, truncated = env.step(control)
          #print("Reward is: ", reward)
          
          next_state = convert_state_to_tensor(next_state)
          episode_reward += reward
        
          # Process the done tensor and reward tensor
          done_tensor = torch.tensor([[1]]).to(device) if terminated or truncated\
                                    else torch.tensor([[0]]).to(device)
          reward_tensor = torch.tensor([[reward]]).to(device).to(torch.float32)

          # Stacking the replay buffer
          if self.rb_size == 0:
            # Initializing the replay buffer
            assert self.current_state_rb is None and self.next_state_rb is None \
              and self.action_rb is None and self.done_list_rb is None and \
              self.reward_rb is None, "Invalid initial replay buffer size"
            self.current_state_rb = deepcopy(current_state)
            self.next_state_rb = deepcopy(next_state)
            self.action_rb = deepcopy(action)
            self.done_list_rb = deepcopy(done_tensor)
            self.reward_rb = deepcopy(reward_tensor)
            self.rb_size += 1
          else:
            # Checking the replay buffer is stacking in the right size
            assert self.current_state_rb.size(0) == self.next_state_rb.size(0)\
              == self.action_rb.size(0) == self.done_list_rb.size(0) ==\
              self.reward_rb.size(0) == self.rb_size, "Invalid stack size during stacking"
            assert self.current_state_rb.size() == self.next_state_rb.size(),\
                "Current state stack size not equal to next state stack size "
            assert self.action_rb.size(1) == self.act_net.action_dim, \
                "Action stack width is incorrect"
            assert self.done_list_rb.size(1) == self.reward_rb.size(1) == 1,\
                "Reward stack width and done list stack width is not eqaul to 1"
            
            # Stack to the existing replay buffer
            self.current_state_rb = torch.vstack((self.current_state_rb, current_state))
            self.next_state_rb = torch.vstack((self.next_state_rb, next_state))
            self.action_rb = torch.vstack((self.action_rb, action))
            self.done_list_rb = torch.vstack((self.done_list_rb, done_tensor))
            self.reward_rb = torch.vstack((self.reward_rb, reward_tensor))
            self.rb_size += 1

            # Under the case where rb exceed max, throw out prevous elements
            if self.rb_size > self.rb_max:
              self.current_state_rb = self.current_state_rb[-self.rb_max:]
              self.next_state_rb = self.next_state_rb[-self.rb_max:]
              self.action_rb = self.action_rb[-self.rb_max:]
              self.done_list_rb = self.done_list_rb[-self.rb_max:]
              self.reward_rb = self.reward_rb[-self.rb_max:]
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
          torch.save(self.act_net.state_dict(), "./actor.pth")
          torch.save(self.critic_net.state_dict(), "./critic.pth")
          x = torch.tensor(self.training_reward_x)
          y = torch.tensor(self.training_reward_y)
          torch.save(x, 'tx.pt')
          torch.save(y, 'ty.pt')

      print("Total training time is: ", total_train_time)
    finally:
      env.close()
    return

  def batch_update(self):
    batch_size = min(self.batch_size, self.rb_size)
    if self.rb_size != 0:
      assert self.current_state_rb.size(0) == self.next_state_rb.size(0)\
            == self.action_rb.size(0) == self.done_list_rb.size(0) ==\
            self.reward_rb.size(0) == self.rb_size, "Invalid stack size (batch)"
    else:
      assert True, "No replay buffer exist!"

    assert self.current_state_rb.size() == self.next_state_rb.size(),\
            "Current state stack size not equal to next state stack size (batch)"
    assert self.action_rb.size(1) == self.act_net.action_dim, \
            "Action stack width is incorrect (batch)"
    assert self.done_list_rb.size(1) == self.reward_rb.size(1) == 1,\
            "Reward stack width and done list stack width is not eqaul to 1 (batch)"
    
    # Compute the batch indices
    batch_idx_range = list(range(0, self.rb_size))
    select_batch_idxs = torch.tensor(random.sample(batch_idx_range, batch_size)).to(device)

    # Load the selected data from the replay buffer
    current_state_stack = self.current_state_rb[select_batch_idxs, ...].to(torch.float32)
    next_state_stack = self.next_state_rb[select_batch_idxs, ...].to(torch.float32)
    action_stack = self.action_rb[select_batch_idxs, ...]
    done_list = self.done_list_rb[select_batch_idxs, ...].view(-1, 1)
    trans_reward = self.reward_rb[select_batch_idxs, ...].view(-1, 1)

    # Get the indices indicate the transaction terminates the trajectory
    done_indices = torch.where(done_list == 1)[0].long()

    # 1. Regress the critic network
    
    # 1-1. Use the critic network to compute the current output
    critic_output = self.critic_net(current_state_stack).view(-1, 1) # Contain gradients

    # 1-2. Use the copy critic network to compute the target
    critic_target = self.critic_net_copy(next_state_stack).detach().view(-1, 1) # No gradients

    # Zero out the case where the end of trajectory reach and no need to compute
    # the future expected reward.
    critic_target[done_indices] = 0

    # 1-3 Compute the traget of the critic network
    critic_target = self.gamma * critic_target + trans_reward

    # 1-4 Regress the critic network
    critic_loss = self.critic_loss_func(critic_output, critic_target)
    self.critic_net.zero_grad()
    critic_loss.backward()
    self.critic_net.optimizer.step()

    # 2. Compute the new critic weight A(s, a) value
    
    # 2-1 To construct Q(s_{t}, a) = r(s_{t}, a) + \gamma V(s_{t+1}), we need 
    # to compute V(s_{t+1})
    expt_next_state_val = self.critic_net(next_state_stack).detach().view(-1, 1)
    expt_next_state_val[done_indices] = 0

    # 2-2 Compose Q(s_{t}, a)
    q_value = self.gamma * expt_next_state_val + trans_reward

    # 2-3 Compute V(s_{t})
    v_value = self.critic_net(current_state_stack).detach().view(-1, 1)

    # 2-4 Compose A(s_t, a_t) = Q(s_t, a_t) - V(s_t)
    a_value = q_value - v_value

    # 3. Use actor network to compute the log prob of the current_state, action pair
    
    # 3-1 Feed the group of trajs into the actor to get the mean and covariance
    mean, cov_mat = self.act_net(current_state_stack) # 
    
    # 3-2 Construct the Gaussian distribution
    distribution = MultivariateNormal(mean, cov_mat) 

    # 3-3 Compute the log prob
    log_prob = distribution.log_prob(action_stack)
    log_prob = log_prob.view(1, -1)#.to(torch.float64)

    # 3-4 Compute the multiplication with a_value
    dot_prod = log_prob @ a_value

    # 3-5 Backprop and update gradients
    self.act_net.zero_grad() 
    dot_prod.backward()
    with torch.no_grad():
      for param in self.act_net.parameters():
        param += self.act_net.lr * torch.clip(param.grad, -1, 1)
    # Done with the single batch update
    return

"""
if __name__ == "__main__":
    #env = RACE_ENV(args, collision_weight=30, distance_weight=5, center_line_weight=5, render=False, round_precision=2, stuck_counter_limit=15)
      
    act_net = Actor_Net_Num(action_dim=2, action_bound=torch.tensor([math.pi / 6, 6]).to(device), lr=0.01, \
                                 input_dim = 356).to(device)
    #loaded_state_dict = torch.load("./actor.pth")
    #act_net.load_state_dict(loaded_state_dict)

    input = torch.randn((356)).to(device)
    print("input size is: ", input.size())
    m, c = act_net(input)
    print(m, c)
    a = act_net.sample_action_from_state_gaussian(input)
    print("action is: ", a)
    a = convert_action_type(a)
    print(a[0])
    
    #torch.save(act_net.state_dict(), "./actor.pth")
    print()
"""
