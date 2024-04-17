import time, warnings
import jax, flax, optax, rlax, chex, gymnax, flashbax as fbx

from flax import linen as nn
from flax.training.train_state import TrainState

from jax_tqdm import scan_tqdm
from jax import lax, numpy as jnp, grad, vmap, jit

import matplotlib.pyplot as plt
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper

# Extended TrainState to allow discount and target_params
class MyTrainState(TrainState):    
    discount: float
    target_params: flax.core.FrozenDict    

@chex.dataclass(frozen=True)
class TimeStep:
    observation: chex.Array
    action: chex.Array
    discount: chex.Array
    reward: chex.Array
    
class DQN(nn.Module):
    num_actions: int

    def setup(self):
        self.l1 = nn.Dense(64)     
        self.l2 = nn.Dense(128)     
        self.l3 = nn.Dense(self.num_actions)
    
    def __call__(self, x):        
        x = nn.relu(self.l1(x))        
        x = nn.relu(self.l2(x))
        x = self.l3(x)
        return x

@jit
def optimal(state, params, obs):
    q = state.apply_fn({'params': params}, obs)
    a = jnp.argmax(q)
    return q, a

@jit
def e_greedy_policy(key, state, n, epsilon, obs):
    _, opt_act = optimal(state, state.params, obs)
        
    key, subkey = jax.random.split(key)
    rand_act = jax.random.randint(subkey, (1,), 0, n)[0]

    key, subkey = jax.random.split(key)
    return lax.cond(jax.random.uniform(subkey) > epsilon, lambda x: x[0], lambda x: x[1], (opt_act, rand_act))

def main(cfg):

    def train(rng):
        
        ############## Env Init #############

        env, env_params = gymnax.make("CartPole-v1")
        dim_a = env.action_space().n
        env = FlattenObservationWrapper(env)
        env = LogWrapper(env)

        ############## DQN Init #############

        dqn = DQN(env.action_space().n)

        rng, _rng = jax.random.split(rng)
        obs_shape = env.observation_space(env_params).shape
        dummy_input = jnp.zeros((1, obs_shape[0]))

        params = dqn.init(_rng, dummy_input)['params'] 

        rng, _rng = jax.random.split(rng)
        target_params = dqn.init(_rng, dummy_input)['params']
        
        lr_scheduler = optax.exponential_decay(init_value=cfg['lr_init'], transition_steps=cfg['num_epochs'], decay_rate=cfg['lr_decay'])        
        opt = optax.adam(learning_rate=lr_scheduler)
        train_state = MyTrainState.create(params=params, tx=opt, apply_fn=dqn.apply, discount=0.99, target_params=target_params)

        ############## Buffer Init #############

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")            
            buffer = fbx.make_flat_buffer(
                    max_length=cfg['buffer_size'],
                    min_length=cfg['num_steps'],
                    sample_batch_size=cfg['sample_size'],
                    add_sequences=False,
                    add_batch_size=cfg['num_envs'],
                )
            buffer = buffer.replace(
                init = jax.jit(buffer.init),
                add = jax.jit(buffer.add, donate_argnums=0),
                sample = jax.jit(buffer.sample),
                can_sample = jax.jit(buffer.can_sample),
            )

        rng, _rng = jax.random.split(rng)
        dummy_obs, _ = env.reset(_rng)    

        dummy_timestep = TimeStep(observation=dummy_obs, action=jnp.int32(0), reward=jnp.float32(0.0), discount=jnp.float32(0.0))
        buffer_state = buffer.init(dummy_timestep)

        #####################################

        rng, _rng = jax.random.split(rng)
        key_reset = jax.random.split(_rng, cfg['num_envs'])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(key_reset, env_params)

        @scan_tqdm(cfg['num_epochs'])
        def _step(state, _):
            
            def _eps_schedule(count):
                slope = 1.25*(cfg['eps_init'] - cfg['eps_min'])/cfg['num_epochs']  # Multiplied the slope with 1.25 to see some steps with pure exploration at the end
                return lax.max(cfg['eps_init'] - slope*count, cfg['eps_min'])      # Otherwise exploration reaches 0 at the end.      
            
            def _update_target_params(train_state):                                
                train_state = train_state.replace(target_params=train_state.params)
                return train_state
              
            # Function to perform one environment step  
            def _env_step(loop_state, unused):        
                rng, train_state, buffer_state, env_state, obs = loop_state

                key = vmap(jax.random.split, in_axes=(0))(rng)[:,:,1]
                rng = vmap(jax.random.split, in_axes=(0))(rng)[:,:,0]
                
                eps = _eps_schedule(train_state.step)
                action = vmap(e_greedy_policy, in_axes=(0, None, None, None, 0))(key, train_state, dim_a, eps, obs)
                        
                key_step = vmap(jax.random.split, in_axes=(0))(rng)[:,:,1]
                rng = vmap(jax.random.split, in_axes=(0))(rng)[:,:,0]
                
                obs1, env_state, reward, done, info = vmap(env.step, in_axes=(0, 0, 0, None))(key_step, env_state, action, env_params)    

                timestep = TimeStep(observation=obs, action=action, reward=reward, discount=info['discount'])
                buffer_state = buffer.add(buffer_state, timestep)

                loop_state = (rng, train_state, buffer_state, env_state, obs1)
                return loop_state, info                                   

            # Function to perform one training step
            def _train_step(train_state, batch):

                def _loss_fn(params, train_state, batch):
                    q_tm1, _ = vmap(optimal, in_axes=(None, None, 0))(train_state, params, batch.first.observation)
                    q_t_selector, _ = vmap(optimal, in_axes=(None, None, 0))(train_state, params, batch.second.observation)
                    q_t_values, _ = vmap(optimal, in_axes=(None, None, 0))(train_state, train_state.target_params, batch.second.observation)            
                    
                    td_error = vmap(rlax.double_q_learning, in_axes=0)(q_tm1, batch.first.action, batch.first.reward, train_state.discount*batch.first.discount, 
                                                                    q_t_values, q_t_selector)            
                    return jnp.mean(rlax.l2_loss(td_error))    
                
                loss, grads = jax.value_and_grad(_loss_fn)(train_state.params, train_state, batch)
                
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, loss 
                        
            rng, _rng = jax.random.split(state[0])        
            _rng = jax.random.split(_rng, cfg['num_envs'])    
            state = (_rng, state[1], state[2], state[3], state[4]) 
                                                
            state, info_batch = jax.lax.scan(_env_step, state, None, cfg["num_steps"])
                    
            rng, _rng = jax.random.split(rng)                 
            batch = buffer.sample(state[2], _rng).experience
                       
            train_state, loss = _train_step(state[1], batch)         
            train_state = lax.cond(train_state.step%10 == 0, _update_target_params, lambda x: x, (train_state))
            state = (rng, train_state, state[2], state[3], state[4]) 

            return state, info_batch
        
        state = (rng, train_state, buffer_state, env_state, obsv)        
        state, info_batch = jax.lax.scan(_step, state, jnp.arange(cfg['num_epochs']), cfg['num_epochs'])
        _, train_state, _, _, _ = state
        
        return train_state, info_batch
    
    return train

cfg = {'num_envs' : 64, 'num_steps': 1, 'buffer_size': 10000, 'sample_size': 512, 'num_epochs': 10000, 'eps_init': 0.65, 'eps_min': 0.0, 'lr_init': 5e-4, 'lr_decay': 0.99}

rng = jax.random.PRNGKey(42)
rngs = jax.random.split(rng, 10)  # Running 10 seeds concurrently
train_vjit = jax.jit(jax.vmap(main(cfg)))  # The whole training loop is jittable

t0 = time.time()
ts, outs = jax.block_until_ready(train_vjit(rngs))
print(f"time: {time.time() - t0:.2f} s")

for i in range(10):
    plt.plot(outs["returned_episode_returns"][i].mean(-1).reshape(-1))
plt.xlabel("Update Step")
plt.ylabel("Return")
plt.show()
plt.savefig('run.png')