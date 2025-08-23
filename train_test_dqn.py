# coding: utf-8

import os
import argparse
import time
import random
from collections import deque

import numpy as np
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print("Using GPUs:", gpus)

from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation,GlobalAveragePooling2D

import gymnasium as gym
# from spec_env import SpecEnv, MAX_STEPS  # uses your provided env
from spec_env2 import SpecEnv, MAX_STEPS  # uses your provided env
from util import (
    setup_logging,
    init_confusion, update_confusion, confusion_accuracy,
    save_confusion_csv, save_confusion_png, _save_curves_and_csv, CLASS_NAMES
)

# ------------------------------ Model ------------------------------

def get_image_model(in_shape, output):
    model = Sequential()
    model.add(Conv2D(64,  (3,3), padding='same', activation='relu', input_shape=in_shape))
    model.add(Conv2D(64,  (3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))    

    model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))    

    model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))    
    model.add(GlobalAveragePooling2D())   

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(output))                    
    return model
# ------------------------------ Utils ------------------------------

def preprocess_obs(obs):
    if obs.ndim == 2:
        obs = obs[..., None]
    obs = obs.astype(np.float32) /255
    return obs

def make_epsilon_schedule(eps_start, eps_end, decay_steps):
    """Linear decay schedule returning a function(step)->epsilon"""
    def eps_fn(step):
        if decay_steps <= 0:
            return eps_end
        ratio = min(1.0, step / float(decay_steps))
        return eps_start + ratio * (eps_end - eps_start)
    return eps_fn

# ------------------------------ Replay Buffer ------------------------------

class ReplayBuffer:
    def __init__(self, capacity, obs_shape, obs_dtype=np.float32):
        self.capacity = int(capacity)
        self.obs_shape = tuple(obs_shape)
        self.obs_buf = np.zeros((self.capacity,) + self.obs_shape, dtype=obs_dtype)
        self.next_obs_buf = np.zeros((self.capacity,) + self.obs_shape, dtype=obs_dtype)
        self.act_buf = np.zeros((self.capacity,), dtype=np.int32)
        self.rew_buf = np.zeros((self.capacity,), dtype=np.float32)
        self.done_buf = np.zeros((self.capacity,), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (self.obs_buf[idxs],
                self.act_buf[idxs],
                self.rew_buf[idxs],
                self.next_obs_buf[idxs],
                self.done_buf[idxs])

# ------------------------------ DQN Agent ------------------------------

class DQNAgentTF:
    def __init__(self, state_shape, n_actions, lr=1e-3, gamma=0.99, huber_delta=1.0):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.gamma = gamma

        self.q_net = get_image_model(state_shape, n_actions)
        self.target_q_net = keras.models.clone_model(self.q_net)
        self.target_q_net.set_weights(self.q_net.get_weights())

        self.optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.huber = keras.losses.Huber(delta=huber_delta, reduction="none")

    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        """
        One gradient update using a batch from replay memory.
        """
        # Compute target Q-values
        next_q = self.target_q_net(next_states, training=False)  # (B, A)
        next_max_q = tf.reduce_max(next_q, axis=1)               # (B,)

        target_q_vals = rewards + (1.0 - dones) * self.gamma * next_max_q  # (B,)

        with tf.GradientTape() as tape:
            q_vals = self.q_net(states, training=True)           # (B, A)
            idx = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)  # (B,2)
            chosen_q = tf.gather_nd(q_vals, idx)                 # (B,)
            loss_each = self.huber(target_q_vals, chosen_q)
            loss = tf.reduce_mean(loss_each)

        grads = tape.gradient(loss, self.q_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_net.trainable_variables))
        return loss

    def act(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        q_vals = self.q_net(state[None, ...], training=False).numpy()[0]
        return int(np.argmax(q_vals))

    def update_target(self):
        self.target_q_net.set_weights(self.q_net.get_weights())


def moving_average(x, window=100):
    if len(x) == 0:
        return []
    import numpy as _np
    x = _np.asarray(x, dtype=float)
    if window <= 1:
        return x
    cumsum = _np.cumsum(_np.insert(x, 0, 0.0))
    ma = (cumsum[window:] - cumsum[:-window]) / float(window)
    # pad to same length by prefixing with first available average
    pad = [ma[0]] * (window - 1)
    return _np.concatenate([pad, ma])
# ------------------------------ Training & Testing ------------------------------
def train_and_test(args):
    os.makedirs(args.out_dir, exist_ok=True)
    logger, _ = setup_logging(log_dir=args.out_dir, log_name_prefix="dqn")
    # call env 
    env = SpecEnv(type='train')
    test_env = SpecEnv(type='test')
    eval_env = SpecEnv(type='eval')
    # Infer input shape from a single reset
    obs0, _ = env.reset(), None
    obs0 = preprocess_obs(obs0)
    in_shape = obs0.shape  # (H, W, 1)
    n_actions = env.action_space.n
    agent = DQNAgentTF(
        state_shape=in_shape,
        n_actions=n_actions,
        lr=args.lr,
        gamma=args.gamma,
        huber_delta=args.huber_delta
    )
    logger.info(agent.q_net.summary())
    # Replay buffer
    buffer = ReplayBuffer(
        capacity=args.replay_capacity,
        obs_shape=in_shape,
        obs_dtype=np.float32
    )
    # Epsilon schedule
    eps_fn = make_epsilon_schedule(args.eps_start, args.eps_end, args.eps_decay_steps)
    global_step = 0
    num_updates = 0
    best_test_acc = 0.8
    ep_avg_qs = []
    train_losses = []  # collect per-gradient-step losses for plotting
    ep_returns = []      # sum of rewards per episode
    ep_success = []      # 1 if final prediction correct, else 0

    for ep in range(1, args.episodes + 1):
        obs = preprocess_obs(env.reset())
        done = False
        ep_reward = 0.0
        steps_this_ep = 0
        ep_q_sum = 0.0
        ep_q_count = 0
        q_values_this_ep = [] 
        last_action = None
        ## start
        while not done and steps_this_ep < args.train_steps_per_episode:
            if args.save_train_frames and (global_step % args.train_frame_interval == 0 or steps_this_ep == 0):
                save_path = os.path.join(args.out_dir, 'frames_train', f'ep{ep:04d}_step{steps_this_ep:03d}.png')
                env.render(show=False, save_path=save_path)

            epsilon = eps_fn(global_step)
            q_vals_now = agent.q_net(obs[None, ...], training=False).numpy()[0]
            ep_q_sum += float(np.max(q_vals_now))
            ep_q_count += 1

            action = agent.act(obs, epsilon=epsilon)
            last_action = action

            next_obs, reward, done, info = env.step(action)
            next_obs = preprocess_obs(next_obs)

            buffer.add(obs, action, reward, next_obs, done)
            ep_reward += reward
            obs = next_obs
            steps_this_ep += 1
            global_step += 1

            # Train
            if buffer.size >= args.learn_start and (global_step % args.train_freq == 0):
                (s_b, a_b, r_b, ns_b, d_b) = buffer.sample(args.batch_size)
                loss = agent.train_step(
                    tf.convert_to_tensor(s_b, dtype=tf.float32),
                    tf.convert_to_tensor(a_b, dtype=tf.int32),
                    tf.convert_to_tensor(r_b, dtype=tf.float32),
                    tf.convert_to_tensor(ns_b, dtype=tf.float32),
                    tf.convert_to_tensor(d_b, dtype=tf.float32),
                )
                num_updates += 1
                train_losses.append(float(loss.numpy() if hasattr(loss, 'numpy') else loss))

            if global_step % args.target_update_freq == 0:
                agent.update_target()

        # --- End of episode ---
        try:
            if last_action is not None and last_action >= 4:
                pred_class = last_action - 4
            else:
                pred_class = -1
            true_class = int(env.Y[env.i])
            success = 1 if pred_class == true_class else 0
        except Exception:
            success = 0

        ep_returns.append(float(ep_reward))
        ep_success.append(int(success))
        ep_avg_qs.append(float(ep_q_sum / max(1, ep_q_count)))
        logger.info(f"Episode {ep:04d} | steps={steps_this_ep:3d} | reward={ep_reward:.3f} "
                    f"| eps={epsilon:.3f} | buffer={buffer.size} | success={success} "
                    f"| pred={pred_class} true={true_class}")

        # ---------------- New Conditional Eval ----------------
        # if len(ep_success) >= args.eval_train_window:
        if ep >= args.eval_train_window:
            # Rolling training acc + F1
            train_acc = float(np.mean(ep_success[-args.eval_train_window:]))
            tp = sum(ep_success[-args.eval_train_window:])
            fn = args.eval_train_window - tp
            precision = tp / max(1, tp + fn)
            recall = precision
            train_f1 = (2*precision*recall)/(precision+recall) if (precision+recall) > 0 else 0.0

            logger.info(f"[TrainCheck] ep={ep:04d} train_acc={train_acc:.3f}, train_f1={train_f1:.3f}")

            if train_acc >= args.train_eval_threshold and train_f1 >= args.train_eval_threshold:
                # ---- Quick test ----
                testacc, cm, precision, recall, testf1 = evaluate(agent, test_env, args.eval_episodes,
                                                          render=args.eval_render,
                                                          out_dir=args.out_dir,
                                                          tag=f"test_ep{ep:04d}",
                                                          save_eval_frames=None)
                # frames_dir = os.path.join(args.out_dir, "frames_eval", f"ep{ep:06d}")
                # os.makedirs(frames_dir, exist_ok=True)
                # agent.q_net.save(os.path.join(frames_dir, f"model_at_ep{ep:06d}.keras"))
                logger.info(f"[TestCheck] ep={ep:04d} test_acc={testacc:.3f}, test_f1={testf1:.3f}")
                logger.info(f"[TestCheck] done: save test model \n ")

                if testacc >= args.test_eval_threshold and testf1 >= args.test_eval_threshold:
                    # ---- Full evaluation ----
                    frames_dir = os.path.join(args.out_dir, "frames_eval", f"ep{ep:06d}")
                    os.makedirs(frames_dir, exist_ok=True)
                    agent.q_net.save(os.path.join(frames_dir, f"model_at_ep{ep:06d}.keras"))
                    valacc, cm, precision, recall, valf1 = evaluate(agent, test_env, episodes =ep,render=args.eval_render,
                                                          out_dir=args.out_dir,
                                                          tag=f"eval_ep{ep:06d}",save_eval_frames=args.save_eval_frames)

                    # model_path, test_env, episodes=50, out_dir="ReportDQN", tag="eval_ep10000"
                    save_confusion_png(cm, os.path.join(frames_dir, f"confusion_ep{ep:06d}.png"),
                                       title=f"Confusion (ep{ep:06d})")
                    agent.q_net.save(os.path.join(frames_dir, f"eval_model_at_ep{ep:06d}.keras"))
                    logger.info(f"[Eval] ep={ep:04d} val_acc={valacc:.3f}, val_f1={valf1:.3f}")
                    logger.info(f"[Eval] done : save eval model")

                    
                    if acc > best_test_acc:
                        best_test_acc = acc
                        agent.q_net.save(os.path.join(args.out_dir, "best_q_model.keras"))
                        logger.info(f"Saved new best model with acc={acc:.3f}")

    # ---------------- Wrap up ----------------
    if len(train_losses) == 0 or num_updates == 0:
        print(f"[warn] No gradient updates performed. Replay size={buffer.size}, learn_start={args.learn_start}")

    # Save curves & csv
    ma_window = getattr(args, "reward_ma_window", 1000)
    ma_ret = moving_average(ep_returns, window=ma_window)
    ma_succ = moving_average(ep_success, window=ma_window)
    _save_curves_and_csv(args, train_losses, ep_returns, ep_success, ep_avg_qs, ma_window, ma_ret, ma_succ)

    # Final model
    agent.q_net.save(os.path.join(args.out_dir, "final_q_model.keras"))
    acc, cm, precision, recall, f1 = evaluate(agent, test_env, args.eval_episodes,
                                              render=args.eval_render,
                                              out_dir=args.out_dir,
                                              tag="final",
                                              save_eval_frames=args.save_eval_frames)
    logger.info(f"no evaluation is maded \n")
    logger.info(f"[Final Eval] acc={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}\nConfusion:\n{cm}")





def evaluate(agent, env, episodes, render=False, out_dir="Report", tag="eval", save_eval_frames=True):
    """Run greedy policy episodes on test env. Compute confusion & save artifacts."""
    cm = init_confusion(num_classes=4)
    total, correct = 0, 0

    for ep in range(1, episodes + 1):
        obs = preprocess_obs(env.reset())
        done = False
        steps = 0

        while not done and steps < MAX_STEPS:
            if render or save_eval_frames:
                save_path = os.path.join(out_dir, 'frames_eval', f'{tag}_ep{ep:04d}_step{steps:03d}.png')
                env.render(show=False, save_path=save_path)
            # Greedy action (epsilon=0)
            q_vals = agent.q_net(obs[None, ...], training=False).numpy()[0]
            action = int(np.argmax(q_vals))
            next_obs, reward, done, info = env.step(action)

            obs = preprocess_obs(next_obs)
            steps += 1
            last_action = action

         # ---- Decode prediction properly ----
        if last_action is not None and last_action >= 4:
            pred_class = last_action - 4   # classify(0–3)
        else:
            pred_class = -1  # no prediction made

        true_class = int(env.Y[env.i])
        # update confusion matrix only if a classification was attempted
        if pred_class >= 0:
            update_confusion(cm, true_class, pred_class)
            total += 1
            if pred_class == true_class:
                correct += 1

    acc = correct / max(1, total)

    # Compute precision, recall, f1
    cm = np.array(cm)
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp

    precision = np.mean([tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0 for i in range(len(tp))])
    recall    = np.mean([tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0 for i in range(len(tp))])
    f1 = (2*precision*recall)/(precision+recall) if (precision+recall)>0 else 0.0
    if f1 > 0.8:
        save_confusion_png(cm, os.path.join(out_dir, f"confusion_{tag}.png"), title=f"Confusion ({tag})")
    else:
        return acc, cm, precision, recall, f1

# ------------------------------ CLI ------------------------------

def save_config_to_file(args, filename='config.txt'):
    # Create the directory if it doesn't exist
    save_dir = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    try:
        with open(filename, 'w') as f:
            f.write("Configuration settings:\n")
            f.write("----------------------\n\n")
            for arg, value in vars(args).items():
                f.write(f"{arg}: {value}\n")
        print(f"Configuration successfully saved to {filename}")
    except IOError as e:
        print(f"Error saving file: {e}")

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default="ReportDQN", help="save logs/models/figures")
    p.add_argument("--episodes", type=int, default=10, help="Training episodes")
    p.add_argument("--train-steps-per-episode", type=int, default=MAX_STEPS, help="Cap on steps per episode")
    p.add_argument("--eval-one", type=int, default=50000, help="Evaluate once after N eps") 
    p.add_argument("--eval-episodes", type=int, default=150, help="number of tested image") ## number of tested image
    p.add_argument("--eval-render", action="store_true", help="Render during evaluation (optional)")
    p.add_argument("--save-train-frames", dest="save_train_frames", action="store_true", help="Save graphs during training")
    p.add_argument("--no-save-train-frames", dest="save_train_frames", action="store_false")
    p.set_defaults(save_train_frames=False)
    p.add_argument("--train-frame-interval", type=int, default=50, help="Save a training frame every N steps")
    p.add_argument("--save-eval-frames", action="store_true", help="Save graphs during evaluation")

    # DQN hyperparams
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--gamma", type=float, default=1)
    p.add_argument("--huber-delta", type=float, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--replay-capacity", type=int, default=200000) ##buffer 
    p.add_argument("--learn-start", type=int, default=50, help="Warmup steps before training")
    p.add_argument("--train-freq", type=int, default=100, help="Gradient step every N env steps")
    p.add_argument("--target-update-freq", type=int, default=20000, help="How often to copy Q->targetQ (in steps)") ## hard update
    p.add_argument("--reward-ma-window", type=int, default=1000, help="Window size for moving-average reward/success")
    p.add_argument("--es-threshold", type=float, default=0.85,help="Early stop if acc, precision, recall all >= this value at eval")
    # Epsilon schedule
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-steps", type=int, default=5000)

    ###training loop 
    p.add_argument("--eval-train-window", type=int, default=500,help="Episodes window for training performance check")
    p.add_argument("--train-eval-threshold", type=float, default=0.95,help="Threshold on training acc/F1 to trigger test eval")
    p.add_argument("--test-eval-threshold", type=float, default=0.8,help="Threshold on test acc/F1 to trigger full evaluation")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    print(f"Training DQN with args: {args}")
    config_filepath = os.path.join(args.out_dir, "config.txt")
    start = time.time()
    train_and_test(args)
    end = time.time()
    print(f"Training completed in {end - start:.2f} seconds.")
    print('end')

