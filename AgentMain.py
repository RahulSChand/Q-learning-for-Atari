import gym
from gym.wrappers import Monitor
import itertools
import numpy as np
import os
import random
import sys
import tensorflow as tf
import copyModelParameters
import deepQLearning

#Select breakout game as environment
env = gym.envs.make("Breakout-v0")

actionList = [0,1,2,3]


class Model():
	def __init__(self,scope="model"):
		with tf.variable_scope(scope):
			self.create_model()
	
	#Model architechure
	def create_model(self):
		self.X_pl = tf.placeholder(shape=[None,84,84,4],dtype = tf.uint8, name="X")
		self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")		
		self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")	
	
		int maxPixel = 255
		X = tf.to_float(self.X_pl) / 255.0
		
		#5 convulational layers
		convLayer1 = tf.contrib.layers.conv2d(X, 32, 8, 4, activation_fn=tf.nn.relu)
        	convLayer2 = tf.contrib.layers.conv2d(convLayer1, 64, 4, 2, activation_fn=tf.nn.relu)
        	convLayer3 = tf.contrib.layers.conv2d(convLayer2, 64, 3, 1, activation_fn=tf.nn.relu)
		convLayer4 = tf.contrib.layers.conv2d(convLayer3, 64, 3, 1, activation_fn=tf.nn.relu)

		convLayer5 = tf.contrib.layers.conv2d(convLayer4, 64, 3, 1, activation_fn=tf.nn.relu)
		
		#Fully connected Layer
		flattened = tf.contrib.layers.flatten(convLayer5)
			
		fcLayer = tf.contrib.layers.fully_connected(flattened, 512)
        	self.predictions = tf.contrib.layers.fully_connected(fcLayer, len(actionList))

		
		gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        	self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)


		#Loss
		self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        	self.loss = tf.reduce_mean(self.losses)

		self.optimizer = tf.train.AdamOptimizer(0.00025, 0.99, 0.0, 1e-6)
        	self.train_op = self.optimizer.minimize(self.loss,global_step=tf.contrib.framework.get_global_step())

	#To predict action
	def predictAction(self,sess,state):
		return sess.run(self.predictions, { self.X_pl: state })

	
	#Update Parameters
	def updateParameter(self,sess,state,action,output):
		feed_dict = { self.X_pl: state, self.y_pl: output, self.actions_pl: action }
        	summaries, global_step, _, loss = sess.run([self.summaries, tf.contrib.framework.get_global_step(), self.train_op, self.loss],
            feed_dict)
        	if self.summary_writer:
            		self.summary_writer.add_summary(summaries, global_step)
        	return loss

	

def epsilonGreedyPolicy(estimator,N):
	M= np.ones(N, dtype=float) * epsilon / N
	qValues = estimator.predict(sess, np.expand_dims(observation, 0))[0]
	best_action = np.argmax(qValues)
        M[best_action] += (1.0 - epsilon)
        return M


tf.reset_default_graph()

# Where we save our checkpoints and graphs
experiment_dir = os.path.abspath("./experiments/{}".format(env.spec.id))

# Create a glboal step variable
global_step = tf.Variable(0, name='global_step', trainable=False)
    
# Create estimators
q_estimator = Estimator(scope="q", summaries_dir=experiment_dir)
target_estimator = Estimator(scope="target_q")

# State processor
state_processor = StateProcessor()

# Run it!
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for t, stats in deep_q_learning(sess,
                                    env,
                                    q_estimator=q_estimator,
                                    target_estimator=target_estimator,
                                    state_processor=state_processor,
                                    experiment_dir=experiment_dir,
                                    num_episodes=10000,
                                    replay_memory_size=500000,
                                    replay_memory_init_size=50000,
                                    update_target_estimator_every=10000,
                                    epsilon_start=1.0,
                                    epsilon_end=0.1,
                                    epsilon_decay_steps=500000,
                                    discount_factor=0.99,
                                    batch_size=32):

        print("\nEpisode Reward: {}".format(stats.episode_rewards[-1]))

