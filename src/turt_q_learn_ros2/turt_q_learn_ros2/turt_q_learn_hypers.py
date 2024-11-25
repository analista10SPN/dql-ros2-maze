#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from gazebo_msgs.srv import SpawnEntity, DeleteEntity, SetEntityState, GetWorldProperties
from collections import deque
from itertools import product
import datetime
import random
import math
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import asyncio

# Constants remain the same
crashDistances = {
    "turtlebot3_burger": .12,
    "turtlebot3_burger_front": .20,
    "turtlebot3_waffle": .22,
    "turtlebot3_waffle_front": .22,
}

ACTION_SPACE = [2.5, 1.25, 0, -1.25, -2.5]
TURTLEBOT_NAME = "turtlebot3_burger"
SCAN_MIN_DISTANCE = crashDistances[TURTLEBOT_NAME]
SCAN_MIN_DISTANCE_FRONT = crashDistances[TURTLEBOT_NAME + "_front"]
GOAL_MIN_DISTANCE = .35
MODEL_SPEED = .12
FINAL_SLICE = 50
ROLLING_AVERAGE_SAMPLES = 25

class ModelClass:
    def __init__(self, hyperParams, env, saveName="myModel - "):
        self.env = env
        self.turt_q_learn_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        if hyperParams["Load Model"]:
            self.model = tf.keras.models.load_model(self.turt_q_learn_path + "/load_model/model")
        else:
            self.model = self.genModel(
                hyperParams["Optimizer"], 
                hyperParams["Loss"],
                hyperParams["Learning Rate"],
                hyperParams["First Activation"],
                hyperParams["Hidden Activations"],
                hyperParams["Last Activation"],
                hyperParams["State Space"],
                hyperParams["Initializer"]
            )
        
        self.targetModel = tf.keras.models.clone_model(self.model)
        self.targetModel.set_weights(self.model.get_weights())
        self.episodes = hyperParams["Episodes"]
        self.episodeLength = hyperParams["Episode Length"]
        self.epsilon = hyperParams["Epsilon Initial"]
        self.epsilonMin = hyperParams["Epsilon Min"]
        self.epsilonDecay = hyperParams["Epsilon Decay"]
        self.gamma = hyperParams["Gamma"]
        self.memory = deque(maxlen=hyperParams["Memory Length"])
        self.batchSize = hyperParams["Batch Size"]
        self.stateSpace = hyperParams["State Space"]
        self.resetTargetMemories = hyperParams["Reset Target"]
        self.saveName = saveName + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.paramsString = self.dictToStr(hyperParams)
        self.memoryCount = 0

    def genModel(self, optimizer, loss, lr, first, hidden, last, stateSpace, initializer):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, input_shape=(stateSpace,), activation=first, kernel_initializer=initializer),
            tf.keras.layers.Dense(32, activation=hidden, kernel_initializer=initializer),
            tf.keras.layers.Dense(16, activation=hidden, kernel_initializer=initializer),
            tf.keras.layers.Dense(len(ACTION_SPACE), activation=last)
        ])
        model.compile(optimizer=optimizer(learning_rate=lr), loss=loss)
        return model

    async def playEpisodes(self):
        scores = []
        for epNum in range(self.episodes):
            self.env.get_logger().info(f"Episode {epNum}:")
            score = 0
            state = await self.env.resetState()
            
            for stepNum in range(self.episodeLength):
                action = self.getAction(state)
                newState, reward, done = await self.env.step(action)
                self.memory.append([state, action, newState, reward, done])
                state = newState
                score += reward
                if len(self.memory) > self.batchSize:
                    self.dqn()
                if done:
                    break
            
            if self.epsilon > self.epsilonMin:
                self.epsilon *= self.epsilonDecay
            
            scores.append(score)
            self.env.get_logger().info(f"\tScore: {score}")
        
        # Save results
        folderPath = f"/dqnmodels/{SESSION_NAME}/"
        savePath = self.turt_q_learn_path + folderPath + self.saveName
        os.makedirs(savePath, exist_ok=True)
        
        self.model.save(savePath + "/model")
        self.plotResults(scores, savePath)
        
        with open(savePath + '/params.txt', "w") as f:
            f.write(self.paramsString)
            
        return np.array(scores), self.movingAverage(scores), self.saveName

    def dqn(self):
        if len(self.memory) < self.batchSize:
            return
            
        mini_batch = random.sample(self.memory, self.batchSize)
        states = np.array([mem[0][0] for mem in mini_batch])
        actions = np.array([mem[1] for mem in mini_batch])
        
        # Predict Q-values for current states
        current_q = self.model.predict(states)
        # Predict Q-values for next states
        next_states = np.array([mem[2][0] for mem in mini_batch])
        future_q = self.targetModel.predict(next_states)
        
        # Update Q-values
        for i, (_, action, _, reward, done) in enumerate(mini_batch):
            if done:
                current_q[i][action] = reward
            else:
                current_q[i][action] = reward + self.gamma * np.amax(future_q[i])
        
        # Train the model
        self.model.fit(states, current_q, batch_size=self.batchSize, verbose=0)
        
        self.memoryCount += 1
        if self.memoryCount >= self.resetTargetMemories:
            self.targetModel.set_weights(self.model.get_weights())
            self.memoryCount = 0
            self.env.get_logger().info("Reset target network weights")

    def getAction(self, state):
        if random.random() > self.epsilon:
            return np.argmax(self.model.predict(state))
        return random.randint(0, len(ACTION_SPACE) - 1)

    def plotResults(self, scores, savePath):
        plt.figure(2)
        averages = self.movingAverage(scores)
        plt.plot(scores, label="Scores")
        plt.plot(np.arange(ROLLING_AVERAGE_SAMPLES - 1, self.episodes), averages, label="Average")
        plt.ylabel("Scores")
        plt.xlabel("Episodes")
        plt.legend()
        plt.savefig(savePath + '/plot.png', bbox_inches='tight')
        plt.close()

    @staticmethod
    def movingAverage(a):
        ret = np.cumsum(a, dtype=float)
        ret[ROLLING_AVERAGE_SAMPLES:] = ret[ROLLING_AVERAGE_SAMPLES:] - ret[:-ROLLING_AVERAGE_SAMPLES]
        return ret[ROLLING_AVERAGE_SAMPLES - 1:] / ROLLING_AVERAGE_SAMPLES

    @staticmethod
    def dictToStr(d):
        return "\n".join(f"{key}: {value}" for key, value in sorted(d.items()))

class EnvWrapper(Node):
    def __init__(self, params):
        super().__init__('env_wrapper')
        self.actionPublisher = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Create service clients
        self.spawn_entity = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete_entity = self.create_client(DeleteEntity, '/delete_entity')
        self.set_entity_state = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.get_world_properties = self.create_client(GetWorldProperties, '/gazebo/get_world_properties')
        
        self.goalX, self.goalY = 0, 0
        self.goalDistanceOld = None
        self.rewardDirection = params["Reward Direction"]
        self.scanRatio = params["Scan Ratio"]
        self.crashPenalty = params["Crash Penalty"]
        self.goalReward = params["Goal Reward"]
        self.directionScalar = params["Direction Scalar"]
        self.stateSpace = params["State Space"]
        self.maxScanRange = params["Max Scan Range"]
        self.scanRewardScalar = params["Scan Reward Scaler"]
        self.modelX, self.modelY = None, None

    async def getState(self, ranges):
        ranges = [(self.maxScanRange if str(r) == 'inf' else min(r, self.maxScanRange)) 
                 for i, r in enumerate(ranges) if not i % self.scanRatio]
        goalInfo = await self.getGoalOdomStateInfo()
        return np.asarray(ranges + goalInfo).reshape(1, self.stateSpace)
        
    async def call_service(self, client, request):
        if not client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error(f'Service {client.srv_name} not available')
            return None
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()




    async def getGoalOdomStateInfo(self):
        """Get goal and odometry state information."""
        try:
            # Create a Future to store the odometry message
            future = rclpy.task.Future()
            
            def odom_callback(msg):
                nonlocal future
                if not future.done():
                    # Calculate robot's position and orientation
                    robot_x = msg.pose.pose.position.x
                    robot_y = msg.pose.pose.position.y
                    
                    # Extract quaternion components
                    qx = msg.pose.pose.orientation.x
                    qy = msg.pose.pose.orientation.y
                    qz = msg.pose.pose.orientation.z
                    qw = msg.pose.pose.orientation.w
                    
                    # Convert quaternion to Euler angles
                    yaw = math.atan2(2.0 * (qw * qz + qx * qy), 
                                    1.0 - 2.0 * (qy * qy + qz * qz))
                    
                    # Calculate distance and angle to goal
                    goal_distance = math.sqrt(
                        (self.goalX - robot_x) ** 2 + 
                        (self.goalY - robot_y) ** 2
                    )
                    
                    goal_angle = math.atan2(
                        self.goalY - robot_y,
                        self.goalX - robot_x
                    )
                    
                    # Calculate relative angle
                    relative_angle = goal_angle - yaw
                    if relative_angle > math.pi:
                        relative_angle -= 2 * math.pi
                    elif relative_angle < -math.pi:
                        relative_angle += 2 * math.pi
                    
                    # Store the old goal distance for reward calculation
                    self.goalDistanceOld = goal_distance
                    
                    # Set the future result
                    future.set_result([goal_distance, relative_angle])
            
            # Create subscription with the callback
            subscription = self.create_subscription(
                Odometry,
                'odom',
                odom_callback,
                10
            )
            
            # Wait for the future to complete
            try:
                result = await future
                return result
            except Exception as e:
                self.get_logger().error(f"Error waiting for odometry data: {str(e)}")
                return [0, 0]
            finally:
                # Cleanup subscription
                self.destroy_subscription(subscription)
                
        except Exception as e:
            self.get_logger().error(f"Failed to get odometry data: {str(e)}")
            return [0, 0]

    async def step(self, action):
        """Execute one step in the environment."""
        # Publish action
        twist_msg = Twist()
        twist_msg.linear.x = MODEL_SPEED
        twist_msg.angular.z = ACTION_SPACE[action]
        self.actionPublisher.publish(twist_msg)
        
        try:
            # Create a Future for the laser scan message
            future = rclpy.task.Future()
            
            def scan_callback(msg):
                nonlocal future
                if not future.done():
                    future.set_result(msg)
            
            # Create subscription with the callback
            subscription = self.create_subscription(
                LaserScan,
                'scan',
                scan_callback,
                10
            )
            
            # Wait for the scan message with timeout
            try:
                scan_msg = await asyncio.wait_for(future, timeout=5.0)
                newState = await self.getState(scan_msg.ranges)
                reward, done = await self.getReward(newState[0])
                return newState, reward, done
            except asyncio.TimeoutError:
                self.get_logger().error("Timeout waiting for laser scan")
                return np.zeros((1, self.stateSpace)), 0, True
            finally:
                # Cleanup subscription
                self.destroy_subscription(subscription)
                
        except Exception as e:
            self.get_logger().error(f"Failed to get laser scan data: {str(e)}")
            return np.zeros((1, self.stateSpace)), 0, True

    async def resetState(self):
        """Reset the environment state."""
        await self.deleteModel("goal")
        self.goalX, self.goalY = self.getGoalCoord(-1, 0)
        await self.teleportModel(TURTLEBOT_NAME, -1.5, 0)
        
        goalSpawned = False
        while not goalSpawned:
            await self.spawnModel("goal", self.goalX, self.goalY)
            goalSpawned = await self.checkModelPresent("goal")
        
        self.get_logger().info(f"New goal at {self.goalX}, {self.goalY}")
        
        try:
            # Create a Future for the laser scan message
            future = rclpy.task.Future()
            
            def scan_callback(msg):
                nonlocal future
                if not future.done():
                    future.set_result(msg)
            
            # Create subscription with the callback
            subscription = self.create_subscription(
                LaserScan,
                'scan',
                scan_callback,
                10
            )
            
            # Wait for the scan message with timeout
            try:
                scan_msg = await asyncio.wait_for(future, timeout=5.0)
                return await self.getState(scan_msg.ranges)
            except asyncio.TimeoutError:
                self.get_logger().error("Timeout waiting for initial laser scan")
                return np.zeros((1, self.stateSpace))
            finally:
                # Cleanup subscription
                self.destroy_subscription(subscription)
                
        except Exception as e:
            self.get_logger().error(f"Failed to get initial laser scan: {str(e)}")
            return np.zeros((1, self.stateSpace))

    # Fixed indentation for these methods:
    async def deleteModel(self, modelName):
        if not await self.checkModelPresent(modelName):
            return
            
        req = DeleteEntity.Request()
        req.name = modelName
        return await self.call_service(self.delete_entity, req)

    async def teleportModel(self, modelName, x, y):
        req = SetEntityState.Request()
        req.state.name = modelName
        req.state.pose.position.x = x
        req.state.pose.position.y = y
        
        # Stop current movement if it's the robot
        if modelName != "goal":
            twist_msg = Twist()
            self.actionPublisher.publish(twist_msg)
        
        try:
            return await self.call_service(self.set_entity_state, req)
        except Exception as e:
            self.get_logger().error(f'Teleport service call failed {str(e)}')

    # async def resetState(self):
    #     await self.deleteModel("goal")
    #     self.goalX, self.goalY = self.getGoalCoord(-1, 0)
    #     await self.teleportModel(TURTLEBOT_NAME, -1.5, 0)
        
    #     goalSpawned = False
    #     while not goalSpawned:
    #         await self.spawnModel("goal", self.goalX, self.goalY)
    #         goalSpawned = await self.checkModelPresent("goal")
        
    #     self.get_logger().info(f"New goal at {self.goalX}, {self.goalY}")
        
    #     try:
    #         scan_msg = await self.create_subscription(
    #             LaserScan,
    #             'scan',
    #             lambda msg: None,
    #             10).wait_for_message(timeout_sec=5.0)
    #         return await self.getState(scan_msg.ranges)
    #     except Exception as e:
    #         self.get_logger().error(f"Failed to get initial laser scan: {str(e)}")
    #         return np.zeros((1, self.stateSpace))

    async def getReward(self, state):
        if min(state[1:len(state) - 3]) < SCAN_MIN_DISTANCE or state[0] < SCAN_MIN_DISTANCE_FRONT:
            self.get_logger().info("Crashed!")
            return self.crashPenalty, True
            
        if state[len(state) - 2] < GOAL_MIN_DISTANCE:
            self.get_logger().info("Reached Goal!")
            self.goalX, self.goalY = self.getGoalCoord(self.goalX, self.goalY)
            await self.deleteModel("goal")
            await self.spawnModel("goal", self.goalX, self.goalY)
            self.get_logger().info(f"New goal at {self.goalX}, {self.goalY}")
            return self.goalReward, False
            
        if self.rewardDirection:
            return self.directionScalar * math.cos(state[len(state) - 1]), False
        else:
            return self.directionScalar * (1 if state[len(state) - 2] < self.goalDistanceOld else -1), False

    @staticmethod
    def getGoalCoord(removeX=None, removeY=None):
        while True:
            x = random.choice([-1.5, -.5, .5, 1.5])
            y = random.choice([-1.5, -.5, .5, 1.5])
            if not (x == removeX and y == removeY):
                break
        return x, y

    async def spawnModel(self, modelName, x, y):
        req = SpawnEntity.Request()
        req.name = modelName
        req.xml = self.getModelSDF(modelName)
        req.robot_namespace = ""
        req.initial_pose.position.x = x
        req.initial_pose.position.y = y
        return await self.call_service(self.spawn_entity, req)

	# async def deleteModel(self, modelName):
    #     if not await self.checkModelPresent(modelName):
    #         return
            
    #     req = DeleteEntity.Request()
    #     req.name = modelName
    #     return await self.call_service(self.delete_entity, req)


	# async def teleportModel(self, modelName, x, y):
    #     req = SetEntityState.Request()
    #     req.state.name = modelName
    #     req.state.pose.position.x = x
    #     req.state.pose.position.y = y
        
    #     # Stop current movement if it's the robot
    #     if modelName != "goal":
    #         twist_msg = Twist()
    #         self.actionPublisher.publish(twist_msg)
        
    #     try:
    #         return await self.call_service(self.set_entity_state, req)
    #     except Exception as e:
    #         self.get_logger().error(f'Teleport service call failed {str(e)}')

    async def checkModelPresent(self, modelName):
        try:
            future = self.get_world_properties.call_async(GetWorldProperties.Request())
            await future
            response = future.result()
            return modelName in response.model_names
        except Exception as e:
            self.get_logger().error(f'Check model service call failed {str(e)}')
            return False

    def getModelSDF(self, modelName):
        # Get the package path
        package_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        
        if modelName == TURTLEBOT_NAME:
            # For TurtleBot3
            model_path = os.path.join(
                "install/turtlebot3_gazebo/share/turtlebot3_gazebo/models",
                "turtlebot3_burger/model.sdf"
            )
        elif modelName == "goal":
            # For goal box
            model_path = os.path.join(
                package_path,
                "models/turtlebot3_square/goal_box/model.sdf"
            )
        else:
            raise ValueError(f"Unknown model name: {modelName}")
            
        try:
            with open(model_path, 'r') as f:
                return f.read()
        except Exception as e:
            self.get_logger().error(f'Failed to read model SDF: {str(e)}')
            return ""


async def run_experiments(node, hyperParameterList, SESSION_NAME):
    # Create results directory
    base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    results_path = os.path.join(base_path, "dqnmodels", SESSION_NAME)
    os.makedirs(results_path, exist_ok=True)

    # Save parameters
    with open(os.path.join(results_path, 'allParams.txt'), 'w') as f:
        f.write("\n".join(f"{k}: {v}" for k, v in hyperParameterList.items()))
    
    with open(os.path.join(results_path, 'envVars.txt'), 'w') as f:
        env_vars = {
            "Action Space": ACTION_SPACE,
            "Turtlebot Name": TURTLEBOT_NAME,
            "Goal Min Distance": GOAL_MIN_DISTANCE,
            "Scan Min Distance": SCAN_MIN_DISTANCE,
            "Model Speed": MODEL_SPEED,
            "Final Slice": FINAL_SLICE,
            "Rolling Average Samples": ROLLING_AVERAGE_SAMPLES
        }
        f.write("\n".join(f"{k}: {v}" for k, v in env_vars.items()))

    maxAvg, maxFinalAvg = -1000000, -1000000
    bestParams, bestFinalParams = {}, {}

    for values in product(*hyperParameterList.values()):
        experiment = dict(zip(hyperParameterList.keys(), values))
        node.get_logger().info(f"Using hyperparams: {experiment}")
        
        # Create model
        model = ModelClass(experiment, node)
        
        # Run episodes
        hyperSetScores, averages, setName = await model.playEpisodes()
        
        # Track best parameters
        currentAvg = np.mean(hyperSetScores)
        if currentAvg > maxAvg:
            maxAvg = currentAvg
            bestParams = experiment
        
        finalAvg = np.mean(hyperSetScores[-FINAL_SLICE:])
        if finalAvg > maxFinalAvg:
            maxFinalAvg = finalAvg
            bestFinalParams = experiment
        
        # Plot results
        plt.figure(1)
        plt.subplot(211)
        plt.plot(hyperSetScores, label=setName)
        plt.subplot(212)
        plt.plot(np.arange(ROLLING_AVERAGE_SAMPLES - 1, experiment["Episodes"]), 
                averages, label=setName)

    # Save final plots and best parameters
    plt.subplot(211)
    plt.ylabel("Scores")
    plt.xlabel("Episodes")
    plt.subplot(212)
    plt.ylabel("Average")
    plt.xlabel("Episodes")
    plt.legend()
    plt.savefig(os.path.join(results_path, 'plot.png'), bbox_inches='tight')
    plt.close()

    with open(os.path.join(results_path, 'bestParams.txt'), 'w') as f:
        f.write("\n".join(f"{k}: {v}" for k, v in bestParams.items()))
        
    with open(os.path.join(results_path, 'bestFinalParams.txt'), 'w') as f:
        f.write("\n".join(f"{k}: {v}" for k, v in bestFinalParams.items()))

def main(args=None):
    rclpy.init(args=args)
    
    # Create session name
    SESSION_NAME = "mySession - " + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    # Define hyperparameters
    hyperParameterList = {
        "Episodes": [500],
        "Episode Length": [350],
        "Crash Penalty": [-2000],
        "Goal Reward": [200],
        "Reward Direction": [True],
        "Epsilon Initial": [1],
        "Epsilon Decay": [.992],
        "Epsilon Min": [.05],
        "Reset Target": [2000],
        "Gamma": [.99],
        "Scan Ratio": [12],
        "Max Scan Range": [1],
        "Scan Reward Scaler": [1],
        "Learning Rate": [0.0002],
        "Optimizer": [tf.keras.optimizers.RMSprop],
        "Loss": [tf.keras.losses.Huber],
        "Batch Size": [100],
        "Memory Length": [1000000],
        "Direction Scalar": [1],
        "First Activation": [tf.keras.activations.relu],
        "Hidden Activations": [tf.keras.activations.relu],
        "Last Activation": [tf.keras.activations.linear],
        "Initializer": [tf.keras.initializers.VarianceScaling(scale=2)],
        "Load Model": [False]
    }

    # Calculate state space based on scan ratio
    hyperParameterList["State Space"] = [360 // r + 3 for r in hyperParameterList["Scan Ratio"]]

    try:
        # Create and initialize the node
        node = EnvWrapper(hyperParameterList)
        
        # Create an event loop
        loop = asyncio.get_event_loop()
        
        # Run the experiments
        loop.run_until_complete(run_experiments(node, hyperParameterList, SESSION_NAME))
        
        # Keep the node spinning
        rclpy.spin(node)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()