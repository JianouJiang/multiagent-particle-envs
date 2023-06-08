import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
from  multiagent.scenarios.flow_around_cylinder import compute_fluid_dynamics

# drag_dict = fac.compute_fluid_dynamics(circles)

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 3
        num_landmarks = 3
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        world.F = None
        world.iteration = 0
        world.ux = None
        world.uy = None
        world.cylinders = None
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        
        #for l in world.landmarks:
            # dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            # we are scaling the coordiantes (from -1 to 1 in x and y) to LBM in flow_around_cylinder.py:
        '''
        Nx                     = 300     # resolution x-dir
        Ny                     = 100     # resolution y-dir
        '''
        circles = [  ( (a.state.p_pos[0]+1)*150, (a.state.p_pos[1]+1)*50, a.size/2 * 50) for a in world.agents]

        drag_dict, F,ux,uy, cylinders = compute_fluid_dynamics(circles, world.F, world.iteration)
        total_drags = sum(drag_dict.values())
        world.F = F
        world.iteration = world.iteration + 1
        world.ux = ux
        world.uy = uy
        world.cylinders

        rew -= total_drags

        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1

        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
            
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
