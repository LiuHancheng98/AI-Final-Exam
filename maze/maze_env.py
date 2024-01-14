import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk
import tkinter.messagebox

UNIT = 40   # pixels per cell (width and height)
MAZE_H = 10  # height of the entire grid in cells
MAZE_W = 10  # width of the entire grid in cells

class Maze(tk.Tk, object):
    def __init__(self, walls=[], pits=[], MAZE_H=10, MAZE_W=10, UNIT=20):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.walls = walls
        self.pits = pits
        self.wallblocks = []
        self.pitblocks=[]
        self.UNIT = UNIT
        # pixels per cell (width and height)
        self.MAZE_H = MAZE_H  # height of the entire grid in cells
        self.MAZE_W = MAZE_W  # width of the entire grid in cells
        self.title('maze')
        self.geometry('{0}x{1}'.format(self.MAZE_H * self.UNIT, self.MAZE_W * self.UNIT))
        self.build_shape_maze(walls, pits)
        self.canvas.bind("<Button-1>", self.set_point)  # 绑定鼠标点击事件
        self.start_point = None
        self.end_point = None
        self.point_set = False  # 添加一个标志来表示终点是否已设置
        tkinter.messagebox.showinfo("提示", "请点击选择起点")
        self.best_path = [] # 保存最佳路径
        self.current_path = [] # 保存当前路径


    def set_point(self, event):
        x, y = event.x // self.UNIT, event.y // self.UNIT
        if not self.start_point:
            if [x, y] not in self.walls:
                self.start_point = (x, y)
                self.add_agent(x, y)
                tkinter.messagebox.showinfo("提示", "请点击选择终点")
            else:
                tkinter.messagebox.showinfo("错误", "起点不能是墙壁")
        elif not self.end_point:
            if [x, y] not in self.walls:
                self.end_point = (x, y)
                self.add_goal(x, y)
                tkinter.messagebox.showinfo("提示", "起点和终点已设置，点击确定开始模拟")
                self.point_set = True  # 标记终点已设置
            else:
                tkinter.messagebox.showinfo("错误", "终点不能是墙壁")

    def build_shape_maze(self,walls,pits):
        self.canvas = tk.Canvas(self, bg='white',
                           height=self.MAZE_H * self.UNIT,
                           width=self.MAZE_W * self.UNIT)

        # create grids
        for c in range(0, self.MAZE_W * self.UNIT, self.UNIT):
            x0, y0, x1, y1 = c, 0, c, self.MAZE_H * self.UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, self.MAZE_H * self.UNIT, self.UNIT):
            x0, y0, x1, y1 = 0, r, self.MAZE_W * self.UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        for x,y in walls:
            self.add_wall(x,y)
        for x,y in pits:
            self.add_pit(x,y)
        self.canvas.pack()

    '''Add a solid wall block at coordinate for centre of bloc'''
    def add_wall(self, x, y):
        origin = np.array([self.UNIT/2, self.UNIT/2])
        wall_center = origin + np.array([self.UNIT * x, self.UNIT*y])
        self.wallblocks.append(self.canvas.create_rectangle(
            wall_center[0] - self.UNIT/2, wall_center[1] - self.UNIT/2,
            wall_center[0] + self.UNIT/2, wall_center[1] + self.UNIT/2,
            fill='black'))

    '''Add a solid pit block at coordinate for centre of bloc'''
    def add_pit(self, x, y):
        origin = np.array([self.UNIT/2, self.UNIT/2])
        pit_center = origin + np.array([self.UNIT * x, self.UNIT*y])
        self.pitblocks.append(self.canvas.create_rectangle(
            pit_center[0] - self.UNIT/2, pit_center[1] - self.UNIT/2,
            pit_center[0] + self.UNIT/2, pit_center[1] + self.UNIT/2,
            fill='blue'))

    '''Add a solid goal for goal at coordinate for centre of bloc'''
    def add_goal(self, x=4, y=4):
        origin = np.array([self.UNIT/2, self.UNIT/2])
        goal_center = origin + np.array([self.UNIT * x, self.UNIT*y])

        self.goal = self.canvas.create_oval(
            goal_center[0] - self.UNIT/2, goal_center[1] - self.UNIT/2,
            goal_center[0] + self.UNIT/2, goal_center[1] + self.UNIT/2,
            fill='yellow')

    '''Add a solid wall red block for agent at coordinate for centre of bloc'''
    def add_agent(self, x=0, y=0):
        origin = np.array([self.UNIT/2, self.UNIT/2])
        agent_center = origin + np.array([self.UNIT * x, self.UNIT*y])

        self.agent = self.canvas.create_rectangle(
            agent_center[0] - self.UNIT/2, agent_center[1] - self.UNIT/2,
            agent_center[0] + self.UNIT/2, agent_center[1] + self.UNIT/2,
            fill='red')

    def state_to_xy(self, state):
        assert state[0] + self.UNIT == state[2]
        assert state[1] + self.UNIT == state[3]
        return [state[0] / self.UNIT, state[1] / self.UNIT]

    def reset(self,  start_point=None, end_point=None, value = 1, resetAgent=True):
        self.update()
        time.sleep(0.2)
        # Set the start and end points if provided
        if(value == 0):
            return self.canvas.coords(self.agent)
        else:
            #Reset Agent
            if(resetAgent):
                self.canvas.delete(self.agent)
            if start_point is not None:
                self.start_point = start_point
                self.add_agent(*self.start_point)

            if end_point is not None:
                self.end_point = end_point
                self.add_goal(*self.end_point)
            self.current_path = []
            return self.canvas.coords(self.agent)

    '''computeReward - definition of reward function'''
    def computeReward(self, currstate, action, nextstate):
            reverse=False
            if nextstate == self.canvas.coords(self.goal):
                reward = 1
                done = True
                nextstate = 'terminal'
            elif nextstate in [self.canvas.coords(w) for w in self.wallblocks]:
                reward = -0.3
                done = False
                nextstate = currstate
                reverse=True
            elif nextstate in [self.canvas.coords(w) for w in self.pitblocks]:
                reward = -10
                done = True
                nextstate = 'terminal'
                reverse=False
            else:
                # end_point = np.array([i for i in self.end_point])
                # reward = np.linalg.norm(end_point - np.array(self.state_to_xy(currstate))) \
                #          - np.linalg.norm(end_point - np.array(self.state_to_xy(nextstate)))
                #reward = - 0.1 + 0.01 * (1 if reward > 0 else -1)
                # if np.linalg.norm(end_point - np.array(self.state_to_xy(currstate))) > 4:
                #     reward = - 0.1 + 0.01 * (1 if reward > 0 else -1)
                # else:
                #     reward = -0.1
                reward = -0.1
                done = False
            return reward, done, reverse

    def sigmoid(self,x):
        s = 1 / (1 + np.exp(-x))
        return s

    '''step - definition of one-step dynamics function'''
    def step(self, action):
        s = self.canvas.coords(self.agent)
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > self.UNIT:
                base_action[1] -= self.UNIT
        elif action == 1:   # down
            if s[1] < (self.MAZE_H - 1) * self.UNIT:
                base_action[1] += self.UNIT
        elif action == 2:   # right
            if s[0] < (self.MAZE_W - 1) * self.UNIT:
                base_action[0] += self.UNIT
        elif action == 3:   # left
            if s[0] > self.UNIT:
                base_action[0] -= self.UNIT

        self.canvas.move(self.agent, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.agent)  # next state

        # call the reward function
        reward, done, reverse = self.computeReward(s, action, s_)
        if(reverse):
            self.canvas.move(self.agent, -base_action[0], -base_action[1])  # move agent back
            s_ = self.canvas.coords(self.agent)
        self.current_path.append(s_)
        return s_, reward, done

    def render(self, sim_speed=.01):
        time.sleep(sim_speed)
        self.update()


def update():
    for t in range(10):
        print("The value of t is", t)
        s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done = env.step(a)
            if done:
                break


if __name__ == '__main__':
    env = Maze()
    env.after(100, update)
    env.mainloop()
