from maze_env import Maze
from RL_q_learning import rlalgorithm as rlalg1
#from RL_sarsa import rlalgorithm as rlalg1
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

DEBUG=1
def debug(debuglevel, msg, **kwargs):
    if debuglevel <= DEBUG:
        if 'printNow' in kwargs:
            if kwargs['printNow']:
                print(msg) 
        else:
            print(msg) 


def plot_rewards(experiments):
    color_list=['blue','green','red','black','magenta']
    label_list=[]
    for i, (env, RL, data) in enumerate(experiments):
        x_values=range(len(data['global_reward']))
        label_list.append(RL.display_name)
        y_values=data['global_reward']
        plt.plot(x_values, y_values, c=color_list[i],label=label_list[-1])
        plt.legend(label_list)
    plt.title("Reward Progress", fontsize=24)
    plt.xlabel("Episode", fontsize=18)
    plt.ylabel("Return", fontsize=18)
    plt.tick_params(axis='both', which='major',
                    labelsize=14)
    plt.show()

def update(env, RL, data, episodes=50, refresh_rate=10):
    global_reward = np.zeros(episodes)
    data['global_reward']=global_reward
    max_reward = float('-inf')

    for episode in range(episodes):
        t=0
        # initial state
        if episode == 0:
            state = env.reset(start_point=env.start_point, end_point=env.end_point, value=0)
        else:
            state = env.reset(start_point=env.start_point, end_point=env.end_point)

        debug(2,'state(ep:{},t:{})={}'.format(episode, t, state))
        step_counter = 0
        # RL choose action based on state
        action = RL.choose_action(str(state))
        while True:
            # fresh env
            if (episode > 100 or (step_counter % refresh_rate == 0)):
                env.render(sim_speed)
            step_counter += 1

            # RL take action and get next state and reward
            state_, reward, done = env.step(action)
            global_reward[episode] += reward
            debug(2,'state(ep:{},t:{})={}'.format(episode, t, state))
            debug(2,'reward_{}=  total return_t ={} Mean50={}'.format(reward, global_reward[episode],np.mean(global_reward[-50:])))


            # RL learn from this transition
            # and determine next state and action
            state, action =  RL.learn(str(state), action, reward, str(state_))

            # break while loop when end of this episode
            if done:
                break
            else:
                t=t+1

        if done:
            # 检查并更新最佳路径
            if global_reward[episode] > max_reward:
                max_reward = global_reward[episode]
                env.best_path = list(env.current_path) # 深拷贝当前路径
            #break
        debug(1,"({}) Episode {}: Length={}  Total return = {} ".format(RL.display_name,episode, t,  global_reward[episode],global_reward[episode]),printNow=(episode%printEveryNth==0))
        if(episode>=100):
            debug(1,"    Median100={} Variance100={}".format(np.median(global_reward[episode-100:episode]),np.var(global_reward[episode-100:episode])),printNow=(episode%printEveryNth==0))
    plot_best_path(env)
    # end of game
    print('game over -- Algorithm {} completed'.format(RL.display_name))
    #env.destroy()


def plot_best_path(env):
    # 绘制最佳路径
    point_begin = env.best_path[0]
    for _, point in enumerate(env.best_path[1:-1]):
        env.canvas.create_oval(point[0], point[1], point[0] + env.UNIT, point[1] + env.UNIT, fill='sky blue')
    env.canvas.create_rectangle(point_begin[0], point_begin[1], point_begin[0] + env.UNIT, point_begin[1] + env.UNIT, fill='red')
    point = env.best_path[-1]
    env.canvas.create_oval(point[0], point[1], point[0] + env.UNIT, point[1] + env.UNIT, fill='yellow')
    env.update()

def load_maze(graph):
    # 打开图片
    img = Image.open(graph).convert('L')

    # 将图片转化为numpy数组
    img_array = np.array(img)

    # 确定每个方格的大小
    grid_size = 8  # 例如，每个方格是10x10像素

    # 获取图片的宽度和高度
    height, width = img_array.shape

    img_array = img_array[48:height - 40, 0 + 50:width - 50]
    height, width = img_array.shape

    # 创建一个新的二维数组来存储结果
    result = np.zeros((height // grid_size, width // grid_size))

    # 遍历每个方格
    for i in range(0, height, grid_size):
        for j in range(0, width, grid_size):
            # 获取当前方格的像素值
            grid = img_array[i:i + grid_size, j:j + grid_size]

            # 计算方格的平均像素值
            avg_value = np.mean(grid)

            # 根据平均像素值判断方格的颜色
            if avg_value > 128 / 2.0:
                result[i // grid_size, j // grid_size] = 1  # 白色方格
            else:
                result[i // grid_size, j // grid_size] = 0  # 黑色方格
    np.set_printoptions(threshold=np.inf)
    with open("1.txt", 'w') as f:
        print(result, file=f)

    x, y = np.where(result == 1)
    wall_shape = []
    for i in range(len(x)):
        wall_shape.append([y[i], x[i]])
    pits = []
    if [63, 44] in wall_shape:
        wall_shape.remove([63, 44])
    return wall_shape, pits, result


if __name__ == "__main__":

    # ---------------------------------------------------------------------------
    wall_shape, pits, result = load_maze('maze.jpg')

    #Exmaple Full Run, you may need to run longer
    sim_speed = 0.05
    episodes=200
    printEveryNth=1
    do_plot_rewards=True
    refresh_rate = 100

    experiments = []
    env1 = Maze(wall_shape, pits, MAZE_H=result.shape[0], MAZE_W=result.shape[1], UNIT=10)
    RL1 = rlalg1(actions=list(range(env1.n_actions)))
    data1={}
    while not env1.point_set:
        env1.update()  # 更新tkinter事件循环
    env1.after(100, update(env1, RL1, data1, episodes, refresh_rate))
    env1.mainloop()
    experiments.append((env1, RL1, data1))

    print("experiments complete")
    for env, RL, data in experiments:
        print("{} : max reward = {} medLast100={} varLast100={}".format(RL.display_name, np.max(data['global_reward']),np.median(data['global_reward'][-100:]), np.var(data['global_reward'][-100:])))
    if(do_plot_rewards):
        #Simple plot of return for each episode and algorithm, you can make more informative plots
        plot_rewards(experiments)



