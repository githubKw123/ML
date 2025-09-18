
class GridWorld:
    """
    网格世界环境
    状态：长乘宽
    动作：上下左右
    终止：左上（0）右下（长乘宽）
    奖励：-1（种植状态为0）
    """


    def __init__(self, height=4, width=4):
        self.height = height
        self.width = width
        self.num_states = height * width
        self.num_actions = 4  # 上、下、左、右

        # 动作定义: 0=上, 1=下, 2=左, 3=右
        self.actions = ['↑', '↓', '←', '→']
        self.action_effects = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # 终止状态 (左上角和右下角)
        self.terminal_states = [0, self.num_states - 1]

    def state_to_coord(self, state):
        """状态索引转坐标"""
        return divmod(state, self.width)

    def coord_to_state(self, row, col):
        """坐标转状态索引"""
        return row * self.width + col

    def get_next_state(self, state, action):
        """获取下一状态"""
        if state in self.terminal_states:
            return state

        row, col = self.state_to_coord(state)
        drow, dcol = self.action_effects[action]

        new_row = max(0, min(self.height - 1, row + drow))
        new_col = max(0, min(self.width - 1, col + dcol))

        return self.coord_to_state(new_row, new_col)

    def get_reward(self, state, action, next_state):
        """获取奖励"""
        if state in self.terminal_states:
            return 0
        return -1  # 每步都有-1的奖励，鼓励尽快到达终止状态

