import networkx as nx
import matplotlib.pyplot as plt

# 创建示例图
G = nx.Graph()
# 添加节点及其组别属性
G.add_node('A', group='1')
G.add_node('B', group='1')
G.add_node('C', group='2')
G.add_node('D', group='2')
G.add_node('E', group='3')
G.add_node('F', group='3')

# 添加边
G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'F')])

# 定义组别颜色
group_colors = {'1': 'red', '2': 'green', '3': 'blue'}

# 获取节点的组别
groups = nx.get_node_attributes(G, 'group')

# 基于组别创建力导向布局，增加组内吸引力
pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

# 手动调整位置，使同组节点更靠近
offset = {'1': (-1, 0), '2': (0, 0), '3': (1, 0)}
for node, (x, y) in pos.items():
    group = groups[node]
    ox, oy = offset[group]
    pos[node] = (x + ox, y + oy)

# 绘制节点
colors = [group_colors[groups[node]] for node in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=500)

# 绘制边
nx.draw_networkx_edges(G, pos)

# 绘制标签
nx.draw_networkx_labels(G, pos, font_color='white')

plt.axis('off')
plt.show()