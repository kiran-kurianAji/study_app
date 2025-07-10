from pyvis.network import Network

net = Network()
net.add_node("A")
net.add_node("B")
net.add_edge("A", "B")
net.show("test.html", notebook=False)