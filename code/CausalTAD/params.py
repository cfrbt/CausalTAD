class Params:

    def __init__(self, city):
        self.batch_size = 32
        self.dropout = 0
        self.hidden_size = 128
        self.layer_rnn = 1
        self.latent_num = 128
        self.label_num = 0

        self.nodes = "../datasets/{}/roadnetwork/nodes.pickle".format(city)
        self.node_adj = "../datasets/{}/roadnetwork/node_adj.pickle".format(city)
        self.train_dataset = "../datasets/{}/trajectories/train.pickle".format(city)
        self.detour_dataset = "../datasets/{}/trajectories/detour.pickle".format(city)
        self.switch_dataset = "../datasets/{}/trajectories/switch.pickle".format(city)
        self.normal_dataset = "../datasets/{}/trajectories/id.pickle".format(city)
        self.ood_dataset = "../datasets/{}/trajectories/ood.pickle".format(city)

        self.save_path = "./save/"
        self.output = "./output/"
        self.lr = 1e-3
        self.weight_decay = 0.01
        self.epochs = 200

        self.num_steps = 49
        self.beta_start = 0.0001
        self.beta_end = 0.5
        self.channels = 64
        self.conditional = False
        self.input_dim = 1
        self.layers_res = 4
        self.graph_layers = 5
        self.graph_path = "../datasets/{}/roadnetwork/graph_partition.pickle".format(city)