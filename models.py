import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import GINConv, GATConv, GCNConv, NNConv, MFConv, GINEConv, GATv2Conv

from params import N_CHEM_NODE_FEAT, N_CHEM_EDGE_FEAT, N_PROT_NODE_FEAT, N_PROT_EDGE_FEAT, N_CHEM_ECFP
from params import LGRAPHDTA_WITHOUT_FP, LGRAPHDTA_WITHOUT_DOMAIN, LGRAPHDTA_WITHOUT_ESM2, LGRAPHDTA_LLAMA_EMBEDDING


class FCLayers(torch.nn.Module):
    def __init__(self, trial, prefix, in_features, layers_range=(2, 3), n_units_list=(128, 256, 512, 1024, 2048, 4096),
                 dropout_range=(0.1, 0.7), **kwargs):
        super(FCLayers, self).__init__()
        self.trial = trial
        self.prefix = prefix
        self.in_features = in_features
        self.layers = None
        self.n_out = None

        self.get_layers(layers_range=layers_range, n_units_list=n_units_list, dropout_range=dropout_range)

    def get_layers(self, layers_range, n_units_list, dropout_range):

        in_features = self.in_features
        fc_layers = []
        n_fc_layers = self.trial.suggest_int(self.prefix + "_n_fc_layers", layers_range[0], layers_range[1])
        activation = nn.ReLU()
        use_batch_norm = self.trial.suggest_categorical(self.prefix + "_fc_use_bn", (True, False))
        out_features = None
        for i in range(n_fc_layers):
            out_features = self.trial.suggest_categorical(self.prefix + f"_fc_n_out_{i}", n_units_list)
            dropout = self.trial.suggest_float(self.prefix + f"_fc_dropout_{i}", dropout_range[0], dropout_range[1])

            fc_layers.append(nn.Linear(in_features, out_features))
            if use_batch_norm:
                fc_layers.append(nn.BatchNorm1d(out_features))
            fc_layers.append(activation)
            fc_layers.append(nn.Dropout(dropout))

            in_features = out_features

        self.layers = nn.Sequential(*fc_layers)
        self.n_out = out_features

    def forward(self, x):
        return self.layers(x)


class GraphPool:
    def __init__(self, trial, prefix,
                 pool_types=("mean", "add", "max", "mean_add", "mean_max", "add_max", "mean_add_max")):
        self.coef_dict = {"mean": 1, "add": 1, "max": 1, "mean_add": 2, "mean_max": 2, "add_max": 2, "mean_add_max": 3}
        self.type_ = trial.suggest_categorical(prefix + "_graph_pool_type", pool_types)
        self.coef = self.coef_dict[self.type_]

    def __call__(self, _graph_out, _graph_batch):
        out = None
        if self.type_ == "mean":
            out = global_mean_pool(_graph_out, _graph_batch)
        elif self.type_ == "add":
            out = global_add_pool(_graph_out, _graph_batch)
        elif self.type_ == "max":
            out = global_max_pool(_graph_out, _graph_batch)
        elif self.type_ == "mean_add":
            out = torch.cat([global_mean_pool(_graph_out, _graph_batch), global_add_pool(_graph_out, _graph_batch)],
                            dim=1)
        elif self.type_ == "mean_max":
            out = torch.cat([global_mean_pool(_graph_out, _graph_batch), global_max_pool(_graph_out, _graph_batch)],
                            dim=1)
        elif self.type_ == "add_max":
            out = torch.cat([global_add_pool(_graph_out, _graph_batch), global_max_pool(_graph_out, _graph_batch)],
                            dim=1)
        elif self.type_ == "mean_add_max":
            out = torch.cat([global_mean_pool(_graph_out, _graph_batch), global_add_pool(_graph_out, _graph_batch),
                             global_max_pool(_graph_out, _graph_batch)], dim=1)
        return out


class GNNLayers(torch.nn.Module):
    def __init__(self, trial, prefix, _node_features_len=None, _edge_features_len=None, use_edges_features=False):
        super(GNNLayers, self).__init__()
        self.trial = trial
        self.prefix = prefix
        self._node_features_len = _node_features_len
        self._edge_features_len = _edge_features_len
        self.use_edges_features = use_edges_features

        self.activation = None
        self.layers_list = None
        self.bn_list = None
        self.n_out = None

    def forward(self, data, **kwargs):
        _graph_out = data.x
        _edges_index = data.edge_index
        _edges_features = data.edge_attr if self.use_edges_features else None

        for _nn, _bn in zip(self.layers_list, self.bn_list):
            _graph_out = _nn(_graph_out, _edges_index, edge_attr=_edges_features) if self.use_edges_features else _nn(
                _graph_out, _edges_index)
            if _bn is not None:
                _graph_out = _bn(_graph_out)
            if self.activation is not None:
                _graph_out = self.activation(_graph_out)

        return _graph_out


class GATLayers(GNNLayers):
    def __init__(self, trial, prefix, gnn, _node_features_len, _edge_features_len, use_edges_features, layers_range,
                 heads_range, dropout_range, **kwargs):
        super(GATLayers, self).__init__(trial, prefix, _node_features_len=_node_features_len,
                                        _edge_features_len=_edge_features_len, use_edges_features=use_edges_features)
        self.gnn = gnn

        self.get_layers(layers_range=layers_range, heads_range=heads_range, dropout_range=dropout_range)

    def get_layers(self, layers_range, heads_range, dropout_range):
        _node_features_len = self._node_features_len
        _edge_features_len = self._edge_features_len

        self.use_edges_features = self.trial.suggest_categorical(self.prefix + "_use_edges_features",
                                                                 (True, False)) if self.use_edges_features else False
        if self.use_edges_features:
            _edge_features_fill = self.trial.suggest_categorical(self.prefix + "_edge_features_fill",
                                                                 ("mean", "add", "max", "mul", "min"))
        else:
            _edge_features_len = None
            _edge_features_fill = "mean"

        use_activation = self.trial.suggest_categorical(self.prefix + "_use_activation", (True, False))
        if use_activation:
            activation_name = self.trial.suggest_categorical(self.prefix + "_activation",
                                                             ("ReLU", "LeakyReLU", "Sigmoid"))
            self.activation = getattr(torch.nn, activation_name)()

        use_bn = self.trial.suggest_categorical(self.prefix + "_use_bn", (True, False))
        _n_layers = self.trial.suggest_int(self.prefix + "_n_layers", layers_range[0], layers_range[1])
        _layers_list = []
        _bn_layers_list = []
        _all_heads = 1

        for i in range(_n_layers):
            _heads = self.trial.suggest_int(self.prefix + f"_heads_{i}", heads_range[0], heads_range[1])
            _dropout = self.trial.suggest_float(self.prefix + f"_dropout_{i}", dropout_range[0], dropout_range[1])

            if self.use_edges_features:
                _gnn = self.gnn(_node_features_len * _all_heads, _node_features_len * _all_heads, heads=_heads,
                                dropout=_dropout, edge_dim=_edge_features_len, fill_value=_edge_features_fill)
            else:
                _gnn = self.gnn(_node_features_len * _all_heads, _node_features_len * _all_heads, heads=_heads,
                                dropout=_dropout)
            _layers_list.append(_gnn)

            _all_heads = _all_heads * _heads

            if use_bn:
                bn = nn.BatchNorm1d(_node_features_len * _all_heads)
                _bn_layers_list.append(bn)

        self.n_out = _node_features_len * _all_heads
        self.layers_list = nn.ModuleList(_layers_list)
        self.bn_list = nn.ModuleList(_bn_layers_list) if use_bn else [None] * _n_layers


class GATv1Layers(GATLayers):
    def __init__(self, trial, prefix, _node_features_len, _edge_features_len, layers_range=(1, 3), heads_range=(1, 5),
                 dropout_range=(0.0, 0.3), **kwargs):
        super(GATv1Layers, self).__init__(trial, prefix=prefix + "_gatv1", gnn=GATConv,
                                          _node_features_len=_node_features_len, _edge_features_len=_edge_features_len,
                                          use_edges_features=False,
                                          layers_range=layers_range, heads_range=heads_range,
                                          dropout_range=dropout_range, **kwargs)


class GATv2Layers(GATLayers):
    def __init__(self, trial, prefix, _node_features_len, _edge_features_len, layers_range=(1, 3), heads_range=(1, 5),
                 dropout_range=(0.0, 0.3), **kwargs):
        super(GATv2Layers, self).__init__(trial, prefix=prefix + "_gatv2", gnn=GATv2Conv,
                                          _node_features_len=_node_features_len, _edge_features_len=_edge_features_len,
                                          use_edges_features=False,
                                          layers_range=layers_range, heads_range=heads_range,
                                          dropout_range=dropout_range, **kwargs)

class GIN_Layers(GNNLayers):
    def __init__(self, trial, prefix, gnn, _node_features_len, _edge_features_len, use_edges_features, layers_range,
                 n_units_list, **kwargs):
        super(GIN_Layers, self).__init__(trial, prefix, _node_features_len=_node_features_len,
                                         _edge_features_len=_edge_features_len, use_edges_features=use_edges_features)
        self.gnn = gnn

        self.get_layers(layers_range=layers_range, n_units_list=n_units_list)

    def get_layers(self, layers_range, n_units_list):
        _n_out = None
        _n_in = self._node_features_len

        use_activation = self.trial.suggest_categorical(self.prefix + "_use_activation", (True, False))
        if use_activation:
            activation_name = self.trial.suggest_categorical(self.prefix + "_activation",
                                                             ("ReLU", "LeakyReLU", "Sigmoid"))
            self.activation = getattr(torch.nn, activation_name)()

        use_bn = self.trial.suggest_categorical(self.prefix + "_use_bn", (True, False))
        _n_layers = self.trial.suggest_int(self.prefix + "_n_layers", layers_range[0], layers_range[1])
        _layers_list = []
        _bn_layers_list = []

        for i in range(_n_layers):
            _n_out = self.trial.suggest_categorical(self.prefix + f"_n_out_{i}", n_units_list)

            _nn = nn.Sequential(nn.Linear(_n_in, _n_out), nn.ReLU(), nn.Linear(_n_out, _n_out))
            _gnn = self.gnn(_nn, edge_dim=self._edge_features_len) if self.use_edges_features else self.gnn(_nn)
            _layers_list.append(_gnn)
            if use_bn:
                bn = nn.BatchNorm1d(_n_out)
                _bn_layers_list.append(bn)

            _n_in = _n_out

        self.n_out = _n_out
        self.layers_list = nn.ModuleList(_layers_list)
        self.bn_list = nn.ModuleList(_bn_layers_list) if use_bn else [None] * _n_layers


class GINLayers(GIN_Layers):
    def __init__(self, trial, prefix, _node_features_len, _edge_features_len, layers_range=(1, 4),
                 n_units_list=(64, 128, 256, 512, 1024,), **kwargs):
        super(GINLayers, self).__init__(trial, prefix=prefix + "_gin", gnn=GINConv,
                                        _node_features_len=_node_features_len, _edge_features_len=_edge_features_len,
                                        use_edges_features=False,
                                        layers_range=layers_range, n_units_list=n_units_list, **kwargs)

class GCNLayers(GNNLayers):
    def __init__(self, trial, prefix, _node_features_len, layers_range=(1, 4), n_units_list=(64, 128, 256, 512, 1024,),
                 **kwargs):
        super(GCNLayers, self).__init__(trial, prefix + "_gcn", _node_features_len=_node_features_len)

        self.get_layers(layers_range=layers_range, n_units_list=n_units_list)

    def get_layers(self, layers_range, n_units_list):
        _n_out = None
        _n_in = self._node_features_len

        use_activation = self.trial.suggest_categorical(self.prefix + "_use_activation", (True, False))
        if use_activation:
            activation_name = self.trial.suggest_categorical(self.prefix + "_activation",
                                                             ("ReLU", "LeakyReLU", "Sigmoid"))
            self.activation = getattr(torch.nn, activation_name)()

        use_bn = self.trial.suggest_categorical(self.prefix + "_use_bn", (True, False))
        _n_layers = self.trial.suggest_int(self.prefix + "_n_layers", layers_range[0], layers_range[1])
        _layers_list = []
        _bn_layers_list = []

        for i in range(_n_layers):
            _n_out = self.trial.suggest_categorical(self.prefix + f"_n_out_{i}", n_units_list)

            _gnn = GCNConv(_n_in, _n_out)
            _layers_list.append(_gnn)

            if use_bn:
                bn = nn.BatchNorm1d(_n_out)
                _bn_layers_list.append(bn)

            _n_in = _n_out

        self.n_out = _n_out
        self.layers_list = nn.ModuleList(_layers_list)
        self.bn_list = nn.ModuleList(_bn_layers_list) if use_bn else [None] * _n_layers

class GMFLayers(GNNLayers):
    def __init__(self, trial, prefix, _node_features_len, layers_range=(1, 4), n_units_list=(64, 128, 256, 512, 1024,),
                 **kwargs):
        super(GMFLayers, self).__init__(trial, prefix + "_gmf", _node_features_len=_node_features_len)

        self.get_layers(layers_range=layers_range, n_units_list=n_units_list)

    def get_layers(self, layers_range, n_units_list):
        _n_out = None
        _n_in = self._node_features_len

        use_activation = self.trial.suggest_categorical(self.prefix + "_use_activation", (True, False))
        if use_activation:
            activation_name = self.trial.suggest_categorical(self.prefix + "_activation",
                                                             ("ReLU", "LeakyReLU", "Sigmoid"))
            self.activation = getattr(torch.nn, activation_name)()

        use_bn = self.trial.suggest_categorical(self.prefix + "_use_bn", (True, False))
        _n_layers = self.trial.suggest_int(self.prefix + "_n_layers", layers_range[0], layers_range[1])
        _layers_list = []
        _bn_layers_list = []

        for i in range(_n_layers):
            _n_out = self.trial.suggest_categorical(self.prefix + f"_n_out_{i}", n_units_list)

            _gnn = MFConv(_n_in, _n_out)
            _layers_list.append(_gnn)

            if use_bn:
                bn = nn.BatchNorm1d(_n_out)
                _bn_layers_list.append(bn)

            _n_in = _n_out

        self.n_out = _n_out
        self.layers_list = nn.ModuleList(_layers_list)
        self.bn_list = nn.ModuleList(_bn_layers_list) if use_bn else [None] * _n_layers

class Node_Attr_Mixer(nn.Module):
    def __init__(self, node_attr_3DProDTA_dim, node_attr_Language_dim):
        super(Node_Attr_Mixer, self).__init__()

        self.node_attr_3DProDTA_dim = node_attr_3DProDTA_dim
        self.node_attr_Language_dim = node_attr_Language_dim

        self.node_attr_language_linear_model = nn.Sequential(
            nn.Linear(node_attr_Language_dim, node_attr_3DProDTA_dim),
            nn.BatchNorm1d(node_attr_3DProDTA_dim),
            nn.ReLU(),
            )
        

    def forward(self, input):
        output_1 = input[:,:self.node_attr_3DProDTA_dim]
        output_2 = self.node_attr_language_linear_model(input[:,self.node_attr_3DProDTA_dim:])
        output = torch.cat((output_1, output_2), dim=1)
        return output

class Reduce_Node_Attribute_Dimensions(nn.Module):
    def __init__(self, in_dim, output_dim):
        super(Reduce_Node_Attribute_Dimensions, self).__init__()

        self.reduce_attr_dim = nn.Sequential(
            nn.Linear(in_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            )
    
    def forward(self, input):
        
        output = self.reduce_attr_dim(input)
        return output
    
class Graph:
    pass

class GraphModel(torch.nn.Module):
    def __init__(self, trial, prefix, _node_features_len, _edge_features_len):
        super(GraphModel, self).__init__()
        _n_out = None

        self._gnn_arch = trial.suggest_categorical(prefix + "_gnn_arch", ("single", "staked"))
        self._use_post_fc = trial.suggest_categorical(prefix + "_gnn_post_fc", (True,))
        prefix = prefix + "_" + self._gnn_arch

        if self._gnn_arch == "single":
            graph_models = ["GATv2Layers", "GINLayers", "GCNLayers", "GMFLayers"]

            _graph_model_type = trial.suggest_categorical(prefix+"_graph_model_type", graph_models)
            self._graph_model = eval(_graph_model_type)(trial, prefix, _node_features_len=_node_features_len,
                                                        _edge_features_len=_edge_features_len)
            _n_out = self._graph_model.n_out
            self._pool = GraphPool(trial, prefix=prefix)
            _n_out = _n_out * self._pool.coef


        if self._use_post_fc:
            self._post_fc = FCLayers(trial, prefix+"_post", _n_out, layers_range=(1, 1),
                                     n_units_list=(256, 512, 1024, 2048))
            _n_out = self._post_fc.n_out

        self.n_out = _n_out

    def forward(self, graph):
        x = None

        if self._gnn_arch == "single":
            x = self._graph_model(graph)
            x = self._pool(x, graph.batch)

        if self._use_post_fc:
            x = self._post_fc(x)

        return x

class GatingMechanism(nn.Module):
    def __init__(self, chem_ecfp_dim, chem_graph_dim, prot_graph_dim):
        super(GatingMechanism, self).__init__()

        self.gate_chem_ecfp = torch.nn.Sequential(
            torch.nn.Linear(chem_ecfp_dim, chem_ecfp_dim//2),
            torch.nn.ReLU(),
            torch.nn.Linear(chem_ecfp_dim//2, chem_ecfp_dim//2),
            torch.nn.ReLU(),
            torch.nn.Linear(chem_ecfp_dim//2, chem_ecfp_dim//4),
            torch.nn.ReLU(),
            torch.nn.Linear(chem_ecfp_dim//4, chem_ecfp_dim//4),
            torch.nn.ReLU(),
            torch.nn.Linear(chem_ecfp_dim//4, chem_ecfp_dim//8),
            torch.nn.ReLU(),
            torch.nn.Linear(chem_ecfp_dim//8, 1),
            torch.nn.Sigmoid(),
        )

        self.gate_chem_graph = torch.nn.Sequential(
            torch.nn.Linear(chem_graph_dim, chem_graph_dim//2),
            torch.nn.ReLU(),
            torch.nn.Linear(chem_graph_dim//2, chem_graph_dim//2),
            torch.nn.ReLU(),
            torch.nn.Linear(chem_graph_dim//2, chem_graph_dim//4),
            torch.nn.ReLU(),
            torch.nn.Linear(chem_graph_dim//4, chem_graph_dim//4),
            torch.nn.ReLU(),
            torch.nn.Linear(chem_graph_dim//4, chem_graph_dim//8),
            torch.nn.ReLU(),
            torch.nn.Linear(chem_graph_dim//8, 1),
            torch.nn.Sigmoid(),
        )

        self.gate_prot_graph = torch.nn.Sequential(
            torch.nn.Linear(prot_graph_dim, prot_graph_dim//2),
            torch.nn.ReLU(),
            torch.nn.Linear(prot_graph_dim//2, prot_graph_dim//2),
            torch.nn.ReLU(),
            torch.nn.Linear(prot_graph_dim//2, prot_graph_dim//4),
            torch.nn.ReLU(),
            torch.nn.Linear(prot_graph_dim//4, prot_graph_dim//4),
            torch.nn.ReLU(),
            torch.nn.Linear(prot_graph_dim//4, prot_graph_dim//8),
            torch.nn.ReLU(),
            torch.nn.Linear(prot_graph_dim//8, 1),
            torch.nn.Sigmoid(),
        )
    
    def forward(self, chem_ecfp, chem_graph, prot_graph):
        gate_chem_ecfp = self.gate_chem_ecfp(chem_ecfp)
        gate_chem_graph = self.gate_chem_graph(chem_graph)
        gate_prot_graph = self.gate_prot_graph(prot_graph)

        return gate_chem_ecfp, gate_chem_graph, gate_prot_graph

class TransformerFusion(nn.Module):
    def __init__(self, embedding_dim=2187, hidden_dim=729, num_heads=9, num_layers=6):
        super(TransformerFusion, self).__init__()
        self.ligand_fp_fc = nn.Linear(256, embedding_dim//3)
        self.ligand_graph_fc = nn.Linear(1024, embedding_dim//3)
        self.protein_graph_fc = nn.Linear(1024, embedding_dim//3)

        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.n_out = embedding_dim

    def forward(self, ligand_fp, ligand_graph, protein_graph):
        ligand_fp = self.ligand_fp_fc(ligand_fp)
        ligand_graph = self.ligand_graph_fc(ligand_graph)
        protein_graph = self.protein_graph_fc(protein_graph)

        x = torch.cat([ligand_fp, ligand_graph, protein_graph], dim=-1)

        x = x.unsqueeze(0)
        x = self.transformer_encoder(x)

        return x.squeeze(0)


class LGraph(torch.nn.Module):
    def __init__(self, trial, chem_node_features_len=N_CHEM_NODE_FEAT, prot_node_features_len=N_PROT_NODE_FEAT,
                 chem_edge_features_len=N_CHEM_EDGE_FEAT, prot_edge_features_len=N_PROT_EDGE_FEAT,
                 chem_ecfp_len=N_CHEM_ECFP):

        super(LGraph, self).__init__()

        self.use_chem_ecfp_post_fc = trial.suggest_categorical("chem_ecfp_post_fc", (True,))
        chem_ecfp_n_out = chem_ecfp_len
        if self.use_chem_ecfp_post_fc and not LGRAPHDTA_WITHOUT_FP:
            self.chem_ecfp_post_fc = FCLayers(trial, "chem_ecfp_post", chem_ecfp_n_out, layers_range=(1, 1),
                                              n_units_list=(256, 512, 1024, 2048))
            chem_ecfp_n_out = self.chem_ecfp_post_fc.n_out

        if LGRAPHDTA_WITHOUT_ESM2:
            prot_node_features_len = 41
        elif LGRAPHDTA_WITHOUT_DOMAIN:
            prot_node_features_len = 1280
        elif LGRAPHDTA_LLAMA_EMBEDDING:
            self.protein_node_attr_mixer = Node_Attr_Mixer(41, 1536)
            prot_node_features_len = 41 + 41
        else:
            self.protein_node_attr_mixer = Node_Attr_Mixer(41, 1280)
            prot_node_features_len = 41 + 41

        self.chem_graph_encoder = GraphModel(trial, "chem", 23, chem_edge_features_len)
        self.prot_graph_encoder = GraphModel(trial, "prot", prot_node_features_len, prot_edge_features_len)

        # # Add GatingMechanism
        # self.gating_mechanism = GatingMechanism(chem_ecfp_n_out, self.chem_graph_encoder.n_out, self.prot_graph_encoder.n_out)

        # Add TransformerFusion
        # self.transformer_fusion = TransformerFusion(embedding_dim=2187, hidden_dim=729, num_heads=9, num_layers=6)

        if not LGRAPHDTA_WITHOUT_FP:
            n_out = chem_ecfp_n_out + self.chem_graph_encoder.n_out + self.prot_graph_encoder.n_out
            # Add TransformerFusion
            # n_out = self.transformer_fusion.n_out
        else:
            n_out = self.chem_graph_encoder.n_out + self.prot_graph_encoder.n_out

        self.fc = FCLayers(trial, "final", n_out, layers_range=(2, 3), n_units_list=(256, 512, 1024, 2048, 4096))
        self.out = torch.nn.Linear(self.fc.n_out, 1)

    def forward(self, data):
        chem_ecfp, chem_graph, prot_graph = data["e1_fp"], data["e1_graph"], data["e2_graph"]

        chem_ecfp_out = chem_ecfp
        if self.use_chem_ecfp_post_fc and not LGRAPHDTA_WITHOUT_FP:
            chem_ecfp_out = self.chem_ecfp_post_fc(chem_ecfp_out)

        if LGRAPHDTA_WITHOUT_ESM2 or LGRAPHDTA_WITHOUT_DOMAIN:
            pass
        else:
            prot_graph.x = self.protein_node_attr_mixer(prot_graph.x)

        chem_graph_out = self.chem_graph_encoder(chem_graph)
        prot_graph_out = self.prot_graph_encoder(prot_graph)

        # # Add GatingMechanism
        # gate_chem_ecfp, gate_chem_graph, gate_prot_graph = self.gating_mechanism(chem_ecfp_out, chem_graph_out, prot_graph_out)
        # chem_ecfp_out = chem_ecfp_out * gate_chem_ecfp
        # chem_graph_out = chem_graph_out * gate_chem_graph
        # prot_graph_out = prot_graph_out * gate_prot_graph

        if not LGRAPHDTA_WITHOUT_FP:
            x = torch.cat([chem_ecfp_out, chem_graph_out, prot_graph_out], dim=1)
            # Add TransformerFusion
            # x = self.transformer_fusion(chem_ecfp_out, chem_graph_out, prot_graph_out)
        else:
            x = torch.cat([chem_graph_out, prot_graph_out], dim=1)
        x = self.out(self.fc(x))

        return x