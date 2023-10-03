import json
import torch
import random
import torch.nn as nn


class GraphTransformer(nn.Module):
    def __init__(
        self,
        n_layers,
        num_heads,
        hidden_dim,
        dropout_rate,
        intput_dropout_rate,
        weight_decay,
        ffn_dim,
        dataset_name,
        warmup_updates,
        tot_updates,
        peak_lr,
        end_lr,
        edge_type,
        multi_hop_max_dist,
        attention_dropout_rate,
        num_class,
        lap_dim,
        svd_dim,
        path,
    ):
        super().__init__()

        self.num_heads = num_heads
        if dataset_name == 'PROTEINS':
            self.atom_encoder = nn.Embedding(8, hidden_dim, padding_idx=0)
            self.edge_encoder = nn.Embedding(4, num_heads, padding_idx=0)
            self.edge_type = edge_type
            self.edge_dis_encoder = nn.Embedding(
                40 * num_heads * num_heads, 1)
            self.spatial_pos_encoder = nn.Embedding(512, num_heads, padding_idx=0)
            self.in_degree_encoder = nn.Embedding(
                64, hidden_dim, padding_idx=0)
            self.out_degree_encoder = nn.Embedding(
                64, hidden_dim, padding_idx=0)
        else:
            self.atom_encoder = nn.Embedding(
                256, hidden_dim, padding_idx=0)
            self.edge_encoder = nn.Embedding(
                256, num_heads, padding_idx=0)
            self.edge_type = edge_type
            self.edge_dis_encoder = nn.Embedding(
                64 * num_heads * num_heads, 1)
            self.spatial_pos_encoder = nn.Embedding(512, num_heads, padding_idx=0)
            self.in_degree_encoder = nn.Embedding(
                128, hidden_dim, padding_idx=0)
            self.out_degree_encoder = nn.Embedding(
                128, hidden_dim, padding_idx=0)

        self.input_dropout = nn.Dropout(intput_dropout_rate)
        encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(hidden_dim)

        self.downstream_out_proj = nn.Linear(hidden_dim, num_class)
        self.pma_linear = nn.Linear(multi_hop_max_dist, num_heads)
        self.lap_linear = nn.Linear(lap_dim, hidden_dim)
        self.svd_linear = nn.Linear(svd_dim, hidden_dim)

        self.graph_token = nn.Embedding(1, hidden_dim)
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        self.warmup_updates = warmup_updates
        self.tot_updates = tot_updates
        self.peak_lr = peak_lr
        self.end_lr = end_lr
        self.weight_decay = weight_decay
        self.multi_hop_max_dist = multi_hop_max_dist

        self.hidden_dim = hidden_dim
        self.apply(lambda module: init_params(module, n_layers=n_layers))
        self.dataset_name = dataset_name
        self.encodings = True
        self.path = path

    def forward(self, batched_data, params=None):
        if params == None and self.encodings:
            # Initialize
            attn_bias, spatial_pos, x = batched_data.attn_bias, batched_data.spatial_pos, batched_data.x
            in_degree, out_degree = batched_data.in_degree, batched_data.in_degree
            edge_input, attn_edge_type = batched_data.edge_input, batched_data.attn_edge_type
            lap_pos_enc, svd_pos_enc, pma_att_enc = batched_data.lap_pos_enc, batched_data.svd_pos_enc, batched_data.pma_att_enc
            # Attention Matrix: Spatial Encoding, Edge Encoding, Proximity-Enhanced Multi-Head Attention
            # Positional Embedding: Centrality Encoding, Laplacian Eigenvector, SVD-based Positional Encoding
            # Mask: Mask-n

            # print("atom_encoder minimal num_embeddings: {} - {} = {}".format(x.max(), x.min(), x.max() - x.min()))
            # print("edge_encoder minimal num_embeddings: {} - {} = {}".format(edge_input.max(), edge_input.min(), edge_input.max() - edge_input.min()))
            # print("spatial_pos_encoder minimal num_embeddings: {} - {} = {}".format(spatial_pos.max(), spatial_pos.min(), spatial_pos.max() - spatial_pos.min()))
            # print("in_degree_encoder minimal num_embeddings: {} - {} = {}".format(in_degree.max(), in_degree.min(), in_degree.max() - in_degree.min()))

            # ==================== Attention Matrix Start ====================
            # graph_attn_bias
            n_graph, n_node = x.size()[:2]
            graph_attn_bias = attn_bias.clone() # attn_bias [n_graph, n_node+1, n_node+1]
            graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # [n_graph, n_head, n_node+1, n_node+1]
            # reset spatial pos here
            t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
            graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
            graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

            # spatial pos: Spatial Encoding
            # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
            spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
            graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias

            # edge feature: Edge Encoding
            spatial_pos_ = spatial_pos.clone()
            spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1
            # set 1 to 1, x > 1 to x - 1
            spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
            spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
            edge_input = edge_input[:, :, :, :self.multi_hop_max_dist, :]
            # [n_graph, n_node, n_node, max_dist, n_head]
            edge_input = self.edge_encoder(edge_input).mean(-2)
            max_dist = edge_input.size(-2)
            edge_input_flat = edge_input.permute(3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
            edge_input_flat = torch.bmm(edge_input_flat, self.edge_dis_encoder.weight.reshape(
                -1, self.num_heads, self.num_heads)[:max_dist, :, :])
            edge_input = edge_input_flat.reshape(max_dist, n_graph, n_node, n_node, self.num_heads).permute(1, 2, 3, 0, 4)
            edge_input = (edge_input.sum(-2) / (spatial_pos_.float().unsqueeze(-1))).permute(0, 3, 1, 2)
            graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input
            graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

            # PMA: Proximity-Enhanced Multi-Head Attention
            proximity_enhanced_bias = self.pma_linear(pma_att_enc).permute(0, 3, 1, 2)
            graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + proximity_enhanced_bias
            # ===================== Attention Matrix End =====================

            # ==================== Positional Embedding Start ====================
            # node feauture + graph token
            node_feature = self.atom_encoder(x).sum(dim=-2) # x [n_graph, n_node, feature_dim]
            # [n_graph, n_node, n_hidden]

            # Centrality Encoding
            # node_feature = node_feature + self.in_degree_encoder(in_degree) + self.out_degree_encoder(out_degree)
            positional_embedding = self.in_degree_encoder(in_degree) + self.out_degree_encoder(out_degree)

            # Laplacian Eigenvector
            lap_feature = self.lap_linear(lap_pos_enc)
            # node_feature = node_feature + lap_feature
            positional_embedding = positional_embedding + lap_feature

            # SVD-based Positional Encoding
            svd_feature = self.svd_linear(svd_pos_enc)
            # node_feature = node_feature + svd_feature
            positional_embedding = positional_embedding + svd_feature

            graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
            graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)
            # [n_graph, n_node+1, hidden_dim]
            # ===================== Positional Embedding End =====================

            # transfomrer encoder
            output = self.input_dropout(graph_node_feature)
            for enc_layer in self.layers:
                output[:, 1:] = output[:, 1:] + positional_embedding
                output = enc_layer(output, graph_attn_bias, spatial_pos=spatial_pos)
            output = self.final_ln(output)

            # output part
            output = self.downstream_out_proj(output[:, 0, :])
            return output
        elif params == None:
            # Initialize
            attn_bias, spatial_pos, x = batched_data.attn_bias, batched_data.spatial_pos, batched_data.x
            in_degree, out_degree = batched_data.in_degree, batched_data.in_degree
            edge_input, attn_edge_type = batched_data.edge_input, batched_data.attn_edge_type
            lap_pos_enc, svd_pos_enc, pma_att_enc = batched_data.lap_pos_enc, batched_data.svd_pos_enc, batched_data.pma_att_enc
            # Attention Matrix: Spatial Encoding, Edge Encoding, Proximity-Enhanced Multi-Head Attention
            # Positional Embedding: Centrality Encoding, Laplacian Eigenvector, SVD-based Positional Encoding
            # Mask: Mask-n

            # ==================== Attention Matrix Start ====================
            # graph_attn_bias
            n_graph, n_node = x.size()[:2]
            graph_attn_bias = attn_bias.clone() # attn_bias [n_graph, n_node+1, n_node+1]
            graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # [n_graph, n_head, n_node+1, n_node+1]
            # reset spatial pos here
            t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
            graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
            graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

            # spatial pos: Spatial Encoding
            # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
            spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
            graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias

            # edge feature: Edge Encoding
            spatial_pos_ = spatial_pos.clone()
            spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1
            # set 1 to 1, x > 1 to x - 1
            spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
            spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
            edge_input = edge_input[:, :, :, :self.multi_hop_max_dist, :]
            # [n_graph, n_node, n_node, max_dist, n_head]
            edge_input = self.edge_encoder(edge_input).mean(-2)
            max_dist = edge_input.size(-2)
            edge_input_flat = edge_input.permute(3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
            edge_input_flat = torch.bmm(edge_input_flat, self.edge_dis_encoder.weight.reshape(
                -1, self.num_heads, self.num_heads)[:max_dist, :, :])
            edge_input = edge_input_flat.reshape(max_dist, n_graph, n_node, n_node, self.num_heads).permute(1, 2, 3, 0, 4)
            edge_input = (edge_input.sum(-2) / (spatial_pos_.float().unsqueeze(-1))).permute(0, 3, 1, 2)
            graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input
            graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset
            # ===================== Attention Matrix End =====================

            # ==================== Positional Embedding Start ====================
            # node feauture + graph token
            node_feature = self.atom_encoder(x).sum(dim=-2) # x [n_graph, n_node, feature_dim]
            # [n_graph, n_node, n_hidden]

            # Centrality Encoding
            node_feature = node_feature + self.in_degree_encoder(in_degree) + self.out_degree_encoder(out_degree)

            graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
            graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)
            # [n_graph, n_node+1, hidden_dim]
            # ===================== Positional Embedding End =====================

            # transfomrer encoder
            output = self.input_dropout(graph_node_feature)
            for enc_layer in self.layers:
                output = enc_layer(output, graph_attn_bias)
            output = self.final_ln(output)

            # output part
            output = self.downstream_out_proj(output[:, 0, :])
            return output
        else:
            # Initialize
            attn_bias, spatial_pos, x = batched_data.attn_bias, batched_data.spatial_pos, batched_data.x
            in_degree, out_degree = batched_data.in_degree, batched_data.in_degree
            edge_input, attn_edge_type = batched_data.edge_input, batched_data.attn_edge_type
            lap_pos_enc, svd_pos_enc, pma_att_enc = batched_data.lap_pos_enc, batched_data.svd_pos_enc, batched_data.pma_att_enc
            # Attention Matrix: Spatial Encoding, Edge Encoding, Proximity-Enhanced Multi-Head Attention
            # Positional Embedding: Centrality Encoding, Laplacian Eigenvector, SVD-based Positional Encoding
            # Mask: Mask-n

            # ==================== Attention Matrix Start ====================
            # graph_attn_bias
            n_graph, n_node = x.size()[:2]
            graph_attn_bias = attn_bias.clone() # attn_bias [n_graph, n_node+1, n_node+1]
            graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # [n_graph, n_head, n_node+1, n_node+1]
            # reset spatial pos here
            t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
            graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
            graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

            # spatial pos: Spatial Encoding
            # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
            spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
            # graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias

            # edge feature: Edge Encoding
            spatial_pos_ = spatial_pos.clone()
            spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1
            # set 1 to 1, x > 1 to x - 1
            spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
            spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
            edge_input = edge_input[:, :, :, :self.multi_hop_max_dist, :]
            # [n_graph, n_node, n_node, max_dist, n_head]
            edge_input = self.edge_encoder(edge_input).mean(-2)
            max_dist = edge_input.size(-2)
            edge_input_flat = edge_input.permute(3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
            edge_input_flat = torch.bmm(edge_input_flat, self.edge_dis_encoder.weight.reshape(
                -1, self.num_heads, self.num_heads)[:max_dist, :, :])
            edge_input = edge_input_flat.reshape(max_dist, n_graph, n_node, n_node, self.num_heads).permute(1, 2, 3, 0, 4)
            edge_input = (edge_input.sum(-2) / (spatial_pos_.float().unsqueeze(-1))).permute(0, 3, 1, 2)
            # graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input
            graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

            # PMA: Proximity-Enhanced Multi-Head Attention
            proximity_enhanced_bias = self.pma_linear(pma_att_enc).permute(0, 3, 1, 2)
            # graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + proximity_enhanced_bias
            # ===================== Attention Matrix End =====================

            # ==================== Positional Embedding Start ====================
            # node feauture + graph token
            node_feature = self.atom_encoder(x).sum(dim=-2) # x [n_graph, n_node, feature_dim]
            # [n_graph, n_node, n_hidden]

            # Centrality Encoding
            cen_feature = self.in_degree_encoder(in_degree) + self.out_degree_encoder(out_degree)

            # Laplacian Eigenvector
            lap_feature = self.lap_linear(lap_pos_enc)

            # SVD-based Positional Encoding
            svd_feature = self.svd_linear(svd_pos_enc)

            graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
            graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)
            # [n_graph, n_node+1, hidden_dim]
            # ===================== Positional Embedding End =====================

            # transfomrer encoder
            output = self.input_dropout(graph_node_feature)
            layers = self.layers[:params[0]]
            for i, enc_layer in enumerate(layers):
                # 这里我们用人话来解释一下生成的 params 都是啥，还有它们是怎么工作的
                # 本函数的返回值是 params = [depth, layers] 其中 depth 为层数， layers 为参数
                # layers = [layer_1, layer_2, ..., layers_depth] 其中 layer_n 为每层的参数
                # layer_n = [shape, PE, AT, mask] 其中 shape 为 Transformer 的架构超参数， PE 与 AT 为 encoding 相关参数， mask 为 mask 相关参数
                # shape = [hidden_in, num_heads, att_size, hidden_mid, ffn_size] 其中 num_heads 为头数，其他均为维度数
                # PE = [cen, eig, svd] 其中三者均为布尔类型变量，表征这一编码是否选取
                # AT = [spa, edg, pma] 其中三者均为布尔类型变量，表征这一编码是否选取
                # mask 表征是否施加，以及施加何种程度的 mask ， 0 表示不施加 mask ，正值则表示 mask 掉所有 hop 数高于 mask 值的边
                layer_i = params[1][i]
                node_feature = 0
                node_feature = node_feature + layer_i[1][0] * cen_feature
                node_feature = node_feature + layer_i[1][1] * lap_feature
                node_feature = node_feature + layer_i[1][2] * svd_feature
                output[:, 1:] = output[:, 1:] + node_feature
                graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[
                    :, :, 1:, 1:] + layer_i[2][0] * spatial_pos_bias
                graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[
                    :, :, 1:, 1:] + layer_i[2][1] * edge_input
                graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[
                    :, :, 1:, 1:] + layer_i[2][2] * proximity_enhanced_bias
                output = enc_layer(output, graph_attn_bias, layer_i[0], spatial_pos)
            output = self.final_ln(output)

            # output part
            output = self.downstream_out_proj(output[:, 0, :])
            return output

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.peak_lr, weight_decay=self.weight_decay)
        lr_scheduler = {
            'scheduler': PolynomialDecayLR(
                optimizer,
                warmup_updates=self.warmup_updates,
                tot_updates=self.tot_updates,
                lr=self.peak_lr,
                end_lr=self.end_lr,
                power=1.0,
            ),
            'name': 'learning_rate',
            'interval': 'step',
            'frequency': 1,
        }
        return optimizer, lr_scheduler

    def gen_params(self):
        # 这里我们用人话来解释一下生成的 params 都是啥，还有它们是怎么工作的
        # 本函数的返回值是 params = [depth, layers] 其中 depth 为层数， layers 为参数
        # layers = [layer_1, layer_2, ..., layers_depth] 其中 layer_n 为每层的参数
        # layer_n = [shape, PE, AT, mask] 其中 shape 为 Transformer 的架构超参数， PE 与 AT 为 encoding 相关参数， mask 为 mask 相关参数
        # shape = [hidden_in, num_heads, att_size, hidden_mid, ffn_size] 其中 num_heads 为头数，其他均为维度数
        # PE = [cen, eig, svd] 其中三者均为布尔类型变量，表征这一编码是否选取
        # AT = [spa, edg, pma] 其中三者均为布尔类型变量，表征这一编码是否选取
        # mask 表征是否施加，以及施加何种程度的 mask ， 0 表示不施加 mask ，正值则表示 mask 掉所有 hop 数高于 mask 值的边
        # 经研究，于2022年5月11日将 mask 移入 shape 的末尾
        assert(self.path != '')
        with open(self.path, 'r') as f:
            dic = json.load(f)
        depth = random.choice(dic['depth'])
        layers = []
        for _ in range(0, depth):
            layer = []
            hidden_in = random.choice(dic['hidden_in'])
            num_heads = random.choice(dic['num_heads'])
            att_size = random.choice(dic['att_size'])
            hidden_mid = random.choice(dic['hidden_mid'])
            ffn_size = random.choice(dic['ffn_size'])
            mask = random.choice(dic['mask'])
            layer.append((hidden_in, num_heads, att_size, hidden_mid, ffn_size, mask))
            cen = random.choice([True, False])
            eig = random.choice([True, False])
            svd = random.choice([True, False])
            layer.append((cen, eig, svd))
            spa = random.choice([True, False])
            edg = random.choice([True, False])
            pma = random.choice([True, False])
            layer.append((spa, edg, pma))
            layers.append(tuple(layer))
        return (depth, tuple(layers))

    def to_graphormer(self):
        self.encodings = False


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x, shape=None):
        if shape == None:
            x = self.layer1(x)
            x = self.gelu(x)
            x = self.layer2(x)
        else:
            x[:, :, shape[3]:] = 0
            x = self.layer1(x)
            x[:, :, shape[4]:] = 0
            x = self.gelu(x)
            x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None, shape=None, spatial_pos=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        if shape != None:
            q[:, :, shape[0]:] = 0
            k[:, :, shape[0]:] = 0
            v[:, :, shape[0]:] = 0

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        if shape != None:
            q[:, :, shape[1]:, shape[2]:] = 0
            k[:, :, shape[1]:, shape[2]:] = 0
            v[:, :, shape[1]:, shape[2]:] = 0

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        if shape != None:
            q = q * (shape[2] ** -0.5)
        else:
            q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias
        if shape is not None and spatial_pos is not None:
            spatial_pos_ = spatial_pos.clone()
            spatial_pos_[spatial_pos_ > shape[5]] = -1
            spatial_pos_[spatial_pos_ != -1] = 0
            spatial_pos_[spatial_pos_ != 0] = -1e8
            x = x.permute(0, 2, 3, 1)
            x[:, 1:, 1:] = x[:, 1:, 1:] + spatial_pos_.unsqueeze(-1)
            x = x.permute(0, 3, 1, 2)

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None, shape=None, spatial_pos=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias, shape, spatial_pos)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y, shape)
        y = self.ffn_dropout(y)
        x = x + y
        return x
