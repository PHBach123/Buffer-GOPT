from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, embed_size, heads):
        super(Attention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, query, keys, values, pad_mask=None):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if pad_mask is not None:
            pad_mask = pad_mask.unsqueeze(-1).expand(N, query_len, key_len)
            pad_mask = pad_mask.unsqueeze(1).repeat(1, self.heads, 1, 1)
            energy = energy.masked_fill(pad_mask == 0, -1e18)

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = Attention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, pad_mask=None):
        attention = self.attention(query, key, value, pad_mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class EncoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(EncoderBlock, self).__init__()
        self.item_embedding = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.ems_embedding = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.ems_on_item = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.item_on_ems = TransformerBlock(embed_size, heads, dropout, forward_expansion)

    def forward(self, item_feature, ems_feature, mask=None):
        item_embedding = self.item_embedding(item_feature, item_feature, item_feature)
        ems_embedding = self.ems_embedding(ems_feature, ems_feature, ems_feature, mask)
        ems_on_item = self.ems_on_item(ems_embedding, item_embedding, item_embedding, mask)
        item_on_ems = self.item_on_ems(item_embedding, ems_embedding, ems_embedding)
        return item_on_ems, ems_on_item

class ActorHead(nn.Module):
    def __init__(
        self,
        preprocess_net: nn.Module,
        embed_size: int,
        num_bins: int = 3,
        k_placement: int = 100,
        padding_mask: bool = False,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.padding_mask = padding_mask
        self.device = device
        self.preprocess = preprocess_net
        self.num_bins = num_bins
        self.k_placement = k_placement
        self.layer_1 = nn.Sequential(
            init_(nn.Linear(embed_size, embed_size)),
            nn.LeakyReLU(),
        )
        self.layer_2 = nn.Sequential(
            init_(nn.Linear(embed_size, embed_size)),
            nn.LeakyReLU(),
        )
        # Đầu ra là num_bins * k_placement
        self.output_layer = nn.Linear(embed_size, k_placement)

    def forward(
        self,
        obs: Union[Dict, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {}
    ) -> Tuple[torch.Tensor, Any]:
        if isinstance(obs, dict):
            obs_tensor = obs["obs"]
            mask = obs["mask"] if self.padding_mask else None
        else:
            obs_tensor = obs
            mask = None
            
        if self.padding_mask:
            mask = torch.as_tensor(obs.mask, dtype=torch.bool, device=self.device)
            mask = torch.sum(mask.reshape(batch_size, -1, 2), dim=-1).bool()
        else:
            mask = None

        batch_size = obs_tensor.shape[0]


        item_embedding, ems_embedding, hidden = self.preprocess(obs_tensor, state, mask)

        item_embedding = self.layer_1(item_embedding)
        ems_embedding = self.layer_2(ems_embedding).permute(0, 2, 1)

        logits = torch.bmm(item_embedding, ems_embedding).reshape(batch_size, -1)

        return logits, hidden

class CriticHead(nn.Module):
    def __init__(
        self,
        k_placement: int,
        preprocess_net: nn.Module,
        embed_size: int,
        num_bins: int = 3,  # Thêm số thùng
        padding_mask: bool = False,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.padding_mask = padding_mask
        self.device = device
        self.preprocess = preprocess_net
        self.k_placement = k_placement
        self.num_bins = num_bins
        self.layer_1 = nn.Sequential(
            init_(nn.Linear(embed_size, embed_size)),
            nn.LeakyReLU(),
        )
        self.layer_2 = nn.Sequential(
            init_(nn.Linear(embed_size, embed_size)),
            nn.LeakyReLU(),
        )
        self.layer_3 = nn.Sequential(
            init_(nn.Linear(2 * embed_size, embed_size)),
            nn.LeakyReLU(),
            init_(nn.Linear(embed_size, embed_size)),
            nn.LeakyReLU(),
            init_(nn.Linear(embed_size, 1))
        )

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor, Dict],
        **kwargs: Any
    ) -> torch.Tensor:
    
        if isinstance(obs, dict):
            obs_tensor = obs["obs"]
            mask = obs["mask"] if self.padding_mask else None
        else:
            obs_tensor = obs
            mask = None

        batch_size = obs_tensor.shape[0]

        if self.padding_mask:
            item_embedding, ems_embedding, _ = self.preprocess(obs_tensor, mask=mask)
        else:
            item_embedding, ems_embedding, _ = self.preprocess(obs_tensor)

        item_embedding = self.layer_1(item_embedding)
        ems_embedding = self.layer_2(ems_embedding)

        item_embedding = torch.sum(item_embedding, dim=-2)
        ems_embedding = torch.sum(ems_embedding * mask[..., None], dim=-2) if mask is not None else torch.sum(ems_embedding, dim=-2)

        joint_embedding = torch.cat((item_embedding, ems_embedding), dim=-1)
        state_value = self.layer_3(joint_embedding)
        return state_value

class ShareNet(nn.Module):
    def __init__(
        self,
        k_placement: int = 100,
        k_buffer: int = 3,
        box_max_size: int = 5,
        container_size: Sequence[int] = [10, 10, 10],
        embed_size: int = 32,
        num_layers: int = 6,
        forward_expansion: int = 4,
        heads: int = 6,
        dropout: float = 0,
        device: Union[str, int, torch.device] = "cpu",
        place_gen: str = "EMS",
        num_bins: int = 3,  # Thêm số thùng
    ) -> None:
        super().__init__()
        self.device = device
        self.k_placement = k_placement
        self.container_size = container_size
        self.place_gen = place_gen
        self.k_buffer = k_buffer
        self.num_bins = num_bins
        if place_gen == "EMS":
            input_size = 6
        else:
            input_size = 3

        self.factor = 1 / max(container_size)
        
        self.item_encoder = nn.Sequential(
            init_(nn.Linear(3, 32)),
            nn.LeakyReLU(),
            init_(nn.Linear(32, embed_size)),
        )
        self.placement_encoder = nn.Sequential(
            init_(nn.Linear(input_size, 32)),
            nn.LeakyReLU(),
            init_(nn.Linear(32, embed_size)),
        )
        
        self.backbone = nn.ModuleList(
            [
                EncoderBlock(
                    embed_size=embed_size,
                    heads=heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        mask: Union[np.ndarray, torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device) * self.factor
        if mask is not None and not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask, dtype=torch.float32, device=self.device)

        # Tách observation cho từng thùng
        obs_hm, obs_next, obs_placements = obs2input(obs, self.container_size, self.place_gen, self.k_buffer, self.num_bins, self.k_placement)
      
        # Mã hóa item và placement
        item_embedding = self.item_encoder(obs_next)  # (batch_size, k_buffer * num_bins, embed_size)
        placement_embedding = self.placement_encoder(obs_placements)  # (batch_size, k_placement * num_bins, embed_size)

        # Áp dụng Transformer cho từng thùng
        for layer in self.backbone:
            item_embedding, placement_embedding = layer(item_embedding, placement_embedding, mask)

        return item_embedding, placement_embedding, state

def obs2input(
    obs: torch.Tensor,
    container_size: Sequence[int],
    place_gen: str = "EMS",
    buffer_size: int = 3,
    num_bins: int = 3,
    k_placement: int = 100
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chuyển đổi observation thành đầu vào cho mạng với multi-bin."""
    if place_gen == "EMS":
      input_size = 6
    else:
      input_size = 3

    batch_size = obs.shape[0]
    area = container_size[0] * container_size[1]
    single_bin_size = area + 6 * buffer_size + input_size * k_placement  # Giả sử k_placement=100
    # Tách observation cho từng thùng
    # print(obs.shape, single_bin_size, area, buffer_size, k_placement)
    hm_list = []
    next_size_list = []
    placements_list = []
    # print('obs:',obs.shape)
    # print('single_bin_size:',single_bin_size)   
    for i in range(num_bins):
        start_idx = i * single_bin_size
        end_hm = start_idx + area
        end_next = end_hm + 6 * buffer_size
        end_placements = start_idx + single_bin_size

        hm = obs[:, start_idx:end_hm].reshape(batch_size, -1, container_size[0], container_size[1])
        next_size = obs[:, end_hm:end_next].reshape(batch_size, -1, 3)  # 2 orientations per item
        placements = obs[:, end_next:end_placements].reshape(batch_size, -1, input_size)

        # print('end_hm:',end_hm)
        # print('end_next:',end_next)
        # print('end_placements:',end_placements)

        # print('hm:',hm)
        # print('next_size:',next_size)   
        # print('placements:',placements)
        # print('--------------------------------------------')
        hm_list.append(hm)
        if i==0:
          next_size_list.append(next_size)
        placements_list.append(placements)

    # Gộp lại
    torch.set_printoptions(threshold=float('inf'))
    hm = torch.cat(hm_list, dim=1)  # (batch_size, num_bins, L, W)
    next_size = torch.cat(next_size_list, dim=1)  # (batch_size, buffer_size * 2, 3)
    placements = torch.cat(placements_list, dim=1)  # (batch_size, num_bins * k_placement, 6 hoặc 3)
    # print('hm:',hm)
    # print('next_size:',next_size)
    # print('placements:',placements)
    # print('hm:',hm.shape)
    # print('next_size:',next_size.shape)
    # print('placements:',placements.shape)
    return hm, next_size, placements

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('leaky_relu'))
