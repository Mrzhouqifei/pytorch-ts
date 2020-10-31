from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Distribution

from pts.core.component import validated
from pts.model import weighted_average
from pts.modules import DistributionOutput, MeanScaler, NOPScaler, FeatureEmbedder


def prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


class LinearSelfAttnSeq(nn.Module):
    def __init__(self, input_size):
        super(LinearSelfAttnSeq, self).__init__()
        self.linear = nn.Sequential(#nn.Linear(input_size, input_size),
                                    #nn.Tanh(),
                                    nn.Linear(input_size, input_size))

    def forward(self, q):
        """
        x = [batch, len, hdim]
        """
        k = v = q
        q = self.linear(q)
        # 计算Q, K的矩阵乘积。 bmm或matmul都可以
        logits = torch.div(torch.bmm(q, k.permute(0, 2, 1)), np.sqrt(q.shape[-1]))
        # 利用softmax将结果归一化。
        weights = nn.functional.softmax(logits, dim=-1)
        # 与V相乘得到加权表示。
        return torch.bmm(weights, v)


class DeepARNetwork(nn.Module):

    @validated()
    def __init__(
        self,
        input_size: int,
        encoder_size: int,
        num_layers: int,
        num_cells: int,
        cell_type: str,
        short_period: int,
        history_length: int,
        context_length: int,    # should be equal to the length of long period
        prediction_length: int,     # should be integer multiples of small period
        distr_output: DistributionOutput,
        dropout_rate: float,
        cardinality: List[int],
        embedding_dimension: List[int],
        lags_seq: List[int],
        scaling: bool = True,
        dtype: np.dtype = np.float32,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.cell_type = cell_type
        self.short_period = short_period
        self.history_length = history_length
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.dropout_rate = dropout_rate
        self.cardinality = cardinality
        self.embedding_dimension = embedding_dimension
        self.num_cat = len(cardinality)
        self.scaling = scaling
        self.dtype = dtype

        self.lags_seq = lags_seq + [0]   # squarenet

        self.distr_output = distr_output
        rnn = {"LSTM": nn.LSTM, "GRU": nn.GRU}[self.cell_type]

        self.long_cell = nn.GRUCell(input_size=input_size, hidden_size=num_cells)
        self.encoder = nn.ModuleList(
            rnn(
                input_size=num_cells,
                hidden_size=num_cells,
                num_layers=num_layers,
                dropout=dropout_rate,
                batch_first=True,
            )
            for _ in range(self.context_length // self.short_period)
        )
        # self.encoder = rnn(
        #         input_size=input_size,
        #         hidden_size=num_cells,
        #         num_layers=num_layers,
        #         dropout=dropout_rate,
        #         batch_first=True,
        #     )
        self.decoder = rnn(
            input_size=encoder_size,
            hidden_size=num_cells,
            num_layers=num_layers,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.out = nn.Linear(num_cells, 1)
        self.criterion = nn.MSELoss()    # l2, l1

        self.target_shape = distr_output.event_shape

        self.proj_distr_args = distr_output.get_args_proj(num_cells)

        self.embedder = FeatureEmbedder(
            cardinalities=cardinality, embedding_dims=embedding_dimension
        )

        if scaling:
            self.scaler = MeanScaler(keepdim=True)
        else:
            self.scaler = NOPScaler(keepdim=True)

    @staticmethod
    def get_lagged_subsequences(
        sequence: torch.Tensor,
        sequence_length: int,
        indices: List[int],
        subsequences_length: int = 1,
    ) -> torch.Tensor:
        """
        Returns lagged subsequences of a given sequence.
        Parameters
        ----------
        sequence : Tensor
            the sequence from which lagged subsequences should be extracted.
            Shape: (N, T, C).
        sequence_length : int
            length of sequence in the T (time) dimension (axis = 1).
        indices : List[int]
            list of lag indices to be used.
        subsequences_length : int
            length of the subsequences to be extracted.
        Returns
        --------
        lagged : Tensor
            a tensor of shape (N, S, C, I), where S = subsequences_length and
            I = len(indices), containing lagged subsequences. Specifically,
            lagged[i, j, :, k] = sequence[i, -indices[k]-S+j, :].
        """
        assert max(indices) + subsequences_length <= sequence_length, (
            f"lags cannot go further than history length, found lag {max(indices)} "
            f"while history length is only {sequence_length}"
        )
        assert all(lag_index >= 0 for lag_index in indices)

        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(sequence[:, begin_index:end_index, ...])
        return torch.stack(lagged_values, dim=-1)

    def unroll_encoder(
        self,
        feat_static_cat: torch.Tensor,  # (batch_size, num_features)
        feat_static_real: torch.Tensor,  # (batch_size, num_features)
        past_time_feat: torch.Tensor,  # (batch_size, history_length, num_features)
        past_target: torch.Tensor,  # (batch_size, history_length, *target_shape)
        past_observed_values: torch.Tensor,  # (batch_size, history_length, *target_shape)
        future_time_feat: Optional[
            torch.Tensor
        ] = None,  # (batch_size, prediction_length, num_features)
        future_target: Optional[
            torch.Tensor
        ] = None,  # (batch_size, prediction_length, *target_shape)
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, List], torch.Tensor, torch.Tensor]:

        time_feat = past_time_feat[
                    :, self.history_length - self.context_length:, ...
                    ]
        sequence = past_target
        sequence_length = self.history_length
        subsequences_length = self.context_length

        lags = self.get_lagged_subsequences(
            sequence=sequence,
            sequence_length=sequence_length,
            indices=self.lags_seq,
            subsequences_length=subsequences_length
        )

        # scale is computed on the context length last units of the past target
        # scale shape is (batch_size, 1, *target_shape)
        _, scale = self.scaler(
            past_target[:, self.context_length:, ...],
            past_observed_values[:, self.context_length:, ...],
        )

        # (batch_size, num_features)
        embedded_cat = self.embedder(feat_static_cat)

        # in addition to embedding features, use the log scale as it can help
        # prediction too
        # (batch_size, num_features + prod(target_shape))
        static_feat = torch.cat(
            (
                embedded_cat,
                feat_static_real,
                scale.log() if len(self.target_shape) == 0 else scale.squeeze(1).log(),
            ),
            dim=1,
        )

        # (batch_size, subsequences_length, num_features + 1)
        repeated_static_feat = static_feat.unsqueeze(1).expand(
            -1, subsequences_length, -1
        )

        # (batch_size, sub_seq_len, *target_shape, num_lags)
        lags_scaled = lags / scale.unsqueeze(-1)

        # from (batch_size, sub_seq_len, *target_shape, num_lags)
        # to (batch_size, sub_seq_len, prod(target_shape) * num_lags)
        input_lags = lags_scaled.reshape(
            (-1, subsequences_length, len(self.lags_seq) * prod(self.target_shape))
        )

        # (batch_size, sub_seq_len, input_dim)
        inputs = torch.cat((input_lags, time_feat, repeated_static_feat), dim=-1)
        # reshape (batch_size, long_period, short_period, input_dim)
        inputs = inputs.reshape(inputs.shape[0], -1, self.short_period, inputs.shape[-1])

        # unroll encoder
        encoder_out = torch.zeros(inputs.shape[0], inputs.shape[2], self.num_cells).to(inputs.device)
        hn = torch.zeros(1, inputs.shape[0], self.num_cells).to(inputs.device)
        cn = torch.zeros(1, inputs.shape[0], self.num_cells).to(inputs.device)
        for i in range(0, inputs.shape[1]):
            hv = self.long_cell(inputs[:, i, :, :].reshape(-1, inputs.shape[-1]),
                                encoder_out.reshape(-1, self.num_cells))
            hv = hv.reshape(inputs.shape[0], self.short_period, -1)
            encoder_out, (hn, cn) = self.encoder[i](hv, (hn, cn))

        # print(encoder_out.shape, time_feat[:, 24:, :].shape)
        query = torch.cat((inputs[:, 0, :, :], encoder_out, future_time_feat, repeated_static_feat[:, -24:, :]), dim=-1)
        outputs, state = self.decoder(query, (hn, cn))
        outputs = self.out(outputs)
        outputs = outputs * scale.unsqueeze(-1)

        # outputs: (batch_size, seq_len, num_cells)
        # state: list of (num_layers, batch_size, num_cells) tensors
        # scale: (batch_size, 1, *target_shape)
        # static_feat: (batch_size, num_features + prod(target_shape))
        return outputs, state, scale, static_feat


class DeepARTrainingNetwork(DeepARNetwork):
    def distribution(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target: torch.Tensor,
        future_observed_values: torch.Tensor,
    ): #-> Distribution:
        rnn_outputs, _, scale, _ = self.unroll_encoder(
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            past_time_feat=past_time_feat,
            past_target=past_target,
            past_observed_values=past_observed_values,
            future_time_feat=future_time_feat,
            future_target=future_target,
        )

        return rnn_outputs.squeeze()

    def forward(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target: torch.Tensor,
        future_observed_values: torch.Tensor,
    ) -> torch.Tensor:
        distr = self.distribution(
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            past_time_feat=past_time_feat,
            past_target=past_target,
            past_observed_values=past_observed_values,
            future_time_feat=future_time_feat,
            future_target=future_target,
            future_observed_values=future_observed_values,
        )

        # put together target sequence
        # (batch_size, seq_len, *target_shape)
        target = torch.cat(
            (
                future_target,
            ),
            dim=1,
        )
        loss = self.criterion(distr, target)
        return loss


class DeepARPredictionNetwork(DeepARNetwork):
    def __init__(self, num_parallel_samples: int = 100, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_parallel_samples = num_parallel_samples

    # noinspection PyMethodOverriding,PyPep8Naming
    def forward(
        self,
        feat_static_cat: torch.Tensor,  # (batch_size, num_features)
        feat_static_real: torch.Tensor,  # (batch_size, num_features)
        past_time_feat: torch.Tensor,  # (batch_size, history_length, num_features)
        past_target: torch.Tensor,  # (batch_size, history_length, *target_shape)
        past_observed_values: torch.Tensor,  # (batch_size, history_length, *target_shape)
        future_time_feat: torch.Tensor,  # (batch_size, prediction_length, num_features)
    ) -> torch.Tensor:
        """
        Predicts samples, all tensors should have NTC layout.
        Parameters
        ----------
        feat_static_cat : (batch_size, num_features)
        feat_static_real : (batch_size, num_features)
        past_time_feat : (batch_size, history_length, num_features)
        past_target : (batch_size, history_length, *target_shape)
        past_observed_values : (batch_size, history_length, *target_shape)
        future_time_feat : (batch_size, prediction_length, num_features)

        Returns
        -------
        Tensor
            Predicted samples
        """

        rnn_outputs, _, scale, _ = self.unroll_encoder(
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            past_time_feat=past_time_feat,
            past_target=past_target,
            past_observed_values=past_observed_values,
            future_time_feat=future_time_feat,
        )
        # (batch_size, num_samples, prediction_length)
        return rnn_outputs.reshape(rnn_outputs.shape[0], 1, self.prediction_length)



