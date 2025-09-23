"""촉각 데이터를 위한 TimeSeries Transformer 오토인코더 유틸리티."""

from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction


def _resolve_device(device: str | None) -> torch.device:
    """가능하면 CUDA를 선택하고, 아니면 CPU를 선택한다."""

    resolved = device or "auto"
    if resolved == "auto":
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(resolved)


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """촉각 모델 체크포인트를 로드하고 필수 키를 검증한다."""

    checkpoint = torch.load(path, map_location=device)
    if "config" not in checkpoint or "state_dict" not in checkpoint:
        raise ValueError("체크포인트에 config/state_dict 키가 필요합니다.")
    return checkpoint


def _select_hidden_from_outputs(outputs: Any) -> torch.Tensor:
    """모델 출력 구조에서 인코더 히든 스테이트를 선택한다."""

    if (
        hasattr(outputs, "encoder_last_hidden_state")
        and outputs.encoder_last_hidden_state is not None
    ):
        return outputs.encoder_last_hidden_state
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    if hasattr(outputs, "hidden_states") and outputs.hidden_states:
        return outputs.hidden_states[-1]
    raise RuntimeError("인코더 히든 스테이트를 찾을 수 없습니다.")


class TimeSeriesAutoencoder(nn.Module):
    """TimeSeriesTransformerForPrediction을 감싸 재구성 기능을 제공한다."""

    def __init__(self, config: TimeSeriesTransformerConfig):
        """체크포인트 설정으로부터 오토인코더 구성요소를 초기화한다."""

        super().__init__()

        # Transformer 본체 구성
        self.model = TimeSeriesTransformerForPrediction(config)
        self.d_model = getattr(config, "d_model", None) or getattr(
            self.model, "d_model", 128
        )
        self.reconstruction_head = nn.Linear(self.d_model, config.input_size)
        self.force_decode_head: bool = False
        self.use_direct_reconstruction: bool = True

        decoder_layers: int = int(getattr(config, "ae_decoder_layers", 2))
        decoder_hidden_mult: float = float(
            getattr(config, "ae_decoder_hidden_mult", 2.0)
        )
        decoder_dropout: float = float(getattr(config, "ae_decoder_dropout", 0.0))
        hidden_dim: int = max(1, int(self.d_model * decoder_hidden_mult))
        in_dim: int = self.d_model * 2
        mlp_layers = []
        if decoder_layers <= 1:
            mlp_layers.extend(
                [
                    nn.Linear(in_dim, self.d_model),
                    nn.GELU(),
                    nn.Dropout(decoder_dropout),
                ]
            )
        else:
            mlp_layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(decoder_dropout),
                ]
            )
            for _ in range(max(0, decoder_layers - 2)):
                mlp_layers.extend(
                    [
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.Dropout(decoder_dropout),
                    ]
                )
            mlp_layers.extend(
                [
                    nn.Linear(hidden_dim, self.d_model),
                    nn.GELU(),
                    nn.Dropout(decoder_dropout),
                ]
            )
        self.decoder_mlp = nn.Sequential(*mlp_layers)

    def _encoder_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """트랜스포머를 실행해 인코더 히든 스테이트를 추출한다."""

        t_past, _, mask = self._build_aux_features(x)
        outputs = self.model(
            past_values=x,
            past_time_features=t_past,
            past_observed_mask=mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return _select_hidden_from_outputs(outputs)

    def _build_aux_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """트랜스포머 입력에 필요한 시간 특징과 마스크를 생성한다."""

        b, seq, feat = x.shape
        device = x.device
        t_past = torch.zeros(b, seq, 1, dtype=x.dtype, device=device)
        t_future = torch.zeros(b, seq, 1, dtype=x.dtype, device=device)
        mask = torch.ones(b, seq, feat, dtype=torch.bool, device=device)
        return t_past, t_future, mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """배치에 대한 재구성 손실을 계산한다."""

        hidden = self._encoder_hidden(x)

        if self.use_direct_reconstruction:
            x_hat = self.reconstruction_head(hidden)
        else:
            z = hidden.mean(dim=1)
            x_hat = self._decode_from_latent(z, sequence_length=x.shape[1])

        return F.mse_loss(x_hat, x)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """그래디언트 없이 시퀀스를 잠재 표현으로 변환한다."""

        hidden = self._encoder_hidden(x)
        return hidden.mean(dim=1)

    @torch.no_grad()
    def decode(self, latent_vector: torch.Tensor, sequence_length: int) -> torch.Tensor:
        """잠재 벡터를 입력 공간으로 복원한다."""

        return self._decode_from_latent(latent_vector, sequence_length)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """전방 연산과 동일하게 입력을 재구성하되 손실 계산은 생략한다."""

        hidden = self._encoder_hidden(x)
        if self.use_direct_reconstruction:
            return self.reconstruction_head(hidden)
        z = hidden.mean(dim=1)
        return self._decode_from_latent(z, sequence_length=x.shape[1])

    def _positional_encoding(
        self, length: int, d_model: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """디코더 경로에서 사용할 사인/코사인 위치 인코딩을 만든다."""

        position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device, dtype=dtype)
            * (-np.log(10000.0) / d_model)
        )
        pe = torch.zeros(length, d_model, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _decode_from_latent(
        self, latent_vector: torch.Tensor, sequence_length: int
    ) -> torch.Tensor:
        """잠재 상태와 위치 인코딩을 결합해 디코딩한다."""

        b, d = latent_vector.shape
        device = latent_vector.device
        dtype = latent_vector.dtype
        pos = self._positional_encoding(sequence_length, d, device, dtype)
        z_seq = latent_vector.unsqueeze(1).expand(b, sequence_length, d)
        pos_seq = pos.unsqueeze(0).expand(b, sequence_length, d)
        h_in = torch.cat([z_seq, pos_seq], dim=-1)
        h = self.decoder_mlp(h_in)
        x_hat = self.reconstruction_head(h)
        return x_hat


def _build_model_from_checkpoint(
    checkpoint: dict[str, Any], device: torch.device
) -> "TimeSeriesAutoencoder":
    """체크포인트 내용을 이용해 :class:`TimeSeriesAutoencoder`를 복원한다."""

    config = TimeSeriesTransformerConfig(**checkpoint["config"])
    model = TimeSeriesAutoencoder(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model


class TactileEncoder:
    """디바이스와 시퀀스 길이 제한을 추적하는 편의 래퍼."""

    def __init__(
        self,
        *,
        checkpoint: Path,
        device: torch.device,
        model: TimeSeriesAutoencoder,
        sequence_length: int | None,
    ) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.model = model
        self.sequence_length = sequence_length

    def encode(
        self, sequence: torch.Tensor, requires_grad: bool = False
    ) -> torch.Tensor:
        """디바이스 배치와 길이 제한을 적용하며 촉각 시퀀스를 인코딩한다."""

        # 배치 차원 보정
        if sequence.dim() == 2:
            sequence = sequence.unsqueeze(0)

        # 모델 디바이스로 이동
        sequence = sequence.to(self.device)

        # 최대 길이 초과 시 잘라내기
        if self.sequence_length is not None and sequence.size(1) > self.sequence_length:
            sequence = sequence[:, : self.sequence_length]

        if requires_grad:
            return self.model.encode(sequence)
        with torch.no_grad():
            return self.model.encode(sequence)


def load_tactile_encoder(
    checkpoint_path: Path,
    device: str | None = None,
    sequence_length: int | None = None,
) -> TactileEncoder:
    """촉각 오토인코더 체크포인트를 로드해 사용 가능한 래퍼를 반환한다.

    Parameters
    ----------
    checkpoint_path:
        촉각 학습 파이프라인이 생성한 체크포인트 경로.
    device:
        실행할 디바이스. ``None`` 또는 ``"auto"``이면 가능한 경우 CUDA를 선택한다.
    sequence_length:
        인코딩 시 적용할 최대 시퀀스 길이. 초과 부분은 잘라낸다.

    Returns
    -------
    TactileEncoder
        로드된 모델과 디바이스, 길이 제한 정보를 포함한 래퍼.

    Raises
    ------
    ValueError
        체크포인트에 ``config`` 혹은 ``state_dict`` 키가 없을 때 발생한다.
    """

    resolved_device = _resolve_device(device)
    checkpoint = _load_checkpoint(checkpoint_path, resolved_device)
    model = _build_model_from_checkpoint(checkpoint, resolved_device)

    return TactileEncoder(
        checkpoint=checkpoint_path,
        device=resolved_device,
        model=model,
        sequence_length=sequence_length,
    )
