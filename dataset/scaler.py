import torch


def _handle_zeros_in_scale(scale, copy=True, constant_mask=None):
    # if we are fitting on 1D arrays, scale might be a scalar
    if constant_mask is None:
        # Detect near constant values to avoid dividing by a very small
        # value that could lead to surprising results and numerical
        # stability issues.
        constant_mask = scale < 10 * torch.finfo(scale.dtype).eps

    if copy:
        # New array to avoid side-effects
        scale = scale.clone()
    scale[constant_mask] = 1.0
    return scale

class MinMaxScaler:
    _parameter_constraints: dict = {
        "feature_range": [tuple],
        "copy": ["boolean"],
        "clip": ["boolean"],
    }

    def __init__(self, feature_range=(0, 1), *, copy=True, clip=False):
        self.feature_range = feature_range
        self.copy = copy
        self.clip = clip

    def _reset(self):
        if hasattr(self, "scale_"):
            del self.scale_
            del self.min_
            del self.n_samples_seen_
            del self.data_min_
            del self.data_max_
            del self.data_range_

    def fit(self, X):
        self._reset()
        return self.partial_fit(X)

    def partial_fit(self, X):
        feature_range = self.feature_range
        if feature_range[0] >= feature_range[1]:
            raise ValueError("Minimum of desired feature range must be smaller than maximum.")

        data_min = torch.min(X, axis=0)[0]
        data_max = torch.max(X, axis=0)[0]

        self.n_samples_seen_ = X.shape[0]

        data_range = data_max - data_min
        # 避免除以 0
        zero_mask = data_range < 10 * torch.finfo(data_range.dtype).eps
        data_range_safe = data_range.clone()
        data_range_safe[zero_mask] = 1.0

        self.scale_ = (feature_range[1] - feature_range[0]) / data_range_safe
        self.min_ = feature_range[0] - data_min * self.scale_
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range
        return self

    def transform(self, X):
        # ⚠️ 修复：禁用 *= 和 += 原地操作，防止 TTO 梯度反传时报 RuntimeError
        X = X * self.scale_.to(X.device)
        X = X + self.min_.to(X.device)
        if self.clip:
            X = torch.clip(X, self.feature_range[0], self.feature_range[1])
        return X

    def inverse_transform(self, X):
        # ⚠️ 修复：移除错误的切片逻辑，直接依赖 PyTorch 最后一维原生广播机制
        X = X - self.min_.to(X.device)
        X = X / self.scale_.to(X.device)
        return X