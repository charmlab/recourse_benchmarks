import numpy as np
import pandas as pd

from data.catalog import DataCatalog
from data.catalog.loadData import loadDataset
from data.pipelining import order_data
from methods.catalog.cfrl.model import CFRL


def test_adapter_roundtrip(
    dataset_name: str = "adult",
    model_type: str = "mlp",
    backend: str = "pytorch",
    train_split: float = 0.8,
    n_samples: int = 256,
) -> None:
    """
    Lightweight sanity check for `_ordered_to_cfrl` and `_cfrl_to_ordered`.

    It:
        1) samples rows from df_train,
        2) maps ordered -> CFRL -> ordered and reports reconstruction error,
        3) maps CFRL -> ordered -> CFRL and reports reconstruction error.
    """
    data = DataCatalog(dataset_name, model_type, train_split)

    class DummyMLModel:
        def __init__(self, data_obj):
            self._data = data_obj
            df_train_local = data_obj.df_train
            target_local = data_obj.target
            if target_local in df_train_local.columns:
                feature_cols = [c for c in df_train_local.columns if c != target_local]
            else:
                feature_cols = list(df_train_local.columns)
            self._feature_input_order = feature_cols

        @property
        def data(self):
            return self._data

        @property
        def feature_input_order(self):
            return self._feature_input_order

        @property
        def backend(self):
            return backend

        @property
        def raw_model(self):
            return None

        def predict(self, x):
            raise NotImplementedError

        def predict_proba(self, x):
            raise NotImplementedError

        def get_ordered_features(self, x):
            if isinstance(x, pd.DataFrame):
                return order_data(self._feature_input_order, x)
            return x

    mlmodel = DummyMLModel(data)

    # Do not train CFRL; we only need the metadata and conversion helpers.
    cfrl = CFRL(mlmodel, {"train": False})  # pyright: ignore[reportArgumentType]

    df_train = data.df_train
    target_name = data.target
    if target_name in df_train.columns:
        df_features = df_train.drop(columns=[target_name])
    else:
        df_features = df_train

    if len(df_features) > n_samples:  # pyright: ignore[reportArgumentType]
        sample = df_features.sample(
            n=n_samples, random_state=0
        )  # pyright: ignore[reportOptionalMemberAccess]
    else:
        sample = df_features

    # Ensure correct model input ordering.
    ordered = mlmodel.get_ordered_features(sample).astype(np.float32)

    print("\n=== Original ordered (ML model input) ===")
    print("shape:", ordered.shape)
    print("columns (first 30):", list(ordered.columns)[:30])
    print("dtypes (first 10):", ordered.dtypes.head(10).to_dict())

    # Summarize per feature based on CFRL metadata
    meta = cfrl._metadata  # type: ignore[attr-defined]
    print("\n[ordered] per-feature summary:")
    for idx, long_name in enumerate(meta.feature_names):
        short = meta.long_to_short[long_name]
        attr_type = meta.attr_types[long_name]
        if attr_type in {"numeric-int", "numeric-real", "binary"}:
            col = ordered[short]
            print(
                f"  NUM  long={long_name:20s} short={short:8s} "
                f"type={attr_type:12s} min={col.min():.4g} max={col.max():.4g}"  # pyright: ignore[reportOptionalMemberAccess]
            )
        else:
            cols = cfrl._encoded_cat_columns.get(idx, [])  # type: ignore[attr-defined]
            if not cols:
                print(
                    f"  CAT  long={long_name:20s} short={short:8s} "
                    f"type={attr_type:12s} (no encoded cols found)"
                )
                continue
            block = ordered[cols]
            min_v = float(
                block.min().min()
            )  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
            max_v = float(
                block.max().max()
            )  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
            row_sums = block.sum(axis=1)  # pyright: ignore[reportOptionalMemberAccess]
            print(
                f"  CAT  long={long_name:20s} short={short:8s} "
                f"type={attr_type:12s} cols={len(cols):2d} "
                f"range=[{min_v:.3g}, {max_v:.3g}] "
                f"row_sum_example={row_sums.head(3).to_list()}"
            )

    # ordered -> CFRL -> ordered
    X_zero = cfrl._ordered_to_cfrl(ordered)  # type: ignore[attr-defined]
    print("\n=== X_zero (CFRL internal representation) ===")
    print("shape:", X_zero.shape)
    print("[X_zero] per-feature summary:")
    for idx, long_name in enumerate(meta.feature_names):
        col = X_zero[:, idx]
        attr_type = meta.attr_types[long_name]
        min_v = float(col.min())
        max_v = float(col.max())
        uniq = np.unique(col)
        uniq_preview = uniq[:10]
        if attr_type in {"numeric-int", "numeric-real", "binary"}:
            print(
                f"  NUM  long={long_name:20s} type={attr_type:12s} "
                f"min={min_v:.4g} max={max_v:.4g}"
            )
        else:
            print(
                f"  CAT  long={long_name:20s} type={attr_type:12s} "
                f"min={min_v:.4g} max={max_v:.4g} "
                f"unique_indices(sample)={uniq_preview}"
            )

    ordered_back = cfrl._cfrl_to_ordered(X_zero).astype(np.float32)  # type: ignore[attr-defined]
    ordered_back = ordered_back[ordered.columns]

    print("\n=== ordered_back (after roundtrip) ===")
    print("shape:", ordered_back.shape)  # pyright: ignore[reportOptionalMemberAccess]
    print(
        "columns (first 30):", list(ordered_back.columns)[:30]
    )  # pyright: ignore[reportOptionalMemberAccess]

    diff = np.abs(
        ordered_back.to_numpy() - ordered.to_numpy()
    )  # pyright: ignore[reportOptionalMemberAccess]
    max_abs_diff = float(diff.max())
    mean_abs_diff = float(diff.mean())

    print(
        "[ordered -> CFRL -> ordered] "
        f"max_abs_diff={max_abs_diff:.6g}, mean_abs_diff={mean_abs_diff:.6g}"
    )

    # CFRL -> ordered -> CFRL
    X_zero_roundtrip = cfrl._ordered_to_cfrl(ordered_back)  # type: ignore[attr-defined]
    diff2 = np.abs(X_zero_roundtrip - X_zero)
    max_abs_diff2 = float(diff2.max())
    mean_abs_diff2 = float(diff2.mean())

    print(
        "[CFRL -> ordered -> CFRL] "
        f"max_abs_diff={max_abs_diff2:.6g}, mean_abs_diff={mean_abs_diff2:.6g}"
    )

    raw_dataset = loadDataset(
        data.name,
        return_one_hot=False,
        load_from_cache=True,
        debug_flag=False,
    )
    feature_names = meta.feature_names
    df_raw_long = raw_dataset.data_frame_long[feature_names]  # type: ignore[attr-defined]
    if len(df_raw_long) > n_samples:
        df_raw_sample = df_raw_long.sample(n=n_samples, random_state=1)
    else:
        df_raw_sample = df_raw_long

    print("\n=== Raw long-form data (from loadDataset, non-hot) ===")
    print("shape:", df_raw_sample.shape)
    print("columns (first 30):", feature_names[:30])

    X_zero_raw = cfrl._ordered_to_cfrl(df_raw_sample)  # type: ignore[attr-defined]

    print("\nAdaptor per-feature change:")
    for idx, long_name in enumerate(feature_names):
        attr_type = meta.attr_types[long_name]
        col_raw = df_raw_sample[long_name].to_numpy()
        col_cfrl = X_zero_raw[:, idx]
        min_raw, max_raw = float(np.min(col_raw)), float(np.max(col_raw))
        min_c, max_c = float(col_cfrl.min()), float(col_cfrl.max())

        if attr_type in {"numeric-int", "numeric-real", "binary"}:
            print(
                f"  NUM  long={long_name:20s} type={attr_type:12s} "
                f"raw[min,max]=[{min_raw:.4g}, {max_raw:.4g}] "
                f"cfrl[min,max]=[{min_c:.4g}, {max_c:.4g}]"
            )
        else:
            uniq_raw = np.unique(col_raw)[:10]
            uniq_idx = np.unique(col_cfrl).astype(int)[
                :10
            ]  # pyright: ignore[reportAttributeAccessIssue]
            idx_to_raw = meta.idx_to_raw[long_name]
            recon = np.vectorize(lambda i: idx_to_raw[int(i)])(col_cfrl)
            mismatch_rate = float(np.mean(recon != col_raw))
            mapping_preview = {
                int(i): idx_to_raw[int(i)] for i in uniq_idx if int(i) in idx_to_raw
            }
            print(
                f"  CAT  long={long_name:20s} type={attr_type:12s} "
                f"raw_vals(sample)={uniq_raw} "
                f"indices(sample)={uniq_idx} "
                f"idx->raw(sample)={mapping_preview} "
                f"mismatch_rate={mismatch_rate:.4g}"
            )

    assert abs(max_abs_diff) + abs(max_abs_diff2) < 1e-6


if __name__ == "__main__":
    test_adapter_roundtrip()
