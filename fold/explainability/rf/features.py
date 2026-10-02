"""Column names for ``Encoder.forest``, in aggregate-then-histogram order."""
from fold.data import (CLIENT_QUERY, PROXY_QUERY, PROXY_RESPONSE, ROLES,
                       TARGET_RESPONSE, Encoder)


def forest_feature_names(encoder: Encoder) -> tuple[str, ...]:
    """Names for the 33 statistics and the encoder's size-bin counts.

    Time refers to the recorded column values, without assuming clock semantics.
    Histogram bin numbers are zero-based and use ``encoder.size_edges``.
    """
    names = ["n_packets", "total_size", "time_last_minus_first",
             "mean_size", "std_size"]
    roles = {CLIENT_QUERY: "client_query", PROXY_RESPONSE: "proxy_response",
             PROXY_QUERY: "proxy_query", TARGET_RESPONSE: "target_response"}
    for role in ROLES:
        names.extend(f"{roles[role]}_{stat}" for stat in
                     ("count", "mean_size", "std_size", "sum_size",
                      "mean_time_value", "std_time_value"))
    names.extend(("mean_consecutive_time_difference", "std_consecutive_time_difference",
                  "mean_absolute_size_difference", "std_absolute_size_difference"))
    names.extend(f"size_bin_{i:02d}_count" for i in range(encoder.histogram_bins))
    return tuple(names)
