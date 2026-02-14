from typing import Any

Record = dict[str, Any]
State = dict[str, Any]
FeatureVector = list[float]
FeatureMatrix = list[FeatureVector]

AdapterStats = dict[str, int]
FeatureStats = dict[str, int]
CsvStats = dict[str, int]
SplitStats = dict[str, int]
RegressionMetrics = dict[str, float]
EvalPolicy = dict[str, float]
