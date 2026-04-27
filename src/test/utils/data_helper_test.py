import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from pytest import fail
from src.utils import data_helper

"""
Run: python -m src.test.utils.data_helper_test
"""


test_apply_dataframe_filters_data = [
    # In the form of (raw_dataframe, filter, expected_filtered_dataframe)
    # Single-column filter
    (
        pd.DataFrame({
            "hidden_size": [[6], [8], [6], [10]],
            "ensemble_size": [2, 4, 6, 12],
            "mse": [0.8, 0.7, 0.85, 0.9]
        }),
        {"hidden_size": [[6]]},
        pd.DataFrame({
            "hidden_size": [[6], [6]],
            "ensemble_size": [2, 6],
            "mse": [0.8, 0.85]
        })
    ),
    # Multiple-columns filter
    (
        pd.DataFrame({
            "hidden_size": [[6], [6], [8], [10]],
            "ensemble_size": [2, 6, 6, 12],
            "mse": [0.8, 0.85, 0.75, 0.9]
        }),
        {"hidden_size": [[6]], "ensemble_size": [2, 6]},
        pd.DataFrame({
            "hidden_size": [[6], [6]],
            "ensemble_size": [2, 6],
            "mse": [0.8, 0.85]
        })
    ),
    # No match
    (
        pd.DataFrame({
            "hidden_size": [[6], [8]],
            "ensemble_size": [2, 4],
            "mse": [0.8, 0.7]
        }),
        {"hidden_size": [[10]]},
        pd.DataFrame(columns=["hidden_size", "ensemble_size", "mse"])
    ),
    # All rows match
    (
        pd.DataFrame({
            "hidden_size": [[6], [6]],
            "ensemble_size": [2, 6],
            "mse": [0.8, 0.85]
        }),
        {"hidden_size": [[6]], "ensemble_size": [2, 6]},
        pd.DataFrame({
            "hidden_size": [[6], [6]],
            "ensemble_size": [2, 6],
            "mse": [0.8, 0.85]
        })
    ),
    # Float column filtering
    (
        pd.DataFrame({
            "learning_rate": [0.01, 0.001, 0.01, 0.1],
            "mse": [0.8, 0.7, 0.85, 0.9]
        }),
        {"learning_rate": [0.01]},
        pd.DataFrame({
            "learning_rate": [0.01, 0.01],
            "mse": [0.8, 0.85]
        })
    )
]

@pytest.mark.skip(f"Need to fix bug due to hidden_size is a list in csv")
@pytest.mark.parametrize("raw_dataframe, filter, expected_filtered_dataframe", test_apply_dataframe_filters_data)
def test_apply_dataframe_filters(raw_dataframe: pd.DataFrame, filter: dict, expected_filtered_dataframe: pd.DataFrame):
    actual_filtered_dataframe = data_helper.apply_dataframe_filter(
        df=raw_dataframe, 
        filter=filter
    ).sort_values(by=list(raw_dataframe.columns)).reset_index(drop=True)
    expected_filtered_dataframe = expected_filtered_dataframe.sort_values(by=list(raw_dataframe.columns)).reset_index(drop=True)

    # Note that row order is not taken into account -> we are only comparing the content of the 2 dataframes

    try:
        data_helper.assert_frame_equal_ignore_order(actual_filtered_dataframe, expected_filtered_dataframe, list_columns=["hidden_size"])
    except Exception as e:
        fail(f"Expected {expected_filtered_dataframe} but received: {actual_filtered_dataframe}")


if __name__ == "__main__":
    # Below run all test cases on all test files
    # pytest.main()

    # Below run all test cases on this test file only
    pytest.main([__file__])
