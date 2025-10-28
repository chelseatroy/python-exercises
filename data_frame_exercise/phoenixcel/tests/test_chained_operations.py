import os

from phoenixcel.src.dataframe import DataFrame
from phoenixcel.src.groupby import GroupBy


class TestDataFrameChainedOperations:
    def test_chained_operations_with_birds_csv(self):
        """Test complex chained DataFrame operations using birds.csv"""
        csv_path = os.path.join(os.path.dirname(__file__), 'test_birds.csv')

        result = DataFrame.from_csv(csv_path) \
            .assign(weight_float=lambda row: float(row["weight"])) \
            .assign(weight_category=lambda row: 'heavy' if float(row["weight"]) > 4.0 else 'light') \
            .assign(species_category=lambda row: f'{row["species"]} ({row["weight_category"]})') \
            .where(lambda row: row['species'] == 'oriole') \
            .group_by('species_category') \
            .describe_with(
                {'agg': 'aggregate', 'column': 'weight_float', 'using_func': max},
                {'agg': 'average', 'column': 'weight_float'},
                {'agg': 'min', 'column': 'weight_float'}
            )

        # Verify the result is a GroupBy object
        assert isinstance(result, GroupBy)

        # Verify we have the correct groups (only oriole birds with their categories)
        assert 'oriole (heavy)' in result.keys() or 'oriole (light)' in result.keys()

        # Verify the structure contains aggregation results
        for group_data in result.values():
            assert isinstance(group_data, dict)
            # Each group should have the three aggregation results
            assert 'weight_float_max' in group_data
            assert 'weight_float_average' in group_data
            assert 'weight_float_min' in group_data

    def test_chained_operations_with_metrics_csv(self):
        """Test complex chained DataFrame operations using metrics.csv"""
        csv_path = os.path.join(os.path.dirname(__file__), 'test_metrics.csv')

        dataframe = DataFrame.from_csv(csv_path) \
            .assign(year=lambda row: row["Period Start"][-4:]) \
            .assign(activity_year=lambda row: f'{row["Activity"]} ({row["year"]})') \
            .assign(average_days_to_complete_activity=lambda row: float(row["Average Days to Complete Activity"])) \
            .where(lambda row: row['Activity'] == "Alley Grading-Unimproved")
        
        result = dataframe \
            .group_by('activity_year') \
            .describe_with(
                {'agg': 'aggregate', 'column': 'Target Response Days', 'using_func': max},
                {'agg': 'average', 'column': 'average_days_to_complete_activity'},
                {'agg': 'min', 'column': 'Total Completed Requests'}
            )

        # Verify the result is a GroupBy object
        assert isinstance(result, GroupBy)

        # Verify we have groups for different years
        # The data contains years like 2017, 2018
        assert any('2017' in key or '2018' in key for key in result.keys())

        # Verify the structure contains aggregation results
        for group_data in result.values():
            assert isinstance(group_data, dict)
            # Each group should have the three aggregation results
            assert 'Target Response Days_max' in group_data
            assert 'average_days_to_complete_activity_average' in group_data
            assert 'Total Completed Requests_min' in group_data
        
        for col in dataframe.columns:
            print(col)

        # This is deliberately written to fail so that the print statement above
        # prints to standard out and you can see all the columns in this dataframe
        # assert dataframe.columns == None 
