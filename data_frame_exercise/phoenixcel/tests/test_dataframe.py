import os
import pytest

from phoenixcel.src.dataframe import DataFrame
from phoenixcel.src.series import Series


class TestDataFrameInit:
    def test_dataframe_initialization(self):
        df = DataFrame()
        assert type(df) is DataFrame

class TestDataFrameFromCSV:
    def test_from_csv_creates_dataframe(self):
        csv_path = os.path.join(os.path.dirname(__file__), 'test_birds.csv')
        df = DataFrame.from_csv(csv_path)
        assert df.shape == (3, 21)  # 3 columns, 21 rows
        assert 'species' in df.columns
        assert 'specimen_id' in df.columns
        assert 'weight' in df.columns

class TestDataFrameFromRows:
    def test_from_rows_empty_creates_dataframe(self):
        rows = []
        df = DataFrame.from_rows(rows)
        assert len(df.columns) == 0
        assert df.columns == []

    def test_from_rows_with_data_creates_dataframe(self):
        rows = [
            {'name': 'Alice', 'age': '30'},
            {'name': 'Bob', 'age': '25'}
        ]
        df = DataFrame.from_rows(rows)
        assert len(df.columns) == 2
        assert 'name' in df.columns
        assert 'age' in df.columns

    def test_from_rows_creates_attributes(self):
        rows = [
            {'name': 'Alice', 'age': '30'},
            {'name': 'Bob', 'age': '25'}
        ]
        df = DataFrame.from_rows(rows)
        assert df["name"] == Series(['Alice', 'Bob'])
        assert df["age"] == Series(['30', '25'])


class TestDataFrameFromDictionary:
    def test_from_dictionary_empty_creates_dataframe(self):
        data = {}
        df = DataFrame.from_dictionary(data)
        assert len(df.columns) == 0
        assert df.columns == []

    def test_from_dictionary_creates_dataframe(self):
        data = {
            'name': ['Alice', 'Bob'],
            'age': [30, 25]
        }
        df = DataFrame.from_dictionary(data)
        assert len(df.columns) == 2
        assert df.shape == (2, 2)


class TestDataFrameProperties:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.df = DataFrame.from_rows([
            {'name': 'Alice', 'age': '30'},
            {'name': 'Bob', 'age': '25'}
        ])

    def test_shape_property(self):
        assert self.df.shape == (2, 2)

    def test_columns_property(self):
        assert len(self.df.columns) == 2
        assert 'name' in self.df.columns


class TestDataFrameGetItem:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.df = DataFrame.from_rows([
            {'name': 'Alice', 'age': '30'},
            {'name': 'Bob', 'age': '25'}
        ])

    def test_getitem_returns_series(self):
        name_column = self.df['name']
        assert isinstance(name_column, Series)

    def test_getitem_returns_correct_values(self):
        name_column = self.df['name']
        assert 'Alice' in name_column
        assert 'Bob' in name_column


class TestDataFrameSetItem:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.df = DataFrame.from_rows([
            {'name': 'Alice', 'age': '30'},
            {'name': 'Bob', 'age': '25'}
        ])

    def test_setitem_adds_new_column(self):
        new_series = Series(['Engineer', 'Designer'])
        self.df['job'] = new_series
        assert 'job' in self.df.columns

    def test_setitem_creates_attribute(self):
        new_series = Series(['Engineer', 'Designer'])
        self.df['Job Title'] = new_series
        assert self.df['Job Title'] == new_series


class TestDataFrameWhere:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.df = DataFrame.from_rows([
            {'name': 'Alice', 'age': '30'},
            {'name': 'Bob', 'age': '25'},
            {'name': 'Charlie', 'age': '35'}
        ])

    def test_where_filters_rows(self):
        filtered = self.df.where(lambda row: int(row['age']) > 27)
        assert filtered.shape[1] == 2  # 2 rows should match


class TestDataFrameAssign:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.df = DataFrame.from_rows([
            {'name': 'Alice', 'age': '30'},
            {'name': 'Bob', 'age': '25'}
        ])

    def test_assign_adds_computed_column(self):
        result = self.df.assign(age_doubled=lambda row: int(row['age']) * 2)
        assert 'age_doubled' in result.columns

class TestDataFrameAppend:
    def test_append_same_columns_add_rows(self):
        first_df = DataFrame.from_rows([
            {"name" : "Alice", "age": 30},
            {"name" : "Bob", "age": 25},
        ])
        update_df = DataFrame.from_rows([
            {"name" : "Chelsea", "age": 35},
            {"name" : "Terry", "age": 61},
        ])
        result = first_df.append(update_df)

        # number of rows
        assert result.shape[1] == 4
        assert "name" in result.columns
        assert "age" in result.columns
        assert result["name"] == Series(["Alice", "Bob", "Chelsea", "Terry"])
        assert result["age"] == Series([30, 25,35, 61])

class TestDataFrameGroupBy:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.df = DataFrame.from_rows([
            {'department': 'Engineering', 'salary': '100000'},
            {'department': 'Engineering', 'salary': '120000'},
            {'department': 'Sales', 'salary': '80000'}
        ])

    def test_group_by_creates_groups(self):
        grouped = self.df.group_by('department')
        assert 'Engineering' in grouped.keys()
        assert 'Sales' in grouped.keys()

    def test_group_by_groups_correct_items(self):
        grouped = self.df.group_by('department')
        assert len(grouped['Engineering']) == 2
        assert len(grouped['Sales']) == 1